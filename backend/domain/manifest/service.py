"""Manifest engine - file walk, ignore handling, language detect, classification, delta."""
import fnmatch
import hashlib
import os
from pathlib import Path

from infrastructure.db.database import get_db
from shared.errors import NotFoundError
from shared.utils import new_id

from .types import (
    BuildManifestRequest,
    BuildManifestResponse,
    FileCategory,
    ManifestFile,
    ManifestFileContentResponse,
    ManifestPreviewResponse,
    ManifestTreeNode,
    ManifestTreeResponse,
)

_DEFAULT_IGNORES = [
    ".git/**", "node_modules/**", ".venv/**", "venv/**", "dist/**", "build/**", "target/**",
    "__pycache__/**", ".mypy_cache/**", ".pytest_cache/**", ".idea/**", ".vscode/**",
]

_LANG_BY_EXT = {
    ".py": "python", ".ts": "typescript", ".tsx": "typescript", ".js": "javascript", ".jsx": "javascript",
    ".java": "java", ".go": "go", ".rs": "rust", ".cpp": "cpp", ".cc": "cpp", ".c": "c", ".h": "c",
    ".cs": "csharp", ".rb": "ruby", ".php": "php", ".swift": "swift", ".kt": "kotlin",
    ".sql": "sql", ".sh": "shell", ".yaml": "yaml", ".yml": "yaml", ".json": "json", ".toml": "toml", ".ini": "ini", ".md": "markdown",
}

_SECRET_FILE_HINTS = [".env", "id_rsa", "id_ed25519", "credentials", "secret", "token", "key"]
_ASSET_EXT = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".mp4", ".mp3", ".wav", ".pdf", ".zip", ".tar", ".gz"}


def _is_binary(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            data = f.read(2048)
        if b"\x00" in data:
            return True
        try:
            data.decode("utf-8")
            return False
        except UnicodeDecodeError:
            return True
    except Exception:
        return True


def _checksum(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _language(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in _LANG_BY_EXT:
        return _LANG_BY_EXT[ext]
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            first = f.readline().strip()
        if first.startswith("#!"):
            if "python" in first:
                return "python"
            if "node" in first or "deno" in first:
                return "javascript"
            if "bash" in first or "sh" in first:
                return "shell"
    except Exception:
        pass
    return None


def _category(rel: str, path: Path) -> FileCategory:
    low = rel.lower()
    name = path.name.lower()
    ext = path.suffix.lower()

    if any(h in low for h in _SECRET_FILE_HINTS):
        return FileCategory.SECRET_RISK
    if ext in _ASSET_EXT:
        return FileCategory.ASSET
    if low.endswith(".min.js") or "generated" in low or low.endswith(".lock"):
        return FileCategory.GENERATED
    if "migration" in low or low.startswith("migrations/"):
        return FileCategory.MIGRATION
    if low.startswith("test/") or low.startswith("tests/") or "_test." in low or ".spec." in low:
        return FileCategory.TEST
    if low.startswith("docs/") or ext in {".md", ".rst"}:
        return FileCategory.DOCS
    if name in {"dockerfile", "makefile"} or low.startswith(".github/") or low.startswith("infra/"):
        return FileCategory.INFRA
    if ext in {".json", ".yaml", ".yml", ".toml", ".ini", ".env", ".cfg", ".xml"}:
        return FileCategory.CONFIG
    if ext in _LANG_BY_EXT:
        return FileCategory.SOURCE
    return FileCategory.OTHER


def _compile_ignores(repo_ignores: list[str]) -> tuple[list[str], list[str]]:
    effective = [*_DEFAULT_IGNORES, *repo_ignores]
    dir_prefixes = [p[:-3] for p in effective if p.endswith("/**")]
    return effective, dir_prefixes


def _ignored(rel: str, pats: list[str], dir_prefixes: list[str], is_dir: bool) -> bool:
    if is_dir:
        return any(rel == prefix or rel.startswith(f"{prefix}/") for prefix in dir_prefixes)
    return any(fnmatch.fnmatch(rel, p) for p in pats)


class ManifestService:
    async def build(self, req: BuildManifestRequest) -> BuildManifestResponse:
        db = get_db()
        async with db.execute("SELECT * FROM repo_snapshots WHERE id=?", (req.snapshot_id,)) as cur:
            snap = await cur.fetchone()
        if snap is None:
            raise NotFoundError("RepoSnapshot", req.snapshot_id)

        repo_id = snap["local_repo_id"]
        root = Path(snap["local_path"])
        if not root.exists() or not root.is_dir():
            raise ValueError("Snapshot path does not exist")

        async with db.execute("SELECT ignore_overrides FROM local_repos WHERE id=?", (repo_id,)) as cur:
            repo_row = await cur.fetchone()
        repo_ignores = []
        if repo_row and repo_row["ignore_overrides"]:
            import json
            try:
                repo_ignores = json.loads(repo_row["ignore_overrides"])
            except Exception:
                repo_ignores = []

        pats, dir_prefixes = _compile_ignores(repo_ignores)

        previous: dict[str, tuple[str, int, int]] = {}
        async with db.execute(
            """
            SELECT rel_path, checksum, size_bytes, mtime_ns
            FROM manifest_files
            WHERE snapshot_id = (
              SELECT id FROM repo_snapshots
              WHERE local_repo_id=? AND id<>?
              ORDER BY created_at DESC LIMIT 1
            )
            """,
            (repo_id, req.snapshot_id),
        ) as cur:
            for row in await cur.fetchall():
                previous[row["rel_path"]] = (row["checksum"], row["size_bytes"], row["mtime_ns"])

        await db.execute("DELETE FROM manifest_files WHERE snapshot_id=?", (req.snapshot_id,))

        total = ignored = new_files = changed = unchanged = 0

        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = os.path.relpath(dirpath, root).replace("\\", "/")
            if rel_dir == ".":
                rel_dir = ""

            keep_dirs: list[str] = []
            for d in dirnames:
                rel = f"{rel_dir}/{d}" if rel_dir else d
                if _ignored(rel, pats, dir_prefixes, True):
                    ignored += 1
                    continue
                keep_dirs.append(d)
            dirnames[:] = keep_dirs

            for name in filenames:
                rel = f"{rel_dir}/{name}" if rel_dir else name
                if _ignored(rel, pats, dir_prefixes, False):
                    ignored += 1
                    continue

                p = root / rel
                if not p.exists() or not p.is_file():
                    continue
                if _is_binary(p):
                    ignored += 1
                    continue

                st = p.stat()
                checksum = _checksum(p)
                size = int(st.st_size)
                mtime_ns = int(st.st_mtime_ns)

                prev = previous.get(rel)
                if prev is None:
                    new_files += 1
                elif prev == (checksum, size, mtime_ns):
                    unchanged += 1
                else:
                    changed += 1

                lang = _language(p)
                cat = _category(rel, p)

                await db.execute(
                    """
                    INSERT INTO manifest_files
                    (id, snapshot_id, rel_path, language, category, size_bytes, mtime_ns, checksum)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (new_id(), req.snapshot_id, rel, lang, cat.value, size, mtime_ns, checksum),
                )
                total += 1

        await db.commit()

        return BuildManifestResponse(
            snapshot_id=req.snapshot_id,
            total_files=total,
            new_files=new_files,
            changed_files=changed,
            unchanged_files=unchanged,
            ignored_files=ignored,
        )

    async def preview(self, snapshot_id: str, limit: int = 200) -> ManifestPreviewResponse:
        db = get_db()
        async with db.execute(
            "SELECT * FROM manifest_files WHERE snapshot_id=? ORDER BY rel_path ASC LIMIT ?",
            (snapshot_id, limit),
        ) as cur:
            rows = await cur.fetchall()

        files = [
            ManifestFile(
                id=r["id"],
                snapshot_id=r["snapshot_id"],
                rel_path=r["rel_path"],
                language=r["language"],
                category=r["category"],
                size_bytes=r["size_bytes"],
                mtime_ns=r["mtime_ns"],
                checksum=r["checksum"],
            )
            for r in rows
        ]
        return ManifestPreviewResponse(snapshot_id=snapshot_id, files=files)

    async def tree(self, snapshot_id: str, limit: int = 5000) -> ManifestTreeResponse:
        db = get_db()
        async with db.execute(
            "SELECT rel_path FROM manifest_files WHERE snapshot_id=? ORDER BY rel_path ASC LIMIT ?",
            (snapshot_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        nodes: list[ManifestTreeNode] = []
        seen: set[str] = set()
        for r in rows:
            rel = r["rel_path"]
            parts = rel.split("/")
            prefix = ""
            for i, part in enumerate(parts):
                prefix = f"{prefix}/{part}" if prefix else part
                is_dir = i < len(parts) - 1
                key = f"{prefix}|{is_dir}"
                if key in seen:
                    continue
                seen.add(key)
                nodes.append(ManifestTreeNode(path=prefix, is_dir=is_dir))
        return ManifestTreeResponse(snapshot_id=snapshot_id, nodes=nodes)

    async def read_file(self, snapshot_id: str, rel_path: str, max_bytes: int = 200_000) -> ManifestFileContentResponse:
        db = get_db()
        async with db.execute("SELECT local_path FROM repo_snapshots WHERE id=?", (snapshot_id,)) as cur:
            snap = await cur.fetchone()
        if snap is None:
            raise NotFoundError("RepoSnapshot", snapshot_id)

        root = Path(snap["local_path"]).resolve()
        target = (root / rel_path).resolve()
        if root not in target.parents and target != root:
            raise ValueError("Invalid file path")
        if not target.exists() or not target.is_file():
            raise ValueError("File not found in snapshot")

        data = target.read_bytes()
        truncated = len(data) > max_bytes
        if truncated:
            data = data[:max_bytes]
        text = data.decode("utf-8", errors="replace")
        return ManifestFileContentResponse(
            snapshot_id=snapshot_id,
            rel_path=rel_path,
            content=text,
            truncated=truncated,
        )
