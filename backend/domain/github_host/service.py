"""GitHubService — OAuth device flow, token storage, and repository discovery."""
import os
from datetime import datetime, timezone

import httpx

from infrastructure.db.database import get_db
from shared.errors import NotFoundError
from shared.logger import logger

from .types import (
    DeviceFlowPollResult,
    DeviceFlowStart,
    GitHubAccount,
    GitHubRepo,
    GitHubRepoListResponse,
)

_GITHUB_OAUTH = "https://github.com"
_GITHUB_API = "https://api.github.com"
_REPOS_PER_PAGE = 30

# GitHub OAuth App client_id (set CODESPECTRA_GITHUB_CLIENT_ID in environment).
# Create an OAuth App at https://github.com/settings/developers, enable Device Flow,
# and set the callback URL to http://localhost (unused for device flow).
_CLIENT_ID = os.getenv("CODESPECTRA_GITHUB_CLIENT_ID", "")


def _get_client_id() -> str:
    if not _CLIENT_ID:
        raise ValueError(
            "GitHub integration is not configured. "
            "Set the CODESPECTRA_GITHUB_CLIENT_ID environment variable. "
            "Create an OAuth App at https://github.com/settings/developers "
            "and enable Device Flow."
        )
    return _CLIENT_ID


class GitHubService:
    # ──────────────────────────────────────────────────────────────────────
    # OAuth Device Flow
    # ──────────────────────────────────────────────────────────────────────

    async def start_device_flow(self) -> DeviceFlowStart:
        """Step 1: request a device code from GitHub."""
        client_id = _get_client_id()
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{_GITHUB_OAUTH}/login/device/code",
                data={"client_id": client_id, "scope": "repo read:user"},
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()

        if "error" in data:
            raise ValueError(f"GitHub device flow error: {data.get('error_description', data['error'])}")

        logger.info("GitHub device flow started")
        return DeviceFlowStart(
            device_code=data["device_code"],
            user_code=data["user_code"],
            verification_uri=data["verification_uri"],
            expires_in=data.get("expires_in", 900),
            interval=data.get("interval", 5),
        )

    async def poll_device_flow(self, device_code: str) -> DeviceFlowPollResult:
        """Step 2: poll GitHub OAuth to check if the user has authorized.

        Returns immediately with the current status — the frontend drives the
        polling loop and handles the interval.
        """
        client_id = _get_client_id()
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{_GITHUB_OAUTH}/login/oauth/access_token",
                data={
                    "client_id": client_id,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()

        if "access_token" in data:
            token = data["access_token"]
            account = await self._fetch_and_save_account(token)
            logger.info(f"GitHub device flow succeeded — connected as {account.login}")
            return DeviceFlowPollResult(status="success", account=account)

        error = data.get("error", "unknown")
        status_map = {
            "authorization_pending": "pending",
            "slow_down": "slow_down",
            "expired_token": "expired",
            "access_denied": "denied",
        }
        status = status_map.get(error, "error")
        logger.debug(f"GitHub device flow poll: {status} ({error})")
        return DeviceFlowPollResult(status=status)

    # ──────────────────────────────────────────────────────────────────────
    # Account management
    # ──────────────────────────────────────────────────────────────────────

    async def get_account(self) -> GitHubAccount | None:
        db = get_db()
        async with db.execute("SELECT * FROM github_accounts LIMIT 1") as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return self._row_to_account(row)

    async def disconnect(self) -> None:
        db = get_db()
        await db.execute("DELETE FROM github_accounts")
        await db.commit()
        logger.info("GitHub account disconnected")

    # ──────────────────────────────────────────────────────────────────────
    # Repository discovery
    # ──────────────────────────────────────────────────────────────────────

    async def list_repos(
        self,
        query: str | None = None,
        page: int = 1,
    ) -> GitHubRepoListResponse:
        token = await self._get_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        async with httpx.AsyncClient(timeout=20) as client:
            if query and query.strip():
                resp = await client.get(
                    f"{_GITHUB_API}/search/repositories",
                    params={
                        "q": f"{query.strip()} user:{await self._get_login()}",
                        "per_page": _REPOS_PER_PAGE,
                        "page": page,
                        "sort": "updated",
                    },
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                items = data.get("items", [])
                total = data.get("total_count", 0)
                has_more = (page * _REPOS_PER_PAGE) < total
            else:
                resp = await client.get(
                    f"{_GITHUB_API}/user/repos",
                    params={
                        "per_page": _REPOS_PER_PAGE,
                        "page": page,
                        "sort": "updated",
                        "affiliation": "owner,collaborator,organization_member",
                    },
                    headers=headers,
                )
                resp.raise_for_status()
                items = resp.json()
                has_more = len(items) == _REPOS_PER_PAGE

        repos = [self._map_repo(r) for r in items]
        return GitHubRepoListResponse(repos=repos, page=page, has_more=has_more)

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    async def _fetch_and_save_account(self, token: str) -> GitHubAccount:
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{_GITHUB_API}/user", headers=headers)
            resp.raise_for_status()
            user = resp.json()

        github_id = str(user["id"])
        login = user["login"]
        display_name = user.get("name")
        avatar_url = user.get("avatar_url")
        now = datetime.now(timezone.utc).isoformat()

        db = get_db()
        # Preserve original created_at on reconnect
        await db.execute(
            """INSERT OR REPLACE INTO github_accounts
               (id, login, display_name, avatar_url, access_token, created_at, updated_at)
               VALUES (
                 ?, ?, ?, ?, ?,
                 COALESCE((SELECT created_at FROM github_accounts WHERE id = ?), ?),
                 ?
               )""",
            (github_id, login, display_name, avatar_url, token, github_id, now, now),
        )
        await db.commit()

        return GitHubAccount(
            id=github_id,
            login=login,
            display_name=display_name,
            avatar_url=avatar_url,
            created_at=now,
            updated_at=now,
        )

    async def _get_token(self) -> str:
        db = get_db()
        async with db.execute("SELECT access_token FROM github_accounts LIMIT 1") as cur:
            row = await cur.fetchone()
        if row is None:
            raise NotFoundError("GitHubAccount", "no_account")
        return row["access_token"]

    async def _get_login(self) -> str:
        db = get_db()
        async with db.execute("SELECT login FROM github_accounts LIMIT 1") as cur:
            row = await cur.fetchone()
        return row["login"] if row else ""

    @staticmethod
    def _row_to_account(row) -> GitHubAccount:
        return GitHubAccount(
            id=row["id"],
            login=row["login"],
            display_name=row["display_name"],
            avatar_url=row["avatar_url"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _map_repo(data: dict) -> GitHubRepo:
        return GitHubRepo(
            id=data["id"],
            full_name=data["full_name"],
            name=data["name"],
            owner_login=data["owner"]["login"],
            is_private=data["private"],
            description=data.get("description"),
            default_branch=data.get("default_branch", "main"),
            html_url=data["html_url"],
            ssh_url=data["ssh_url"],
            clone_url=data["clone_url"],
            updated_at=data.get("updated_at", ""),
        )
