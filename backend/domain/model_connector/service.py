"""ProviderConfigService — persistence + live adapter routing for LLM providers."""
import json
import uuid
from datetime import datetime, timezone

from infrastructure.db.database import get_db
from shared.errors import ConflictError, NotFoundError
from shared.logger import logger

from .errors import ProviderError
from .lmstudio.adapter import LMStudioAdapter
from .ollama.adapter import OllamaAdapter
from .types import ProviderCapabilities, ProviderConfig, ProviderKind


class TestConnectionResult:
    def __init__(self, ok: bool, message: str, warning: str | None = None) -> None:
        self.ok = ok
        self.message = message
        self.warning = warning


def _get_adapter(config: ProviderConfig) -> OllamaAdapter | LMStudioAdapter:
    if config.kind == ProviderKind.OLLAMA:
        return OllamaAdapter(config)
    if config.kind == ProviderKind.LM_STUDIO:
        return LMStudioAdapter(config)
    raise ValueError(f"Unknown provider kind: {config.kind}")


class ProviderConfigService:
    async def list_all(self) -> list[ProviderConfig]:
        db = get_db()
        async with db.execute(
            "SELECT * FROM provider_configs ORDER BY created_at ASC"
        ) as cur:
            rows = await cur.fetchall()
        return [self._row_to_config(r) for r in rows]

    async def get_by_id(self, provider_id: str) -> ProviderConfig:
        db = get_db()
        async with db.execute(
            "SELECT * FROM provider_configs WHERE id = ?", (provider_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            raise NotFoundError("ProviderConfig", provider_id)
        return self._row_to_config(row)

    async def create(
        self,
        kind: ProviderKind,
        display_name: str,
        base_url: str,
        model_id: str,
        capabilities: ProviderCapabilities | None = None,
        extra: dict | None = None,
    ) -> ProviderConfig:
        db = get_db()

        # Prevent duplicate display names
        async with db.execute(
            "SELECT 1 FROM provider_configs WHERE display_name = ?", (display_name,)
        ) as cur:
            if await cur.fetchone():
                raise ConflictError(f"A provider named '{display_name}' already exists")

        cfg_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        caps = (capabilities or ProviderCapabilities()).model_dump()
        ext = extra or {}

        await db.execute(
            """INSERT INTO provider_configs
               (id, kind, display_name, base_url, model_id, capabilities, extra, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                cfg_id,
                kind.value,
                display_name,
                base_url,
                model_id,
                json.dumps(caps),
                json.dumps(ext),
                now,
                now,
            ),
        )
        await db.commit()
        logger.info(f"Created provider config '{display_name}' ({cfg_id})")

        return ProviderConfig(
            id=cfg_id,
            kind=kind,
            display_name=display_name,
            base_url=base_url,
            model_id=model_id,
            capabilities=capabilities or ProviderCapabilities(),
            extra=ext,
        )

    async def update(
        self,
        provider_id: str,
        display_name: str | None = None,
        base_url: str | None = None,
        model_id: str | None = None,
        capabilities: ProviderCapabilities | None = None,
        extra: dict | None = None,
    ) -> ProviderConfig:
        existing = await self.get_by_id(provider_id)
        db = get_db()

        new_name = display_name or existing.display_name
        new_url = base_url or existing.base_url
        new_model = model_id or existing.model_id
        new_caps = (capabilities or existing.capabilities).model_dump()
        new_extra = extra if extra is not None else existing.extra

        if display_name and display_name != existing.display_name:
            async with db.execute(
                "SELECT 1 FROM provider_configs WHERE display_name = ? AND id != ?",
                (display_name, provider_id),
            ) as cur:
                if await cur.fetchone():
                    raise ConflictError(f"A provider named '{display_name}' already exists")

        now = datetime.now(timezone.utc).isoformat()
        await db.execute(
            """UPDATE provider_configs
               SET display_name=?, base_url=?, model_id=?, capabilities=?, extra=?, updated_at=?
               WHERE id=?""",
            (new_name, new_url, new_model, json.dumps(new_caps), json.dumps(new_extra), now, provider_id),
        )
        await db.commit()
        logger.info(f"Updated provider config {provider_id}")
        return await self.get_by_id(provider_id)

    async def delete(self, provider_id: str) -> None:
        db = get_db()
        async with db.execute(
            "DELETE FROM provider_configs WHERE id = ?", (provider_id,)
        ) as cur:
            if cur.rowcount == 0:
                raise NotFoundError("ProviderConfig", provider_id)
        await db.commit()
        logger.info(f"Deleted provider config {provider_id}")

    async def test_connection(self, provider_id: str) -> TestConnectionResult:
        config = await self.get_by_id(provider_id)
        adapter = _get_adapter(config)
        try:
            ok, message, warning = await adapter.test_connection()
            return TestConnectionResult(ok=ok, message=message, warning=warning)
        except ProviderError as e:
            return TestConnectionResult(ok=False, message=e.message)
        finally:
            await adapter.aclose()

    async def list_models(self, provider_id: str) -> list[str]:
        config = await self.get_by_id(provider_id)
        adapter = _get_adapter(config)
        try:
            return await adapter.list_models()
        finally:
            await adapter.aclose()

    @staticmethod
    def _row_to_config(row) -> ProviderConfig:
        caps_raw = row["capabilities"] or "{}"
        extra_raw = row["extra"] or "{}"
        caps_dict = json.loads(caps_raw) if isinstance(caps_raw, str) else caps_raw
        extra_dict = json.loads(extra_raw) if isinstance(extra_raw, str) else extra_raw
        return ProviderConfig(
            id=row["id"],
            kind=ProviderKind(row["kind"]),
            display_name=row["display_name"],
            base_url=row["base_url"],
            model_id=row["model_id"],
            capabilities=ProviderCapabilities(**caps_dict),
            extra=extra_dict,
        )
