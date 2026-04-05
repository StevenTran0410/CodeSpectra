"""ProviderConfigService — persistence + live adapter routing for LLM providers."""
import json

from infrastructure.db.database import get_db
from shared.errors import ConflictError, NotFoundError
from shared.logger import logger
from shared.utils import new_id, utc_now_iso

from .anthropic.adapter import AnthropicAdapter
from .deepseek.adapter import DeepSeekAdapter
from .errors import ProviderError
from .gemini.adapter import GeminiAdapter
from .lmstudio.adapter import LMStudioAdapter
from .ollama.adapter import OllamaAdapter
from .openai.adapter import OpenAIAdapter
from .types import ChatRequest, ChatResponse, ProviderCapabilities, ProviderConfig, ProviderKind


class TestConnectionResult:
    def __init__(self, ok: bool, message: str, warning: str | None = None) -> None:
        self.ok = ok
        self.message = message
        self.warning = warning


def _get_adapter(config: ProviderConfig):
    match config.kind:
        case ProviderKind.OLLAMA:
            return OllamaAdapter(config)
        case ProviderKind.LM_STUDIO:
            return LMStudioAdapter(config)
        case ProviderKind.OPENAI:
            return OpenAIAdapter(config)
        case ProviderKind.ANTHROPIC:
            return AnthropicAdapter(config)
        case ProviderKind.GEMINI:
            return GeminiAdapter(config)
        case ProviderKind.DEEPSEEK:
            return DeepSeekAdapter(config)
        case _:
            raise ValueError(f"Unknown provider kind: {config.kind}")


def _mask_extra(extra: dict) -> dict:
    """Remove api_key from extra; replace with has_api_key flag."""
    masked = dict(extra)
    if "api_key" in masked:
        masked["has_api_key"] = bool(masked.pop("api_key"))
    return masked


class ProviderConfigService:
    async def list_all(self) -> list[ProviderConfig]:
        db = get_db()
        async with db.execute(
            "SELECT * FROM provider_configs ORDER BY created_at ASC"
        ) as cur:
            rows = await cur.fetchall()
        configs = [self._row_to_config(r) for r in rows]
        return [c.model_copy(update={"extra": _mask_extra(dict(c.extra))}) for c in configs]

    async def get_by_id(self, provider_id: str) -> ProviderConfig:
        """Returns masked config (no api_key in extra) — safe for API responses."""
        config = await self._get_by_id_full(provider_id)
        return config.model_copy(update={"extra": _mask_extra(dict(config.extra))})

    async def _get_by_id_full(self, provider_id: str) -> ProviderConfig:
        """Returns unmasked config with api_key — for internal/adapter use only."""
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
        api_key: str | None = None,
    ) -> ProviderConfig:
        db = get_db()

        async with db.execute(
            "SELECT 1 FROM provider_configs WHERE display_name = ?", (display_name,)
        ) as cur:
            if await cur.fetchone():
                raise ConflictError(f"A provider named '{display_name}' already exists")

        cfg_id = new_id()
        now = utc_now_iso()
        caps = (capabilities or ProviderCapabilities()).model_dump()
        ext = dict(extra or {})
        if api_key:
            ext["api_key"] = api_key

        await db.execute(
            """INSERT INTO provider_configs
               (id, kind, display_name, base_url, model_id, capabilities, extra, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (cfg_id, kind.value, display_name, base_url, model_id,
             json.dumps(caps), json.dumps(ext), now, now),
        )
        await db.commit()
        logger.info(f"Created provider config '{display_name}' ({cfg_id})")

        return ProviderConfig(
            id=cfg_id, kind=kind, display_name=display_name,
            base_url=base_url, model_id=model_id,
            capabilities=capabilities or ProviderCapabilities(),
            extra=_mask_extra(ext),
        )

    async def update(
        self,
        provider_id: str,
        display_name: str | None = None,
        base_url: str | None = None,
        model_id: str | None = None,
        capabilities: ProviderCapabilities | None = None,
        extra: dict | None = None,
        api_key: str | None = None,
    ) -> ProviderConfig:
        existing = await self._get_by_id_full(provider_id)
        db = get_db()

        new_name = display_name or existing.display_name
        new_url = base_url or existing.base_url
        new_model = model_id or existing.model_id
        new_caps = (capabilities or existing.capabilities).model_dump()
        new_extra = dict(existing.extra)  # full extra with api_key
        if extra is not None:
            new_extra.update(extra)
        if api_key:  # only update key if explicitly provided
            new_extra["api_key"] = api_key

        if display_name and display_name != existing.display_name:
            async with db.execute(
                "SELECT 1 FROM provider_configs WHERE display_name = ? AND id != ?",
                (display_name, provider_id),
            ) as cur:
                if await cur.fetchone():
                    raise ConflictError(f"A provider named '{display_name}' already exists")

        now = utc_now_iso()
        await db.execute(
            """UPDATE provider_configs
               SET display_name=?, base_url=?, model_id=?, capabilities=?, extra=?, updated_at=?
               WHERE id=?""",
            (new_name, new_url, new_model, json.dumps(new_caps), json.dumps(new_extra), now, provider_id),
        )
        await db.commit()
        logger.info(f"Updated provider config {provider_id}")
        return await self.get_by_id(provider_id)  # masked version

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
        config = await self._get_by_id_full(provider_id)
        adapter = _get_adapter(config)
        try:
            ok, message, warning = await adapter.test_connection()
            return TestConnectionResult(ok=ok, message=message, warning=warning)
        except ProviderError as e:
            return TestConnectionResult(ok=False, message=e.message)
        finally:
            await adapter.aclose()

    async def list_models(self, provider_id: str) -> list[str]:
        config = await self._get_by_id_full(provider_id)
        adapter = _get_adapter(config)
        try:
            return await adapter.list_models()
        finally:
            await adapter.aclose()

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Route a chat request to selected provider (with optional model override)."""
        config = await self._get_by_id_full(request.provider_id)
        if request.model_id:
            config = config.model_copy(update={"model_id": request.model_id})
        adapter = _get_adapter(config)
        try:
            return await adapter.chat(request)
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
