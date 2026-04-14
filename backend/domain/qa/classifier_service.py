"""
Classifier service — manages user-labelled examples in DB and triggers retraining.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from infrastructure.db.database import get_db
from shared.logger import logger

from . import intent_classifier as _clf_module


class ClassifierService:
    # ------------------------------------------------------------------
    # Examples — DB CRUD
    # ------------------------------------------------------------------

    async def add_example(self, text: str, is_deep_research: bool) -> dict:
        """Persist a new labelled example and retrain the classifier."""
        text = text.strip()
        if not text:
            raise ValueError("text must not be empty")

        example_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        db = get_db()
        await db.execute(
            """INSERT INTO qa_classifier_examples (id, text, is_deep_research, created_at)
               VALUES (?, ?, ?, ?)""",
            (example_id, text, int(is_deep_research), created_at),
        )
        await db.commit()

        logger.info(
            "[ClassifierService] added example id=%s is_deep=%s text=%r",
            example_id, is_deep_research, text[:60],
        )

        # Retrain with the new example included
        await self._retrain()

        return {"id": example_id, "text": text, "is_deep_research": is_deep_research, "created_at": created_at}

    async def delete_example(self, example_id: str) -> None:
        """Delete a user example and retrain."""
        db = get_db()
        await db.execute("DELETE FROM qa_classifier_examples WHERE id=?", (example_id,))
        await db.commit()
        await self._retrain()

    async def list_examples(self) -> list[dict]:
        """Return all user-added examples ordered by creation time."""
        db = get_db()
        async with db.execute(
            "SELECT id, text, is_deep_research, created_at FROM qa_classifier_examples ORDER BY created_at DESC"
        ) as cur:
            rows = await cur.fetchall()
        return [
            {
                "id": r["id"],
                "text": r["text"],
                "is_deep_research": bool(r["is_deep_research"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Retraining
    # ------------------------------------------------------------------

    async def _retrain(self) -> None:
        """Load all user examples from DB and retrain the classifier singleton."""
        examples = await self.list_examples()
        extra: list[tuple[str, bool]] = [(e["text"], e["is_deep_research"]) for e in examples]
        _clf_module.retrain_with_examples(extra)
        logger.info(
            "[ClassifierService] retrained classifier with %d user examples (%d total)",
            len(extra), len(extra) + len(_clf_module._DATASET),
        )

    async def retrain(self) -> dict:
        """Public trigger — retrain and return status."""
        await self._retrain()
        examples = await self.list_examples()
        return {
            **_clf_module.get_status(),
            "user_examples": len(examples),
        }

    async def get_status(self) -> dict:
        """Return classifier status including user example count."""
        examples = await self.list_examples()
        return {
            **_clf_module.get_status(),
            "user_examples": len(examples),
        }

    # ------------------------------------------------------------------
    # Startup warm-up (called from lifespan)
    # ------------------------------------------------------------------

    async def warm_up_with_db_examples(self) -> None:
        """Load persisted model (or train fresh) at startup. Non-blocking — failures logged.

        Strategy:
        - If classifier_model.pkl exists on disk → classify_intent() loads it automatically
          (lazy, on first call). Just call warm_up() to force-load it now.
        - If the file is absent → train from scratch using DB examples + built-in dataset.
        """
        try:
            from . import intent_classifier as _m
            if _m._model_path().exists():
                # Model already persisted — just warm the singleton
                _clf_module.warm_up()
                logger.info("[ClassifierService] loaded persisted model from disk")
            else:
                # No saved model — train fresh with DB examples, then save
                await self._retrain()
                logger.info("[ClassifierService] trained fresh model and saved to disk")
        except Exception as exc:
            logger.warning("[ClassifierService] warm-up failed, will train on first request: %s", exc)
