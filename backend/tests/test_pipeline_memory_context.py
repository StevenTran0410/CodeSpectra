from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call

import pytest

from domain.analysis.agents._context_builders import prefetch_pipeline_context
from domain.analysis.profiles import LARGE_PROFILE, NORMAL_PROFILE, get_profile
from domain.retrieval.types import RetrievalBundle, RetrievalMode


@pytest.mark.asyncio
async def test_prefetch_pipeline_context_populates_all_fields(
    mock_retrieval: MagicMock,
    canned_retrieval_bundle: RetrievalBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from domain.analysis.agents import _context_builders as cb

    async def fake_tree(_snapshot_id: str, _max: int = 60) -> str:
        return "src/a.py\nsrc/b.py"

    async def fake_patterns(
        _snapshot_id: str,
        _patterns: tuple[str, ...],
        char_limit: int = 0,
        max_rows: int = 6,
    ) -> str:
        return "doc-content"

    monkeypatch.setattr(cb, "fetch_folder_tree", fake_tree)
    monkeypatch.setattr(cb, "_fetch_files_by_pattern", fake_patterns)

    ctx = await prefetch_pipeline_context(mock_retrieval, "snap-xyz", mode=RetrievalMode.HYBRID)
    assert ctx.arch_bundle is canned_retrieval_bundle
    assert "a.py" in ctx.folder_tree
    assert ctx.doc_files == "doc-content"
    assert ctx.manifest_files == "doc-content"
    mock_retrieval.retrieve.assert_awaited()


@pytest.mark.asyncio
async def test_prefetch_failure_graceful_none_like(mock_retrieval: MagicMock) -> None:
    mock_retrieval.retrieve = AsyncMock(side_effect=RuntimeError("retrieve failed"))
    mem_ctx = None
    try:
        mem_ctx = await prefetch_pipeline_context(mock_retrieval, "snap-fail")
    except Exception:
        mem_ctx = None
    assert mem_ctx is None


# ---------------------------------------------------------------------------
# CS-012: profiles module tests
# ---------------------------------------------------------------------------


def test_get_profile_normal_returns_normal_profile() -> None:
    p = get_profile(False)
    assert p is NORMAL_PROFILE
    assert p.mode == "normal"


def test_get_profile_large_returns_large_profile() -> None:
    p = get_profile(True)
    assert p is LARGE_PROFILE
    assert p.mode == "large"


def test_normal_profile_values_unchanged() -> None:
    """NORMAL_PROFILE must match pre-CS-012 hardcoded values exactly."""
    assert NORMAL_PROFILE.retrieval_max_results == 30
    assert NORMAL_PROFILE.retrieval_manifest_char_limit == 3000
    assert NORMAL_PROFILE.tokens_project_identity == 2000
    assert NORMAL_PROFILE.tokens_architecture == 2500
    assert NORMAL_PROFILE.tokens_structure == 2000
    assert NORMAL_PROFILE.tokens_conventions == 3000
    assert NORMAL_PROFILE.tokens_violations == 2000
    assert NORMAL_PROFILE.tokens_feature_map == 5000
    assert NORMAL_PROFILE.tokens_important_files == 2000
    assert NORMAL_PROFILE.tokens_onboarding == 4000
    assert NORMAL_PROFILE.tokens_glossary == 3000
    assert NORMAL_PROFILE.tokens_risk == 3000
    assert NORMAL_PROFILE.tokens_auditor == 2000
    assert NORMAL_PROFILE.tokens_synthesizer == 4000
    assert NORMAL_PROFILE.concurrency_scale == 1.0


def test_large_profile_budgets_exceed_normal() -> None:
    """LARGE_PROFILE must have higher budgets than NORMAL_PROFILE."""
    assert LARGE_PROFILE.retrieval_max_results >= NORMAL_PROFILE.retrieval_max_results
    assert LARGE_PROFILE.retrieval_manifest_char_limit > NORMAL_PROFILE.retrieval_manifest_char_limit
    assert LARGE_PROFILE.tokens_architecture > NORMAL_PROFILE.tokens_architecture
    assert LARGE_PROFILE.tokens_feature_map > NORMAL_PROFILE.tokens_feature_map
    assert LARGE_PROFILE.concurrency_scale > NORMAL_PROFILE.concurrency_scale


def test_profiles_respect_provider_ceiling() -> None:
    """No profile value must exceed provider hard limits."""
    _MAX_TOKENS = 8192
    _MAX_RESULTS = 60
    for profile in (NORMAL_PROFILE, LARGE_PROFILE):
        for attr in (
            "tokens_project_identity",
            "tokens_architecture",
            "tokens_structure",
            "tokens_conventions",
            "tokens_violations",
            "tokens_feature_map",
            "tokens_important_files",
            "tokens_onboarding",
            "tokens_glossary",
            "tokens_risk",
            "tokens_auditor",
            "tokens_synthesizer",
        ):
            assert getattr(profile, attr) <= _MAX_TOKENS, (
                f"{profile.mode}.{attr}={getattr(profile, attr)} exceeds {_MAX_TOKENS}"
            )
        assert profile.retrieval_max_results <= _MAX_RESULTS
        assert profile.retrieval_arch_max_results <= _MAX_RESULTS


@pytest.mark.asyncio
async def test_prefetch_uses_profile_arch_max_results(
    mock_retrieval: MagicMock,
    canned_retrieval_bundle: RetrievalBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """prefetch_pipeline_context must pass profile.retrieval_arch_max_results to retrieve."""
    from domain.analysis.agents import _context_builders as cb
    from domain.retrieval.types import RetrievalSection

    monkeypatch.setattr(cb, "fetch_folder_tree", AsyncMock(return_value=""))
    monkeypatch.setattr(cb, "_fetch_files_by_pattern", AsyncMock(return_value=""))

    await prefetch_pipeline_context(
        mock_retrieval, "snap-large", profile=LARGE_PROFILE
    )
    # _fetch_docs_with_fallback may fire an extra retrieve call (IMPORTANT_FILES fallback)
    # when no doc files are found — find the ARCHITECTURE call specifically.
    arch_req = next(
        c[0][0]
        for c in mock_retrieval.retrieve.call_args_list
        if c[0][0].section == RetrievalSection.ARCHITECTURE
    )
    assert arch_req.max_results == LARGE_PROFILE.retrieval_arch_max_results


@pytest.mark.asyncio
async def test_prefetch_defaults_to_normal_profile(
    mock_retrieval: MagicMock,
    canned_retrieval_bundle: RetrievalBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """prefetch_pipeline_context with profile=None must behave like NORMAL_PROFILE."""
    from domain.analysis.agents import _context_builders as cb
    from domain.retrieval.types import RetrievalSection

    monkeypatch.setattr(cb, "fetch_folder_tree", AsyncMock(return_value=""))
    monkeypatch.setattr(cb, "_fetch_files_by_pattern", AsyncMock(return_value=""))

    await prefetch_pipeline_context(mock_retrieval, "snap-default")
    arch_req = next(
        c[0][0]
        for c in mock_retrieval.retrieve.call_args_list
        if c[0][0].section == RetrievalSection.ARCHITECTURE
    )
    assert arch_req.max_results == NORMAL_PROFILE.retrieval_arch_max_results
