"""Analysis budget profiles for normal and large-codebase modes.

All agent retrieval and token budgets are centralised here.
Normal mode must not regress from pre-CS-012 behaviour.
Large mode applies multipliers and raises retrieval depth.

Provider hard limits (conservative maximums):
  max_results per retrieve call  : 60  (avoid oversized vector scans)
  max_completion_tokens per call : 8192 (safe ceiling for all supported LLMs)
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Provider / infrastructure ceilings — never exceed these regardless of mode
# ---------------------------------------------------------------------------
_MAX_RETRIEVAL_RESULTS: int = 60
_MAX_COMPLETION_TOKENS: int = 8192


@dataclass(frozen=True)
class AnalysisProfile:
    """Immutable set of scaling parameters for one analysis mode."""

    mode: str

    # Retrieval
    retrieval_max_results: int          # max_results for RetrieveRequest
    retrieval_doc_char_limit: int       # char limit passed to _fetch_files_by_pattern docs (0 = unlimited)
    retrieval_manifest_char_limit: int  # char limit for manifest files
    retrieval_arch_max_results: int     # prefetch arch bundle size (shared by A/B/C)

    # Token budgets per agent call
    tokens_project_identity: int
    tokens_architecture: int
    tokens_structure: int
    tokens_conventions: int
    tokens_violations: int
    tokens_feature_map: int
    tokens_important_files: int
    tokens_onboarding: int
    tokens_glossary: int
    tokens_risk: int
    tokens_auditor: int
    tokens_synthesizer: int

    # Pipeline concurrency scaling (applied to _default_concurrency())
    concurrency_scale: float

    # Max additional retrieval rounds per agent when blind_spots signal missing/truncated data
    # Normal: 2, Large: 5
    retrieval_augment_rounds: int


def _clamp_results(n: int) -> int:
    return min(n, _MAX_RETRIEVAL_RESULTS)


def _clamp_tokens(n: int) -> int:
    return min(n, _MAX_COMPLETION_TOKENS)


# ---------------------------------------------------------------------------
# NORMAL_PROFILE — baseline, identical to pre-CS-012 hard-coded values
# ---------------------------------------------------------------------------
NORMAL_PROFILE = AnalysisProfile(
    mode="normal",
    # Retrieval
    retrieval_max_results=_clamp_results(30),
    retrieval_doc_char_limit=0,           # unlimited (original behaviour)
    retrieval_manifest_char_limit=3000,
    retrieval_arch_max_results=_clamp_results(30),
    # Completion tokens — unchanged from agent defaults
    tokens_project_identity=_clamp_tokens(2000),
    tokens_architecture=_clamp_tokens(2500),
    tokens_structure=_clamp_tokens(2000),
    tokens_conventions=_clamp_tokens(3000),
    tokens_violations=_clamp_tokens(2000),
    tokens_feature_map=_clamp_tokens(5000),
    tokens_important_files=_clamp_tokens(2000),
    tokens_onboarding=_clamp_tokens(4000),
    tokens_glossary=_clamp_tokens(3000),
    tokens_risk=_clamp_tokens(3000),
    tokens_auditor=_clamp_tokens(2000),
    tokens_synthesizer=_clamp_tokens(4000),
    # No concurrency change in normal mode
    concurrency_scale=1.0,
    retrieval_augment_rounds=2,
)

# ---------------------------------------------------------------------------
# LARGE_PROFILE — 1.5× retrieval depth, modest token increases, +1 concurrency
# ---------------------------------------------------------------------------
LARGE_PROFILE = AnalysisProfile(
    mode="large",
    # Retrieval — 1.5× depth, clamped to provider ceiling
    retrieval_max_results=_clamp_results(45),
    retrieval_doc_char_limit=0,           # still unlimited for docs
    retrieval_manifest_char_limit=6000,   # 2× for larger manifest sets
    retrieval_arch_max_results=_clamp_results(45),
    # Completion tokens — ~1.25× increase for agents that benefit most,
    # clamped to provider ceiling
    tokens_project_identity=_clamp_tokens(2500),
    tokens_architecture=_clamp_tokens(3500),
    tokens_structure=_clamp_tokens(3000),
    tokens_conventions=_clamp_tokens(4000),
    tokens_violations=_clamp_tokens(3000),
    tokens_feature_map=_clamp_tokens(6500),
    tokens_important_files=_clamp_tokens(3000),
    tokens_onboarding=_clamp_tokens(5000),
    tokens_glossary=_clamp_tokens(4000),
    tokens_risk=_clamp_tokens(4000),
    tokens_auditor=_clamp_tokens(3000),
    tokens_synthesizer=_clamp_tokens(5500),
    # Scale concurrency up by ~45 % for large repos
    concurrency_scale=1.45,
    retrieval_augment_rounds=5,
)


def get_profile(large_codebase_mode: bool) -> AnalysisProfile:
    """Return the appropriate :class:`AnalysisProfile` for a run.

    Args:
        large_codebase_mode: ``True`` to select the LARGE_PROFILE.

    Returns:
        :data:`LARGE_PROFILE` when *large_codebase_mode* is ``True``,
        :data:`NORMAL_PROFILE` otherwise.
    """
    return LARGE_PROFILE if large_codebase_mode else NORMAL_PROFILE
