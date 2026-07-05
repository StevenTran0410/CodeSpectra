"""Tests for cross-encoder reranking (CS-254).

Covers:
- Mocked-boundary regression test: real _louvain_fallback.py content fed as passage
- Assertion: passage is longer than 200 chars and contains 'sigma_tot'
- Assertion: rerank_fused_entries output is sorted by rerank_score descending
- Unit test: passage passed to .rerank() differs from excerpt for same chunk_id
- GPU-gated real end-to-end smoke test (requires CUDA)
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from domain.retrieval.cross_encoder_rerank import (
    _MAX_SEQ_LENGTH,
    is_gpu_reranker_enabled,
    rerank_fused_entries,
)
from infrastructure.db.database import get_db
from domain.retrieval.types import FusedRankEntry


class TestRerankerMockedBoundary:
    """Mocked .rerank() boundary tests with real file content."""

    def test_louvain_fallback_real_content_fed_to_stub(self):
        """Regression test (CS-254 claim #1): real _louvain_fallback.py content is fed
        to the reranker, proving full content (not just 200-char excerpt) reaches the
        model.

        The test:
        1. Loads real _louvain_fallback.py content via Path.read_text()
        2. Mocks the model's .rerank() method to capture what's passed
        3. Asserts the passage is longer than 200 chars (exceeds old excerpt cap)
        4. Asserts it contains 'sigma_tot' (confirmed present at line 100, past char 200)
        5. Returns a high rerank_score from the stub to test rank promotion
        """
        louvain_path = (
            Path(__file__).parent.parent / "domain" / "structural_graph" / "_louvain_fallback.py"
        )
        assert louvain_path.exists(), f"Louvain fallback not found at {louvain_path}"

        real_content = louvain_path.read_text()
        assert len(real_content) > 200, (
            f"Real louvain_fallback.py content must be > 200 chars, got {len(real_content)}"
        )
        assert "sigma_tot" in real_content, (
            "Real content must contain 'sigma_tot' (confirmed at line 100, past char 200)"
        )

        # Build fused entry referencing this file
        fused_entry = FusedRankEntry(
            chunk_id="chunk_louvain",
            rel_path="backend/domain/structural_graph/_louvain_fallback.py",
            fused_score=0.5,
            per_signal_ranks={"bm25": 10, "graph": 5},
            excerpt=real_content[:200],  # Traditional excerpt cap
            token_estimate=2000,
        )

        # Stub content lookup
        chunk_content_by_id = {"chunk_louvain": real_content}

        # Mock the model
        with patch("domain.retrieval.cross_encoder_rerank._model") as mock_model:
            with patch(
                "domain.retrieval.cross_encoder_rerank._check_gpu_available",
                return_value=True,
            ):
                # Capture what's passed to .rerank()
                captured_passages = []

                def stub_rerank(query, passages, top_n=None):
                    captured_passages.extend(passages)
                    # Return a mock score object with high score for the louvain chunk
                    return [{"index": 0, "relevance_score": 0.95}]

                mock_model.rerank = stub_rerank

                # Call the reranker
                reranked, status = rerank_fused_entries(
                    "louvain community detection clustering",
                    [fused_entry],
                    chunk_content_by_id,
                )

                assert status == "ok"
                assert len(captured_passages) == 1

                # The passage passed to stub must be the full content (or truncated at max_seq_len)
                stub_passage = captured_passages[0]
                assert len(stub_passage) > 200, (
                    f"Passage passed to stub must be > 200 chars, got {len(stub_passage)}. "
                    "This proves full content threading works (claim #1 fix)."
                )
                assert "sigma_tot" in stub_passage, (
                    "Full content must be passed, containing 'sigma_tot'"
                )

                # Reranked list should have the entry with new score
                assert len(reranked) == 1
                assert reranked[0].chunk_id == "chunk_louvain"
                assert reranked[0].rerank_score == 0.95

    def test_passage_longer_than_excerpt_for_same_chunk(self):
        """Unit test (anti-regression guard for claim #1): the passage passed to
        .rerank() must be longer than the FusedRankEntry.excerpt for the same chunk_id.
        """
        long_content = "X" * 500  # 500 chars
        short_excerpt = "X" * 200  # 200-char excerpt

        fused_entry = FusedRankEntry(
            chunk_id="chunk_1",
            rel_path="file.py",
            fused_score=0.7,
            per_signal_ranks={"bm25": 1},
            excerpt=short_excerpt,
            token_estimate=100,
        )

        chunk_content_by_id = {"chunk_1": long_content}

        with patch("domain.retrieval.cross_encoder_rerank._model") as mock_model:
            with patch(
                "domain.retrieval.cross_encoder_rerank._check_gpu_available",
                return_value=True,
            ):
                captured_passages = []

                def stub_rerank(query, passages, top_n=None):
                    captured_passages.extend(passages)
                    return [{"index": 0, "relevance_score": 0.8}]

                mock_model.rerank = stub_rerank

                reranked, status = rerank_fused_entries("test query", [fused_entry], chunk_content_by_id)

                assert status == "ok"
                assert len(captured_passages) == 1

                # Passage must be longer than excerpt
                passage_len = len(captured_passages[0])
                excerpt_len = len(fused_entry.excerpt)
                assert passage_len > excerpt_len, (
                    f"Passage ({passage_len} chars) must be longer than excerpt ({excerpt_len} chars). "
                    "Proves full content threading (claim #1 fix)."
                )


class TestRerankerSortOrder:
    """Regression tests for claim #2: output must be sorted by rerank_score descending."""

    def test_rerank_output_sorted_by_rerank_score_descending(self):
        """Regression test (CS-254 claim #2): rerank_fused_entries() must sort its output
        by rerank_score descending before returning, not return in insertion order.
        """
        # Build 3 entries with distinct fused_scores but we'll give them rerank_scores
        # in different order via the stub
        fused_entries = [
            FusedRankEntry(
                chunk_id="chunk_1",
                rel_path="file1.py",
                fused_score=10.0,  # High fused score
                per_signal_ranks={"bm25": 1},
                excerpt="content 1",
                token_estimate=50,
            ),
            FusedRankEntry(
                chunk_id="chunk_2",
                rel_path="file2.py",
                fused_score=5.0,  # Medium fused score
                per_signal_ranks={"bm25": 2},
                excerpt="content 2",
                token_estimate=50,
            ),
            FusedRankEntry(
                chunk_id="chunk_3",
                rel_path="file3.py",
                fused_score=1.0,  # Low fused score
                per_signal_ranks={"bm25": 3},
                excerpt="content 3",
                token_estimate=50,
            ),
        ]

        chunk_content_by_id = {
            "chunk_1": "content 1 extended",
            "chunk_2": "content 2 extended",
            "chunk_3": "content 3 extended",
        }

        with patch("domain.retrieval.cross_encoder_rerank._model") as mock_model:
            with patch(
                "domain.retrieval.cross_encoder_rerank._check_gpu_available",
                return_value=True,
            ):
                # Stub returns scores in reverse order compared to fused_score
                # (to prove the final sort is actually happening)
                def stub_rerank(query, passages, top_n=None):
                    return [
                        {"index": 2, "relevance_score": 0.9},  # chunk_3 gets high rerank_score
                        {"index": 1, "relevance_score": 0.5},  # chunk_2 gets medium rerank_score
                        {"index": 0, "relevance_score": 0.1},  # chunk_1 gets low rerank_score
                    ]

                mock_model.rerank = stub_rerank

                reranked, status = rerank_fused_entries("test query", fused_entries, chunk_content_by_id)

                assert status == "ok"
                assert len(reranked) == 3

                # Output must be sorted by rerank_score descending
                scores = [e.rerank_score for e in reranked]
                assert scores == [0.9, 0.5, 0.1], (
                    f"rerank_fused_entries must sort output by rerank_score descending. "
                    f"Got {scores}, expected [0.9, 0.5, 0.1]"
                )

                # Fused scores should be preserved (side-by-side comparison)
                fused_scores = [e.fused_score for e in reranked]
                # After reranking and sorting by rerank_score, order is: chunk_3 (0.9), chunk_2 (0.5), chunk_1 (0.1)
                # So fused_scores should be: [1.0, 5.0, 10.0] (in that new order)
                expected_fused_order = [1.0, 5.0, 10.0]
                assert fused_scores == expected_fused_order, (
                    f"Fused scores must be preserved in reranked order. "
                    f"Got {fused_scores}, expected {expected_fused_order}"
                )


class TestRerankerErrorHandling:
    """Tests for error handling and fallback paths."""

    def test_no_gpu_returns_empty_list(self):
        """Test that when no GPU is available, rerank returns empty list with 'no_gpu' status."""
        fused = [
            FusedRankEntry(
                chunk_id="chunk_1",
                rel_path="file.py",
                fused_score=0.5,
                per_signal_ranks={"bm25": 1},
                excerpt="content",
                token_estimate=50,
            )
        ]

        chunk_content_by_id = {"chunk_1": "content"}

        with patch(
            "domain.retrieval.cross_encoder_rerank._check_gpu_available", return_value=False
        ):
            reranked, status = rerank_fused_entries("test", fused, chunk_content_by_id)

            assert status == "no_gpu"
            assert reranked == []

    def test_empty_fused_list_returns_empty_reranked(self):
        """Test that empty fused list returns empty reranked list."""
        reranked, status = rerank_fused_entries("test query", [], {})

        assert status == "ok"
        assert reranked == []

    def test_max_seq_length_truncation_logs(self):
        """Test that truncation at max_seq_length logs a debug message."""
        # Create content longer than max_seq_length
        long_content = "X" * (_MAX_SEQ_LENGTH + 100)

        fused = [
            FusedRankEntry(
                chunk_id="chunk_long",
                rel_path="file.py",
                fused_score=0.5,
                per_signal_ranks={"bm25": 1},
                excerpt="excerpt",
                token_estimate=5000,
            )
        ]

        chunk_content_by_id = {"chunk_long": long_content}

        with patch("domain.retrieval.cross_encoder_rerank._model") as mock_model:
            with patch(
                "domain.retrieval.cross_encoder_rerank._check_gpu_available", return_value=True
            ):
                with patch("domain.retrieval.cross_encoder_rerank.logger") as mock_logger:
                    captured_passages = []

                    def stub_rerank(query, passages, top_n=None):
                        captured_passages.extend(passages)
                        return [{"index": 0, "score": 0.8}]

                    mock_model.rerank = stub_rerank

                    reranked, status = rerank_fused_entries("test", fused, chunk_content_by_id)

                    # Check that truncation was logged
                    mock_logger.debug.assert_called()

                    # Passage passed to stub should be truncated
                    assert len(captured_passages[0]) == _MAX_SEQ_LENGTH


class TestRerankerPreservesMetadata:
    """Test that reranking preserves original fused metadata."""

    def test_fused_rank_preserved(self):
        """Test that fused_rank (original position in fused list) is preserved."""
        fused = [
            FusedRankEntry(
                chunk_id="chunk_1",
                rel_path="file1.py",
                fused_score=0.5,
                per_signal_ranks={"bm25": 1},
                excerpt="content 1",
                token_estimate=50,
            ),
            FusedRankEntry(
                chunk_id="chunk_2",
                rel_path="file2.py",
                fused_score=0.3,
                per_signal_ranks={"bm25": 2},
                excerpt="content 2",
                token_estimate=50,
            ),
        ]

        chunk_content_by_id = {"chunk_1": "content 1", "chunk_2": "content 2"}

        with patch("domain.retrieval.cross_encoder_rerank._model") as mock_model:
            with patch(
                "domain.retrieval.cross_encoder_rerank._check_gpu_available", return_value=True
            ):
                def stub_rerank(query, passages, top_n=None):
                    # Return in reverse order
                    return [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.7}]

                mock_model.rerank = stub_rerank

                reranked, status = rerank_fused_entries("test", fused, chunk_content_by_id)

                assert status == "ok"
                assert len(reranked) == 2

                # After sorting by rerank_score, order should be [chunk_2, chunk_1]
                # But fused_rank should still reflect original positions
                assert reranked[0].chunk_id == "chunk_2"
                assert reranked[0].fused_rank == 2  # Was position 2 in fused list

                assert reranked[1].chunk_id == "chunk_1"
                assert reranked[1].fused_rank == 1  # Was position 1 in fused list

    def test_excerpt_and_token_estimate_preserved(self):
        """Test that excerpt (200-char UI display) and token_estimate are preserved."""
        excerpt_text = "Short excerpt for UI display"
        token_est = 123

        fused = [
            FusedRankEntry(
                chunk_id="chunk_1",
                rel_path="file.py",
                fused_score=0.5,
                per_signal_ranks={"bm25": 1},
                excerpt=excerpt_text,
                token_estimate=token_est,
            )
        ]

        chunk_content_by_id = {"chunk_1": "Much longer content"}

        with patch("domain.retrieval.cross_encoder_rerank._model") as mock_model:
            with patch(
                "domain.retrieval.cross_encoder_rerank._check_gpu_available", return_value=True
            ):
                def stub_rerank(query, passages, top_n=None):
                    return [{"index": 0, "relevance_score": 0.8}]

                mock_model.rerank = stub_rerank

                reranked, status = rerank_fused_entries("test", fused, chunk_content_by_id)

                assert status == "ok"
                assert reranked[0].excerpt == excerpt_text
                assert reranked[0].token_estimate == token_est


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.getenv("RUN_GPU_SMOKE_TEST") != "1",
    reason="Requires GPU (CUDA) AND explicit opt-in via RUN_GPU_SMOKE_TEST=1 — "
    "CUDA availability alone isn't consent to download/load a real model "
    "every time the suite runs; a dev machine having a GPU shouldn't make "
    "this fire unattended.",
)
class TestRerankerGPUEndToEnd:
    """Real GPU end-to-end smoke test (requires CUDA + RUN_GPU_SMOKE_TEST=1)."""

    def test_real_model_inference_smoke_test(self):
        """Smoke test: load the real model and run inference on 2-3 docs.

        Opt-in only (RUN_GPU_SMOKE_TEST=1) — never fires in a routine `pytest`
        run even on a machine with CUDA, since it downloads/loads a real model.
        """
        from domain.retrieval.cross_encoder_rerank import load_reranker

        # Try to load the real model
        if not load_reranker():
            pytest.skip("Failed to load jina-reranker-v3 model")

        fused = [
            FusedRankEntry(
                chunk_id="chunk_1",
                rel_path="file1.py",
                fused_score=0.5,
                per_signal_ranks={"bm25": 1},
                excerpt="Python function definitions and class hierarchies",
                token_estimate=100,
            ),
            FusedRankEntry(
                chunk_id="chunk_2",
                rel_path="file2.py",
                fused_score=0.3,
                per_signal_ranks={"bm25": 2},
                excerpt="Database schema and migration files",
                token_estimate=100,
            ),
            FusedRankEntry(
                chunk_id="chunk_3",
                rel_path="file3.py",
                fused_score=0.2,
                per_signal_ranks={"bm25": 3},
                excerpt="Test fixtures and mock utilities",
                token_estimate=100,
            ),
        ]

        chunk_content_by_id = {
            "chunk_1": "def foo(x): return x * 2\nclass Bar: pass",
            "chunk_2": "CREATE TABLE users (id INT, name VARCHAR)",
            "chunk_3": "def mock_service(): return MagicMock()",
        }

        # Call without mocking — use the real model
        reranked, status = rerank_fused_entries(
            "python function and class definitions",
            fused,
            chunk_content_by_id,
        )

        # Should succeed (GPU available)
        assert status == "ok"
        assert len(reranked) == 3

        # Output should be sorted by rerank_score
        scores = [e.rerank_score for e in reranked]
        assert scores == sorted(scores, reverse=True), (
            f"Reranked output must be sorted by rerank_score descending, got {scores}"
        )

        # All entries should have been scored (may be positive or negative)
        for entry in reranked:
            assert isinstance(entry.rerank_score, float), "rerank_score should be a float"


class TestGpuRerankerGlobalToggle:
    """Tests for the global GPU Reranker on/off flag (app_metadata, CS-254 follow-up)."""

    async def _set_flag(self, value: str | None) -> None:
        db = get_db()
        if value is None:
            await db.execute("DELETE FROM app_metadata WHERE key = 'gpu_reranker_enabled'")
        else:
            await db.execute(
                "INSERT OR REPLACE INTO app_metadata (key, value) VALUES ('gpu_reranker_enabled', ?)",
                (value,),
            )
        await db.commit()

    async def test_defaults_to_disabled_when_flag_never_set(self):
        await self._set_flag(None)
        with patch(
            "domain.retrieval.cross_encoder_rerank.detect_gpu", return_value=(True, 8.0)
        ):
            assert await is_gpu_reranker_enabled() is False

    async def test_enabled_when_flag_true_and_gpu_available(self):
        await self._set_flag("true")
        with patch(
            "domain.retrieval.cross_encoder_rerank.detect_gpu", return_value=(True, 8.0)
        ):
            assert await is_gpu_reranker_enabled() is True
        await self._set_flag(None)

    async def test_disabled_when_flag_true_but_no_gpu(self):
        """A stale 'true' flag (e.g. carried over from a different machine) must
        never be trusted blindly — GPU availability is re-checked every time."""
        await self._set_flag("true")
        with patch(
            "domain.retrieval.cross_encoder_rerank.detect_gpu", return_value=(False, None)
        ):
            assert await is_gpu_reranker_enabled() is False
        await self._set_flag(None)

    async def test_disabled_when_flag_explicitly_false(self):
        await self._set_flag("false")
        with patch(
            "domain.retrieval.cross_encoder_rerank.detect_gpu", return_value=(True, 8.0)
        ):
            assert await is_gpu_reranker_enabled() is False
        await self._set_flag(None)
