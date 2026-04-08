from __future__ import annotations

from domain.analysis.model_guard import check_model_capability
from domain.retrieval.service import _chunk_size_for, _ends_mid_function, _token_estimate


def test_ends_mid_function_brace_unbalanced() -> None:
    assert _ends_mid_function("function foo() { return 1;", "typescript") is True


def test_ends_mid_function_brace_balanced() -> None:
    assert _ends_mid_function("function foo() { return 1; }", "javascript") is False


def test_ends_mid_function_python_mid_body() -> None:
    src = "def foo():\n    x = 1\n    return x"
    assert _ends_mid_function(src, "python") is True


def test_ends_mid_function_python_complete() -> None:
    src = "def foo():\n    return 1\n\nprint(foo())"
    assert _ends_mid_function(src, "python") is False


def test_chunk_size_for_categories_and_lang() -> None:
    assert _chunk_size_for("docs", None) == 1800
    assert _chunk_size_for("config", "python") == 1200
    assert _chunk_size_for("test", "go") == 1400
    assert _chunk_size_for("source", "python") == 1500
    assert _chunk_size_for("source", "rust") == 1300


def test_token_estimate_minimum_one() -> None:
    assert _token_estimate("") == 1
    assert _token_estimate("abab") == 1
    assert _token_estimate("abcd" * 10) == 10


def test_check_model_capability_small_models_warn() -> None:
    w = check_model_capability("qwen2.5:1.5b")
    assert w is not None
    assert w["code"] == "model_too_small"
    assert w["severity"] == "warn"
    assert check_model_capability("phi3:mini") is not None
    assert check_model_capability("TinyLlama/TinyLlama-1.1B") is not None


def test_check_model_capability_larger_ok() -> None:
    assert check_model_capability("llama3.1:8b") is None
    assert check_model_capability("qwen2.5:72b") is None
