"""Pure-function tests for analysis report section extractors (no async, no mocks)."""
from __future__ import annotations

import pytest

from agent_eval_harness.discovery.analysis_context_extractors import (
    _extract_important_files,
    _filter_risk_findings,
    _extract_identity,
    _extract_synthesis,
)


def test_extract_important_files_with_nested_lists() -> None:
    """Extract file paths from various nested list values in section dict."""
    section = {
        "entrypoint": ["main.py", "app.py"],
        "backbone": ["core/engine.py"],
        "read_first": ["README.md"],
        "other_key": "not a list",
    }
    result = _extract_important_files(section)
    assert set(result) == {"main.py", "app.py", "core/engine.py", "README.md"}


def test_extract_important_files_with_none() -> None:
    """Return empty list when section is None."""
    result = _extract_important_files(None)
    assert result == []


def test_extract_important_files_with_empty_section() -> None:
    """Return empty list when section has no list values."""
    section = {"key1": "value", "key2": 123}
    result = _extract_important_files(section)
    assert result == []


def test_filter_risk_findings_keeps_high_and_medium() -> None:
    """Keep only findings with high or medium severity."""
    section = {
        "findings": [
            {
                "severity": "high",
                "title": "Critical issue",
                "evidence": ["file1.py", "file2.py"],
            },
            {
                "severity": "medium",
                "title": "Warning",
                "evidence": ["file3.py"],
            },
            {
                "severity": "low",
                "title": "Info",
                "evidence": ["file4.py"],
            },
        ]
    }
    result = _filter_risk_findings(section)
    assert len(result) == 2
    assert result[0]["severity"] == "high"
    assert result[1]["severity"] == "medium"


def test_filter_risk_findings_preserves_evidence() -> None:
    """Preserve the evidence key when filtering."""
    section = {
        "findings": [
            {
                "severity": "high",
                "evidence": ["file1.py", "file2.py"],
            }
        ]
    }
    result = _filter_risk_findings(section)
    assert len(result) == 1
    assert result[0]["evidence"] == ["file1.py", "file2.py"]


def test_filter_risk_findings_with_none() -> None:
    """Return empty list when section is None."""
    result = _filter_risk_findings(None)
    assert result == []


def test_filter_risk_findings_with_no_findings_key() -> None:
    """Return empty list when findings key is absent."""
    section = {"other_key": "value"}
    result = _filter_risk_findings(section)
    assert result == []


def test_extract_identity_with_all_fields() -> None:
    """Extract all identity fields from section dict."""
    section = {
        "purpose": "Process documents",
        "domain": "NLP",
        "runtime_type": "CLI",
        "tech_stack": "Python 3.11, LangChain",
        "key_frameworks": ["langchain"],
        "extra_key": "ignored",
    }
    result = _extract_identity(section)
    assert result == {
        "purpose": "Process documents",
        "domain": "NLP",
        "runtime_type": "CLI",
        "tech_stack": "Python 3.11, LangChain",
        "key_frameworks": ["langchain"],
    }


def test_extract_identity_partial_fields() -> None:
    """Extract only present identity fields."""
    section = {
        "purpose": "Test",
        "domain": "Testing",
    }
    result = _extract_identity(section)
    assert result == {
        "purpose": "Test",
        "domain": "Testing",
    }


def test_extract_identity_with_none() -> None:
    """Return empty dict when section is None."""
    result = _extract_identity(None)
    assert result == {}


def test_extract_synthesis_with_all_fields() -> None:
    """Extract all synthesis fields from section dict."""
    section = {
        "executive_summary": "High-level overview",
        "architecture_narrative": "Detailed architecture",
        "reading_path": "Start with main.py",
        "extra_key": "ignored",
    }
    result = _extract_synthesis(section)
    assert result == {
        "executive_summary": "High-level overview",
        "architecture_narrative": "Detailed architecture",
        "reading_path": "Start with main.py",
    }


def test_extract_synthesis_partial_fields() -> None:
    """Extract only present synthesis fields."""
    section = {
        "executive_summary": "Overview",
    }
    result = _extract_synthesis(section)
    assert result == {
        "executive_summary": "Overview",
    }


def test_extract_synthesis_with_none() -> None:
    """Return empty dict when section is None."""
    result = _extract_synthesis(None)
    assert result == {}
