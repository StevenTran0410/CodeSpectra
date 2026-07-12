"""Tests for the field_downstream_consumers static harvest (CS-289 Workstream A3).

Fixtures deliberately mirror the real shape of backend/domain/analysis/agents/agent_auditor.py
+ agent_synthesis.py + _section_compressor.py (loop-over-letters -> per-letter section dict ->
direct .get() reads plus a cross-file preview-keys-dict-driven helper call) without depending
on the real backend source, so the harvest logic is exercised generically.
"""
from __future__ import annotations

from pathlib import Path

from agent_eval_harness.mapping.agent_flow import AgentFlow, AgentFlowMap
from agent_eval_harness.mapping.builder.contract_harvest import (
    _parse_files,
    harvest_field_downstream_consumers,
)
from agent_eval_harness.mapping.system_map import Component, SystemMap

COMPRESSOR_SRC = '''
_PREVIEW_KEYS = {
    "A": ["purpose", "domain"],
    "B": ["main_layers"],
    "C": ["summary", "folders"],
}

def compress_section(letter, section, char_cap=500):
    keys = _PREVIEW_KEYS.get(letter, [])
    preview = {}
    for key in keys:
        val = section.get(key)
        if val is not None:
            preview[key] = val
    return str(preview)[:char_cap]


def compress_audit(section_k, char_cap=800):
    subset = {
        "overall_confidence": section_k.get("overall_confidence"),
        "notes": section_k.get("notes"),
    }
    return str(subset)[:char_cap]
'''

AGENT_K_SRC = '''
from .compressor import compress_section

def _build_k_input(sections):
    compressed = {}
    for letter in "ABC":
        s = sections.get(letter) or {}
        compressed[letter] = {
            "confidence": s.get("confidence", "medium"),
            "blind_spots": (s.get("blind_spots") or [])[:3],
            "content_preview": compress_section(letter, s, char_cap=500),
        }
    return compressed


class KAgent:
    async def run(self, provider_id: str, model_id: str, all_sections: dict) -> dict:
        return _build_k_input(all_sections)
'''

AGENT_L_SRC = '''
from .compressor import compress_audit, compress_section

def _build_l_input(sections):
    compact = {}
    for letter in "AB":
        s = sections.get(letter) or {}
        compact[letter] = compress_section(letter, s, char_cap=800)
    compact["K"] = compress_audit(sections.get("K") or {}, char_cap=800)
    return compact


class LAgent:
    async def run(self, provider_id: str, model_id: str, all_sections: dict) -> dict:
        return _build_l_input(all_sections)
'''

TWO_PARAM_AGENT_SRC = '''
class TwoParamAgent:
    async def run(self, provider_id: str, model_id: str, all_sections: dict, extra: dict) -> dict:
        return {}
'''

DYNAMIC_LETTERS_AGENT_SRC = '''
def _letters():
    return "XY"

class DynamicAgent:
    async def run(self, provider_id: str, model_id: str, all_sections: dict) -> dict:
        out = {}
        for letter in _letters():
            s = all_sections.get(letter) or {}
            out[letter] = s.get("summary")
        return out
'''


def _write(tmp_path: Path, rel: str, src: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(src, encoding="utf-8")
    return p


def _component(agent_id: str, class_name: str, file_rel: str) -> Component:
    return Component(id=agent_id, role="unknown", entry_point=f"{file_rel[:-3].replace('/', '.')}:{class_name}", file=file_rel)


def test_direct_and_cross_file_preview_dict_reads_like_auditor(tmp_path: Path) -> None:
    files = [
        _write(tmp_path, "pkg/compressor.py", COMPRESSOR_SRC),
        _write(tmp_path, "pkg/agent_k.py", AGENT_K_SRC),
    ]
    asts = _parse_files(files)
    system_map = SystemMap(target_system_id="t", components=[_component("K", "KAgent", "pkg/agent_k.py")])
    flow = AgentFlowMap(target_system_id="t", agents=[AgentFlow(id="K", component_ids=["K"])])

    by_agent, notes = harvest_field_downstream_consumers(flow, system_map, asts)

    assert notes == []
    assert by_agent["K"] == {
        "A": ["blind_spots", "confidence", "domain", "purpose"],
        "B": ["blind_spots", "confidence", "main_layers"],
        "C": ["blind_spots", "confidence", "folders", "summary"],
    }


def test_literal_letter_and_second_helper_like_synthesizer(tmp_path: Path) -> None:
    files = [
        _write(tmp_path, "pkg/compressor.py", COMPRESSOR_SRC),
        _write(tmp_path, "pkg/agent_l.py", AGENT_L_SRC),
    ]
    asts = _parse_files(files)
    system_map = SystemMap(target_system_id="t", components=[_component("L", "LAgent", "pkg/agent_l.py")])
    flow = AgentFlowMap(target_system_id="t", agents=[AgentFlow(id="L", component_ids=["L"])])

    by_agent, notes = harvest_field_downstream_consumers(flow, system_map, asts)

    assert notes == []
    assert by_agent["L"] == {
        "A": ["domain", "purpose"],
        "B": ["main_layers"],
        "K": ["notes", "overall_confidence"],
    }


def test_agent_with_two_required_params_is_out_of_scope_silently(tmp_path: Path) -> None:
    files = [_write(tmp_path, "pkg/agent_two.py", TWO_PARAM_AGENT_SRC)]
    asts = _parse_files(files)
    system_map = SystemMap(target_system_id="t", components=[_component("TWO", "TwoParamAgent", "pkg/agent_two.py")])
    flow = AgentFlowMap(target_system_id="t", agents=[AgentFlow(id="TWO", component_ids=["TWO"])])

    by_agent, notes = harvest_field_downstream_consumers(flow, system_map, asts)

    assert by_agent == {}
    assert notes == []  # not fan-in shaped -> silently out of scope, not an error


def test_dynamic_letters_source_yields_no_resolvable_fields_and_a_note(tmp_path: Path) -> None:
    files = [_write(tmp_path, "pkg/agent_dyn.py", DYNAMIC_LETTERS_AGENT_SRC)]
    asts = _parse_files(files)
    system_map = SystemMap(target_system_id="t", components=[_component("DYN", "DynamicAgent", "pkg/agent_dyn.py")])
    flow = AgentFlowMap(target_system_id="t", agents=[AgentFlow(id="DYN", component_ids=["DYN"])])

    by_agent, notes = harvest_field_downstream_consumers(flow, system_map, asts)

    assert by_agent == {}
    assert any("DYN" in n and "no statically-resolvable" in n for n in notes)
