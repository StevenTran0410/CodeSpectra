"""Deterministic report diff (section hashes + per-field deltas)."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field


def compute_section_hash(section_data: dict) -> str:
    payload = json.dumps(section_data, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _item_sig(item: object) -> str:
    if isinstance(item, dict):
        return json.dumps(item, sort_keys=True, default=str)
    return json.dumps(item, sort_keys=True, default=str)


def _conf_rank(v: str) -> int:
    t = (v or "").strip().lower()
    if t == "high":
        return 3
    if t == "medium":
        return 2
    if t == "low":
        return 1
    return 0


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _collect_list_diffs(old_d: dict, new_d: dict) -> tuple[list[str], list[str]]:
    added: list[str] = []
    removed: list[str] = []
    keys = set(old_d.keys()) | set(new_d.keys())
    for key in keys:
        ov, nv = old_d.get(key), new_d.get(key)
        if not (isinstance(ov, list) and isinstance(nv, list)):
            continue
        osig = {_item_sig(x) for x in ov}
        nsig = {_item_sig(x) for x in nv}
        for s in nsig - osig:
            added.append(f"{key}:{s[:240]}")
        for s in osig - nsig:
            removed.append(f"{key}:{s[:240]}")
    return added, removed


def _j_risk_keys(f: dict) -> str:
    return json.dumps(
        {"title": f.get("title", ""), "severity": f.get("severity", "")},
        sort_keys=True,
    )


def _diff_j_findings(old_d: dict, new_d: dict) -> tuple[list[str], list[str]]:
    old_f = [x for x in (old_d.get("findings") or []) if isinstance(x, dict)]
    new_f = [x for x in (new_d.get("findings") or []) if isinstance(x, dict)]
    ok = {_j_risk_keys(f): _item_sig(f) for f in old_f}
    nk = {_j_risk_keys(f): _item_sig(f) for f in new_f}
    added = [nk[s] for s in nk.keys() - ok.keys()]
    removed = [ok[s] for s in ok.keys() - nk.keys()]
    return [a[:300] for a in added], [r[:300] for r in removed]


def _k_score_changes(old_scores: dict, new_scores: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    letters = set(old_scores.keys()) | set(new_scores.keys())
    for letter in letters:
        o = str(old_scores.get(letter, "") or "").lower()
        n = str(new_scores.get(letter, "") or "").lower()
        if o == n:
            continue
        if _conf_rank(n) > _conf_rank(o):
            out[str(letter)] = f"{o or '?'} → {n or '?'}"
        elif _conf_rank(n) < _conf_rank(o):
            out[str(letter)] = f"{o or '?'} → {n or '?'}"
        else:
            out[str(letter)] = f"{o or '?'} → {n or '?'}"
    return out


def diff_section(section_letter: str, old_data: dict, new_data: dict) -> dict:
    oc = old_data.get("confidence")
    nc = new_data.get("confidence")
    conf_delta: str | None = None
    if isinstance(oc, str) and isinstance(nc, str) and oc != nc:
        conf_delta = f"{oc} → {nc}"
    elif isinstance(oc, str) ^ isinstance(nc, str):
        conf_delta = f"{oc or '?'} → {nc or '?'}"

    otxt = old_data.get("content")
    ntxt = new_data.get("content")
    wc_delta_pct: float | None = None
    if isinstance(otxt, str) and isinstance(ntxt, str):
        ow = max(1, _word_count(otxt))
        nw = _word_count(ntxt)
        pct = abs(nw - ow) / ow
        if pct > 0.1:
            wc_delta_pct = round(pct * 100, 1)

    added: list[str] = []
    removed: list[str] = []
    if section_letter == "J":
        added, removed = _diff_j_findings(old_data, new_data)
    else:
        la, lr = _collect_list_diffs(old_data, new_data)
        added, removed = la, lr

    score_changes: dict[str, str] = {}
    if section_letter == "K":
        os_ = old_data.get("section_scores") or {}
        ns_ = new_data.get("section_scores") or {}
        if isinstance(os_, dict) and isinstance(ns_, dict):
            score_changes = _k_score_changes(os_, ns_)
        ooc = str(old_data.get("overall_confidence") or "")
        noc = str(new_data.get("overall_confidence") or "")
        if ooc != noc:
            if conf_delta is None:
                conf_delta = f"{ooc or '?'} → {noc or '?'}"

    improvement: bool | None = None
    if section_letter == "K":
        ro = _conf_rank(str(old_data.get("overall_confidence") or ""))
        rn = _conf_rank(str(new_data.get("overall_confidence") or ""))
        if ro and rn:
            if rn > ro:
                improvement = True
            elif rn < ro:
                improvement = False
    elif isinstance(oc, str) and isinstance(nc, str):
        ro, rn = _conf_rank(oc), _conf_rank(nc)
        if ro and rn:
            if rn > ro:
                improvement = True
            elif rn < ro:
                improvement = False

    return {
        "letter": section_letter,
        "changed": True,
        "skipped_by_hash": False,
        "confidence_delta": conf_delta,
        "content_word_delta_pct": wc_delta_pct,
        "list_added": added[:80],
        "list_removed": removed[:80],
        "section_score_changes": score_changes,
        "improvement": improvement,
    }


def _avg_k_score(section_k: dict | None) -> float | None:
    if not isinstance(section_k, dict):
        return None
    scores = section_k.get("section_scores") or {}
    if not isinstance(scores, dict) or not scores:
        return None
    nums = [_conf_rank(str(v).lower()) for v in scores.values()]
    return sum(nums) / len(nums)


def _sections_dict(report: dict) -> dict[str, dict]:
    ver = report.get("version")
    blob = report.get("sections") if ver in (2, 3) else report.get("sections_v2")
    if not isinstance(blob, dict):
        return {}
    out: dict[str, dict] = {}
    for letter, data in blob.items():
        if isinstance(data, dict):
            out[str(letter)] = data
    return out


@dataclass
class CompareResult:
    quality_trend: str
    sections_changed: int
    section_diffs: dict[str, dict] = field(default_factory=dict)
    identical: bool = False


def compare_reports(report_a: dict, report_b: dict) -> CompareResult:
    sec_a = _sections_dict(report_a)
    sec_b = _sections_dict(report_b)
    hashes_a = report_a.get("section_hashes")
    hashes_b = report_b.get("section_hashes")
    letters = sorted(set(sec_a.keys()) | set(sec_b.keys()), key=lambda x: (len(x) > 1, x))

    diffs: dict[str, dict] = {}
    changed_count = 0

    for letter in letters:
        da, db = sec_a.get(letter), sec_b.get(letter)
        if not isinstance(da, dict) and not isinstance(db, dict):
            continue
        if not isinstance(da, dict):
            da = {}
        if not isinstance(db, dict):
            db = {}

        skip = False
        if isinstance(hashes_a, dict) and isinstance(hashes_b, dict):
            ha, hb = hashes_a.get(letter), hashes_b.get(letter)
            if isinstance(ha, str) and isinstance(hb, str) and ha == hb:
                skip = True

        if skip:
            diffs[letter] = {
                "letter": letter,
                "changed": False,
                "skipped_by_hash": True,
                "confidence_delta": None,
                "content_word_delta_pct": None,
                "list_added": [],
                "list_removed": [],
                "section_score_changes": {},
                "improvement": None,
            }
            continue

        if compute_section_hash(da) == compute_section_hash(db):
            diffs[letter] = {
                "letter": letter,
                "changed": False,
                "skipped_by_hash": False,
                "confidence_delta": None,
                "content_word_delta_pct": None,
                "list_added": [],
                "list_removed": [],
                "section_score_changes": {},
                "improvement": None,
            }
            continue

        entry = diff_section(letter, da, db)
        diffs[letter] = entry
        changed_count += 1

    ka = sec_a.get("K")
    kb = sec_b.get("K")
    av_a = _avg_k_score(ka if isinstance(ka, dict) else None)
    av_b = _avg_k_score(kb if isinstance(kb, dict) else None)
    trend = "stable"
    if av_a is not None and av_b is not None:
        if av_b > av_a + 0.15:
            trend = "improving"
        elif av_b < av_a - 0.15:
            trend = "degrading"

    identical = changed_count == 0

    return CompareResult(
        quality_trend=trend,
        sections_changed=changed_count,
        section_diffs=diffs,
        identical=identical,
    )
