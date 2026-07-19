"""Assertion: referential_integrity — evidence/file-path fields are subset of snapshot manifest + retrieved chunks."""
from __future__ import annotations

from agent_eval_harness.metrics.assertions.registry import find_component_output_json, register
from agent_eval_harness.metrics.types import MetricResult


def _extract_referenced_paths(obj: dict, field_names: list[str] | None = None) -> set[str]:
    """Extract file paths from specified fields in an object (any depth)."""
    if not field_names:
        return set()
    paths: set[str] = set()
    for field in field_names:
        _extract_field_paths(obj, field, paths)
    return paths


def _extract_field_paths(obj: dict, field_name: str, paths: set[str]) -> None:
    """Recursively extract values of a named field from nested objects/lists."""
    if isinstance(obj, dict):
        if field_name in obj:
            val = obj[field_name]
            if isinstance(val, str):
                paths.add(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, str):
                        paths.add(item)
                    elif isinstance(item, dict) and field_name in item:
                        _extract_field_paths(item, field_name, paths)
        for v in obj.values():
            if isinstance(v, dict):
                _extract_field_paths(v, field_name, paths)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        _extract_field_paths(item, field_name, paths)


@register("referential_integrity")
def referential_integrity(spans: list[dict], component_id: str, params: dict) -> MetricResult:
    """Check that output paths are subset of allowed paths (manifest + retrieved chunks)."""
    field_names = params.get("path_fields", [])
    if not field_names:
        return MetricResult(
            metric_name="assertion.referential_integrity",
            metric_class="assertion",
            score=None,
            passed=None,
            details={"reason": "no_path_fields_specified"},
            component_id=component_id,
        )

    output_data, early_result = find_component_output_json(
        spans, component_id, "assertion.referential_integrity"
    )
    if early_result is not None:
        return early_result

    if output_data is None:
        return MetricResult(
            metric_name="assertion.referential_integrity",
            metric_class="assertion",
            score=None,
            passed=None,
            details={"reason": "no_result"},
            component_id=component_id,
        )

    referenced_paths = _extract_referenced_paths(output_data, field_names)

    allowed_paths: set[str] = set()
    manifest_paths = params.get("snapshot_manifest_paths", [])
    if manifest_paths:
        allowed_paths.update(manifest_paths)
    retrieved_paths = params.get("retrieved_chunk_rel_paths", [])
    if retrieved_paths:
        allowed_paths.update(retrieved_paths)

    if not allowed_paths:
        return MetricResult(
            metric_name="assertion.referential_integrity",
            metric_class="assertion",
            score=None,
            passed=None,
            details={"reason": "no_allowed_paths_available"},
            component_id=component_id,
        )

    violations = referenced_paths - allowed_paths
    passed = len(violations) == 0

    return MetricResult(
        metric_name="assertion.referential_integrity",
        metric_class="assertion",
        score=None,
        passed=passed,
        details={
            "referenced_paths": sorted(referenced_paths),
            "allowed_paths": sorted(allowed_paths),
            "violations": sorted(violations),
        },
        component_id=component_id,
    )
