"""Observe-first experiment: runtime vs static component diff.

Standalone diagnostic script that captures component_ids at runtime via HarnessTracer
and compares against the static SystemMap to detect scanner over-detection and missed
components.
"""
from __future__ import annotations

import asyncio
import importlib
import logging
import sys

from haystack.tracing import disable_tracing, enable_tracing

from agent_eval_harness.instrumentation.tier1_haystack import HarnessTracer, _normalize
from agent_eval_harness.mapping.engine import map_spans_to_components
from agent_eval_harness.mapping.system_map import load_system_map

logger = logging.getLogger("agent_eval_harness.discovery.a11_runtime_diff")


async def main(system_map_path: str, target_module: str | None = None) -> None:
    """
    Load system_map from YAML, run a single trace pass through target_module,
    collect component_ids from captured spans, diff against static components.

    Args:
        system_map_path: Path to system_map.yaml
        target_module: Optional module name to import and run (must have main() or run_pipeline())

    Prints three sections: MATCHED, STATIC-ONLY, RUNTIME-ONLY.
    Logs WARNING if RUNTIME-ONLY is non-empty (scanner missed components).
    """
    system_map = load_system_map(system_map_path)
    static_ids = {c.id for c in system_map.components}

    logger.info(
        f"Loaded system map: {system_map.target_system_id} "
        f"with {len(static_ids)} components"
    )

    tracer = HarnessTracer()
    enable_tracing(tracer)

    try:
        if target_module:
            try:
                module = importlib.import_module(target_module)
                if hasattr(module, "run_pipeline"):
                    result = module.run_pipeline()
                    if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                        await result
                elif hasattr(module, "main"):
                    result = module.main()
                    if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                        await result
                else:
                    logger.warning(
                        f"No run_pipeline() or main() found in {target_module}"
                    )
            except Exception as e:
                logger.warning(f"Failed to run target module {target_module}: {e}")
        else:
            logger.info(
                "No target module provided; no spans will be captured. "
                "Pass --target-module to capture runtime components."
            )

        normalized_spans = [_normalize(s) for s in tracer.captured]
        if normalized_spans:
            map_spans_to_components(normalized_spans, system_map)

        # Collect runtime component IDs (set by map_spans_to_components)
        runtime_ids = set()
        for span in normalized_spans:
            if span.component_id:
                runtime_ids.add(span.component_id)

        if not runtime_ids and tracer.captured:
            logger.warning(
                f"Captured {len(tracer.captured)} spans but no component_ids mapped"
            )

        matched = static_ids & runtime_ids
        static_only = static_ids - runtime_ids
        runtime_only = runtime_ids - static_ids

        print("\n=== MATCHED ===")
        for cid in sorted(matched):
            print(f"  {cid}")
        if not matched:
            print("  (none)")

        print("\n=== STATIC-ONLY (scanner over-detected) ===")
        for cid in sorted(static_only):
            component = system_map.component_by_id(cid)
            role = component.role if component else "?"
            print(f"  {cid} ({role})")
        if not static_only:
            print("  (none)")

        print("\n=== RUNTIME-ONLY (scanner missed) ===")
        for cid in sorted(runtime_only):
            print(f"  {cid}")
        if not runtime_only:
            print("  (none)")

        if runtime_only:
            logger.warning(
                f"Scanner missed {len(runtime_only)} components at runtime: "
                f"{sorted(runtime_only)}"
            )

    finally:
        disable_tracing()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )

    if len(sys.argv) < 2:
        print(
            "Usage: python -m agent_eval_harness.discovery.a11_runtime_diff "
            "<system_map.yaml> [--target-module <module>]"
        )
        sys.exit(1)

    system_map_path = sys.argv[1]
    target_module = None

    if "--target-module" in sys.argv:
        idx = sys.argv.index("--target-module")
        if idx + 1 < len(sys.argv):
            target_module = sys.argv[idx + 1]

    asyncio.run(main(system_map_path, target_module))
