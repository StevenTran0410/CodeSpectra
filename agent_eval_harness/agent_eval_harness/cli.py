"""`aeh` CLI entry point.

CRITICAL: HAYSTACK_CONTENT_TRACING_ENABLED must be set before `haystack` is
imported anywhere in this process — it's read exactly once, at haystack.tracing's
first import. These two lines MUST stay above every other import in this file,
including this package's own modules (several transitively import haystack).
"""
import os

os.environ.setdefault("HAYSTACK_CONTENT_TRACING_ENABLED", "true")
# We register our own HarnessTracer explicitly, so auto-detection is unneeded.
os.environ.setdefault("HAYSTACK_AUTO_TRACE_ENABLED", "false")

import argparse  # noqa: E402
import asyncio  # noqa: E402
import sys  # noqa: E402

from agent_eval_harness.config import AEHConfig  # noqa: E402
from agent_eval_harness.llm.client import LLMResponse  # noqa: E402
from agent_eval_harness.llm.fake_client import FakeLLMClient  # noqa: E402
from agent_eval_harness.llm.proxy_client import CodeSpectraProxyClient  # noqa: E402
from agent_eval_harness.mapping.system_map import load_system_map  # noqa: E402
from agent_eval_harness.reporting import (  # noqa: E402
    DEFAULT_UNMATCHED_WARN_THRESHOLD,
    format_trace,
)
from agent_eval_harness.runner import execute_run  # noqa: E402
from agent_eval_harness.store.database import close_db, init_db  # noqa: E402
from test_targets._shared.defects import DefectConfig  # noqa: E402


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aeh")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a query through an instrumented target")
    run_parser.add_argument(
        "--target", required=True, help="module:function reference to the target's build_pipeline"
    )
    run_parser.add_argument("--map", dest="map_path", required=True, help="path to system_map.yaml")
    run_parser.add_argument("--query", help="single query string")
    run_parser.add_argument("--input-file", dest="input_file", help="file of queries, one per line")
    run_parser.add_argument("--tier", default="auto", choices=["auto", "1", "2"])
    run_parser.add_argument("--provider-id", dest="provider_id", default=None)
    run_parser.add_argument("--backend-url", dest="backend_url", default=None)
    run_parser.add_argument("--backend-token", dest="backend_token", default=None)
    run_parser.add_argument(
        "--unmatched-warn-threshold",
        dest="unmatched_warn_threshold",
        type=float,
        default=DEFAULT_UNMATCHED_WARN_THRESHOLD,
    )
    run_parser.add_argument("--data-dir", dest="data_dir", default=None)
    run_parser.add_argument("--json", action="store_true")

    # dataset parser group
    dataset_parser = subparsers.add_parser(
        "dataset", help="Dataset synthesis and management commands"
    )
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command", required=True)

    gen_parser = dataset_subparsers.add_parser("generate", help="Generate a new dataset version")
    gen_parser.add_argument(
        "--kind",
        required=True,
        choices=["guard_classification", "qa_testset", "decomposition_gold", "sufficiency_labeled"],
    )
    gen_parser.add_argument(
        "--config", required=True, help="Path to generator configuration YAML file"
    )
    gen_parser.add_argument("--seed", type=int, default=None, help="Random seed for generation")
    gen_parser.add_argument("--provider-id", dest="provider_id", default=None)
    gen_parser.add_argument("--backend-url", dest="backend_url", default=None)
    gen_parser.add_argument("--backend-token", dest="backend_token", default=None)
    gen_parser.add_argument("--data-dir", dest="data_dir", default=None)

    rev_parser = dataset_subparsers.add_parser("review", help="Review synthetic cases in a dataset")
    rev_parser.add_argument("dataset_id", help="Dataset ID to review")
    rev_parser.add_argument("--data-dir", dest="data_dir", default=None)

    ls_parser = dataset_subparsers.add_parser("ls", help="List all datasets and their summaries")
    ls_parser.add_argument("--data-dir", dest="data_dir", default=None)

    return parser


def _read_queries(args: argparse.Namespace) -> list[str]:
    if args.input_file:
        with open(args.input_file, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    if args.query:
        return [args.query]
    raise SystemExit("aeh run requires --query or --input-file")


async def _run_command(args: argparse.Namespace) -> int:
    if args.data_dir:
        os.environ["AEH_DATA_DIR"] = args.data_dir

    config = AEHConfig.load()
    provider_id = args.provider_id or config.provider_id

    if provider_id:
        backend_url = args.backend_url or config.backend_url
        backend_token = args.backend_token or config.backend_token
        if not backend_url or not backend_token:
            raise SystemExit(
                "--provider-id requires --backend-url/--backend-token (or .aeh/config.yaml)"
            )
        llm_client = CodeSpectraProxyClient(
            backend_url, backend_token, provider_id, config.model_id
        )
    else:
        llm_client = FakeLLMClient(
            LLMResponse(content="This is a fallback offline demo answer.", model="fake-default")
        )

    await init_db()
    try:
        system_map = load_system_map(args.map_path)
        defects = DefectConfig.from_env()
        active = defects.active_names()
        print(f"[aeh] run target={args.target} map={args.map_path} tier={args.tier}")
        print(f"[aeh] active defects: {', '.join(active) if active else 'none'}")

        queries = _read_queries(args)
        outcomes = await execute_run(args.target, args.map_path, llm_client, queries, args.tier)

        for i, outcome in enumerate(outcomes, start=1):
            print(f'[aeh] trace {i}  query="{outcome.trace_result.root_input}"')
            print(format_trace(outcome.trace_result, system_map, args.unmatched_warn_threshold))

        run_id = outcomes[0].run_id if outcomes else "n/a"
        print(f"[aeh] run completed  status=completed  run_id={run_id}")
        return 0
    finally:
        await close_db()


async def _dataset_command(args: argparse.Namespace) -> int:
    if args.data_dir:
        os.environ["AEH_DATA_DIR"] = args.data_dir

    await init_db()
    try:
        from agent_eval_harness.store import repository

        if args.dataset_command == "generate":
            from pathlib import Path

            import yaml

            from agent_eval_harness.datasets.jsonl_io import write_jsonl
            from agent_eval_harness.datasets.registry import get_generator
            from agent_eval_harness.datasets.versioning import next_version

            config = AEHConfig.load()
            provider_id = args.provider_id or config.provider_id

            if provider_id:
                backend_url = args.backend_url or config.backend_url
                backend_token = args.backend_token or config.backend_token
                if not backend_url or not backend_token:
                    raise SystemExit(
                        "--provider-id requires --backend-url/--backend-token (or .aeh/config.yaml)"
                    )
                llm_client = CodeSpectraProxyClient(
                    backend_url, backend_token, provider_id, config.model_id
                )
            else:
                llm_client = FakeLLMClient(
                    LLMResponse(
                        content="This is a fallback offline demo answer.",
                        model="fake-default",
                    )
                )

            with open(args.config, encoding="utf-8") as f:
                gen_config = yaml.safe_load(f)

            base_name = gen_config.get("dataset_name", args.kind)
            dataset_id = await next_version(base_name)
            gen_config["dataset_name"] = dataset_id

            generator_fn = get_generator(args.kind)
            cases = await generator_fn(gen_config, llm_client, args.seed)

            output_dir = Path(os.environ.get("AEH_DATA_DIR", "."))
            output_path = output_dir / f"{dataset_id}.jsonl"
            write_jsonl(cases, output_path)

            await repository.insert_dataset_cases_bulk(dataset_id, cases)

            print(f"[aeh] Generated {len(cases)} cases for dataset {dataset_id}")
            print(f"[aeh] JSONL written to {output_path}")
            return 0

        elif args.dataset_command == "review":
            from agent_eval_harness.datasets.review import run_review_loop
            await run_review_loop(args.dataset_id)
            return 0

        elif args.dataset_command == "ls":
            import json
            dataset_summaries = await repository.list_dataset_ids()
            if not dataset_summaries:
                print("[aeh] No datasets found in database.")
                return 0

            print("Registered Datasets:\n")
            for ds in dataset_summaries:
                dataset_id = ds["dataset_id"]
                cases = await repository.get_dataset_cases(dataset_id)

                categories = {}
                for case in cases:
                    cat = None
                    if case["labels_json"]:
                        try:
                            cat = json.loads(case["labels_json"]).get("category")
                        except Exception:
                            pass
                    if not cat and case["input_json"]:
                        try:
                            cat = json.loads(case["input_json"]).get("category")
                        except Exception:
                            pass
                    if not cat:
                        cat = "unknown"
                    categories[cat] = categories.get(cat, 0) + 1

                if ds["synthetic_count"] == 0:
                    status = "fully reviewed"
                elif ds["reviewed_count"] > 0:
                    status = "partially reviewed"
                else:
                    status = "pending review"

                cats_str = ", ".join(f"{k}:{v}" for k, v in categories.items())
                print(f"Dataset: {dataset_id}")
                print(f"  Total Cases: {ds['total_count']}")
                print(
                    f"  Provenance:  synthetic={ds['synthetic_count']}, "
                    f"handwritten={ds['handwritten_count']}, "
                    f"reviewed={ds['reviewed_count']}"
                )
                print(f"  Status:      {status}")
                if cats_str:
                    print(f"  Categories:  {cats_str}")
                print()
            return 0
    finally:
        await close_db()


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return asyncio.run(_run_command(args))
    elif args.command == "dataset":
        return asyncio.run(_dataset_command(args))
    parser.error(f"unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
