"""`aeh` CLI entry point. CRITICAL: HAYSTACK_CONTENT_TRACING_ENABLED must be set before haystack is imported."""
import os

os.environ.setdefault("HAYSTACK_CONTENT_TRACING_ENABLED", "true")
# We register our own HarnessTracer explicitly, so auto-detection is unneeded.
os.environ.setdefault("HAYSTACK_AUTO_TRACE_ENABLED", "false")

import argparse  # noqa: E402
import asyncio  # noqa: E402
import sys  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

from agent_eval_harness.config import AEHConfig  # noqa: E402
from agent_eval_harness.llm.client import LLMClient, LLMResponse  # noqa: E402
from agent_eval_harness.llm.embedding_client import EmbeddingClient  # noqa: E402
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


def _add_provider_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider-id", dest="provider_id", default=None)
    parser.add_argument("--backend-url", dest="backend_url", default=None)
    parser.add_argument("--backend-token", dest="backend_token", default=None)


def _add_embedding_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedding-provider-id", dest="embedding_provider_id", default=None)
    parser.add_argument("--embedding-model-id", dest="embedding_model_id", default=None)
    parser.add_argument("--use-local-embedding", dest="use_local_embedding", action="store_true")


def _add_data_dir_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", dest="data_dir", default=None)


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
    _add_provider_args(run_parser)
    run_parser.add_argument(
        "--unmatched-warn-threshold",
        dest="unmatched_warn_threshold",
        type=float,
        default=DEFAULT_UNMATCHED_WARN_THRESHOLD,
    )
    _add_data_dir_arg(run_parser)
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
        choices=[
            "guard_classification", "qa_testset", "decomposition_gold", "sufficiency_labeled",
            "snapshot_fixture", "field_match_gold",
        ],
    )
    gen_parser.add_argument(
        "--config", required=True, help="Path to generator configuration YAML file"
    )
    gen_parser.add_argument("--seed", type=int, default=None, help="Random seed for generation")
    _add_provider_args(gen_parser)
    _add_embedding_args(gen_parser)
    _add_data_dir_arg(gen_parser)

    ls_parser = dataset_subparsers.add_parser("ls", help="List all datasets and their summaries")
    _add_data_dir_arg(ls_parser)

    # eval subcommand group
    eval_parser = subparsers.add_parser("eval", help="Run an evaluation sweep against a target")
    eval_parser.add_argument(
        "--target", required=True, help="module:function reference to the target's build_pipeline"
    )
    eval_parser.add_argument(
        "--map", dest="map_path", required=True, help="path to system_map.yaml"
    )
    eval_parser.add_argument("--suite", help="path to suite.yaml")
    eval_parser.add_argument("--plan", help="path to eval_plan.yaml (alias for --suite)")
    eval_parser.add_argument("--tier", default="auto", choices=["auto", "1", "2"])
    _add_provider_args(eval_parser)
    _add_data_dir_arg(eval_parser)
    eval_parser.add_argument("--concurrency", type=int, default=4)
    eval_parser.add_argument("--json", action="store_true")

    # plan subcommand group
    plan_parser = subparsers.add_parser("plan", help="Evaluation plan commands")
    plan_subparsers = plan_parser.add_subparsers(dest="plan_command", required=False)

    # validate subcommand: aeh plan validate --plan <path>
    validate_parser = plan_subparsers.add_parser("validate", help="Validate an evaluation plan")
    validate_parser.add_argument("--plan", required=True, help="path to eval_plan.yaml")
    _add_data_dir_arg(validate_parser)

    # plan generator command (when plan_command is not validate)
    plan_parser.add_argument("--map", dest="map_path", help="path to system_map.yaml")
    plan_parser.add_argument("--output", dest="output_path", help="path to output eval_plan.yaml")
    _add_provider_args(plan_parser)
    _add_data_dir_arg(plan_parser)

    # report subcommand group
    report_parser = subparsers.add_parser("report", help="Print an evaluation report for a run")
    report_parser.add_argument("--run", dest="run_id", required=True, help="run ID to report on")
    _add_data_dir_arg(report_parser)
    report_parser.add_argument("--json", action="store_true")

    # map subcommand group
    map_parser = subparsers.add_parser("map", help="Build a system_map.yaml from target source")
    map_parser.add_argument("--target", required=True, help="directory of Python source to scan")
    map_parser.add_argument("--docs", dest="docs_path", default=None)
    map_parser.add_argument("--output", dest="output_path", default=None)
    map_parser.add_argument(
        "--confidence-threshold", dest="confidence_threshold", type=float, default=0.7
    )
    _add_provider_args(map_parser)

    # ui subcommand group
    ui_parser = subparsers.add_parser("ui", help="Start the evaluation dashboard web server")
    ui_parser.add_argument("--port", type=int, default=8321, help="Port to bind the server to")
    ui_parser.add_argument("--host", default="127.0.0.1", help="Host address to bind the server to")
    _add_data_dir_arg(ui_parser)

    # discover subcommand group
    discover_parser = subparsers.add_parser("discover", help="Discover candidate agentic systems")
    discover_parser.add_argument("--snapshot", required=True, help="Snapshot ID to scan")
    discover_parser.add_argument("--repo-ref", help="Optional reference repository path/identifier")
    _add_provider_args(discover_parser)
    _add_data_dir_arg(discover_parser)
    discover_parser.add_argument("--json", action="store_true")

    # enrich subcommand group
    enrich_parser = subparsers.add_parser("enrich", help="Enrich discovered agents with LLM analysis")
    enrich_parser.add_argument("session_id", help="Expansion session ID")
    enrich_parser.add_argument("--depth", choices=["normal", "deep"], default="normal")
    enrich_parser.add_argument("--agent-ids", dest="agent_ids", nargs="+", help="Specific agents to enrich")
    enrich_parser.add_argument("--force", action="store_true", help="Force re-enrichment")
    enrich_parser.add_argument("--offline", action="store_true", help="Offline mode: list cached knowledge")
    _add_provider_args(enrich_parser)
    _add_data_dir_arg(enrich_parser)

    return parser


def _read_queries(args: argparse.Namespace) -> list[str]:
    if args.input_file:
        with open(args.input_file, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    if args.query:
        return [args.query]
    raise SystemExit("aeh run requires --query or --input-file")


def _apply_data_dir(args: argparse.Namespace) -> None:
    """Point the store at a custom data directory before init_db() reads AEH_DATA_DIR."""
    if args.data_dir:
        os.environ["AEH_DATA_DIR"] = args.data_dir


def _build_llm_client(args: argparse.Namespace, config: AEHConfig) -> LLMClient:
    """Resolve the LLM client: live CodeSpectraProxyClient if provider set, else FakeLLMClient."""
    provider_id = args.provider_id or config.provider_id

    if provider_id:
        backend_url = args.backend_url or config.backend_url
        backend_token = args.backend_token or config.backend_token
        if not backend_url or not backend_token:
            raise SystemExit(
                "--provider-id requires --backend-url/--backend-token (or .aeh/config.yaml)"
            )
        return CodeSpectraProxyClient(
            backend_url, backend_token, provider_id, config.model_id,
            reasoning_effort=config.reasoning_effort, thinking_budget=config.thinking_budget,
        )

    return FakeLLMClient(
        LLMResponse(content="This is a fallback offline demo answer.", model="fake-default")
    )


def _build_embedding_client(args: argparse.Namespace, config: AEHConfig) -> EmbeddingClient | None:
    """Resolve the embedding client: live CodeSpectraEmbeddingProxyClient or None."""
    use_local = getattr(args, "use_local_embedding", False)
    provider_id = getattr(args, "embedding_provider_id", None)
    model_id = getattr(args, "embedding_model_id", None)

    # Fail early at argparse/UI resolve level if qa_testset is requested but no embed flags provided
    if getattr(args, "kind", None) == "qa_testset" and not use_local and not provider_id:
        raise SystemExit(
            "[aeh] Error: --kind qa_testset requires either --use-local-embedding or --embedding-provider-id."
        )

    if use_local or provider_id:
        backend_url = args.backend_url or config.backend_url
        backend_token = args.backend_token or config.backend_token
        if not backend_url or not backend_token:
            raise SystemExit(
                "[aeh] Error: embedding configuration requires --backend-url/--backend-token (or .aeh/config.yaml)"
            )
        from agent_eval_harness.llm.embedding_client import CodeSpectraEmbeddingProxyClient
        return CodeSpectraEmbeddingProxyClient(
            backend_url,
            backend_token,
            provider_id=provider_id,
            model_id=model_id,
            use_local=use_local,
        )
    return None


@asynccontextmanager
async def _db_session():
    """Open the store DB for a command's lifetime and close it after."""
    await init_db()
    try:
        yield
    finally:
        await close_db()


async def _run_command(args: argparse.Namespace) -> int:
    _apply_data_dir(args)
    config = AEHConfig.load()
    llm_client = _build_llm_client(args, config)

    async with _db_session():
        system_map = load_system_map(args.map_path)
        defects = DefectConfig.from_env()
        active = defects.active_names()
        print(f"[aeh] run target={args.target} map={args.map_path} tier={args.tier}")
        print(f"[aeh] active defects: {', '.join(active) if active else 'none'}")

        queries = _read_queries(args)
        outcomes = await execute_run(
            args.target,
            args.map_path,
            llm_client,
            queries,
            args.tier,
            active_defects=active,
        )

        for i, outcome in enumerate(outcomes, start=1):
            print(f'[aeh] trace {i}  query="{outcome.trace_result.root_input}"')
            print(format_trace(outcome.trace_result, system_map, args.unmatched_warn_threshold))

        run_id = outcomes[0].run_id if outcomes else "n/a"
        print(f"[aeh] run completed  status=completed  run_id={run_id}")
        return 0


async def _dataset_command(args: argparse.Namespace) -> int:
    _apply_data_dir(args)

    async with _db_session():
        from agent_eval_harness.store import repository

        if args.dataset_command == "generate":
            from pathlib import Path

            import yaml

            from agent_eval_harness.datasets.jsonl_io import write_jsonl
            from agent_eval_harness.datasets.registry import get_generator
            from agent_eval_harness.datasets.versioning import next_version

            config = AEHConfig.load()
            llm_client = _build_llm_client(args, config)

            with open(args.config, encoding="utf-8") as f:
                gen_config = yaml.safe_load(f)

            base_name = gen_config.get("dataset_name", args.kind)
            dataset_id = await next_version(base_name)
            gen_config["dataset_name"] = dataset_id

            embedding_client = None
            try:
                generator_fn = get_generator(args.kind)
                if args.kind == "qa_testset":
                    embedding_client = _build_embedding_client(args, config)
                    cases = await generator_fn(
                        gen_config, llm_client, args.seed, embedding_client=embedding_client
                    )
                else:
                    cases = await generator_fn(gen_config, llm_client, args.seed)
            finally:
                if embedding_client and hasattr(embedding_client, "aclose"):
                    await embedding_client.aclose()

            output_dir = Path(os.environ.get("AEH_DATA_DIR", "."))
            output_path = output_dir / f"{dataset_id}.jsonl"
            write_jsonl(cases, output_path)

            await repository.insert_dataset_cases_bulk(dataset_id, cases)

            print(f"[aeh] Generated {len(cases)} cases for dataset {dataset_id}")
            print(f"[aeh] JSONL written to {output_path}")
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


async def _eval_command(args: argparse.Namespace) -> int:
    _apply_data_dir(args)
    config = AEHConfig.load()
    llm_client = _build_llm_client(args, config)

    suite_path = args.suite or args.plan
    if not suite_path:
        raise SystemExit("--suite or --plan is required for eval sweep")

    async with _db_session():
        from agent_eval_harness.metrics.reporting import format_sweep_results, results_to_json
        from agent_eval_harness.metrics.sweep import run_sweep

        defects = DefectConfig.from_env()
        active = defects.active_names()
        print(f"[aeh] eval target={args.target} map={args.map_path} plan={suite_path}")
        print(f"[aeh] active defects: {', '.join(active) if active else 'none'}")

        sweep_result = await run_sweep(
            target=args.target,
            map_path=args.map_path,
            suite_path=suite_path,
            llm_client=llm_client,
            concurrency=args.concurrency,
            tier=args.tier,
            active_defects=active,
        )

        if args.json:
            print(results_to_json(sweep_result.results, sweep_result.run_id))
        else:
            print(format_sweep_results(sweep_result.results))
            if sweep_result.errors:
                print(f"\n[aeh] {len(sweep_result.errors)} entry error(s):")
                for err in sweep_result.errors:
                    print(f"  {err['entry_id']}: {err['error']}")

        print(f"[aeh] eval completed  run_id={sweep_result.run_id}")
        return 0


async def _report_command(args: argparse.Namespace) -> int:
    _apply_data_dir(args)

    async with _db_session():
        from agent_eval_harness.metrics.reporting import render_run_report

        report = await render_run_report(args.run_id, as_json=args.json)
        print(report)
        return 0


async def _map_command(args: argparse.Namespace) -> int:
    """Build a system_map.yaml from target source code."""
    from pathlib import Path

    import yaml

    from agent_eval_harness.mapping.builder.pipeline import SystemMapBuilder

    config = AEHConfig.load()
    llm_client = _build_llm_client(args, config)

    target_path = Path(args.target)
    docs_path = Path(args.docs_path) if args.docs_path else None

    builder = SystemMapBuilder(llm_client, confidence_threshold=args.confidence_threshold)
    system_map, summary = await builder.build(target_path, docs_path)

    # Write output
    output_path = Path(args.output_path) if args.output_path else target_path / "system_map.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.dump(system_map.model_dump(), default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )

    print(summary)
    print(f"[aeh] system_map written to {output_path}")

    return 0


async def _plan_command(args: argparse.Namespace) -> int:
    _apply_data_dir(args)

    from pathlib import Path

    import yaml

    from agent_eval_harness.planning.planner import generate_plan
    from agent_eval_harness.planning.validation import validate_plan

    config = AEHConfig.load()
    llm_client = _build_llm_client(args, config)

    if args.plan_command == "validate":
        async with _db_session():
            report = await validate_plan(args.plan)
            if report.errors:
                print("[aeh] Validation failed:")
                for err in report.errors:
                    print(f"  - {err}")
                return 1
            print(f"[aeh] Plan {args.plan} is VALID.")
            return 0

    # Plan generation
    if not args.map_path:
        print("[aeh] Error: --map is required for plan generation.")
        return 1

    async with _db_session():
        plan = await generate_plan(args.map_path, llm_client)

        output_path = (
            Path(args.output_path)
            if args.output_path
            else Path(args.map_path).parent / "eval_plan.yaml"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plan_dict = plan.model_dump(exclude_none=True)
        output_path.write_text(
            yaml.dump(plan_dict, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )

        print("\n=== Evaluation Plan Summary ===")
        comp_entries = {}
        for entry in plan.entries:
            comp_entries.setdefault(entry.component, []).append(entry)

        for comp_id, entries in sorted(comp_entries.items()):
            print(f"Component: {comp_id}")
            for entry in entries:
                ds_str = ""
                if entry.dataset:
                    if entry.dataset.ref:
                        ds_str = f" (dataset ref: {entry.dataset.ref})"
                    elif entry.dataset.required:
                        ds_str = f" (dataset required: {entry.dataset.required.get('kind')})"
                    elif entry.dataset.waived:
                        ds_str = f" (dataset waived: {entry.dataset.waived})"
                status_str = f" [status: {entry.status}]" if entry.status else ""
                print(f"  - Metric: {entry.metric} ({entry.metric_class}){ds_str}{status_str}")

        needs_human_entries = [
            e for e in plan.entries if e.status == "needs_human" or e.metric == "unknown"
        ]
        if needs_human_entries:
            print("\nNeeds Human Review:")
            for entry in needs_human_entries:
                print(
                    f"  - Component: {entry.component}, Metric: {entry.metric} "
                    f"({entry.rationale})"
                )
        else:
            print("\nNeeds Human Review: None")

        print(f"\n[aeh] eval_plan written to {output_path}")
        return 0


async def _discover_command(args: argparse.Namespace) -> int:
    import json
    _apply_data_dir(args)
    config = AEHConfig.load()

    backend_url = args.backend_url or config.backend_url
    backend_token = args.backend_token or config.backend_token
    if not backend_url or not backend_token:
        raise SystemExit(
            "discover command requires --backend-url/--backend-token (or .aeh/config.yaml)"
        )

    llm_client = _build_llm_client(args, config)

    from agent_eval_harness.discovery.client import CodeSpectraClient
    from agent_eval_harness.discovery.engine import discover_agentic_systems
    from agent_eval_harness.store import repository

    async with _db_session():
        client = CodeSpectraClient(backend_url, backend_token)
        try:
            repo_ref = args.repo_ref or args.snapshot
            print(f"[aeh] discover starting  snapshot={args.snapshot} repo_ref={repo_ref}")

            session_id = await repository.insert_discovery_session(repo_ref, args.snapshot)

            try:
                candidates = await discover_agentic_systems(args.snapshot, repo_ref, client, llm_client)
                await repository.insert_discovery_candidates_bulk(session_id, candidates)
                await repository.finish_discovery_session(session_id, "completed")
                print(f"[aeh] discover completed  session_id={session_id} candidates={len(candidates)}")
            except Exception as e:
                import traceback
                err_msg = "".join(traceback.format_exception(None, e, e.__traceback__))
                await repository.finish_discovery_session(session_id, "failed", err_msg[:2000])
                print(f"[aeh] discover failed  session_id={session_id} error={e}")
                raise

            # Print output
            if args.json:
                out = {
                    "session_id": session_id,
                    "status": "completed",
                    "candidates": candidates
                }
                print(json.dumps(out, indent=2))
            else:
                print("\nDiscovered Candidate Agentic Systems:")
                print(f"{'Name':<25} | {'Frameworks':<20} | {'Confidence':<10} | {'Entry Point'}")
                print("-" * 80)
                for cand in candidates:
                    frameworks = ", ".join(cand.get("frameworks", []))
                    entry = ", ".join(cand.get("entry_points", []))
                    print(f"{cand['name'][:25]:<25} | {frameworks[:20]:<20} | {cand['confidence']:<10} | {entry}")
                print()
        finally:
            await client.aclose()

    return 0


async def _enrich_command(args: argparse.Namespace) -> int:
    import json
    _apply_data_dir(args)

    from agent_eval_harness.store import repository

    async with _db_session():
        if args.offline:
            # Offline mode: list cached knowledge
            knowledge_list = await repository.list_agent_knowledge(args.session_id)
            if args.json:
                print(json.dumps(knowledge_list, indent=2))
            else:
                print(f"[aeh] Cached agent knowledge for session {args.session_id}:")
                for item in knowledge_list:
                    print(f"  - {item['agent_id']}: confidence={item['confidence']}, queries={item['query_count']}")
            return 0

        # Online enrichment
        config = AEHConfig.load()
        backend_url = args.backend_url or config.backend_url
        backend_token = args.backend_token or config.backend_token
        if not backend_url or not backend_token:
            raise SystemExit(
                "enrich command requires --backend-url/--backend-token (or .aeh/config.yaml)"
            )

        llm_client = _build_llm_client(args, config)

        from agent_eval_harness.discovery.client import CodeSpectraClient
        from agent_eval_harness.discovery.enrichment import enrich_agents
        from agent_eval_harness.mapping.agent_flow import load_agent_flow_map
        from agent_eval_harness.mapping.system_map import load_system_map

        client = CodeSpectraClient(backend_url, backend_token)

        try:
            # Load session and related data
            sess = await repository.get_expansion_session(args.session_id)
            if not sess:
                raise SystemExit(f"Session {args.session_id} not found")

            system_map = load_system_map(sess["map_path"])
            agent_flow_map = load_agent_flow_map(sess["agent_flows_path"])

            force_ids = [args.session_id] if args.force else []

            knowledge_list = await enrich_agents(
                session_id=args.session_id,
                agent_flow_map=agent_flow_map,
                system_map=system_map,
                accepted_with_annotations=sess.get("accepted", []),
                accepted_edges=sess.get("accepted_edges", []),
                client=client,
                llm_client=llm_client,
                depth=args.depth,
                agent_ids=args.agent_ids,
                force_agent_ids=force_ids,
            )

            enriched_count = sum(1 for k in knowledge_list if not k.degraded)
            degraded_count = sum(1 for k in knowledge_list if k.degraded)

            print(f"[aeh] enrich completed  session_id={args.session_id}")
            print(f"  enriched: {enriched_count}")
            print(f"  degraded: {degraded_count}")
            return 0

        finally:
            await client.aclose()
            if hasattr(llm_client, "aclose"):
                await llm_client.aclose()


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return asyncio.run(_run_command(args))
    elif args.command == "discover":
        return asyncio.run(_discover_command(args))
    elif args.command == "enrich":
        return asyncio.run(_enrich_command(args))
    elif args.command == "dataset":
        return asyncio.run(_dataset_command(args))
    elif args.command == "eval":
        return asyncio.run(_eval_command(args))
    elif args.command == "report":
        return asyncio.run(_report_command(args))
    elif args.command == "map":
        return asyncio.run(_map_command(args))
    elif args.command == "plan":
        return asyncio.run(_plan_command(args))
    elif args.command == "ui":
        return _ui_command(args)
    parser.error(f"unknown command: {args.command}")
    return 1


def _ui_command(args) -> int:
    from pathlib import Path

    import uvicorn

    if args.data_dir:
        os.environ["AEH_DATA_DIR"] = str(Path(args.data_dir).absolute())
    elif "AEH_DATA_DIR" not in os.environ:
        # Only default to cwd for standalone CLI usage — a parent process (the
        # Electron AEHProcessManager) already sets this to a proper per-user data
        # directory before spawning us, and must never be silently overridden here.
        os.environ["AEH_DATA_DIR"] = str(Path(os.getcwd()).absolute())

    print(f"Starting AEH Dashboard server on http://{args.host}:{args.port}")
    uvicorn.run("agent_eval_harness.ui.server:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
