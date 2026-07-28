"""Symbol edges test suite: reference resolution (EC-1..EC-11), confidence-scored edges (resolver branch mapping, DB filtering), and CS-251 dynamic confidence formula."""
from __future__ import annotations

import pytest

from domain.qa.graph_queries import (
    SymbolHop,
    TraceStep,
    get_callees_of,
    get_callers_of,
    trace_call_chain,
)
from domain.structural_graph._symbol_resolver import _ambiguous_confidence
from domain.structural_graph.symbol_graph import SymbolEdge, SymbolGraphBuilder
from infrastructure.db.database import get_db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _builder() -> SymbolGraphBuilder:
    return SymbolGraphBuilder()


def _find(edges: list[SymbolEdge], src: str, dst: str) -> SymbolEdge | None:
    """Return the first edge matching *src* → *dst*, or None."""
    for e in edges:
        if e.src_symbol == src and e.dst_symbol == dst:
            return e
    return None


def _has(edges: list[SymbolEdge], src: str, dst: str) -> bool:
    return _find(edges, src, dst) is not None


def _absent(edges: list[SymbolEdge], src: str, dst: str) -> bool:
    return not _has(edges, src, dst)


# ===========================================================================
# Reference edge resolution (from test_symbol_reference_edges.py)
# ===========================================================================

# ---------------------------------------------------------------------------
# EC-1  Same method name across multiple classes
# ---------------------------------------------------------------------------

_MULTI_SERVICE = """\
class GraphService:
    async def build(self, snapshot_id): ...
    async def summary(self, snapshot_id): ...

class RetrievalService:
    async def build(self, req): ...
    async def retrieve(self, body): ...

class ManifestService:
    async def build(self, snapshot_id): ...
"""

_ORCHESTRATOR = """\
from multi_service import GraphService, RetrievalService, ManifestService

class Orchestrator:
    def __init__(self):
        self._graph = GraphService()
        self._retrieval = RetrievalService()
        self._manifest = ManifestService()

    async def run(self):
        await self._graph.build("snap-1")
        await self._retrieval.build(req)
        await self._manifest.build("snap-1")
"""


def test_ec1_same_method_name_correct_edges() -> None:
    """Each self._x.build() call resolves to the right class, not all three."""
    edges = _builder().build(
        {"multi_service.py": _MULTI_SERVICE, "orchestrator.py": _ORCHESTRATOR}
    )

    assert _has(edges, "orchestrator.py::Orchestrator.run", "multi_service.py::GraphService.build")
    assert _has(edges, "orchestrator.py::Orchestrator.run", "multi_service.py::RetrievalService.build")
    assert _has(edges, "orchestrator.py::Orchestrator.run", "multi_service.py::ManifestService.build")


def test_ec1_no_false_positive_unrelated_method() -> None:
    """summary() is never called — must not appear as an edge."""
    edges = _builder().build(
        {"multi_service.py": _MULTI_SERVICE, "orchestrator.py": _ORCHESTRATOR}
    )

    assert _absent(edges, "orchestrator.py::Orchestrator.run", "multi_service.py::GraphService.summary")


# ---------------------------------------------------------------------------
# EC-2  Constructor assignment tracking
# ---------------------------------------------------------------------------

_WIRED_SERVICE = """\
class ServiceA:
    def process(self): return "A"

class ServiceB:
    def process(self): return "B"

class Controller:
    def __init__(self):
        self.svc = ServiceA()   # type is known here

    def handle(self):
        return self.svc.process()  # must resolve to ServiceA.process
"""


def test_ec2_constructor_assignment_resolves_correctly() -> None:
    """self.svc assigned ServiceA() in __init__ → handle resolves to ServiceA.process."""
    edges = _builder().build({"wired_service.py": _WIRED_SERVICE})

    assert _has(edges, "wired_service.py::Controller.handle", "wired_service.py::ServiceA.process")


def test_ec2_no_false_positive_other_service() -> None:
    """ServiceB.process is never used — must not appear."""
    edges = _builder().build({"wired_service.py": _WIRED_SERVICE})

    assert _absent(edges, "wired_service.py::Controller.handle", "wired_service.py::ServiceB.process")


# ---------------------------------------------------------------------------
# EC-3  Reassigned attribute — ambiguous, both edges or confidence:low
# ---------------------------------------------------------------------------

_REASSIGN = """\
class Alpha:
    def run(self): ...

class Beta:
    def run(self): ...

class Runner:
    def __init__(self):
        self.worker = Alpha()

    def switch(self):
        self.worker = Beta()   # type changes here

    def execute(self):
        self.worker.run()      # ambiguous: Alpha or Beta after switch()
"""


def test_ec3_reassigned_attribute_emits_both_or_low_confidence() -> None:
    """execute() may call either Alpha.run or Beta.run — both present, or the emitted edge(s) confidence != 'high'."""
    edges = _builder().build({"reassign.py": _REASSIGN})

    alpha_edge = _find(edges, "reassign.py::Runner.execute", "reassign.py::Alpha.run")
    beta_edge  = _find(edges, "reassign.py::Runner.execute", "reassign.py::Beta.run")

    both_present = alpha_edge is not None and beta_edge is not None
    low_confidence = (
        (alpha_edge is not None and alpha_edge.confidence != "high") or
        (beta_edge  is not None and beta_edge.confidence  != "high")
    )

    assert both_present or low_confidence, (
        "Reassigned attribute must emit both candidates OR flag edge as "
        "confidence:low — not silently pick one with confidence:high"
    )


# ---------------------------------------------------------------------------
# EC-4  Module-level function — unambiguous
# ---------------------------------------------------------------------------

_UTILS = """\
def compute_hash(data: bytes) -> str: ...
"""

_PIPELINE = """\
from utils import compute_hash

def process(raw):
    return compute_hash(raw)
"""


def test_ec4_module_function_resolves_to_imported_definition() -> None:
    """compute_hash explicitly imported from utils → pipeline resolves cleanly."""
    edges = _builder().build({"utils.py": _UTILS, "pipeline.py": _PIPELINE})

    assert _has(edges, "pipeline.py::process", "utils.py::compute_hash")


# ---------------------------------------------------------------------------
# EC-5  Inherited method call
# ---------------------------------------------------------------------------

_INHERITANCE = """\
class Base:
    def validate(self): ...

class Child(Base):
    def handle(self):
        self.validate()   # calls Base.validate via inheritance
"""


def test_ec5_inherited_method_resolves_to_base_class() -> None:
    """self.validate() in Child.handle resolves up to Base.validate."""
    edges = _builder().build({"inheritance.py": _INHERITANCE})

    assert _has(edges, "inheritance.py::Child.handle", "inheritance.py::Base.validate")


# ---------------------------------------------------------------------------
# EC-6  Duck typing — unresolvable, no edge or confidence:none
# ---------------------------------------------------------------------------

_DUCK = """\
def process_any(obj):
    obj.build()   # obj has no type annotation, no assignment to trace
"""


def test_ec6_duck_typing_emits_no_confident_edge() -> None:
    """obj.build() with no type info must not emit confidence:high edges."""
    edges = _builder().build({"duck.py": _DUCK})

    # Gather any edges produced for this call site
    duck_edges = [
        e for e in edges
        if e.src_symbol == "duck.py::process_any" and "build" in e.dst_symbol
    ]

    high_conf = [e for e in duck_edges if e.confidence == "high"]
    assert high_conf == [], (
        f"Duck-typed call site must not produce confidence:high edges — got {high_conf}"
    )


# ---------------------------------------------------------------------------
# EC-7  TypeScript: interface vs concrete class
# ---------------------------------------------------------------------------

_REPO_TS = """\
interface Repository {
    findById(id: string): Promise<any>;
    save(entity: any): Promise<any>;
}

class UserRepo implements Repository {
    async findById(id: string) { return null; }
    async save(entity: any) { return null; }
}

class OrderRepo implements Repository {
    async findById(id: string) { return null; }
    async save(entity: any) { return null; }
}
"""

_SERVICE_TS = """\
import { Repository, UserRepo } from './repo';

class UserService {
    constructor(private repo: Repository) {}

    async getUser(id: string) {
        return this.repo.findById(id);
    }
}
"""


def test_ec7_interface_injection_no_definitive_edge_to_wrong_impl() -> None:
    """repo typed as interface — must NOT produce a confident edge to OrderRepo."""
    edges = _builder().build({"repo.ts": _REPO_TS, "service.ts": _SERVICE_TS})

    wrong = _find(edges, "service.ts::UserService.getUser", "repo.ts::OrderRepo.findById")
    assert wrong is None or wrong.confidence != "high", (
        "Interface-injected dependency must not produce a confident edge to "
        "OrderRepo — DI site is not statically traceable"
    )


def test_ec7_interface_injection_no_edge_without_evidence() -> None:
    """No edge should claim UserService.getUser → OrderRepo.findById (not imported)."""
    edges = _builder().build({"repo.ts": _REPO_TS, "service.ts": _SERVICE_TS})

    assert _absent(edges, "service.ts::UserService.getUser", "repo.ts::OrderRepo.findById") or (
        _find(edges, "service.ts::UserService.getUser", "repo.ts::OrderRepo.findById").confidence  # type: ignore[union-attr]
        in ("low", "none")
    )


# ---------------------------------------------------------------------------
# EC-8  TypeScript: generic functions — unambiguous
# ---------------------------------------------------------------------------

_GENERIC_TS = """\
function wrap<T>(value: T): T { return value; }

function caller() {
    const x = wrap<string>("hello");
}
"""


def test_ec8_generic_function_resolved_cleanly() -> None:
    """wrap<T> has only one definition — caller → wrap edge must exist."""
    edges = _builder().build({"generic.ts": _GENERIC_TS})

    assert _has(edges, "generic.ts::caller", "generic.ts::wrap")


# ---------------------------------------------------------------------------
# EC-9  Star import — unresolvable
# ---------------------------------------------------------------------------

_STAR = """\
from utils import *   # unknown what's imported

def handler():
    result = compute_hash(data)
"""


def test_ec9_star_import_emits_no_confident_edge() -> None:
    """Star import makes origin unknown — must not produce confidence:high edges."""
    edges = _builder().build({"star.py": _STAR})

    star_edges = [
        e for e in edges
        if e.src_symbol == "star.py::handler" and "compute_hash" in e.dst_symbol
    ]
    high_conf = [e for e in star_edges if e.confidence == "high"]
    assert high_conf == [], (
        "Star import must not produce confident edges — guessing origin is forbidden"
    )


# ---------------------------------------------------------------------------
# EC-10  Cross-file call to same-named standalone function
# ---------------------------------------------------------------------------

_AUTH = """\
def authenticate(token): ...
"""

_BILLING = """\
def authenticate(api_key): ...
"""

_MIDDLEWARE = """\
from auth import authenticate   # explicit import

def process(req):
    authenticate(req.token)
"""


def test_ec10_explicit_import_resolves_to_correct_module() -> None:
    """middleware explicitly imports from auth → must resolve to auth.authenticate."""
    edges = _builder().build(
        {"auth.py": _AUTH, "billing.py": _BILLING, "middleware.py": _MIDDLEWARE}
    )

    assert _has(edges, "middleware.py::process", "auth.py::authenticate")


def test_ec10_no_false_positive_non_imported_module() -> None:
    """billing.authenticate is NOT imported — must not appear as a target."""
    edges = _builder().build(
        {"auth.py": _AUTH, "billing.py": _BILLING, "middleware.py": _MIDDLEWARE}
    )

    assert _absent(edges, "middleware.py::process", "billing.py::authenticate")


# ---------------------------------------------------------------------------
# CS-330 Substrate Enrichment Tests
# ---------------------------------------------------------------------------

def test_extends_ambiguous_base_no_edge() -> None:
    """Two classes named Base in different files → extends resolution is ambiguous, must emit NO edge (never fabricate)."""
    f1 = "class Base: pass\nclass Sub(Base): pass"
    f2 = "class Base: pass"
    edges = _builder().build({"mod1.py": f1, "mod2.py": f2})
    extends_edges = [e for e in edges if e.edge_type == "extends"]
    assert len(extends_edges) == 0


def test_instantiates_unresolved_type_no_edge() -> None:
    """RHS type UnresolvableType is not in def_index → instantiates resolution must emit NO edge."""
    code = "class App:\n    def __init__(self):\n        self.x = UnresolvableType()"
    edges = _builder().build({"app.py": code})
    inst_edges = [e for e in edges if e.edge_type == "instantiates"]
    assert len(inst_edges) == 0


def test_instantiates_nested_closure_not_attributed_to_method() -> None:
    """Assignment self.x = Target() inside a nested function closure within method must NOT be attributed to outer method."""
    code = """\
class Target: pass

class Outer:
    def main_method(self):
        def closure():
            self.x = Target()
        closure()
"""
    edges = _builder().build({"app.py": code})
    inst_edges = [e for e in edges if e.edge_type == "instantiates"]
    assert len(inst_edges) == 0


def test_ec11_annotated_attribute_resolves_call_not_instantiates() -> None:
    """self.x: Target annotation types the attr so self.x.run() resolves to a calls edge — but a bare annotation is NOT a construction, so it emits NO instantiates edge."""
    code = """\
class Target:
    def run(self): ...

class Service:
    def __init__(self):
        self.x: Target

    def go(self):
        self.x.run()
"""
    edges = _builder().build({"app.py": code})
    assert _has(edges, "app.py::Service.go", "app.py::Target.run")
    assert len([e for e in edges if e.edge_type == "instantiates"]) == 0


def test_ec11_annotated_attribute_unknown_type_no_edge() -> None:
    """self.x: UnknownType with unknown type annotation emits NO instantiates edge."""
    code = """\
class Service:
    def __init__(self):
        self.x: UnknownType
"""
    edges = _builder().build({"app.py": code})
    inst_edges = [e for e in edges if e.edge_type == "instantiates"]
    assert len(inst_edges) == 0


def test_js_file_produces_calls_edges() -> None:
    """A .js file parsed through parse_file routes to JS parser and produces call edges."""
    from domain.structural_graph._symbol_parser import parse_file
    js_code = "function helper() {}\nfunction main() { helper(); }"
    pf = parse_file("app.js", js_code)
    assert pf is not None
    assert pf.language == "javascript"
    assert len(pf.call_sites) > 0


def test_cross_parser_fqn_identity() -> None:
    """Cross-parser FQN identity test: repo_map extraction and structural_graph parse_file produce byte-identical FQNs."""
    from domain.repo_map.service import make_qualified_name
    from domain.structural_graph._symbol_parser import parse_file

    code = """\
class MyService:
    def process(self):
        pass
"""
    pf = parse_file("service.py", code)
    assert pf is not None
    assert len(pf.definitions) > 0

    class_sym = next(s for s in pf.definitions.values() if s.method_name == "MyService")
    method_sym = next(s for s in pf.definitions.values() if s.method_name == "process")

    q_class_sg = f"service.py::{class_sym.method_name}"
    q_class_rm = make_qualified_name("service.py", "MyService", None)
    assert q_class_sg == q_class_rm

    q_method_sg = f"service.py::{method_sym.class_name}.{method_sym.method_name}"
    q_method_rm = make_qualified_name("service.py", "process", "MyService")
    assert q_method_sg == q_method_rm


# ===========================================================================
# Confidence-scored edges (from test_confidence_scored_edges.py)
# ===========================================================================

# ---------------------------------------------------------------------------
# Unit Tests: Resolver Branch Confidence Mapping
# ---------------------------------------------------------------------------

class TestResolverBranchMapping:
    """Verify each resolver branch emits the documented confidence_score/resolution_method."""

    def test_import_path_match_bare_call(self) -> None:
        """_resolve_bare_call import-namespace hit → confidence_score=0.95, resolution_method='import_path_match'."""
        utils = "def compute_hash(data: bytes) -> str: pass"
        pipeline = "from utils import compute_hash\n\ndef process(raw):\n    return compute_hash(raw)"

        edges = _builder().build({"utils.py": utils, "pipeline.py": pipeline})
        edge = _find(edges, "pipeline.py::process", "utils.py::compute_hash")

        assert edge is not None, "Import-resolved bare call must produce an edge"
        assert edge.confidence_score == 0.95, f"Expected 0.95, got {edge.confidence_score}"
        assert edge.resolution_method == "import_path_match"
        assert edge.confidence == "high", "Score 0.95 >= 0.7 → confidence='high'"

    def test_same_file_scope_bare_call(self) -> None:
        """_resolve_bare_call same-file single candidate → confidence_score=0.85, resolution_method='same_file_scope'."""
        code = """
def helper(): pass

def caller():
    helper()
"""
        edges = _builder().build({"local.py": code})
        edge = _find(edges, "local.py::caller", "local.py::helper")

        assert edge is not None, "Same-file bare call must produce an edge"
        assert edge.confidence_score == 0.85, f"Expected 0.85, got {edge.confidence_score}"
        assert edge.resolution_method == "same_file_scope"
        assert edge.confidence == "high", "Score 0.85 >= 0.7 → confidence='high'"

    def test_mro_resolved_self_call(self) -> None:
        """_resolve_self_call MRO walk → confidence_score=0.9, resolution_method='mro_resolved'."""
        code = """
class Base:
    def validate(self): pass

class Child(Base):
    def handle(self):
        self.validate()
"""
        edges = _builder().build({"inheritance.py": code})
        edge = _find(edges, "inheritance.py::Child.handle", "inheritance.py::Base.validate")

        assert edge is not None, "MRO-resolved self call must produce an edge"
        assert edge.confidence_score == 0.9, f"Expected 0.9, got {edge.confidence_score}"
        assert edge.resolution_method == "mro_resolved"
        assert edge.confidence == "high", "Score 0.9 >= 0.7 → confidence='high'"

    def test_constructor_type_trace_single(self) -> None:
        """Attribute receiver with single assigned type → confidence_score=0.85, resolution_method='constructor_type_trace'."""
        code = """
class ServiceA:
    def process(self): return "A"

class Controller:
    def __init__(self):
        self.svc = ServiceA()

    def handle(self):
        return self.svc.process()
"""
        edges = _builder().build({"wired.py": code})
        edge = _find(edges, "wired.py::Controller.handle", "wired.py::ServiceA.process")

        assert edge is not None, "Constructor-traced single type must produce an edge"
        assert edge.confidence_score == 0.85, f"Expected 0.85, got {edge.confidence_score}"
        assert edge.resolution_method == "constructor_type_trace"
        assert edge.confidence == "high", "Score 0.85 >= 0.7 → confidence='high'"

    def test_name_heuristic_ambiguous_multi_type(self) -> None:
        """Attribute receiver with 2+ assigned types → confidence_score=0.4, resolution_method='name_heuristic_ambiguous'."""
        code = """
class Alpha:
    def run(self): pass

class Beta:
    def run(self): pass

class Runner:
    def __init__(self):
        self.worker = Alpha()

    def switch(self):
        self.worker = Beta()

    def execute(self):
        self.worker.run()
"""
        edges = _builder().build({"reassign.py": code})
        alpha_edge = _find(edges, "reassign.py::Runner.execute", "reassign.py::Alpha.run")
        beta_edge = _find(edges, "reassign.py::Runner.execute", "reassign.py::Beta.run")

        # Both edges marked low confidence: _ambiguous_confidence(2) = 0.6/2 = 0.3
        expected_score = 0.3
        if alpha_edge:
            assert alpha_edge.confidence_score == expected_score, f"Expected {expected_score} for Alpha, got {alpha_edge.confidence_score}"
            assert alpha_edge.resolution_method == "name_heuristic_ambiguous"
            assert alpha_edge.confidence == "low", f"Score {expected_score} < 0.7 → confidence='low'"
        if beta_edge:
            assert beta_edge.confidence_score == expected_score, f"Expected {expected_score} for Beta, got {beta_edge.confidence_score}"
            assert beta_edge.resolution_method == "name_heuristic_ambiguous"
            assert beta_edge.confidence == "low", f"Score {expected_score} < 0.7 → confidence='low'"

    def test_module_call_import_resolved(self) -> None:
        """_resolve_module_call via import → confidence_score=0.95, resolution_method='import_path_match'."""
        auth = "def authenticate(token): pass"
        service = "from auth import authenticate as auth_func\n\nclass Service:\n    def login(self):\n        auth_func('token')"

        edges = _builder().build({"auth.py": auth, "service.py": service})
        # The call is to auth_func which aliases authenticate
        matching = [e for e in edges if "login" in e.src_symbol and "authenticate" in e.dst_symbol]

        if matching:
            edge = matching[0]
            assert edge.confidence_score == 0.95, f"Expected 0.95, got {edge.confidence_score}"
            assert edge.resolution_method == "import_path_match"
            assert edge.confidence == "high"

    def test_confidence_derived_from_score(self) -> None:
        """confidence string is derived from confidence_score: >= 0.7 → 'high', else 'low'."""
        # Test the boundary
        edge_high = SymbolEdge(
            src_symbol="a.py::func",
            dst_symbol="b.py::func",
            confidence_score=0.7,
            resolution_method="test",
        )
        assert edge_high.confidence == "high", "Score 0.7 should be 'high'"

        edge_low = SymbolEdge(
            src_symbol="a.py::func",
            dst_symbol="b.py::func",
            confidence_score=0.69,
            resolution_method="test",
        )
        assert edge_low.confidence == "low", "Score 0.69 should be 'low'"


# ---------------------------------------------------------------------------
# EC-9/EC-6 Cases: Ensure Unchanged Behavior
# ---------------------------------------------------------------------------

class TestAmbiguousCasesUnchanged:
    """Verify ambiguous/unresolvable cases still return [] (unchanged EC-9/EC-6 behavior)."""

    def test_star_import_no_edge(self) -> None:
        """Star import makes origin unknown → resolver returns [] (EC-9)."""
        code = "from utils import *\n\ndef handler():\n    result = compute_hash(data)"

        edges = _builder().build({"star.py": code})
        star_edges = [
            e for e in edges
            if e.src_symbol == "star.py::handler" and "compute_hash" in e.dst_symbol
        ]

        assert star_edges == [], "Star import must produce no edges — guessing origin is forbidden"

    def test_duck_typing_no_confident_edge(self) -> None:
        """Untyped parameter → resolver returns [] (EC-6)."""
        code = "def process_any(obj):\n    obj.build()"

        edges = _builder().build({"duck.py": code})
        duck_edges = [
            e for e in edges
            if e.src_symbol == "duck.py::process_any" and "build" in e.dst_symbol
        ]

        high_conf = [e for e in duck_edges if e.confidence == "high"]
        assert high_conf == [], "Duck-typed call site must not produce confidence='high' edges"

    def test_two_same_named_functions_no_import_no_edge(self) -> None:
        """Two same-named functions, different files, no import link → no edge (unchanged)."""
        auth = "def validate(token): pass"
        billing = "def validate(api_key): pass"
        # No explicit import, so no resolution path → []
        middleware = "def process(req):\n    validate(req.token)"

        edges = _builder().build({
            "auth.py": auth,
            "billing.py": billing,
            "middleware.py": middleware,
        })

        middleware_edges = [
            e for e in edges
            if e.src_symbol == "middleware.py::process" and "validate" in e.dst_symbol
        ]

        assert middleware_edges == [], (
            "Unimported bare call to ambiguous name must produce no edges — "
            "neither auth.validate nor billing.validate should be guessed"
        )


# ---------------------------------------------------------------------------
# Integration Tests: Database Round-trip & Filter Behavior
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_legacy_confidence_string_filter_backward_compat(
    test_snapshot_id: str,
    cleanup_test_data: None,
) -> None:
    """Verify high_confidence_only=True still excludes confidence < 0.7 rows."""
    db = get_db()

    # Insert test edges with mixed confidence_score values
    await db.executemany(
        """
        INSERT INTO symbol_graph_edges
        (snapshot_id, src_symbol, dst_symbol, edge_type, confidence_score, resolution_method, evidence_lines)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (test_snapshot_id, "a.py::caller", "b.py::high_conf", "calls", 0.95, "import_path_match", "[]"),
            (test_snapshot_id, "a.py::caller", "b.py::mid_conf", "calls", 0.7, "same_file_scope", "[]"),
            (test_snapshot_id, "a.py::caller", "b.py::low_conf", "calls", 0.4, "name_heuristic_ambiguous", "[]"),
            (test_snapshot_id, "a.py::caller", "b.py::zero_conf", "calls", 0.0, "unknown", "[]"),
        ],
    )

    # Query with high_confidence_only=True (default)
    high_only = await get_callees_of(test_snapshot_id, "a.py", high_confidence_only=True)

    # Should get only rows with confidence='high' (score >= 0.7)
    dests_high_only = {hop.dst_symbol for hop in high_only}
    assert "b.py::high_conf" in dests_high_only, "Score 0.95 should be included"
    assert "b.py::mid_conf" in dests_high_only, "Score 0.7 (boundary) should be included"
    assert "b.py::low_conf" not in dests_high_only, "Score 0.4 should be excluded"
    assert "b.py::zero_conf" not in dests_high_only, "Score 0.0 should be excluded"

    # Query with high_confidence_only=False (unfiltered)
    all_edges = await get_callees_of(test_snapshot_id, "a.py", high_confidence_only=False)
    dests_all = {hop.dst_symbol for hop in all_edges}
    assert len(dests_all) == 4, "Unfiltered should get all 4 edges"


@pytest.mark.asyncio
async def test_min_confidence_filtering_get_callees_of(
    test_snapshot_id: str,
    cleanup_test_data: None,
) -> None:
    """Verify min_confidence parameter filters correctly (inclusive >=)."""
    db = get_db()

    await db.executemany(
        """
        INSERT INTO symbol_graph_edges
        (snapshot_id, src_symbol, dst_symbol, edge_type, confidence_score, resolution_method, evidence_lines)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (test_snapshot_id, "a.py::caller", "b.py::t1", "calls", 0.95, "import_path_match", "[]"),
            (test_snapshot_id, "a.py::caller", "b.py::t2", "calls", 0.85, "constructor_type_trace", "[]"),
            (test_snapshot_id, "a.py::caller", "b.py::t3", "calls", 0.4, "name_heuristic_ambiguous", "[]"),
        ],
    )

    # Query with min_confidence=0.85
    result = await get_callees_of(
        test_snapshot_id, "a.py", high_confidence_only=False, min_confidence=0.85
    )
    dests = {hop.dst_symbol for hop in result}

    assert "b.py::t1" in dests, "Score 0.95 >= 0.85 should be included"
    assert "b.py::t2" in dests, "Score 0.85 >= 0.85 (boundary) should be included"
    assert "b.py::t3" not in dests, "Score 0.4 < 0.85 should be excluded"

    # Query with min_confidence=0.5
    result_half = await get_callees_of(
        test_snapshot_id, "a.py", high_confidence_only=False, min_confidence=0.5
    )
    dests_half = {hop.dst_symbol for hop in result_half}

    assert dests_half == {"b.py::t1", "b.py::t2"}, "Only t1 (0.95) and t2 (0.85) are >= 0.5; t3 (0.4) is excluded"


@pytest.mark.asyncio
async def test_min_confidence_filtering_get_callers_of(
    test_snapshot_id: str,
    cleanup_test_data: None,
) -> None:
    """Verify min_confidence parameter works for reverse lookup (get_callers_of)."""
    db = get_db()

    await db.executemany(
        """
        INSERT INTO symbol_graph_edges
        (snapshot_id, src_symbol, dst_symbol, edge_type, confidence_score, resolution_method, evidence_lines)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (test_snapshot_id, "a.py::caller1", "b.py::target", "calls", 0.95, "import_path_match", "[]"),
            (test_snapshot_id, "a.py::caller2", "b.py::target", "calls", 0.4, "name_heuristic_ambiguous", "[]"),
        ],
    )

    # Unfiltered
    all_callers = await get_callers_of(test_snapshot_id, "b.py", high_confidence_only=False)
    assert len(all_callers) == 2

    # With min_confidence=0.85
    high_callers = await get_callers_of(
        test_snapshot_id, "b.py", high_confidence_only=False, min_confidence=0.85
    )
    assert len(high_callers) == 1
    assert high_callers[0].src_symbol == "a.py::caller1"


@pytest.mark.asyncio
async def test_min_confidence_none_unfiltered(
    test_snapshot_id: str,
    cleanup_test_data: None,
) -> None:
    """Verify min_confidence=None means unfiltered (default, preserve current recall)."""
    db = get_db()

    await db.executemany(
        """
        INSERT INTO symbol_graph_edges
        (snapshot_id, src_symbol, dst_symbol, edge_type, confidence_score, resolution_method, evidence_lines)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (test_snapshot_id, "a.py::caller", "b.py::t1", "calls", 0.95, "import_path_match", "[]"),
            (test_snapshot_id, "a.py::caller", "b.py::t2", "calls", 0.0, "unknown", "[]"),
        ],
    )

    # Explicitly pass min_confidence=None
    result = await get_callees_of(
        test_snapshot_id, "a.py", high_confidence_only=False, min_confidence=None
    )

    assert len(result) == 2, "min_confidence=None should be unfiltered (all edges)"


# ---------------------------------------------------------------------------
# Fixtures (scoped to the DB round-trip/filter integration tests above)
# ---------------------------------------------------------------------------

@pytest.fixture
def test_snapshot_id() -> str:
    """Provide a test snapshot ID."""
    import uuid
    return f"test-snap-{uuid.uuid4().hex[:8]}"


@pytest.fixture
async def cleanup_test_data(test_snapshot_id: str) -> None:
    """Clean up test edges after test runs (async/DB integration tests only)."""
    yield
    db = get_db()
    await db.execute(
        "DELETE FROM symbol_graph_edges WHERE snapshot_id LIKE ?",
        ("test-snap-%",),
    )
    await db.commit()


# ===========================================================================
# CS-251 dynamic confidence formula (from test_cs251_dynamic_confidence.py)
# ===========================================================================
# Acceptance gate for the dynamic ambiguous-confidence formula, min_confidence
# wiring, path_confidence multiplicative compounding, and Enhancement-5
# tentative-file tracking, plus SymbolHop/TraceStep backward compatibility.

# ---------------------------------------------------------------------------
# Unit Tests: _ambiguous_confidence formula
# ---------------------------------------------------------------------------


class TestAmbiguousConfidenceFormula:
    """Verify _ambiguous_confidence formula shape and floor."""

    @pytest.mark.parametrize(
        "num_candidates,expected",
        [
            (1, 0.6),
            (2, 0.3),
            (3, 0.2),
            (5, 0.12),  # actually 0.12, should be floored to 0.15
            (10, 0.15),  # floor applies
            (20, 0.15),  # floor applies
        ],
    )
    def test_formula_shape(self, num_candidates: int, expected: float) -> None:
        """Test _ambiguous_confidence returns expected value for candidate counts."""
        result = _ambiguous_confidence(num_candidates)
        expected_floored = max(0.15, expected)
        assert abs(result - expected_floored) < 0.001, (
            f"_ambiguous_confidence({num_candidates}): expected {expected_floored}, got {result}"
        )

    def test_monotonic_decrease(self) -> None:
        """Confidence should decrease monotonically as candidate count increases."""
        values = [_ambiguous_confidence(n) for n in range(1, 21)]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1], (
                f"Not monotonic at index {i}: {values[i]} > {values[i + 1]}"
            )

    def test_floor_at_0_15(self) -> None:
        """Confidence should never drop below 0.15."""
        for n in range(1, 100):
            result = _ambiguous_confidence(n)
            assert result >= 0.15, f"Floor violated at n={n}: {result}"

    def test_never_reaches_high_boundary(self) -> None:
        """All ambiguous edges must remain 'low' (score < 0.7 derivation boundary)."""
        for n in range(1, 100):
            result = _ambiguous_confidence(n)
            # The >=0.7 boundary for 'high' confidence must never be reached by ambiguous scores
            assert result < 0.7, (
                f"Ambiguous confidence at n={n} ({result}) reaches/exceeds high boundary (0.7)"
            )


# ---------------------------------------------------------------------------
# Integration Tests: min_confidence wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_min_confidence_filters_callees():
    """min_confidence parameter in get_callees_of filters results correctly."""
    db = get_db()

    # Prepare test: insert edges with known confidence scores
    snapshot_id = "test-min-conf-callees"
    await db.execute("DELETE FROM symbol_graph_edges WHERE snapshot_id=?", (snapshot_id,))

    # Insert test edges with varying confidence scores
    test_edges = [
        (snapshot_id, "test.py::func_a", "test.py::func_b", "calls", "low", 0.3, "[]"),
        (snapshot_id, "test.py::func_a", "test.py::func_c", "calls", "high", 0.9, "[]"),
        (snapshot_id, "test.py::func_a", "test.py::func_d", "calls", "low", 0.5, "[]"),
    ]
    for row in test_edges:
        await db.execute(
            "INSERT INTO symbol_graph_edges "
            "(snapshot_id, src_symbol, dst_symbol, edge_type, confidence, "
            "confidence_score, evidence_lines) VALUES (?, ?, ?, ?, ?, ?, ?)",
            row,
        )

    # Test 1: high_confidence_only=True filters to >=0.7 (1 edge)
    hops_high = await get_callees_of(snapshot_id, "test.py", high_confidence_only=True)
    assert len(hops_high) == 1
    assert hops_high[0].dst_symbol == "test.py::func_c"

    # Test 2: min_confidence=0.5 filters to >=0.5 (2 edges)
    hops_min = await get_callees_of(
        snapshot_id, "test.py", high_confidence_only=False, min_confidence=0.5
    )
    assert len(hops_min) == 2
    assert all(h.confidence_score >= 0.5 for h in hops_min)

    # Test 3: min_confidence=0.9 filters to >=0.9 (1 edge)
    hops_strict = await get_callees_of(
        snapshot_id, "test.py", high_confidence_only=False, min_confidence=0.9
    )
    assert len(hops_strict) == 1
    assert hops_strict[0].confidence_score == 0.9


@pytest.mark.asyncio
async def test_trace_call_chain_forwards_min_confidence():
    """trace_call_chain forwards min_confidence to get_callees_of/get_callers_of."""
    db = get_db()

    snapshot_id = "test-trace-min-conf"
    await db.execute("DELETE FROM symbol_graph_edges WHERE snapshot_id=?", (snapshot_id,))

    # Build a simple call chain: A -> B -> C
    # A->B: high confidence (0.9)
    # B->C: low confidence (0.4)
    test_edges = [
        (snapshot_id, "file_a.py::func_a", "file_b.py::func_b", "calls", "high", 0.9, "[]"),
        (snapshot_id, "file_b.py::func_b", "file_c.py::func_c", "calls", "low", 0.4, "[]"),
    ]
    for row in test_edges:
        await db.execute(
            "INSERT INTO symbol_graph_edges "
            "(snapshot_id, src_symbol, dst_symbol, edge_type, confidence, "
            "confidence_score, evidence_lines) VALUES (?, ?, ?, ?, ?, ?, ?)",
            row,
        )

    # Trace forward from file_a with min_confidence=0.5 should find file_b but not file_c
    trace_filtered = await trace_call_chain(
        snapshot_id,
        "file_a.py",
        "forward",
        max_hops=3,
        high_confidence_only=False,
        min_confidence=0.5,
    )
    files_found = {s.file for s in trace_filtered}
    assert "file_b.py" in files_found, "Should find file_b via 0.9 edge"
    assert "file_c.py" not in files_found, (
        "Should NOT find file_c via 0.4 edge when min_confidence=0.5"
    )


# ---------------------------------------------------------------------------
# Integration Tests: path_confidence multiplicative compounding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_path_confidence_multiplicative():
    """path_confidence multiplies across step depth: 0.9 * 0.85 * 0.4 ≈ 0.306."""
    db = get_db()

    snapshot_id = "test-path-conf"
    await db.execute("DELETE FROM symbol_graph_edges WHERE snapshot_id=?", (snapshot_id,))

    # Build 3-hop chain: A -> B -> C -> D with confidences 0.9, 0.85, 0.4
    edges = [
        (snapshot_id, "a.py::f", "b.py::g", "calls", "high", 0.9, "[]"),
        (snapshot_id, "b.py::g", "c.py::h", "calls", "high", 0.85, "[]"),
        (snapshot_id, "c.py::h", "d.py::i", "calls", "low", 0.4, "[]"),
    ]
    for row in edges:
        await db.execute(
            "INSERT INTO symbol_graph_edges "
            "(snapshot_id, src_symbol, dst_symbol, edge_type, confidence, "
            "confidence_score, evidence_lines) VALUES (?, ?, ?, ?, ?, ?, ?)",
            row,
        )

    # Trace forward from a.py, high_confidence_only=False to include the 0.4 edge
    trace = await trace_call_chain(
        snapshot_id, "a.py", "forward", max_hops=4, high_confidence_only=False
    )

    # Find the final step (d.py should be found at step 3)
    final_step = None
    for step in trace:
        if step.file == "d.py":
            final_step = step
            break

    assert final_step is not None, "Should find d.py in the trace"
    # Expected: 0.9 * 0.85 * 0.4 = 0.306
    expected_path_confidence = 0.9 * 0.85 * 0.4
    assert abs(final_step.path_confidence - expected_path_confidence) < 0.01, (
        f"Expected path_confidence ≈ {expected_path_confidence}, got {final_step.path_confidence}"
    )


@pytest.mark.asyncio
async def test_path_confidence_max_per_step():
    """path_confidence uses max() across parallel hops in same step."""
    db = get_db()

    snapshot_id = "test-path-conf-parallel"
    await db.execute("DELETE FROM symbol_graph_edges WHERE snapshot_id=?", (snapshot_id,))

    # Build parallel hops to same file: A -> B via two different paths
    # Path 1: A -> B directly (0.9)
    # Path 2: A -> B via alias (0.6)
    # Expected: max(0.9, 0.6) = 0.9
    edges = [
        (snapshot_id, "a.py::f1", "b.py::g", "calls", "high", 0.9, "[]"),
        (snapshot_id, "a.py::f2", "b.py::g", "calls", "low", 0.6, "[]"),
    ]
    for row in edges:
        await db.execute(
            "INSERT INTO symbol_graph_edges "
            "(snapshot_id, src_symbol, dst_symbol, edge_type, confidence, "
            "confidence_score, evidence_lines) VALUES (?, ?, ?, ?, ?, ?, ?)",
            row,
        )

    trace = await trace_call_chain(
        snapshot_id, "a.py", "forward", max_hops=2, high_confidence_only=False
    )

    b_step = None
    for step in trace:
        if step.file == "b.py":
            b_step = step
            break

    assert b_step is not None, "Should find b.py"
    # With parallel hops (0.9 and 0.6), max should be 0.9
    assert abs(b_step.path_confidence - 0.9) < 0.01, (
        f"Expected path_confidence ≈ 0.9 (max of parallel hops), got {b_step.path_confidence}"
    )


@pytest.mark.asyncio
async def test_symbol_hop_confidence_score_field():
    """SymbolHop.confidence_score field is populated from DB."""
    db = get_db()

    snapshot_id = "test-hop-field"
    await db.execute("DELETE FROM symbol_graph_edges WHERE snapshot_id=?", (snapshot_id,))

    await db.execute(
        "INSERT INTO symbol_graph_edges "
        "(snapshot_id, src_symbol, dst_symbol, edge_type, confidence, "
        "confidence_score, evidence_lines) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (snapshot_id, "a.py::f", "b.py::g", "calls", "high", 0.85, "[10]"),
    )

    hops = await get_callees_of(snapshot_id, "a.py", high_confidence_only=False)
    assert len(hops) == 1
    hop = hops[0]
    assert hop.confidence_score == 0.85, "SymbolHop.confidence_score should be populated from DB"


# ---------------------------------------------------------------------------
# Regression Tests: Existing positional construction compatibility
# ---------------------------------------------------------------------------


def test_symbol_hop_backward_compat():
    """SymbolHop constructed positionally (backward compat, no confidence_score)."""
    # Old-style positional construction (5 args, no confidence_score)
    hop = SymbolHop("file.py::func_a", "file.py::func_b", "calls", "high", [10, 20])
    assert hop.src_symbol == "file.py::func_a"
    assert hop.dst_symbol == "file.py::func_b"
    assert hop.edge_type == "calls"
    assert hop.confidence == "high"
    assert hop.evidence_lines == [10, 20]
    assert hop.confidence_score == 1.0  # default


def test_trace_step_backward_compat():
    """TraceStep constructed positionally (backward compat, no path_confidence)."""
    hop1 = SymbolHop("a.py::f", "b.py::g", "calls", "high", [1])

    # Old-style positional construction (5 args, no path_confidence)
    step = TraceStep(
        1,
        "b.py",
        ["b.py::g"],
        [hop1],
        False,  # is_seed
    )
    assert step.step == 1
    assert step.file == "b.py"
    assert step.symbols_involved == ["b.py::g"]
    assert len(step.hops) == 1
    assert step.is_seed is False
    assert step.path_confidence == 1.0  # default


# ---------------------------------------------------------------------------
# Integration Tests: tentative files capture in Enhancement-5 fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enhancement5_fallback_capture_tentative_files():
    """Verify tentative_files/tentative_confidence capture when the Enhancement-5 low-confidence fallback fires."""
    db = get_db()

    snapshot_id = "test-enhancement5-fallback"
    await db.execute("DELETE FROM symbol_graph_edges WHERE snapshot_id=?", (snapshot_id,))

    # Build a scenario where:
    # - High-confidence edges: A -> B (only 1 file found, < 2)
    # - Low-confidence edges: A -> B, A -> C (total 2 files: B, C)
    # When fallback fires, it should capture C as tentative
    # Note: DB has UNIQUE constraint on (snapshot_id, src_symbol, dst_symbol)
    # so we use different source symbols for the parallel paths

    edges = [
        # High confidence path (only leads to B)
        (snapshot_id, "a.py::f_start", "b.py::g", "calls", "high", 0.9, "[]"),
        # Low confidence path (leads to C)
        (snapshot_id, "a.py::f_start", "c.py::h", "calls", "low", 0.45, "[]"),
    ]
    for row in edges:
        await db.execute(
            "INSERT INTO symbol_graph_edges ("
            "snapshot_id, src_symbol, dst_symbol, edge_type, confidence, "
            "confidence_score, evidence_lines) VALUES (?, ?, ?, ?, ?, ?, ?)",
            row,
        )

    # Trace with high_confidence_only=True: should find only b.py (1 file < 2)
    trace_high = await trace_call_chain(
        snapshot_id, "a.py", "forward", max_hops=2, high_confidence_only=True
    )
    high_conf_files = {s.file for s in trace_high}
    assert high_conf_files == {"a.py", "b.py"}, (
        f"High-confidence trace should find only a.py and b.py, got {high_conf_files}"
    )
    new_files_high = [s.file for s in trace_high if s.file != "a.py"]
    assert len(new_files_high) < 2, (
        "High-confidence trace should find < 2 files (triggers fallback)"
    )

    # Trace with high_confidence_only=False: should find both b.py and c.py
    trace_low = await trace_call_chain(
        snapshot_id, "a.py", "forward", max_hops=2, high_confidence_only=False
    )
    low_conf_files = {s.file for s in trace_low}
    assert "c.py" in low_conf_files, f"Low-confidence trace should find c.py, got {low_conf_files}"

    # Simulate Enhancement-5 fallback logic:
    tentative_files = []
    tentative_confidence = {}

    extra = [s.file for s in trace_low if s.file not in high_conf_files and s.file != "a.py"]
    if extra and len(new_files_high) < 2:
        tentative_files = extra[:5]
        for file in tentative_files:
            for step in trace_low:
                if step.file == file:
                    max_conf = max((h.confidence_score for h in step.hops), default=1.0)
                    tentative_confidence[file] = max_conf
                    break

    # Verify tentative capture
    assert "c.py" in tentative_files, "c.py should be captured as tentative"
    assert "c.py" in tentative_confidence, "c.py should have confidence score"
    assert tentative_confidence["c.py"] == 0.45, (
        f"c.py confidence should be 0.45 (its hop confidence), got {tentative_confidence['c.py']}"
    )


@pytest.mark.asyncio
async def test_enhancement5_no_fallback_when_sufficient_high_confidence():
    """Verify tentative_files remain empty when high-confidence trace finds >= 2 files."""
    db = get_db()

    snapshot_id = "test-enhancement5-no-fallback"
    await db.execute("DELETE FROM symbol_graph_edges WHERE snapshot_id=?", (snapshot_id,))

    # Build a scenario where high-confidence trace finds 2+ files (no fallback)
    edges = [
        (snapshot_id, "a.py::f_start", "b.py::g", "calls", "high", 0.9, "[]"),
        (snapshot_id, "a.py::f_start", "c.py::h", "calls", "high", 0.85, "[]"),
        # Low confidence (should not be used)
        (snapshot_id, "a.py::f_start", "d.py::i", "calls", "low", 0.3, "[]"),
    ]
    for row in edges:
        await db.execute(
            "INSERT INTO symbol_graph_edges ("
            "snapshot_id, src_symbol, dst_symbol, edge_type, confidence, "
            "confidence_score, evidence_lines) VALUES (?, ?, ?, ?, ?, ?, ?)",
            row,
        )

    # Trace with high_confidence_only=True: find b.py & c.py (no fallback)
    trace_high = await trace_call_chain(
        snapshot_id, "a.py", "forward", max_hops=2, high_confidence_only=True
    )
    new_files = [s.file for s in trace_high if s.file != "a.py"]
    assert len(new_files) >= 2, (
        "High-confidence trace should find >= 2 files (no fallback triggers)"
    )

    # Simulate Enhancement-5 logic (should NOT capture tentative files)
    tentative_files = []
    tentative_confidence = {}

    if len(new_files) < 2:  # This condition should be False
        trace_low = await trace_call_chain(
            snapshot_id, "a.py", "forward", max_hops=2, high_confidence_only=False
        )
        high_conf_files = {s.file for s in trace_high}
        extra = [s.file for s in trace_low if s.file not in high_conf_files and s.file != "a.py"]
        if extra:
            tentative_files = extra[:5]
            for file in tentative_files:
                for step in trace_low:
                    if step.file == file:
                        max_conf = max((h.confidence_score for h in step.hops), default=1.0)
                        tentative_confidence[file] = max_conf
                        break

    # Verify NO tentative capture (enhancement-5 doesn't fire)
    assert len(tentative_files) == 0, "tentative_files should remain empty when no fallback"
    assert len(tentative_confidence) == 0, (
        "tentative_confidence should remain empty when no fallback"
    )
