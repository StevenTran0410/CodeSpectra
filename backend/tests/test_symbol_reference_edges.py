"""CS-213 — Symbol Reference Edge Verification Test Suite.

Acceptance gate for CS-202 (symbol_graph.py).  Every test here MUST pass
before CS-202 is considered shippable.

After all 10 edge cases pass on a real codebase snapshot:
  1. Run this suite one final time.
  2. Confirm false-positive rate is acceptable.
  3. Delete this file.
  4. Close ticket CS-213.

This file is TEMPORARY — do NOT merge into the permanent test suite.
"""
from __future__ import annotations

import pytest

from domain.structural_graph.symbol_graph import SymbolEdge, SymbolGraphBuilder


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
    """execute() may call either Alpha.run or Beta.run — both must be present
    OR the emitted edge(s) must carry confidence != 'high'."""
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
