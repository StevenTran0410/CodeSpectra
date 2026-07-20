"""Tests for recursive schema resolution in contract harvest."""
from __future__ import annotations

from pathlib import Path

from agent_eval_harness.mapping.builder.contract_harvest import (
    _SchemaResolveCtx,
    _parse_files,
    _resolve_class_schema,
)


# Nested TypedDict test
NESTED_TYPEDDICT_SRC = '''
from typing import TypedDict

class Address(TypedDict):
    street: str
    city: str
    zip_code: str

class Person(TypedDict):
    name: str
    age: int
    address: Address
'''

# Pydantic test
PYDANTIC_SRC = '''
from pydantic import BaseModel

class Address(BaseModel):
    street: str
    city: str
    zip_code: str | None = None

class Person(BaseModel):
    name: str
    age: int
    address: Address
'''

# Dataclass test
DATACLASS_SRC = '''
from dataclasses import dataclass

@dataclass
class Address:
    street: str
    city: str
    zip_code: str | None = None

@dataclass
class Person:
    name: str
    age: int
    address: Address
'''

# Cycle test - A references B which references A
CYCLE_SRC = '''
from typing import TypedDict

class A(TypedDict):
    name: str
    b_ref: 'B'

class B(TypedDict):
    name: str
    a_ref: 'A'
'''

# List and dict generic test
GENERICS_SRC = '''
from typing import TypedDict

class Item(TypedDict):
    id: int
    name: str

class Container(TypedDict):
    items: list[Item]
    metadata: dict[str, str]
'''


def _write_fixture(tmp_path: Path, name: str, src: str) -> Path:
    p = tmp_path / f"{name}.py"
    p.write_text(src, encoding="utf-8")
    return p


def test_nested_typeddict(tmp_path: Path):
    """Test resolving nested TypedDict with recursive refs."""
    file_path = _write_fixture(tmp_path, "nested_td", NESTED_TYPEDDICT_SRC)
    asts = _parse_files([file_path])
    ctx = _SchemaResolveCtx(asts=asts, files_root=tmp_path, visited=set(), depth=0, conventions=None)

    schema, source = _resolve_class_schema("Person", ctx)
    assert schema is not None
    assert schema["type"] == "object"
    assert "name" in schema["properties"]
    assert "age" in schema["properties"]
    assert "address" in schema["properties"]
    assert schema["properties"]["address"]["type"] == "object"
    assert "street" in schema["properties"]["address"]["properties"]


def test_pydantic_model(tmp_path: Path):
    """Test resolving Pydantic BaseModel with optional fields."""
    file_path = _write_fixture(tmp_path, "pydantic_model", PYDANTIC_SRC)
    asts = _parse_files([file_path])
    ctx = _SchemaResolveCtx(asts=asts, files_root=tmp_path, visited=set(), depth=0, conventions=None)

    schema, source = _resolve_class_schema("Person", ctx)
    assert schema is not None
    assert schema["type"] == "object"
    assert "name" in schema["required"]
    assert "age" in schema["required"]
    assert "address" in schema["properties"]


def test_dataclass_resolution(tmp_path: Path):
    """Test resolving dataclass with optional fields."""
    file_path = _write_fixture(tmp_path, "dataclass_model", DATACLASS_SRC)
    asts = _parse_files([file_path])
    ctx = _SchemaResolveCtx(asts=asts, files_root=tmp_path, visited=set(), depth=0, conventions=None)

    schema, source = _resolve_class_schema("Person", ctx)
    assert schema is not None
    assert schema["type"] == "object"
    assert "name" in schema["properties"]
    assert "address" in schema["properties"]


def test_cycle_detection(tmp_path: Path):
    """Test that cycles are detected and don't cause infinite loops."""
    file_path = _write_fixture(tmp_path, "cycle_test", CYCLE_SRC)
    asts = _parse_files([file_path])
    ctx = _SchemaResolveCtx(asts=asts, files_root=tmp_path, visited=set(), depth=0, conventions=None)

    schema, source = _resolve_class_schema("A", ctx)
    assert schema is not None
    # The cycle should be handled gracefully — A and B resolve but without infinite recursion
    assert schema["type"] == "object"
    assert "name" in schema["properties"]


def test_list_and_dict_generics(tmp_path: Path):
    """Test resolving list[Class] and dict[str, Class] with nested types."""
    file_path = _write_fixture(tmp_path, "generics_test", GENERICS_SRC)
    asts = _parse_files([file_path])
    ctx = _SchemaResolveCtx(asts=asts, files_root=tmp_path, visited=set(), depth=0, conventions=None)

    schema, source = _resolve_class_schema("Container", ctx)
    assert schema is not None
    assert "items" in schema["properties"]
    assert schema["properties"]["items"]["type"] == "array"
    assert "metadata" in schema["properties"]


def test_depth_overflow(tmp_path: Path):
    """Test that depth overflow is handled gracefully."""
    file_path = _write_fixture(tmp_path, "nested_td", NESTED_TYPEDDICT_SRC)
    asts = _parse_files([file_path])
    ctx = _SchemaResolveCtx(asts=asts, files_root=tmp_path, visited=set(), depth=6, conventions=None)

    result = _resolve_class_schema("Person", ctx)
    # Should return None when depth is exceeded
    assert result is None
