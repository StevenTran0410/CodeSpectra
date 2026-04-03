"""Manifest engine types."""
from enum import Enum

from pydantic import BaseModel


class FileCategory(str, Enum):
    SOURCE = "source"
    TEST = "test"
    CONFIG = "config"
    MIGRATION = "migration"
    DOCS = "docs"
    INFRA = "infra"
    GENERATED = "generated"
    ASSET = "asset"
    SECRET_RISK = "secret-risk"
    OTHER = "other"


class ManifestFile(BaseModel):
    id: str
    snapshot_id: str
    rel_path: str
    language: str | None
    category: FileCategory
    size_bytes: int
    mtime_ns: int
    checksum: str


class BuildManifestRequest(BaseModel):
    snapshot_id: str


class BuildManifestResponse(BaseModel):
    snapshot_id: str
    total_files: int
    new_files: int
    changed_files: int
    unchanged_files: int
    ignored_files: int


class ManifestPreviewResponse(BaseModel):
    snapshot_id: str
    files: list[ManifestFile]


class ManifestTreeNode(BaseModel):
    path: str
    is_dir: bool


class ManifestTreeResponse(BaseModel):
    snapshot_id: str
    nodes: list[ManifestTreeNode]


class ManifestFileContentResponse(BaseModel):
    snapshot_id: str
    rel_path: str
    content: str
    truncated: bool
