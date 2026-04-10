from pydantic import BaseModel, Field, field_validator


class Workspace(BaseModel):
    id: str
    name: str
    description: str | None = None
    created_at: str
    updated_at: str
    settings: dict = Field(default_factory=dict)


class CreateWorkspaceRequest(BaseModel):
    name: str
    description: str | None = None

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Workspace name cannot be empty")
        if len(v) > 100:
            raise ValueError("Workspace name cannot exceed 100 characters")
        return v


class RenameWorkspaceRequest(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Workspace name cannot be empty")
        if len(v) > 100:
            raise ValueError("Workspace name cannot exceed 100 characters")
        return v
