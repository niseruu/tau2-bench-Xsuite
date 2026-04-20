"""Data models for the AssistXSuite domain."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from tau2.domains.assistxsuite.utils import ASSISTXSUITE_DB_PATH
from tau2.environment.db import DB


class Chunk(BaseModel):
    """A searchable chunk within a legal document."""

    id: str = Field(description="Unique chunk identifier")
    content: str = Field(description="Chunk text")
    positions: list[list[int]] = Field(
        default_factory=list,
        description="Mock positional information for the chunk",
    )


class Dataset(BaseModel):
    """Dataset metadata."""

    id: str = Field(description="Unique dataset identifier")
    name: str = Field(description="Display name")
    description: str = Field(description="Dataset summary")
    document_ids: list[str] = Field(
        default_factory=list,
        description="Documents included in the dataset",
    )


class LegalDocument(BaseModel):
    """A mock legal document stored in the domain."""

    id: str = Field(description="Unique document identifier")
    dataset_id: str = Field(description="Owning dataset")
    title: str = Field(description="Document title")
    content: str = Field(description="Full document text")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Document metadata used for filtering",
    )
    doc_type: str = Field(default="", description="Mock document type")
    chunks: list[Chunk] = Field(
        default_factory=list,
        description="Document chunks used by retrieval",
    )


class ChatAssistant(BaseModel):
    """Mock chat assistant configuration."""

    id: str = Field(description="Unique chat assistant identifier")
    name: str = Field(description="Display name")
    description: str = Field(description="Assistant summary")
    dataset_ids: list[str] = Field(
        default_factory=list,
        description="Datasets available to the assistant",
    )
    opener: str = Field(description="Default greeting")


class AgentDefinition(BaseModel):
    """Mock agent configuration."""

    id: str = Field(description="Unique agent identifier")
    title: str = Field(description="Display title")
    description: str = Field(description="Agent summary")
    dataset_ids: list[str] = Field(
        default_factory=list,
        description="Datasets available to the agent",
    )


class AssistXSuiteDB(DB):
    """Mock AssistXSuite database."""

    datasets: dict[str, Dataset] = Field(
        default_factory=dict,
        description="Datasets keyed by dataset id",
    )
    documents: dict[str, LegalDocument] = Field(
        default_factory=dict,
        description="Legal documents keyed by document id",
    )
    chat_assistants: dict[str, ChatAssistant] = Field(
        default_factory=dict,
        description="Chat assistants keyed by assistant id",
    )
    agents: dict[str, AgentDefinition] = Field(
        default_factory=dict,
        description="Agent definitions keyed by agent id",
    )


def get_db() -> AssistXSuiteDB:
    """Load the default AssistXSuite database."""

    return AssistXSuiteDB.load(ASSISTXSUITE_DB_PATH)

