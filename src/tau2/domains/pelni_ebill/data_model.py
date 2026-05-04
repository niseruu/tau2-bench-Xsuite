"""Data models for the PELNI e-Billing domain."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tau2.domains.pelni_ebill.utils import PELNI_EBILL_DB_PATH
from tau2.environment.db import DB


class KnowledgeChunk(BaseModel):
    """A local mock knowledge-base chunk returned by semantic search."""

    id: str = Field(description="Unique chunk identifier")
    title: str = Field(description="Short source title for the chunk")
    content: str = Field(description="Chunk text")
    topic: str = Field(description="High-level e-Billing topic")
    keywords: list[str] = Field(
        default_factory=list,
        description="Extra search keywords for deterministic lexical retrieval",
    )


class InvoiceRecord(BaseModel):
    """A local mock invoice progress record."""

    invoice_number: str = Field(description="PELNI invoice number")
    vendor_bill_num: str = Field(description="Vendor bill or tagihan number")
    division_name: str = Field(description="Division display name")
    division_code: str = Field(description="Division code used for access filtering")
    location: str = Field(description="Operating unit or OU location")
    description: str = Field(description="Invoice description")
    supplier_name: str = Field(description="Supplier display name")
    grand_total: int | float | str | None = Field(description="Invoice grand total")
    vat_total: int | float | str | None = Field(description="Invoice VAT total")
    status: str = Field(description="Invoice workflow status")
    updated_date: str = Field(description="Last updated timestamp")


class SessionContext(BaseModel):
    """Mock chatbot runtime context used by invoice progress filtering."""

    utk_kode: str = Field(default="Null", description="Work unit code")
    is_all: str = Field(default="false", description="Admin/all-division access flag")
    segment_divisi: str = Field(
        default="Null",
        description="Division segment used when is_all is false",
    )


class PelniEbillDB(DB):
    """Mock PELNI e-Billing chatbot database."""

    session_context: SessionContext = Field(
        default_factory=SessionContext,
        description="Mock runtime session context",
    )
    knowledge_chunks: list[KnowledgeChunk] = Field(
        default_factory=list,
        description="Local knowledge-base chunks",
    )
    invoices: list[InvoiceRecord] = Field(
        default_factory=list,
        description="Local invoice progress records",
    )


def get_db() -> PelniEbillDB:
    """Load the default PELNI e-Billing database."""

    return PelniEbillDB.load(PELNI_EBILL_DB_PATH)
