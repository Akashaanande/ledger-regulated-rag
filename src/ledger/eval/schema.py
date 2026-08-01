"""Gold question set schema.

A gold record is the unit the whole eval harness is built around: one
question, one human-checkable answer, and a precise pointer back to the
filing text that answer came from. Everything in eval/ consumes this shape.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class DifficultyBand(StrEnum):
    DIRECT_LOOKUP = "direct_lookup"
    CROSS_TABLE = "cross_table"
    FOOTNOTE_DEPENDENT = "footnote_dependent"
    MULTI_DOCUMENT = "multi_document"


class SourceRef(BaseModel):
    """One pointer into a specific filing: which document, which fact."""

    cik: str = Field(description="SEC CIK, e.g. '0000320193'")
    accession_number: str = Field(description="e.g. '0000320193-24-000123'")
    form_type: str = Field(description="'10-K' or '10-Q'")
    fiscal_period: str = Field(description="e.g. 'FY2024' or 'Q2FY2025'")
    section_or_concept: str = Field(
        description="Filing section (e.g. 'Item 7 MD&A') or XBRL concept "
        "(e.g. 'us-gaap:Revenues') the answer was pulled from"
    )
    excerpt: str = Field(description="The verbatim text/value the answer is grounded in")
    filing_url: str = Field(description="Direct URL to the filing or fact")


class GoldQuestion(BaseModel):
    id: str = Field(description="Stable id, e.g. 'dl-001'")
    band: DifficultyBand
    question: str
    answer: str
    sources: list[SourceRef] = Field(
        min_length=1,
        description="One entry for direct/footnote lookups; 2+ for cross-table/multi-doc",
    )
    verified: bool = Field(
        default=False,
        description="True once a human has spot-checked this record against the source filing",
    )
    notes: str | None = Field(default=None, description="Why this question is hard, if non-obvious")

    @model_validator(mode="after")
    def _band_matches_source_count(self) -> GoldQuestion:
        multi_source_bands = {DifficultyBand.CROSS_TABLE, DifficultyBand.MULTI_DOCUMENT}
        if self.band in multi_source_bands and len(self.sources) < 2:
            msg = f"{self.band} questions require at least 2 sources, got {len(self.sources)}"
            raise ValueError(msg)
        return self
