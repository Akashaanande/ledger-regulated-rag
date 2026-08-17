"""Table serialiser: fuse a routed TABLE element into one atomic markdown
unit -- see README's "Table serialiser" node and DECISIONS.md ADR-005.

docling_parser.py already exports each table's full markdown body (a
multi-page table is merged across its provenance entries by docling itself)
and extracts its caption onto `DocElement.table_caption`. This module's job
is narrower: fuse caption + body into a single string so the
structural/contextual chunkers (LEDGER-020) can treat a table as one
indivisible unit -- never splitting it by token count the way the naive
512-token chunker splits prose.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ledger.ingestion.docling_parser import DocElement, ElementKind


@dataclass(frozen=True)
class SerialisedTable:
    """A table element fused into one atomic markdown unit, ready for chunking."""

    markdown: str
    section_path: tuple[str, ...] = ()
    page_number: int | None = None


def serialize_table(element: DocElement) -> SerialisedTable:
    """Fuse one TABLE element's caption and body into a single markdown unit."""
    if element.kind != ElementKind.TABLE:
        raise ValueError(f"expected a TABLE element, got {element.kind!r}")

    markdown = (
        f"**{element.table_caption}**\n\n{element.text}"
        if element.table_caption
        else element.text
    )

    return SerialisedTable(
        markdown=markdown,
        section_path=element.section_path,
        page_number=element.page_number,
    )


def serialize_tables(elements: Iterable[DocElement]) -> tuple[SerialisedTable, ...]:
    """Serialise every TABLE element in `elements`, preserving document order."""
    return tuple(serialize_table(e) for e in elements)
