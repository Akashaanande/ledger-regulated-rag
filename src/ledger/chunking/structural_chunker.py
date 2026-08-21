"""Structural chunker: chunk boundaries respect section structure -- see
README's "Structural: section-aware" node.

The naive chunker (naive_chunker.py) treats the whole prose stream as one
blob and lets a chunk boundary fall wherever the token count runs out, even
mid-section. This chunker never does that: prose is first grouped into
consecutive runs that share the same `section_path`, and only then is each
run windowed by token count. A chunk from "Income Taxes" and a chunk from
"Segment Information" are never the same chunk, no matter how short either
section is.

Token windowing itself is not reimplemented here -- `chunk_text` from
naive_chunker.py is reused as-is. The only thing this module adds is where
the boundaries fall before that windowing happens.

Tables never go through that windowing at all (`structural_chunk_filing`
below). ADR-005 already settled that a table is preserved as one unit
rather than split by token count, so a serialised table becomes exactly one
StructuralChunk, verbatim -- its caption is already its context, the same
role a section boundary plays for prose.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ledger.chunking.naive_chunker import Tokenizer, chunk_text
from ledger.ingestion.docling_parser import DocElement
from ledger.ingestion.element_router import RoutedElements
from ledger.ingestion.table_serializer import serialize_tables


@dataclass(frozen=True)
class StructuralChunk:
    """One token window that never crosses a section boundary."""

    text: str
    section_path: tuple[str, ...] = ()


def group_by_consecutive_section(
    elements: Iterable[DocElement],
) -> tuple[tuple[tuple[str, ...], tuple[DocElement, ...]], ...]:
    """Split `elements` into consecutive runs sharing the same section_path.

    Two runs with an identical section_path that aren't adjacent (e.g. a
    heading repeated in a different part of the filing) stay separate --
    grouping is positional, not a dict keyed by path.

    Public because the contextual chunker (contextual_chunker.py) reuses it
    too: both chunkers share the same notion of a section boundary, only
    what happens inside each run differs.
    """
    groups: list[tuple[tuple[str, ...], list[DocElement]]] = []
    for element in elements:
        if groups and groups[-1][0] == element.section_path:
            groups[-1][1].append(element)
        else:
            groups.append((element.section_path, [element]))
    return tuple((path, tuple(es)) for path, es in groups)


def structural_chunk_prose(
    elements: Iterable[DocElement], tokenizer: Tokenizer, *, chunk_size: int = 512
) -> tuple[StructuralChunk, ...]:
    """Chunk a filing's routed prose elements, one section run at a time."""
    chunks: list[StructuralChunk] = []
    for section_path, section_elements in group_by_consecutive_section(elements):
        text = " ".join(e.text for e in section_elements)
        for chunk in chunk_text(text, tokenizer, chunk_size=chunk_size):
            chunks.append(StructuralChunk(text=chunk.text, section_path=section_path))
    return tuple(chunks)


def structural_chunk_filing(
    routed: RoutedElements, tokenizer: Tokenizer, *, chunk_size: int = 512
) -> tuple[StructuralChunk, ...]:
    """Structurally chunk a filing's prose, then append its tables verbatim.

    Each serialised table becomes exactly one StructuralChunk -- never
    windowed by token count, per ADR-005.
    """
    prose_chunks = structural_chunk_prose(routed.prose, tokenizer, chunk_size=chunk_size)
    table_chunks = tuple(
        StructuralChunk(text=table.markdown, section_path=table.section_path)
        for table in serialize_tables(routed.tables)
    )
    return prose_chunks + table_chunks
