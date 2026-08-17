"""Element router: split a parsed filing's elements onto their downstream
path -- see the README architecture diagram's "Element router" node.

Prose (TEXT, FOOTNOTE) goes straight to the chunkers (LEDGER-017/018/019).
Tables go to the table serialiser (table_serializer.py) first; its markdown
output is then wired back into the structural/contextual chunkers by
LEDGER-020, not here -- this module only does the initial split.

SECTION_HEADER and OTHER elements are dropped: heading text is already
carried on every element's `section_path`, so chunking the heading itself
would just produce a near-empty chunk, and OTHER covers things like page
furniture that has no retrieval value.
"""

from __future__ import annotations

from dataclasses import dataclass

from ledger.ingestion.docling_parser import DocElement, ElementKind, ParsedFiling

_PROSE_KINDS = frozenset({ElementKind.TEXT, ElementKind.FOOTNOTE})


@dataclass(frozen=True)
class RoutedElements:
    """A filing's elements split onto their downstream path, in document order."""

    prose: tuple[DocElement, ...] = ()
    tables: tuple[DocElement, ...] = ()


def route_elements(filing: ParsedFiling) -> RoutedElements:
    """Split a parsed filing's elements into the prose and table paths."""
    prose = tuple(e for e in filing.elements if e.kind in _PROSE_KINDS)
    tables = filing.tables
    return RoutedElements(prose=prose, tables=tables)
