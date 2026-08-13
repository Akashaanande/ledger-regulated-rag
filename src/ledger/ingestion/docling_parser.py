"""Docling integration: turn a downloaded filing into ledger's own structured
document representation (sections, tables, footnotes) -- see README's
Ingestion stage.

This module is the only place that imports docling, so its (fast-moving)
API surface can change without rippling into the element router
(LEDGER-015), table serialiser (LEDGER-016), or chunkers -- all of those
depend only on the ElementKind/DocElement/ParsedFiling types below, and the
one function that actually walks a DoclingDocument's tree
(`_elements_from_docling_document`) is isolated so a docling upgrade that
renames something has exactly one place to fix.

docling is imported lazily inside `_convert_with_docling`, not at module
scope: it's a heavy dependency (torch, transformers, OCR models) and the
eval-harness-only workflow (metrics.py, gold_set.py, schema.py) shouldn't
need to pull it in.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class ElementKind(StrEnum):
    SECTION_HEADER = "section_header"
    TEXT = "text"
    TABLE = "table"
    FOOTNOTE = "footnote"
    OTHER = "other"


_DOCLING_LABEL_TO_KIND: dict[str, ElementKind] = {
    "section_header": ElementKind.SECTION_HEADER,
    "title": ElementKind.SECTION_HEADER,
    "text": ElementKind.TEXT,
    "paragraph": ElementKind.TEXT,
    "list_item": ElementKind.TEXT,
    "footnote": ElementKind.FOOTNOTE,
    "table": ElementKind.TABLE,
}


def docling_label_to_kind(label: str) -> ElementKind:
    """Map a docling DocItemLabel's string value to ledger's ElementKind."""
    return _DOCLING_LABEL_TO_KIND.get(label.lower(), ElementKind.OTHER)


@dataclass(frozen=True)
class DocElement:
    """One element of a parsed filing, in document order."""

    kind: ElementKind
    text: str
    section_path: tuple[str, ...] = ()
    page_number: int | None = None
    table_caption: str | None = None


@dataclass(frozen=True)
class ParsedFiling:
    """A filing after Docling parsing, before element routing or chunking."""

    cik: str
    accession_number: str
    form_type: str
    elements: tuple[DocElement, ...] = field(default_factory=tuple)

    @property
    def sections(self) -> tuple[DocElement, ...]:
        return tuple(e for e in self.elements if e.kind == ElementKind.SECTION_HEADER)

    @property
    def tables(self) -> tuple[DocElement, ...]:
        return tuple(e for e in self.elements if e.kind == ElementKind.TABLE)

    @property
    def footnotes(self) -> tuple[DocElement, ...]:
        return tuple(e for e in self.elements if e.kind == ElementKind.FOOTNOTE)


def pop_section_stack_to_level(
    stack: tuple[tuple[int, str], ...], level: int
) -> tuple[tuple[int, str], ...]:
    """Drop any entries at or below `level`; what's left is the parent breadcrumb.

    Pure and docling-independent on purpose, so the section-breadcrumb logic
    can be tested without a real parsed document.
    """
    return tuple(entry for entry in stack if entry[0] < level)


def update_section_stack(
    stack: tuple[tuple[int, str], ...], level: int, text: str
) -> tuple[tuple[int, str], ...]:
    """Push a heading onto the section stack, popping any deeper-or-equal headings first."""
    return (*pop_section_stack_to_level(stack, level), (level, text))


def _table_caption(item: Any, docling_document: Any) -> str | None:
    if hasattr(item, "caption_text"):
        caption = item.caption_text(docling_document)
        return caption or None
    captions = getattr(item, "captions", None)
    if captions:
        resolved = captions[0].resolve(docling_document)
        return getattr(resolved, "text", None)
    return None


def _page_number(item: Any) -> int | None:
    prov = getattr(item, "prov", None)
    if prov:
        return getattr(prov[0], "page_no", None)
    return None


def _elements_from_docling_document(docling_document: Any) -> tuple[DocElement, ...]:
    """Flatten a DoclingDocument's item tree into ledger's DocElement list.

    The single point of contact with docling's document schema
    (`iterate_items`, `.label`, `.text`, `.prov`, table export/captions) --
    see the module docstring for why this is deliberately isolated here.
    """
    elements: list[DocElement] = []
    section_stack: tuple[tuple[int, str], ...] = ()

    for item, level in docling_document.iterate_items():
        label = getattr(item, "label", None)
        label_name = getattr(label, "value", str(label)) if label is not None else ""
        kind = docling_label_to_kind(label_name)
        page_number = _page_number(item)

        if kind == ElementKind.SECTION_HEADER:
            text = getattr(item, "text", "") or ""
            parent_path = tuple(t for _, t in pop_section_stack_to_level(section_stack, level))
            elements.append(
                DocElement(
                    kind=kind, text=text, section_path=parent_path, page_number=page_number
                )
            )
            section_stack = update_section_stack(section_stack, level, text)
            continue

        current_path = tuple(text for _, text in section_stack)

        if kind == ElementKind.TABLE:
            elements.append(
                DocElement(
                    kind=kind,
                    text=item.export_to_markdown(docling_document),
                    section_path=current_path,
                    page_number=page_number,
                    table_caption=_table_caption(item, docling_document),
                )
            )
            continue

        text = getattr(item, "text", "") or ""
        if not text:
            continue
        elements.append(
            DocElement(kind=kind, text=text, section_path=current_path, page_number=page_number)
        )

    return tuple(elements)


def _convert_with_docling(content: bytes, filename: str) -> Any:
    """The one place this module calls docling's DocumentConverter."""
    from docling.document_converter import DocumentConverter

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / filename
        tmp_path.write_bytes(content)
        result = DocumentConverter().convert(str(tmp_path))
        return result.document


def parse_filing(
    content: bytes, filename: str, *, cik: str, accession_number: str, form_type: str
) -> ParsedFiling:
    """Parse a downloaded filing's raw bytes into ledger's structured representation."""
    docling_document = _convert_with_docling(content, filename)
    elements = _elements_from_docling_document(docling_document)
    return ParsedFiling(
        cik=cik, accession_number=accession_number, form_type=form_type, elements=elements
    )
