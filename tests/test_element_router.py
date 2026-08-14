from __future__ import annotations

from ledger.ingestion.docling_parser import DocElement, ElementKind, ParsedFiling
from ledger.ingestion.element_router import RoutedElements, route_elements


def _filing(*elements: DocElement) -> ParsedFiling:
    return ParsedFiling(
        cik="0000320193",
        accession_number="0000320193-24-000123",
        form_type="10-K",
        elements=elements,
    )


def test_route_elements_splits_prose_and_tables():
    filing = _filing(
        DocElement(kind=ElementKind.SECTION_HEADER, text="Item 8"),
        DocElement(kind=ElementKind.TEXT, text="Total net sales rose."),
        DocElement(kind=ElementKind.TABLE, text="| a | b |", table_caption="Revenue by segment"),
        DocElement(kind=ElementKind.FOOTNOTE, text="Restated to reflect a change in estimate."),
    )

    routed = route_elements(filing)

    assert routed == RoutedElements(
        prose=(
            DocElement(kind=ElementKind.TEXT, text="Total net sales rose."),
            DocElement(kind=ElementKind.FOOTNOTE, text="Restated to reflect a change in estimate."),
        ),
        tables=(
            DocElement(
                kind=ElementKind.TABLE, text="| a | b |", table_caption="Revenue by segment"
            ),
        ),
    )


def test_route_elements_drops_section_headers_and_other():
    filing = _filing(
        DocElement(kind=ElementKind.SECTION_HEADER, text="Item 9"),
        DocElement(kind=ElementKind.OTHER, text="Page 214 of 302"),
    )

    routed = route_elements(filing)

    assert routed == RoutedElements(prose=(), tables=())


def test_route_elements_preserves_document_order_within_each_path():
    filing = _filing(
        DocElement(kind=ElementKind.TEXT, text="first"),
        DocElement(kind=ElementKind.TABLE, text="| t1 |"),
        DocElement(kind=ElementKind.TEXT, text="second"),
        DocElement(kind=ElementKind.TABLE, text="| t2 |"),
    )

    routed = route_elements(filing)

    assert [e.text for e in routed.prose] == ["first", "second"]
    assert [e.text for e in routed.tables] == ["| t1 |", "| t2 |"]


def test_route_elements_on_empty_filing():
    assert route_elements(_filing()) == RoutedElements()
