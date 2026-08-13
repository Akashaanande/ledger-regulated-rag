from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from ledger.ingestion.docling_parser import (
    DocElement,
    ElementKind,
    ParsedFiling,
    _elements_from_docling_document,
    docling_label_to_kind,
    pop_section_stack_to_level,
    update_section_stack,
)

# --- pure logic: ElementKind mapping and the section-breadcrumb stack ---


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("section_header", ElementKind.SECTION_HEADER),
        ("title", ElementKind.SECTION_HEADER),
        ("text", ElementKind.TEXT),
        ("paragraph", ElementKind.TEXT),
        ("list_item", ElementKind.TEXT),
        ("footnote", ElementKind.FOOTNOTE),
        ("table", ElementKind.TABLE),
        ("picture", ElementKind.OTHER),
        ("SECTION_HEADER", ElementKind.SECTION_HEADER),
    ],
)
def test_docling_label_to_kind(label, expected):
    assert docling_label_to_kind(label) == expected


def test_pop_section_stack_to_level_drops_deeper_and_equal_entries():
    stack = ((0, "Item 8"), (1, "Income Taxes"), (2, "Reconciliation"))
    assert pop_section_stack_to_level(stack, 1) == ((0, "Item 8"),)
    assert pop_section_stack_to_level(stack, 0) == ()
    assert pop_section_stack_to_level(stack, 3) == stack


def test_update_section_stack_pushes_and_pops():
    stack: tuple = ()
    stack = update_section_stack(stack, 0, "Item 8")
    stack = update_section_stack(stack, 1, "Income Taxes")
    assert stack == ((0, "Item 8"), (1, "Income Taxes"))

    stack = update_section_stack(stack, 0, "Item 9")
    assert stack == ((0, "Item 9"),)


# --- ParsedFiling filtering properties ---


def test_parsed_filing_filters_by_kind():
    filing = ParsedFiling(
        cik="0000320193",
        accession_number="0000320193-24-000123",
        form_type="10-K",
        elements=(
            DocElement(kind=ElementKind.SECTION_HEADER, text="Item 8"),
            DocElement(kind=ElementKind.TEXT, text="Some prose"),
            DocElement(kind=ElementKind.TABLE, text="| a | b |"),
            DocElement(kind=ElementKind.FOOTNOTE, text="See note 3"),
        ),
    )

    assert [e.text for e in filing.sections] == ["Item 8"]
    assert [e.text for e in filing.tables] == ["| a | b |"]
    assert [e.text for e in filing.footnotes] == ["See note 3"]


# --- flattening a fake DoclingDocument tree ---


@dataclass
class FakeLabel:
    value: str


@dataclass
class FakeProv:
    page_no: int


@dataclass
class FakeTextItem:
    label: FakeLabel
    text: str = ""
    prov: list[FakeProv] = field(default_factory=list)


@dataclass
class FakeTableItem:
    label: FakeLabel = field(default_factory=lambda: FakeLabel("table"))
    prov: list[FakeProv] = field(default_factory=list)
    markdown: str = ""
    caption: str | None = None

    def export_to_markdown(self, doc: Any) -> str:
        return self.markdown

    def caption_text(self, doc: Any) -> str | None:
        return self.caption


@dataclass
class FakeDoclingDocument:
    items: list[tuple[Any, int]]

    def iterate_items(self):
        return iter(self.items)


def test_elements_from_docling_document_tracks_nested_section_paths():
    doc = FakeDoclingDocument(
        items=[
            (FakeTextItem(FakeLabel("section_header"), "Item 8", [FakeProv(60)]), 0),
            (FakeTextItem(FakeLabel("section_header"), "Income Taxes", [FakeProv(61)]), 1),
            (FakeTextItem(FakeLabel("text"), "Effective tax rate rose.", [FakeProv(61)]), 2),
            (FakeTableItem(prov=[FakeProv(61)], markdown="| a | b |", caption="Tax recon"), 2),
            (FakeTextItem(FakeLabel("section_header"), "Item 9", [FakeProv(70)]), 0),
            (FakeTextItem(FakeLabel("footnote"), "See note 3.", [FakeProv(70)]), 1),
        ]
    )

    elements = _elements_from_docling_document(doc)

    assert elements == (
        DocElement(kind=ElementKind.SECTION_HEADER, text="Item 8", section_path=(), page_number=60),
        DocElement(
            kind=ElementKind.SECTION_HEADER,
            text="Income Taxes",
            section_path=("Item 8",),
            page_number=61,
        ),
        DocElement(
            kind=ElementKind.TEXT,
            text="Effective tax rate rose.",
            section_path=("Item 8", "Income Taxes"),
            page_number=61,
        ),
        DocElement(
            kind=ElementKind.TABLE,
            text="| a | b |",
            section_path=("Item 8", "Income Taxes"),
            page_number=61,
            table_caption="Tax recon",
        ),
        # "Item 9" is a sibling of "Item 8" (level 0), not nested under it --
        # this is the case the earlier buggy version got wrong.
        DocElement(kind=ElementKind.SECTION_HEADER, text="Item 9", section_path=(), page_number=70),
        DocElement(
            kind=ElementKind.FOOTNOTE, text="See note 3.", section_path=("Item 9",), page_number=70
        ),
    )


def test_elements_from_docling_document_skips_empty_text_items():
    doc = FakeDoclingDocument(
        items=[
            (FakeTextItem(FakeLabel("text"), "", []), 0),
            (FakeTextItem(FakeLabel("text"), "Real content", []), 0),
        ]
    )

    elements = _elements_from_docling_document(doc)

    assert [e.text for e in elements] == ["Real content"]


def test_elements_from_docling_document_falls_back_to_captions_list():
    @dataclass
    class FakeCaptionRef:
        target: FakeTextItem

        def resolve(self, doc: Any) -> FakeTextItem:
            return self.target

    @dataclass
    class FakeTableItemWithoutCaptionText:
        label: FakeLabel = field(default_factory=lambda: FakeLabel("table"))
        prov: list[FakeProv] = field(default_factory=list)
        captions: list[FakeCaptionRef] = field(default_factory=list)

        def export_to_markdown(self, doc: Any) -> str:
            return "| x |"

    caption_item = FakeTextItem(FakeLabel("text"), "Legacy caption")
    doc = FakeDoclingDocument(
        items=[
            (
                FakeTableItemWithoutCaptionText(captions=[FakeCaptionRef(caption_item)]),
                0,
            )
        ]
    )

    elements = _elements_from_docling_document(doc)

    assert elements[0].table_caption == "Legacy caption"


def test_parse_filing_runs_a_real_docling_conversion_end_to_end():
    """Exercises the real DocumentConverter, not the fake tree above.

    docling doesn't install locally on this Windows/Python 3.14 dev box (see
    edgar_client.py-adjacent notes in the commit history); this runs for
    real in CI (Python 3.11/Ubuntu), where it does.
    """
    pytest.importorskip("docling")
    from ledger.ingestion.docling_parser import parse_filing

    html = (
        b"<html><body>"
        b"<h1>Item 8. Financial Statements</h1>"
        b"<p>Total net sales were $391,035 million.</p>"
        b"</body></html>"
    )

    filing = parse_filing(
        html,
        "filing.html",
        cik="0000320193",
        accession_number="0000320193-24-000123",
        form_type="10-K",
    )

    assert filing.cik == "0000320193"
    assert filing.accession_number == "0000320193-24-000123"
    assert filing.form_type == "10-K"
    assert len(filing.elements) > 0
    assert any("391,035" in e.text for e in filing.elements)
