from __future__ import annotations

import pytest

from ledger.ingestion.docling_parser import DocElement, ElementKind
from ledger.ingestion.table_serializer import (
    SerialisedTable,
    serialize_table,
    serialize_tables,
)


def test_serialize_table_prefixes_caption_onto_markdown():
    element = DocElement(
        kind=ElementKind.TABLE,
        text="| segment | revenue |\n| --- | --- |\n| Americas | 167,046 |",
        section_path=("Item 8", "Segment Information"),
        page_number=61,
        table_caption="Revenue by segment",
    )

    result = serialize_table(element)

    assert result == SerialisedTable(
        markdown="**Revenue by segment**\n\n"
        "| segment | revenue |\n| --- | --- |\n| Americas | 167,046 |",
        section_path=("Item 8", "Segment Information"),
        page_number=61,
    )


def test_serialize_table_without_caption_uses_body_only():
    element = DocElement(kind=ElementKind.TABLE, text="| a | b |", page_number=12)

    result = serialize_table(element)

    assert result.markdown == "| a | b |"


def test_serialize_table_rejects_non_table_elements():
    element = DocElement(kind=ElementKind.TEXT, text="Total net sales rose.")

    with pytest.raises(ValueError, match="TABLE"):
        serialize_table(element)


def test_serialize_tables_preserves_document_order():
    elements = (
        DocElement(kind=ElementKind.TABLE, text="| t1 |", table_caption="First"),
        DocElement(kind=ElementKind.TABLE, text="| t2 |", table_caption="Second"),
    )

    result = serialize_tables(elements)

    assert [t.markdown for t in result] == [
        "**First**\n\n| t1 |",
        "**Second**\n\n| t2 |",
    ]


def test_serialize_tables_on_empty_input():
    assert serialize_tables(()) == ()
