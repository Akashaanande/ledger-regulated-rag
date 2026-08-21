from __future__ import annotations

from ledger.chunking.structural_chunker import (
    StructuralChunk,
    structural_chunk_filing,
    structural_chunk_prose,
)
from ledger.ingestion.docling_parser import DocElement, ElementKind
from ledger.ingestion.element_router import RoutedElements


class _CharTokenizer:
    """Toy tokenizer: one token per character. Deterministic, no ML deps."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


def _text(s: str, section_path: tuple[str, ...] = ()) -> DocElement:
    return DocElement(kind=ElementKind.TEXT, text=s, section_path=section_path)


def test_structural_chunk_prose_splits_at_section_boundary_even_when_small():
    elements = (_text("ab", ("Item 8",)), _text("cd", ("Item 9",)))

    result = structural_chunk_prose(elements, _CharTokenizer(), chunk_size=512)

    # "ab" and "cd" would easily fit in one 512-token window together, but
    # they belong to different sections so they must stay separate chunks.
    assert result == (
        StructuralChunk(text="ab", section_path=("Item 8",)),
        StructuralChunk(text="cd", section_path=("Item 9",)),
    )


def test_structural_chunk_prose_merges_consecutive_elements_within_same_section():
    elements = (_text("ab", ("Item 8",)), _text("cd", ("Item 8",)))

    result = structural_chunk_prose(elements, _CharTokenizer(), chunk_size=512)

    assert result == (StructuralChunk(text="ab cd", section_path=("Item 8",)),)


def test_structural_chunk_prose_splits_a_long_section_by_token_count():
    elements = (_text("abcdef", ("Income Taxes",)),)

    result = structural_chunk_prose(elements, _CharTokenizer(), chunk_size=3)

    assert result == (
        StructuralChunk(text="abc", section_path=("Income Taxes",)),
        StructuralChunk(text="def", section_path=("Income Taxes",)),
    )


def test_structural_chunk_prose_preserves_document_order_across_sections():
    elements = (
        _text("first", ("Item 8",)),
        _text("second", ("Item 9",)),
        _text("third", ("Item 10",)),
    )

    result = structural_chunk_prose(elements, _CharTokenizer(), chunk_size=512)

    assert [c.text for c in result] == ["first", "second", "third"]
    assert [c.section_path for c in result] == [("Item 8",), ("Item 9",), ("Item 10",)]


def test_structural_chunk_prose_does_not_merge_nonadjacent_runs_of_same_section():
    elements = (
        _text("a", ("Overview",)),
        _text("b", ("Detail",)),
        _text("c", ("Overview",)),
    )

    result = structural_chunk_prose(elements, _CharTokenizer(), chunk_size=512)

    assert result == (
        StructuralChunk(text="a", section_path=("Overview",)),
        StructuralChunk(text="b", section_path=("Detail",)),
        StructuralChunk(text="c", section_path=("Overview",)),
    )


def test_structural_chunk_prose_on_empty_elements_returns_no_chunks():
    assert structural_chunk_prose((), _CharTokenizer(), chunk_size=512) == ()


# --- structural_chunk_filing: prose + tables wired together (LEDGER-020) ---


def _table(markdown: str, section_path: tuple[str, ...] = ()) -> DocElement:
    return DocElement(kind=ElementKind.TABLE, text=markdown, section_path=section_path)


def test_structural_chunk_filing_appends_tables_after_prose_chunks():
    routed = RoutedElements(
        prose=(_text("ab", ("Item 8",)),),
        tables=(_table("| a | b |", ("Item 8",)),),
    )

    result = structural_chunk_filing(routed, _CharTokenizer(), chunk_size=512)

    assert result == (
        StructuralChunk(text="ab", section_path=("Item 8",)),
        StructuralChunk(text="| a | b |", section_path=("Item 8",)),
    )


def test_structural_chunk_filing_never_splits_a_table_by_token_count():
    long_table = "| " + "x" * 20 + " |"
    routed = RoutedElements(prose=(), tables=(_table(long_table, ("Item 8",)),))

    result = structural_chunk_filing(routed, _CharTokenizer(), chunk_size=3)

    assert result == (StructuralChunk(text=long_table, section_path=("Item 8",)),)


def test_structural_chunk_filing_fuses_table_caption_via_the_serialiser():
    routed = RoutedElements(
        prose=(),
        tables=(
            DocElement(
                kind=ElementKind.TABLE,
                text="| a | b |",
                section_path=("Item 8",),
                table_caption="Revenue by segment",
            ),
        ),
    )

    result = structural_chunk_filing(routed, _CharTokenizer(), chunk_size=512)

    assert result == (
        StructuralChunk(text="**Revenue by segment**\n\n| a | b |", section_path=("Item 8",)),
    )


def test_structural_chunk_filing_on_empty_routed_elements_returns_no_chunks():
    assert structural_chunk_filing(RoutedElements(), _CharTokenizer(), chunk_size=512) == ()
