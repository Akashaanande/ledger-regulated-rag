from __future__ import annotations

from dataclasses import dataclass, field

from ledger.chunking.contextual_chunker import (
    ContextualChunk,
    contextual_chunk_filing,
    contextual_chunk_prose,
)
from ledger.ingestion.docling_parser import DocElement, ElementKind
from ledger.ingestion.element_router import RoutedElements


class _CharTokenizer:
    """Toy tokenizer: one token per character. Deterministic, no ML deps."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


@dataclass
class _FakeContextGenerator:
    """Deterministic stand-in for a real LLM call. Records every call it saw."""

    calls: list[dict] = field(default_factory=list)
    response: str = "CTX"

    def generate_context(
        self, *, section_path: tuple[str, ...], section_text: str, chunk_text: str
    ) -> str:
        self.calls.append(
            {"section_path": section_path, "section_text": section_text, "chunk_text": chunk_text}
        )
        return self.response


def _text(s: str, section_path: tuple[str, ...] = ()) -> DocElement:
    return DocElement(kind=ElementKind.TEXT, text=s, section_path=section_path)


def test_contextual_chunk_prose_prefixes_generated_context_onto_chunk():
    elements = (_text("ab", ("Item 8",)),)
    generator = _FakeContextGenerator(response="Situates ab in Item 8.")

    result = contextual_chunk_prose(elements, _CharTokenizer(), generator, chunk_size=512)

    assert result == (
        ContextualChunk(text="Situates ab in Item 8.\n\nab", section_path=("Item 8",)),
    )


def test_contextual_chunk_prose_calls_generator_once_per_chunk_with_section_text():
    elements = (_text("abcdef", ("Income Taxes",)),)
    generator = _FakeContextGenerator()

    contextual_chunk_prose(elements, _CharTokenizer(), generator, chunk_size=3)

    assert len(generator.calls) == 2
    assert generator.calls[0] == {
        "section_path": ("Income Taxes",),
        "section_text": "abcdef",
        "chunk_text": "abc",
    }
    assert generator.calls[1] == {
        "section_path": ("Income Taxes",),
        "section_text": "abcdef",
        "chunk_text": "def",
    }


def test_contextual_chunk_prose_respects_section_boundaries():
    elements = (_text("ab", ("Item 8",)), _text("cd", ("Item 9",)))
    generator = _FakeContextGenerator(response="")

    result = contextual_chunk_prose(elements, _CharTokenizer(), generator, chunk_size=512)

    assert [c.section_path for c in result] == [("Item 8",), ("Item 9",)]


def test_contextual_chunk_prose_falls_back_to_bare_chunk_when_context_is_blank():
    elements = (_text("ab", ("Item 8",)),)
    generator = _FakeContextGenerator(response="   ")

    result = contextual_chunk_prose(elements, _CharTokenizer(), generator, chunk_size=512)

    assert result == (ContextualChunk(text="ab", section_path=("Item 8",)),)


def test_contextual_chunk_prose_on_empty_elements_returns_no_chunks_and_never_calls_generator():
    generator = _FakeContextGenerator()

    result = contextual_chunk_prose((), _CharTokenizer(), generator, chunk_size=512)

    assert result == ()
    assert generator.calls == []


# --- contextual_chunk_filing: prose + tables wired together (LEDGER-020) ---


def test_contextual_chunk_filing_appends_tables_after_prose_chunks():
    routed = RoutedElements(
        prose=(_text("ab", ("Item 8",)),),
        tables=(
            DocElement(kind=ElementKind.TABLE, text="| a | b |", section_path=("Item 8",)),
        ),
    )
    generator = _FakeContextGenerator(response="Situates ab.")

    result = contextual_chunk_filing(routed, _CharTokenizer(), generator, chunk_size=512)

    assert result == (
        ContextualChunk(text="Situates ab.\n\nab", section_path=("Item 8",)),
        ContextualChunk(text="| a | b |", section_path=("Item 8",)),
    )


def test_contextual_chunk_filing_never_sends_tables_to_the_context_generator():
    routed = RoutedElements(
        prose=(),
        tables=(
            DocElement(kind=ElementKind.TABLE, text="| a | b |", section_path=("Item 8",)),
        ),
    )
    generator = _FakeContextGenerator()

    contextual_chunk_filing(routed, _CharTokenizer(), generator, chunk_size=512)

    assert generator.calls == []


def test_contextual_chunk_filing_fuses_table_caption_via_the_serialiser():
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

    result = contextual_chunk_filing(
        routed, _CharTokenizer(), _FakeContextGenerator(), chunk_size=512
    )

    assert result == (
        ContextualChunk(text="**Revenue by segment**\n\n| a | b |", section_path=("Item 8",)),
    )


def test_contextual_chunk_filing_on_empty_routed_elements_returns_no_chunks():
    generator = _FakeContextGenerator()

    result = contextual_chunk_filing(RoutedElements(), _CharTokenizer(), generator, chunk_size=512)

    assert result == ()
    assert generator.calls == []
