from __future__ import annotations

import pytest

from ledger.chunking.naive_chunker import Chunk, chunk_text, naive_chunk_prose
from ledger.ingestion.docling_parser import DocElement, ElementKind


class _CharTokenizer:
    """Toy tokenizer: one token per character. Deterministic, no ML deps."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


# --- chunk_text: pure splitting logic ---


def test_chunk_text_splits_into_fixed_size_windows():
    result = chunk_text("abcdefghijkl", _CharTokenizer(), chunk_size=5)

    assert result == (Chunk(text="abcde"), Chunk(text="fghij"), Chunk(text="kl"))


def test_chunk_text_exact_multiple_has_no_trailing_short_chunk():
    result = chunk_text("abcdef", _CharTokenizer(), chunk_size=3)

    assert result == (Chunk(text="abc"), Chunk(text="def"))


def test_chunk_text_shorter_than_chunk_size_is_one_chunk():
    result = chunk_text("ab", _CharTokenizer(), chunk_size=512)

    assert result == (Chunk(text="ab"),)


def test_chunk_text_on_empty_string_returns_no_chunks():
    assert chunk_text("", _CharTokenizer(), chunk_size=512) == ()


def test_chunk_text_rejects_non_positive_chunk_size():
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_text("abc", _CharTokenizer(), chunk_size=0)


# --- naive_chunk_prose: element-stream wiring ---


def test_naive_chunk_prose_joins_elements_with_a_space():
    elements = (
        DocElement(kind=ElementKind.TEXT, text="ab"),
        DocElement(kind=ElementKind.FOOTNOTE, text="cd"),
    )

    result = naive_chunk_prose(elements, _CharTokenizer(), chunk_size=512)

    assert result == (Chunk(text="ab cd"),)


def test_naive_chunk_prose_ignores_section_boundaries_across_elements():
    elements = (
        DocElement(kind=ElementKind.TEXT, text="ab", section_path=("Item 8",)),
        DocElement(kind=ElementKind.TEXT, text="cd", section_path=("Item 9",)),
    )

    result = naive_chunk_prose(elements, _CharTokenizer(), chunk_size=3)

    # "ab cd" split blindly at 3 chars -- a chunk boundary falls mid-stream,
    # straddling both sections, because this chunker has no idea they exist.
    assert result == (Chunk(text="ab "), Chunk(text="cd"))


def test_naive_chunk_prose_on_empty_elements_returns_no_chunks():
    assert naive_chunk_prose((), _CharTokenizer(), chunk_size=512) == ()


# --- bge_tokenizer: real integration, skipped when transformers isn't installed ---


def test_bge_tokenizer_round_trips_real_text():
    pytest.importorskip("transformers")
    from ledger.chunking.naive_chunker import bge_tokenizer

    tokenizer = bge_tokenizer()
    ids = tokenizer.encode("Total net sales rose.")

    assert len(ids) > 0
    assert isinstance(tokenizer.decode(ids), str)
