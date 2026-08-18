"""Naive chunker: the baseline the other two strategies are measured
against -- see README's "Naive: 512 tokens, fixed" node.

Deliberately ignorant of section boundaries, table structure, and element
boundaries: it concatenates the routed prose stream into one blob and cuts
it into fixed-size token windows. That's the point -- it's the chunker the
recall curve in RESULTS.md is supposed to beat.

Tokenization is delegated to a `Tokenizer` the caller supplies, so the
splitting logic here is testable without pulling in bge-large's real
tokenizer (a heavy transformers/torch dependency). `bge_tokenizer()` is the
one production implementation, imported lazily for the same reason
docling_parser.py imports docling lazily -- see that module's docstring.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from ledger.ingestion.docling_parser import DocElement

_BGE_MODEL_NAME = "BAAI/bge-large-en-v1.5"


class Tokenizer(Protocol):
    """Anything that can round-trip text through token ids."""

    def encode(self, text: str) -> Sequence[int]: ...
    def decode(self, ids: Sequence[int]) -> str: ...


@dataclass(frozen=True)
class Chunk:
    """One fixed-size slice of the prose token stream."""

    text: str


def chunk_text(text: str, tokenizer: Tokenizer, *, chunk_size: int = 512) -> tuple[Chunk, ...]:
    """Split `text` into non-overlapping windows of `chunk_size` tokens."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not text:
        return ()

    token_ids = tokenizer.encode(text)
    return tuple(
        Chunk(text=tokenizer.decode(token_ids[start : start + chunk_size]))
        for start in range(0, len(token_ids), chunk_size)
    )


def naive_chunk_prose(
    elements: Iterable[DocElement], tokenizer: Tokenizer, *, chunk_size: int = 512
) -> tuple[Chunk, ...]:
    """Chunk a filing's routed prose elements as one continuous token stream.

    No section awareness, no per-element boundaries -- elements are joined
    with a single space and chunked as if the whole prose stream were one
    document. Contrast with the structural chunker (LEDGER-018).
    """
    text = " ".join(e.text for e in elements)
    return chunk_text(text, tokenizer, chunk_size=chunk_size)


def bge_tokenizer() -> Tokenizer:
    """The real tokenizer this pipeline embeds with -- see EPIC-05 Indexing.

    Imported lazily: transformers/torch are heavy and the pure chunking
    logic above has no need for them.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(_BGE_MODEL_NAME)
