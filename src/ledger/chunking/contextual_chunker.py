"""Contextual chunker: each chunk is prefixed with an LLM-generated summary
of its surrounding context before embedding -- see README's "Contextual:
LLM-prefixed summary" node and DECISIONS.md ADR-003.

Built on the structural chunker's section boundaries (structural_chunker.py):
a chunk's "surrounding context" is the section it was drawn from, not the
whole filing. Sending an entire 10-K through an LLM for every chunk would
make every ingest run expensive and slow for no proportional gain, and the
section is already the most relevant scope a chunk sits inside.

Like ragas_metrics.py, this module has no opinion on which LLM backs the
context generator: callers inject an already-configured ContextGenerator
(see .env.example -- ANTHROPIC_API_KEY is the project's chosen provider for
both this and answer generation), so importing this module never requires
network access or API keys. ADR-003 is exactly why: the recall/faithfulness
gain has to be measured against structural chunking alone before this
strategy's extra LLM-call cost is worth defending, so the LLM call has to
stay swappable for a deterministic fake in that measurement.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from ledger.chunking.naive_chunker import Tokenizer, chunk_text
from ledger.chunking.structural_chunker import group_by_consecutive_section
from ledger.ingestion.docling_parser import DocElement


class ContextGenerator(Protocol):
    """Anything that can situate one chunk within its section."""

    def generate_context(
        self, *, section_path: tuple[str, ...], section_text: str, chunk_text: str
    ) -> str: ...


@dataclass(frozen=True)
class ContextualChunk:
    """A chunk with its LLM-generated situating context prefixed onto the text."""

    text: str
    section_path: tuple[str, ...] = ()


def _prefix_with_context(context: str, chunk_body: str) -> str:
    context = context.strip()
    return f"{context}\n\n{chunk_body}" if context else chunk_body


def contextual_chunk_prose(
    elements: Iterable[DocElement],
    tokenizer: Tokenizer,
    context_generator: ContextGenerator,
    *,
    chunk_size: int = 512,
) -> tuple[ContextualChunk, ...]:
    """Chunk prose along section boundaries, then prefix each chunk with a
    generated sentence situating it within its own section's full text."""
    chunks: list[ContextualChunk] = []
    for section_path, section_elements in group_by_consecutive_section(elements):
        section_text = " ".join(e.text for e in section_elements)
        for base_chunk in chunk_text(section_text, tokenizer, chunk_size=chunk_size):
            context = context_generator.generate_context(
                section_path=section_path,
                section_text=section_text,
                chunk_text=base_chunk.text,
            )
            chunks.append(
                ContextualChunk(
                    text=_prefix_with_context(context, base_chunk.text),
                    section_path=section_path,
                )
            )
    return tuple(chunks)
