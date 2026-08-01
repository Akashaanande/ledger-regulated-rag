"""Rank-quality metrics: recall@k, MRR, nDCG@k.

Deliberately independent of Ragas and of any retrieval/index implementation.
Each function takes a ranked list of retrieved doc ids and the set of ids
considered relevant for that query, so the same code scores dense-only,
BM25-only, hybrid, and reranked runs identically.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant ids present in the top-k retrieved ids."""
    if not relevant:
        raise ValueError("relevant set must be non-empty")
    hits = set(retrieved[:k]) & relevant
    return len(hits) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant id, or 0.0 if none of the top results are relevant."""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG@k."""
    if not relevant:
        raise ValueError("relevant set must be non-empty")

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved[:k], start=1)
        if doc_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def mean_recall_at_k(runs: Iterable[tuple[Sequence[str], set[str]]], k: int) -> float:
    values = [recall_at_k(retrieved, relevant, k) for retrieved, relevant in runs]
    return sum(values) / len(values) if values else 0.0


def mean_reciprocal_rank(runs: Iterable[tuple[Sequence[str], set[str]]]) -> float:
    values = [reciprocal_rank(retrieved, relevant) for retrieved, relevant in runs]
    return sum(values) / len(values) if values else 0.0


def mean_ndcg_at_k(runs: Iterable[tuple[Sequence[str], set[str]]], k: int) -> float:
    values = [ndcg_at_k(retrieved, relevant, k) for retrieved, relevant in runs]
    return sum(values) / len(values) if values else 0.0
