"""Ragas wiring: context recall, context precision, and faithfulness against the gold set.

Deliberately thin, mirroring metrics.py's independence: this module only shapes
GoldQuestion + system-output pairs into a ragas EvaluationDataset and reports back
per-metric means. It has no opinion on which LLM backs the judge -- callers inject
an already-configured ragas LLM (e.g. `LangchainLLMWrapper(ChatAnthropic(...))`),
so importing this module never requires network access or API keys.

ragas is imported lazily inside the functions that need it, not at module scope:
the eval-harness-only workflow (metrics.py, gold_set.py, schema.py) has to keep
working even before the full, heavier dependency set is installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ledger.eval.schema import GoldQuestion

if TYPE_CHECKING:
    from ragas import EvaluationDataset


@dataclass
class RagasCase:
    """One retrieval-plus-generation run to score: a gold question plus what the system produced."""

    gold: GoldQuestion
    retrieved_contexts: list[str]
    generated_answer: str


def cases_from_runs(
    gold_questions: list[GoldQuestion],
    retrieved_contexts: list[list[str]],
    generated_answers: list[str],
) -> list[RagasCase]:
    """Zip parallel per-question run outputs into RagasCases, one per gold question."""
    if not (len(gold_questions) == len(retrieved_contexts) == len(generated_answers)):
        raise ValueError(
            "gold_questions, retrieved_contexts, and generated_answers must be the same length: "
            f"got {len(gold_questions)}, {len(retrieved_contexts)}, {len(generated_answers)}"
        )
    return [
        RagasCase(gold=gold, retrieved_contexts=contexts, generated_answer=answer)
        for gold, contexts, answer in zip(
            gold_questions, retrieved_contexts, generated_answers, strict=True
        )
    ]


def build_evaluation_dataset(cases: list[RagasCase]) -> EvaluationDataset:
    """Shape gold questions + system outputs into a ragas EvaluationDataset."""
    if not cases:
        raise ValueError("cases must be non-empty")

    from ragas import EvaluationDataset, SingleTurnSample

    samples = [
        SingleTurnSample(
            user_input=case.gold.question,
            retrieved_contexts=case.retrieved_contexts,
            response=case.generated_answer,
            reference=case.gold.answer,
        )
        for case in cases
    ]
    return EvaluationDataset(samples=samples)


def default_metrics() -> list[Any]:
    """The three metrics LEDGER-011 asks for: context recall, context precision, faithfulness.

    Isolated in its own function because ragas has renamed its metric classes across
    releases -- if an upgrade breaks this, it's the one place to fix.
    """
    from ragas.metrics import context_precision, context_recall, faithfulness

    return [context_recall, context_precision, faithfulness]


def score(cases: list[RagasCase], llm: Any, metrics: list[Any] | None = None) -> dict[str, float]:
    """Run ragas metrics over `cases` and return each metric's mean score.

    `llm` must already be a ragas-compatible LLM (e.g.
    `LangchainLLMWrapper(ChatAnthropic(...))`) -- this module has no opinion on which
    provider or model backs it.
    """
    from ragas import evaluate

    dataset = build_evaluation_dataset(cases)
    metrics = metrics if metrics is not None else default_metrics()
    result = evaluate(dataset=dataset, metrics=metrics, llm=llm)
    df = result.to_pandas()
    return {metric.name: float(df[metric.name].mean()) for metric in metrics}
