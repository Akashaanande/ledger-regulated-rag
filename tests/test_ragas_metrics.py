import pytest

from ledger.eval.ragas_metrics import RagasCase, build_evaluation_dataset, cases_from_runs
from ledger.eval.schema import DifficultyBand, GoldQuestion, SourceRef


def _gold(id_: str = "dl-001") -> GoldQuestion:
    return GoldQuestion(
        id=id_,
        band=DifficultyBand.DIRECT_LOOKUP,
        question="What was total net sales for fiscal year 2024?",
        answer="$391,035 million",
        sources=[
            SourceRef(
                cik="0000320193",
                accession_number="0000320193-24-000123",
                form_type="10-K",
                fiscal_period="FY2024",
                section_or_concept="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
                excerpt="Total net sales $391,035 million for the year ended September 28, 2024",
                filing_url="https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/R3.htm",
            )
        ],
    )


def test_cases_from_runs_zips_parallel_lists():
    gold = [_gold("dl-001"), _gold("dl-002")]
    contexts = [["ctx a"], ["ctx b"]]
    answers = ["answer a", "answer b"]

    cases = cases_from_runs(gold, contexts, answers)

    assert len(cases) == 2
    assert cases[0] == RagasCase(
        gold=gold[0], retrieved_contexts=["ctx a"], generated_answer="answer a"
    )
    assert cases[1] == RagasCase(
        gold=gold[1], retrieved_contexts=["ctx b"], generated_answer="answer b"
    )


def test_cases_from_runs_rejects_mismatched_lengths():
    gold = [_gold("dl-001")]
    with pytest.raises(ValueError, match="same length"):
        cases_from_runs(gold, [["ctx a"], ["ctx b"]], ["answer a"])


def test_build_evaluation_dataset_rejects_empty_cases():
    """Guards against calling ragas.evaluate on nothing -- checked before ragas is even imported."""
    with pytest.raises(ValueError, match="non-empty"):
        build_evaluation_dataset([])


def test_build_evaluation_dataset_shapes_one_sample_per_case():
    pytest.importorskip("ragas")
    cases = cases_from_runs([_gold()], [["retrieved context text"]], ["generated answer text"])

    dataset = build_evaluation_dataset(cases)

    assert len(dataset) == 1


def test_default_metrics_returns_the_three_ticket_metrics():
    pytest.importorskip("ragas")
    from ledger.eval.ragas_metrics import default_metrics

    metrics = default_metrics()

    names = {metric.name for metric in metrics}
    assert names == {"context_recall", "context_precision", "faithfulness"}
