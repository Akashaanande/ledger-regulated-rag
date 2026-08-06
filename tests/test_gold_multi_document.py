from pathlib import Path

from ledger.eval.gold_set import load_gold_set
from ledger.eval.schema import DifficultyBand

GOLD_PATH = Path(__file__).parent.parent / "eval" / "gold_questions" / "multi_document.jsonl"


def test_multi_document_has_30_questions():
    questions = load_gold_set(GOLD_PATH)
    assert len(questions) == 30


def test_multi_document_ids_are_unique():
    questions = load_gold_set(GOLD_PATH)
    ids = [q.id for q in questions]
    assert len(ids) == len(set(ids))


def test_multi_document_all_tagged_correct_band():
    questions = load_gold_set(GOLD_PATH)
    assert all(q.band == DifficultyBand.MULTI_DOCUMENT for q in questions)


def test_multi_document_every_question_spans_at_least_two_filings():
    """The whole point of the band: no single filing answers the question."""
    questions = load_gold_set(GOLD_PATH)
    for q in questions:
        accession_numbers = {source.accession_number for source in q.sources}
        assert len(accession_numbers) >= 2, f"{q.id} only cites one accession number"


def test_multi_document_every_source_cites_an_apple_10k():
    questions = load_gold_set(GOLD_PATH)
    for q in questions:
        for source in q.sources:
            assert source.cik == "0000320193"
            assert source.form_type == "10-K"


def test_multi_document_not_yet_human_verified():
    """AI-drafted from primary-source filings; verified flips to True once spot-checked."""
    questions = load_gold_set(GOLD_PATH)
    assert all(q.verified is False for q in questions)
