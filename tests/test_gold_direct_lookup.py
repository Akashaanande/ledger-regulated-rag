from pathlib import Path

from ledger.eval.gold_set import load_gold_set
from ledger.eval.schema import DifficultyBand

GOLD_PATH = Path(__file__).parent.parent / "eval" / "gold_questions" / "direct_lookup.jsonl"


def test_direct_lookup_has_30_questions():
    questions = load_gold_set(GOLD_PATH)
    assert len(questions) == 30


def test_direct_lookup_ids_are_unique():
    questions = load_gold_set(GOLD_PATH)
    ids = [q.id for q in questions]
    assert len(ids) == len(set(ids))


def test_direct_lookup_all_tagged_correct_band():
    questions = load_gold_set(GOLD_PATH)
    assert all(q.band == DifficultyBand.DIRECT_LOOKUP for q in questions)


def test_direct_lookup_every_source_cites_the_fy2024_10k():
    questions = load_gold_set(GOLD_PATH)
    for q in questions:
        for source in q.sources:
            assert source.cik == "0000320193"
            assert source.accession_number == "0000320193-24-000123"
            assert source.form_type == "10-K"


def test_direct_lookup_not_yet_human_verified():
    """AI-drafted from SEC XBRL facts; verified flips to True once spot-checked."""
    questions = load_gold_set(GOLD_PATH)
    assert all(q.verified is False for q in questions)
