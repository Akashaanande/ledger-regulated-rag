import pytest
from pydantic import ValidationError

from ledger.eval.gold_set import load_gold_set, save_gold_set
from ledger.eval.schema import DifficultyBand, GoldQuestion, SourceRef


def _source(**overrides) -> SourceRef:
    defaults = dict(
        cik="0000320193",
        accession_number="0000320193-24-000123",
        form_type="10-K",
        fiscal_period="FY2024",
        section_or_concept="us-gaap:Revenues",
        excerpt="Total net sales were $391,035 million.",
        filing_url="https://www.sec.gov/Archives/edgar/data/320193/000032019324000123",
    )
    defaults.update(overrides)
    return SourceRef(**defaults)


def test_direct_lookup_accepts_single_source():
    q = GoldQuestion(
        id="dl-001",
        band=DifficultyBand.DIRECT_LOOKUP,
        question="What was total net sales for FY2024?",
        answer="$391,035 million",
        sources=[_source()],
    )
    assert q.verified is False


def test_cross_table_requires_two_sources():
    with pytest.raises(ValidationError):
        GoldQuestion(
            id="ct-001",
            band=DifficultyBand.CROSS_TABLE,
            question="How did segment operating margin change YoY?",
            answer="It rose 2.1 points.",
            sources=[_source()],
        )


def test_roundtrip_jsonl(tmp_path):
    q = GoldQuestion(
        id="dl-001",
        band=DifficultyBand.DIRECT_LOOKUP,
        question="What was total net sales for FY2024?",
        answer="$391,035 million",
        sources=[_source()],
    )
    path = tmp_path / "direct_lookup.jsonl"
    save_gold_set([q], path)
    loaded = load_gold_set(path)
    assert loaded == [q]
