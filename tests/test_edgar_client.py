from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from ledger.ingestion.edgar_client import (
    DEFAULT_FORM_TYPES,
    EdgarClient,
    Filing,
    RateLimiter,
)

VALID_USER_AGENT = "Ledger Portfolio Project akash.anande66@gmail.com"


@dataclass
class FakeResponse:
    _json: Any = None
    content: bytes = b""
    status: int = 200

    def json(self) -> Any:
        return self._json

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")


@dataclass
class FakeSession:
    response: FakeResponse
    calls: list[dict[str, Any]] = field(default_factory=list)

    def get(self, url, *, headers, params, timeout):
        self.calls.append(
            {"url": url, "headers": headers, "params": params, "timeout": timeout}
        )
        return self.response


SUBMISSIONS_JSON = {
    "filings": {
        "recent": {
            "form": ["10-K", "8-K", "10-Q", "10-Q"],
            "accessionNumber": [
                "0000320193-24-000123",
                "0000320193-24-000100",
                "0000320193-25-000008",
                "0000320193-25-000057",
            ],
            "filingDate": ["2024-11-01", "2024-10-15", "2025-01-31", "2025-05-02"],
            "reportDate": ["2024-09-28", "2024-10-01", "2024-12-28", "2025-03-29"],
            "primaryDocument": [
                "aapl-20240928.htm",
                "a8-k.htm",
                "aapl-20241228.htm",
                "aapl-20250329.htm",
            ],
        }
    }
}


def make_client(response: FakeResponse) -> tuple[EdgarClient, FakeSession]:
    session = FakeSession(response=response)
    no_op_limiter = RateLimiter.__new__(RateLimiter)
    no_op_limiter._min_interval = 0.0
    no_op_limiter._last_call = None
    client = EdgarClient(
        user_agent=VALID_USER_AGENT, session=session, rate_limiter=no_op_limiter
    )
    return client, session


def test_constructor_rejects_missing_user_agent():
    session = FakeSession(response=FakeResponse())
    with pytest.raises(ValueError, match="descriptive User-Agent"):
        EdgarClient(user_agent="", session=session)


def test_constructor_rejects_user_agent_without_contact_email():
    session = FakeSession(response=FakeResponse())
    with pytest.raises(ValueError, match="descriptive User-Agent"):
        EdgarClient(user_agent="Just A Name", session=session)


def test_get_submissions_sends_required_user_agent_header():
    client, session = make_client(FakeResponse(_json=SUBMISSIONS_JSON))

    client.get_submissions("320193")

    assert session.calls[0]["headers"] == {"User-Agent": VALID_USER_AGENT}


def test_get_submissions_zero_pads_cik_in_url():
    client, session = make_client(FakeResponse(_json=SUBMISSIONS_JSON))

    client.get_submissions("320193")

    assert session.calls[0]["url"] == "https://data.sec.gov/submissions/CIK0000320193.json"


def test_list_filings_filters_to_requested_form_types():
    client, _ = make_client(FakeResponse(_json=SUBMISSIONS_JSON))

    filings = client.list_filings("320193", form_types=DEFAULT_FORM_TYPES)

    assert [f.form_type for f in filings] == ["10-K", "10-Q", "10-Q"]
    assert all(f.cik == "0000320193" for f in filings)


def test_list_filings_builds_correct_filing_records():
    client, _ = make_client(FakeResponse(_json=SUBMISSIONS_JSON))

    filings = client.list_filings("320193", form_types=("10-K",))

    assert filings == [
        Filing(
            cik="0000320193",
            accession_number="0000320193-24-000123",
            form_type="10-K",
            filing_date="2024-11-01",
            report_date="2024-09-28",
            primary_document="aapl-20240928.htm",
        )
    ]


def test_filing_document_url_strips_leading_zeros_and_dashes():
    filing = Filing(
        cik="0000320193",
        accession_number="0000320193-24-000123",
        form_type="10-K",
        filing_date="2024-11-01",
        report_date="2024-09-28",
        primary_document="aapl-20240928.htm",
    )

    assert filing.document_url() == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
    )


def test_download_filing_returns_raw_content():
    client, session = make_client(FakeResponse(content=b"<html>filing</html>"))
    filing = Filing(
        cik="0000320193",
        accession_number="0000320193-24-000123",
        form_type="10-K",
        filing_date="2024-11-01",
        report_date="2024-09-28",
        primary_document="aapl-20240928.htm",
    )

    content = client.download_filing(filing)

    assert content == b"<html>filing</html>"
    assert session.calls[0]["url"] == filing.document_url()


def test_search_full_text_builds_query_params_and_parses_hits():
    hits_json = {"hits": {"hits": [{"_id": "0000320193-24-000123"}]}}
    client, session = make_client(FakeResponse(_json=hits_json))

    results = client.search_full_text("State Aid Decision", form_types=("10-K",))

    assert results == [{"_id": "0000320193-24-000123"}]
    assert session.calls[0]["params"] == {"q": "State Aid Decision", "forms": "10-K"}


def test_get_raises_on_http_error_status():
    client, _ = make_client(FakeResponse(status=403))
    with pytest.raises(RuntimeError, match="HTTP 403"):
        client.get_submissions("320193")


def test_from_env_reads_edgar_user_agent(monkeypatch):
    monkeypatch.setenv("EDGAR_USER_AGENT", VALID_USER_AGENT)
    client = EdgarClient.from_env()
    assert client.user_agent == VALID_USER_AGENT


def test_from_env_raises_without_edgar_user_agent(monkeypatch):
    monkeypatch.delenv("EDGAR_USER_AGENT", raising=False)
    with pytest.raises(ValueError, match="descriptive User-Agent"):
        EdgarClient.from_env()


class TestRateLimiter:
    def test_rejects_non_positive_rate(self):
        with pytest.raises(ValueError, match="must be positive"):
            RateLimiter(max_per_second=0)

    def test_waits_between_calls_to_stay_under_the_limit(self, monkeypatch):
        clock = [0.0]
        sleeps: list[float] = []

        monkeypatch.setattr("time.monotonic", lambda: clock[0])
        monkeypatch.setattr("time.sleep", lambda seconds: sleeps.append(seconds))

        limiter = RateLimiter(max_per_second=10)
        limiter.wait()
        clock[0] += 0.02  # only 20ms elapsed, need 100ms between calls at 10/s
        limiter.wait()

        assert sleeps == [pytest.approx(0.08)]

    def test_does_not_wait_once_enough_time_has_passed(self, monkeypatch):
        clock = [0.0]
        sleeps: list[float] = []

        monkeypatch.setattr("time.monotonic", lambda: clock[0])
        monkeypatch.setattr("time.sleep", lambda seconds: sleeps.append(seconds))

        limiter = RateLimiter(max_per_second=10)
        limiter.wait()
        clock[0] += 1.0
        limiter.wait()

        assert sleeps == []
