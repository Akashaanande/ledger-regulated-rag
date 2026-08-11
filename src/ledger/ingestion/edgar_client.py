"""SEC EDGAR API client: submissions lookup, full-text search, and filing download.

Every request carries the descriptive User-Agent SEC requires (a name plus a
contact email -- requests without one are routinely rejected with a 403) and
is throttled to stay within EDGAR's published fair-access rate limit. The
HTTP layer is injected via an `HttpClient` protocol so tests never need a
live network call.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
FULL_TEXT_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"

MAX_REQUESTS_PER_SECOND = 10.0
DEFAULT_FORM_TYPES: tuple[str, ...] = ("10-K", "10-Q")
DEFAULT_TIMEOUT_SECONDS = 10.0


class HttpResponse(Protocol):
    content: bytes

    def json(self) -> Any: ...
    def raise_for_status(self) -> None: ...


class HttpClient(Protocol):
    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        params: dict[str, Any] | None,
        timeout: float,
    ) -> HttpResponse: ...


@dataclass(frozen=True)
class Filing:
    """One filing's metadata, as returned by the submissions API."""

    cik: str
    accession_number: str
    form_type: str
    filing_date: str
    report_date: str
    primary_document: str

    @property
    def accession_number_no_dashes(self) -> str:
        return self.accession_number.replace("-", "")

    def document_url(self) -> str:
        cik_no_leading_zeros = self.cik.lstrip("0") or "0"
        return (
            "https://www.sec.gov/Archives/edgar/data/"
            f"{cik_no_leading_zeros}/{self.accession_number_no_dashes}/{self.primary_document}"
        )


class RateLimiter:
    """Sleeps as needed so calls through it never exceed `max_per_second`."""

    def __init__(self, max_per_second: float = MAX_REQUESTS_PER_SECOND) -> None:
        if max_per_second <= 0:
            raise ValueError("max_per_second must be positive")
        self._min_interval = 1.0 / max_per_second
        self._last_call: float | None = None

    def wait(self) -> None:
        now = time.monotonic()
        if self._last_call is not None:
            remaining = self._min_interval - (now - self._last_call)
            if remaining > 0:
                time.sleep(remaining)
        self._last_call = time.monotonic()


def _validate_user_agent(user_agent: str) -> None:
    if not user_agent or "@" not in user_agent:
        raise ValueError(
            "EDGAR requires a descriptive User-Agent with a contact email, e.g. "
            "'Your Name your.email@example.com' (see .env.example: EDGAR_USER_AGENT)"
        )


class EdgarClient:
    """Thin wrapper over SEC EDGAR's submissions, full-text search, and document APIs."""

    def __init__(
        self,
        user_agent: str,
        session: HttpClient,
        rate_limiter: RateLimiter | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        _validate_user_agent(user_agent)
        self._user_agent = user_agent
        self._session = session
        self._rate_limiter = rate_limiter or RateLimiter()
        self._timeout = timeout

    @property
    def user_agent(self) -> str:
        return self._user_agent

    @classmethod
    def from_env(cls, rate_limiter: RateLimiter | None = None) -> EdgarClient:
        """Build a client from EDGAR_USER_AGENT (loading .env first if present)."""
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        user_agent = os.environ.get("EDGAR_USER_AGENT", "")
        _validate_user_agent(user_agent)

        import requests

        return cls(user_agent=user_agent, session=requests.Session(), rate_limiter=rate_limiter)

    def _get(self, url: str, params: dict[str, Any] | None = None) -> HttpResponse:
        self._rate_limiter.wait()
        response = self._session.get(
            url,
            headers={"User-Agent": self._user_agent},
            params=params,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response

    def get_submissions(self, cik: str) -> dict[str, Any]:
        """Raw submissions JSON for a company, given any CIK (zero-padded internally)."""
        padded = cik.zfill(10)
        response = self._get(SUBMISSIONS_URL.format(cik=padded))
        return response.json()

    def list_filings(
        self, cik: str, form_types: tuple[str, ...] = DEFAULT_FORM_TYPES
    ) -> list[Filing]:
        """10-K/10-Q filings for a company, in submissions-API order (newest first)."""
        data = self.get_submissions(cik)
        recent = data.get("filings", {}).get("recent", {})
        padded_cik = cik.zfill(10)

        filings = []
        for i, form in enumerate(recent.get("form", [])):
            if form not in form_types:
                continue
            filings.append(
                Filing(
                    cik=padded_cik,
                    accession_number=recent["accessionNumber"][i],
                    form_type=form,
                    filing_date=recent["filingDate"][i],
                    report_date=recent["reportDate"][i],
                    primary_document=recent["primaryDocument"][i],
                )
            )
        return filings

    def search_full_text(
        self, query: str, form_types: tuple[str, ...] = DEFAULT_FORM_TYPES
    ) -> list[dict[str, Any]]:
        """Full-text search across EDGAR filings; returns raw per-hit dicts."""
        params = {"q": query, "forms": ",".join(form_types)}
        response = self._get(FULL_TEXT_SEARCH_URL, params=params)
        return response.json().get("hits", {}).get("hits", [])

    def download_filing(self, filing: Filing) -> bytes:
        """Raw bytes of a filing's primary document, ready for the parser to consume."""
        response = self._get(filing.document_url())
        return response.content
