"""Shared fixtures. The suite is hermetic: no network, no API key."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
import pytest

from equity_agent.sources import SourceClient


def mock_source_client(routes: dict[str, tuple[int, str]]) -> SourceClient:
    """A SourceClient whose transport answers from ``routes``.

    Keys are substrings matched against the request URL; values are
    ``(status_code, body)`` pairs. Unmatched URLs return 404.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        for fragment, (status, body) in routes.items():
            if fragment in url:
                return httpx.Response(status, text=body)
        return httpx.Response(404, text=f"no route for {url}")

    transport = httpx.MockTransport(handler)
    return SourceClient(user_agent="test-agent", client=httpx.Client(transport=transport))


TICKER_INDEX = json.dumps(
    {
        "0": {"cik_str": 909832, "ticker": "COST", "title": "COSTCO WHOLESALE CORP /NEW"},
        "1": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "2": {"cik_str": 1067983, "ticker": "BRK-B", "title": "BERKSHIRE HATHAWAY INC"},
    }
)


def _duration(year: int, value: float, frame: bool = True) -> dict[str, Any]:
    entry = {
        "start": f"{year}-01-01",
        "end": f"{year}-12-31",
        "val": value,
        "form": "10-K",
        "fp": "FY",
        "fy": year,
        "filed": f"{year + 1}-02-15",
    }
    if frame:
        entry["frame"] = f"CY{year}"
    return entry


def _instant(year: int, value: float, frame: bool = True) -> dict[str, Any]:
    entry = {
        "end": f"{year}-12-31",
        "val": value,
        "form": "10-K",
        "fp": "FY",
        "fy": year,
        "filed": f"{year + 1}-02-15",
    }
    if frame:
        entry["frame"] = f"CY{year}Q4I"
    return entry


def build_company_facts(frames: bool = True) -> dict[str, Any]:
    """A miniature company-facts document covering three fiscal years."""
    years = [2022, 2023, 2024]
    revenue = {2022: 900e9, 2023: 1000e9, 2024: 1100e9}
    income = {2022: 90e9, 2023: 100e9, 2024: 121e9}
    return {
        "cik": 909832,
        "entityName": "COSTCO WHOLESALE CORP /NEW",
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": [_duration(y, revenue[y], frames) for y in years]}},
                "NetIncomeLoss": {
                    "units": {"USD": [_duration(y, income[y], frames) for y in years]}
                },
                "OperatingIncomeLoss": {
                    "units": {"USD": [_duration(y, 150e9, frames) for y in years]}
                },
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {"USD": [_duration(y, 130e9, frames) for y in years]}
                },
                "Assets": {"units": {"USD": [_instant(y, 2000e9, frames) for y in years]}},
                "Liabilities": {"units": {"USD": [_instant(y, 1200e9, frames) for y in years]}},
                "StockholdersEquity": {
                    "units": {"USD": [_instant(y, 800e9, frames) for y in years]}
                },
                "EarningsPerShareDiluted": {
                    "units": {"USD/shares": [_duration(y, 6.05, frames) for y in years]}
                },
                "WeightedAverageNumberOfDilutedSharesOutstanding": {
                    "units": {"shares": [_duration(y, 20e9, frames) for y in years]}
                },
            }
        },
    }


SUBMISSIONS = json.dumps(
    {
        "cik": "0000909832",
        "filings": {
            "recent": {
                "form": ["10-K", "8-K", "10-Q"],
                "accessionNumber": [
                    "0000909832-24-000012",
                    "0000909832-24-000030",
                    "0000909832-24-000044",
                ],
                "filingDate": ["2024-10-09", "2024-11-01", "2024-12-12"],
                "reportDate": ["2024-09-01", "2024-11-01", "2024-11-24"],
                "primaryDocument": ["cost-20240901.htm", "cost-8k.htm", "cost-10q.htm"],
                "primaryDocDescription": ["10-K", "8-K", "10-Q"],
            }
        },
    }
)


def build_price_chart(
    closes: list[float | None] | None = None,
    currency: str = "USD",
    start: date = date(2024, 1, 1),
) -> dict[str, Any]:
    """A miniature Yahoo chart document, one daily bar per element of ``closes``.

    Bars are stamped at 14:30 UTC — US market open — so the epoch second and the
    trading date agree.
    """
    if closes is None:
        closes = [100.0 + day for day in range(1, 32)]
    midday = datetime(start.year, start.month, start.day, 14, 30, tzinfo=timezone.utc)
    stamps = [int((midday + timedelta(days=offset)).timestamp()) for offset in range(len(closes))]
    return {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {"symbol": "COST", "currency": currency},
                    "timestamp": stamps,
                    "indicators": {"quote": [{"close": list(closes)}]},
                }
            ],
        }
    }


PRICE_CHART = json.dumps(build_price_chart())


@pytest.fixture
def company_facts() -> dict[str, Any]:
    return build_company_facts()


@pytest.fixture
def source_client() -> SourceClient:
    return mock_source_client(
        {
            "company_tickers.json": (200, TICKER_INDEX),
            "companyfacts": (200, json.dumps(build_company_facts())),
            "submissions": (200, SUBMISSIONS),
            "finance.yahoo.com": (200, PRICE_CHART),
            "Archives/edgar": (200, "<html><body><p>Risk Factors</p></body></html>"),
        }
    )
