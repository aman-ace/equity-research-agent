"""SEC EDGAR access: ticker lookup, XBRL company facts, and filing metadata.

Everything here reads primary sources published by the SEC. No vendor feed sits
in between, which is what makes the resulting memo auditable.
"""

from __future__ import annotations

import re
from typing import Any

from .sources import SourceClient, SourceError
from .valuation import FundamentalSnapshot

TICKER_INDEX_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
FILING_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{document}"

_ANNUAL_DURATION_FRAME = re.compile(r"^CY\d{4}$")
_ANNUAL_INSTANT_FRAME = re.compile(r"^CY\d{4}Q4I$")

# US-GAAP concepts vary by filer, and a filer may switch tags between years. Each
# field lists candidates in priority order; values are merged per year, with the
# highest-priority concept winning any year it covers.
CONCEPTS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "operating_income": ("OperatingIncomeLoss",),
    "total_assets": ("Assets",),
    "total_liabilities": ("Liabilities",),
    "stockholders_equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "operating_cash_flow": (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    "diluted_eps": ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"),
    "shares_diluted": (
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ),
}

INSTANT_FIELDS = frozenset({"total_assets", "total_liabilities", "stockholders_equity"})
_UNITS = {"diluted_eps": "USD/shares", "shares_diluted": "shares"}


def pad_cik(cik: int | str) -> str:
    """EDGAR's JSON endpoints expect a zero-padded ten-digit CIK."""
    return str(int(cik)).zfill(10)


def resolve_ticker(client: SourceClient, query: str) -> dict[str, Any]:
    """Resolve a ticker symbol or company name to its CIK.

    Exact ticker matches win; otherwise the first case-insensitive substring
    match on the company name is returned. Class shares are hyphenated in the
    SEC's index (``BRK-B``), so a dotted ``BRK.B`` is accepted as well.
    """
    index = client.get_json(
        TICKER_INDEX_URL,
        "U.S. Securities and Exchange Commission, company ticker index",
    )
    needle = query.strip().upper()
    exact = {needle, needle.replace(".", "-")}
    partial: dict[str, Any] | None = None
    for entry in index.values():
        ticker = str(entry.get("ticker", "")).upper()
        title = str(entry.get("title", ""))
        if ticker in exact:
            return {"ticker": ticker, "name": title, "cik": pad_cik(entry["cik_str"])}
        if partial is None and needle in title.upper():
            partial = {"ticker": ticker, "name": title, "cik": pad_cik(entry["cik_str"])}
    if partial is not None:
        return partial
    raise SourceError(f"no SEC registrant matches {query!r}")


def company_facts(client: SourceClient, cik: str, name: str = "") -> dict[str, Any]:
    """Fetch the full XBRL company-facts document for a CIK."""
    label = f"{name or 'Registrant'}, XBRL company facts (SEC EDGAR)"
    return client.get_json(COMPANY_FACTS_URL.format(cik=pad_cik(cik)), label)


def _annual_points(
    facts: dict[str, Any], concept: str, unit: str, instant: bool
) -> dict[int, float]:
    """Pull one concept's annual values out of a company-facts document.

    Prefers the SEC's own ``frame`` labels (``CY2023``, ``CY2023Q4I``), which are
    already deduplicated and calendar-aligned. Falls back to 10-K full-year
    entries when a filer has no frames, keeping the most recently filed value
    for each period.
    """
    units = facts.get("facts", {}).get("us-gaap", {}).get(concept, {}).get("units", {})
    entries = units.get(unit, [])
    pattern = _ANNUAL_INSTANT_FRAME if instant else _ANNUAL_DURATION_FRAME

    framed: dict[int, float] = {}
    for entry in entries:
        frame = entry.get("frame")
        if frame and pattern.match(frame):
            framed[int(frame[2:6])] = float(entry["val"])
    if framed:
        return framed

    fallback: dict[int, tuple[str, float]] = {}
    for entry in entries:
        if entry.get("form") != "10-K" or entry.get("fp") != "FY":
            continue
        end = entry.get("end")
        if not end:
            continue
        if not instant:
            start = entry.get("start")
            if not start or not _is_full_year(start, end):
                continue
        year = int(end[:4])
        filed = str(entry.get("filed", ""))
        if year not in fallback or filed >= fallback[year][0]:
            fallback[year] = (filed, float(entry["val"]))
    return {year: value for year, (_, value) in fallback.items()}


def _is_full_year(start: str, end: str) -> bool:
    """True when a reporting period spans roughly twelve months."""
    from datetime import date

    try:
        days = (date.fromisoformat(end) - date.fromisoformat(start)).days
    except ValueError:  # pragma: no cover - malformed upstream date
        return False
    return 340 <= days <= 400


def fundamentals(facts: dict[str, Any], years: int = 5) -> list[FundamentalSnapshot]:
    """Build a series of annual snapshots, oldest first.

    Filers change tags over time: NVIDIA reports older years under
    ``RevenueFromContractWithCustomerExcludingAssessedTax`` and recent ones under
    ``Revenues``. Candidates are therefore merged per year rather than settling
    on the first concept that has any data at all — otherwise one concept
    covering only stale years would shadow another covering the years actually
    being reported, and the field would read as untagged.

    Args:
        facts: A company-facts document from :func:`company_facts`.
        years: How many of the most recent fiscal years to keep.
    """
    by_field: dict[str, dict[int, float]] = {}
    for field, candidates in CONCEPTS.items():
        unit = _UNITS.get(field, "USD")
        merged: dict[int, float] = {}
        for concept in candidates:
            # setdefault keeps the highest-priority concept's value for a year
            # and lets a lower-priority one fill only the years it left empty.
            for year, value in _annual_points(
                facts, concept, unit, field in INSTANT_FIELDS
            ).items():
                merged.setdefault(year, value)
        by_field[field] = merged

    all_years = sorted({year for points in by_field.values() for year in points})
    if not all_years:
        raise SourceError("company facts contained no annual US-GAAP data")
    selected = all_years[-years:]
    return [
        FundamentalSnapshot(
            fiscal_year=year,
            **{field: by_field[field].get(year) for field in CONCEPTS},
        )
        for year in selected
    ]


def recent_filings(
    client: SourceClient,
    cik: str,
    forms: tuple[str, ...] = ("10-K", "10-Q", "8-K"),
    limit: int = 10,
) -> list[dict[str, str]]:
    """List recent filings with direct document URLs.

    Args:
        client: Source client used for the request.
        cik: Registrant CIK, padded or not.
        forms: Filing types to keep, e.g. ``("10-K",)``.
        limit: Maximum number of filings returned.
    """
    padded = pad_cik(cik)
    data = client.get_json(SUBMISSIONS_URL.format(cik=padded), "SEC EDGAR filing history")
    recent = data.get("filings", {}).get("recent", {})
    wanted = {form.upper() for form in forms}
    results: list[dict[str, str]] = []
    for index, form in enumerate(recent.get("form", [])):
        if form.upper() not in wanted:
            continue
        accession = recent["accessionNumber"][index].replace("-", "")
        document = recent.get("primaryDocument", [""] * (index + 1))[index]
        results.append(
            {
                "form": form,
                "filing_date": recent.get("filingDate", [""])[index],
                "report_date": recent.get("reportDate", [""])[index],
                "description": recent.get("primaryDocDescription", [""])[index],
                "url": FILING_INDEX_URL.format(
                    cik_int=int(padded), accession=accession, document=document
                ),
            }
        )
        if len(results) >= limit:
            break
    return results


_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def filing_text(client: SourceClient, url: str, max_chars: int = 60_000) -> str:
    """Fetch a filing and strip it to readable text.

    Filings run to hundreds of thousands of characters; the result is truncated
    so a single tool result cannot swamp the context window.
    """
    raw = client.get_text(url, "SEC EDGAR filing document")
    body = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    text = _TAG.sub(" ", body)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&#8217;", "'")
        .replace("&#8212;", "—")
    )
    text = _WHITESPACE.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[truncated]"
    return text
