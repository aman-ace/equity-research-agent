"""The agent's tool surface.

Each function here is one thing the agent can do. They are deliberately narrow:
the model chooses *which* company data to pull and *how* to interpret it, but
never computes a ratio itself and never states a figure that did not come back
from one of these calls.

Tool errors are returned to the model as ``{"error": ...}`` rather than raised,
so a missing filing or an unrecognized ticker lets the agent adapt instead of
ending the run.

The functions are plain Python and the schemas are written out explicitly, so
nothing in this module depends on a particular model provider.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from . import market, sec, valuation
from .sources import SourceClient, SourceError
from .toolspec import ToolSpec, integer, schema, string


@dataclass
class ResearchContext:
    """Per-run state shared by every tool.

    Company-facts documents run to several megabytes, so they are fetched once
    per ticker and reused for the rest of the run.
    """

    client: SourceClient
    default_years: int = 5
    _companies: dict[str, dict[str, Any]] = field(default_factory=dict)
    _facts: dict[str, dict[str, Any]] = field(default_factory=dict)

    def company(self, query: str) -> dict[str, Any]:
        key = query.strip().upper()
        if key not in self._companies:
            self._companies[key] = sec.resolve_ticker(self.client, query)
        return self._companies[key]

    def facts(self, query: str) -> tuple[dict[str, Any], dict[str, Any]]:
        company = self.company(query)
        cik = company["cik"]
        if cik not in self._facts:
            self._facts[cik] = sec.company_facts(self.client, cik, company["name"])
        return company, self._facts[cik]


_context: ResearchContext | None = None


def configure(context: ResearchContext) -> None:
    """Bind the tools to a run's context. Called by the agent at startup."""
    global _context
    _context = context


def _ctx() -> ResearchContext:
    if _context is None:  # pragma: no cover - guarded by the agent
        raise RuntimeError("tools used before configure() was called")
    return _context


def _json(payload: Any) -> str:
    return json.dumps(payload, default=str, indent=2)


def _snapshot_rows(series: list[valuation.FundamentalSnapshot]) -> list[dict[str, Any]]:
    return [
        {
            "fiscal_year": item.fiscal_year,
            "revenue": item.revenue,
            "operating_income": item.operating_income,
            "net_income": item.net_income,
            "operating_cash_flow": item.operating_cash_flow,
            "total_assets": item.total_assets,
            "total_liabilities": item.total_liabilities,
            "stockholders_equity": item.stockholders_equity,
            "diluted_eps": item.diluted_eps,
            "shares_diluted": item.shares_diluted,
        }
        for item in series
    ]


def lookup_company(query: str) -> str:
    try:
        return _json(_ctx().company(query))
    except SourceError as exc:
        return _json({"error": str(exc)})


def get_fundamentals(ticker: str, years: int = 5) -> str:
    context = _ctx()
    try:
        company, facts = context.facts(ticker)
        series = sec.fundamentals(facts, years=years or context.default_years)
    except SourceError as exc:
        return _json({"error": str(exc)})
    return _json(
        {
            "company": company,
            "units": "as reported (USD unless the filer reports otherwise)",
            "fiscal_years": _snapshot_rows(series),
        }
    )


def get_price_history(ticker: str) -> str:
    try:
        return _json(market.price_history(_ctx().client, ticker))
    except SourceError as exc:
        return _json({"error": str(exc)})


def get_valuation_metrics(ticker: str, years: int = 5) -> str:
    context = _ctx()
    try:
        company, facts = context.facts(ticker)
        series = sec.fundamentals(facts, years=years or context.default_years)
    except SourceError as exc:
        return _json({"error": str(exc)})

    price: float | None = None
    price_note = "share price unavailable; multiples omitted"
    try:
        prices = market.price_history(context.client, company["ticker"])
        price = float(prices["last_close"])  # type: ignore[arg-type]
        price_note = f"last close {price} as of {prices['as_of']}"
    except (SourceError, KeyError, TypeError, ValueError):
        pass

    return _json(
        {
            "company": company,
            "price_basis": price_note,
            "metrics": valuation.analyze(series, price),
            "fiscal_years_used": [item.fiscal_year for item in series],
        }
    )


def list_filings(ticker: str, form: str = "10-K", limit: int = 5) -> str:
    context = _ctx()
    try:
        company = context.company(ticker)
        filings = sec.recent_filings(context.client, company["cik"], forms=(form,), limit=limit)
    except SourceError as exc:
        return _json({"error": str(exc)})
    return _json({"company": company, "filings": filings})


def read_filing(url: str, max_chars: int = 40000) -> str:
    if "sec.gov" not in url:
        return _json({"error": "only sec.gov filing URLs can be read"})
    try:
        return sec.filing_text(_ctx().client, url, max_chars=max_chars)
    except SourceError as exc:
        return _json({"error": str(exc)})


ALL_TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="lookup_company",
        description=(
            "Resolve a ticker symbol or company name to its SEC registrant record and CIK. "
            "Call this first when the user names a company rather than a ticker, or to "
            "confirm which registrant a ticker maps to."
        ),
        input_schema=schema(
            {"query": string('A ticker such as "COST" or a company name such as "Costco".')},
            ["query"],
        ),
        run=lookup_company,
    ),
    ToolSpec(
        name="get_fundamentals",
        description=(
            "Return annual income-statement, balance-sheet, and cash-flow figures as filed "
            "with the SEC in XBRL. Values are in reporting currency units (dollars, not "
            "millions). Fields the filer did not tag come back as null — report them as "
            "untagged rather than substituting a proxy."
        ),
        input_schema=schema(
            {
                "ticker": string("Ticker symbol or company name."),
                "years": integer("Recent fiscal years to return, oldest first.", 5),
            },
            ["ticker"],
        ),
        run=get_fundamentals,
    ),
    ToolSpec(
        name="get_price_history",
        description=(
            "Return end-of-day price levels, trailing returns, and annualized volatility. "
            "Prices are end-of-day and may lag the current session. Use this for market "
            "context and for the share price behind any multiple."
        ),
        input_schema=schema(
            {"ticker": string("Ticker symbol of a US-listed security.")},
            ["ticker"],
        ),
        run=get_price_history,
    ),
    ToolSpec(
        name="get_valuation_metrics",
        description=(
            "Compute margins, returns on capital, leverage, multi-year growth, and trailing "
            "multiples. Every figure is calculated from the filings rather than estimated. "
            "Multiples are trailing: they use the last reported fiscal year and the latest "
            "close, not forward estimates or consensus."
        ),
        input_schema=schema(
            {
                "ticker": string("Ticker symbol or company name."),
                "years": integer("Fiscal years to include in the growth calculation.", 5),
            },
            ["ticker"],
        ),
        run=get_valuation_metrics,
    ),
    ToolSpec(
        name="list_filings",
        description="List recent SEC filings with direct links to the filed documents.",
        input_schema=schema(
            {
                "ticker": string("Ticker symbol or company name."),
                "form": string('Filing type, such as "10-K", "10-Q", or "8-K".'),
                "limit": integer("Maximum filings to return, most recent first.", 5),
            },
            ["ticker"],
        ),
        run=list_filings,
    ),
    ToolSpec(
        name="read_filing",
        description=(
            "Fetch a filing from EDGAR and return it as plain text. Use this to read risk "
            "factors, management's discussion, or segment detail in the issuer's own words. "
            "Long filings are truncated, so prefer reading one relevant document over "
            "surveying many."
        ),
        input_schema=schema(
            {
                "url": string("A document URL returned by list_filings."),
                "max_chars": integer("Characters to return before truncation.", 40000),
            },
            ["url"],
        ),
        run=read_filing,
    ),
]

TOOLS_BY_NAME = {tool.name: tool for tool in ALL_TOOLS}
