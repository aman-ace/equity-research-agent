"""Deterministic financial arithmetic.

The agent is not asked to do arithmetic in its head. Every ratio, growth rate,
and margin a memo cites is computed here and handed back as a tool result, so
the numbers in the output are reproducible and unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass

Number = float | int | None


def _usable(*values: Number) -> bool:
    return all(value is not None for value in values)


def ratio(numerator: Number, denominator: Number) -> float | None:
    """Divide, returning ``None`` when the inputs are missing or degenerate."""
    if not _usable(numerator, denominator):
        return None
    assert numerator is not None and denominator is not None
    if denominator == 0:
        return None
    return numerator / denominator


def percent(numerator: Number, denominator: Number) -> float | None:
    """A ratio expressed in percent."""
    value = ratio(numerator, denominator)
    return None if value is None else value * 100.0


def growth(previous: Number, current: Number) -> float | None:
    """Period-over-period growth in percent.

    Returns ``None`` when the base period is zero or negative, where a
    percentage change is not meaningful.
    """
    if not _usable(previous, current):
        return None
    assert previous is not None and current is not None
    if previous <= 0:
        return None
    return (current - previous) / previous * 100.0


def cagr(first: Number, last: Number, years: float) -> float | None:
    """Compound annual growth rate in percent over ``years`` periods."""
    if not _usable(first, last) or years <= 0:
        return None
    assert first is not None and last is not None
    if first <= 0 or last <= 0:
        return None
    return ((last / first) ** (1.0 / years) - 1.0) * 100.0


@dataclass(frozen=True)
class FundamentalSnapshot:
    """One fiscal year of the line items a memo actually needs."""

    fiscal_year: int
    revenue: Number = None
    net_income: Number = None
    operating_income: Number = None
    total_assets: Number = None
    total_liabilities: Number = None
    stockholders_equity: Number = None
    operating_cash_flow: Number = None
    diluted_eps: Number = None
    shares_diluted: Number = None


def profitability(snapshot: FundamentalSnapshot) -> dict[str, float | None]:
    """Margin and return metrics for a single fiscal year."""
    return {
        "net_margin_pct": percent(snapshot.net_income, snapshot.revenue),
        "operating_margin_pct": percent(snapshot.operating_income, snapshot.revenue),
        "return_on_equity_pct": percent(snapshot.net_income, snapshot.stockholders_equity),
        "return_on_assets_pct": percent(snapshot.net_income, snapshot.total_assets),
        "cash_conversion_pct": percent(snapshot.operating_cash_flow, snapshot.net_income),
    }


def leverage(snapshot: FundamentalSnapshot) -> dict[str, float | None]:
    """Balance-sheet structure for a single fiscal year."""
    return {
        "debt_to_equity": ratio(snapshot.total_liabilities, snapshot.stockholders_equity),
        "equity_to_assets": ratio(snapshot.stockholders_equity, snapshot.total_assets),
    }


def multiples(price: Number, snapshot: FundamentalSnapshot) -> dict[str, float | None]:
    """Market multiples, given a share price and a fiscal year of fundamentals.

    Trailing figures are used throughout: these are last reported annual
    results, not forward estimates.
    """
    market_cap = None
    if _usable(price, snapshot.shares_diluted):
        assert price is not None and snapshot.shares_diluted is not None
        market_cap = price * snapshot.shares_diluted
    return {
        "market_cap": market_cap,
        "trailing_pe": ratio(price, snapshot.diluted_eps),
        "price_to_sales": ratio(market_cap, snapshot.revenue),
        "price_to_book": ratio(market_cap, snapshot.stockholders_equity),
    }


def trend(series: list[FundamentalSnapshot]) -> dict[str, float | None]:
    """Multi-year growth for a series ordered oldest to newest."""
    if len(series) < 2:
        return {"revenue_cagr_pct": None, "net_income_cagr_pct": None, "years": len(series)}
    first, last = series[0], series[-1]
    span = float(last.fiscal_year - first.fiscal_year)
    return {
        "revenue_cagr_pct": cagr(first.revenue, last.revenue, span),
        "net_income_cagr_pct": cagr(first.net_income, last.net_income, span),
        "years": span,
    }


def analyze(series: list[FundamentalSnapshot], price: Number = None) -> dict[str, object]:
    """Full metric set for a company: latest-year ratios plus multi-year trend.

    Args:
        series: Fiscal years ordered oldest to newest.
        price: Latest share price, if a market view is wanted.
    """
    if not series:
        raise ValueError("at least one fiscal year is required")
    latest = series[-1]
    previous = series[-2] if len(series) >= 2 else None
    return {
        "latest_fiscal_year": latest.fiscal_year,
        "profitability": profitability(latest),
        "leverage": leverage(latest),
        "multiples": multiples(price, latest),
        "trend": trend(series),
        "year_over_year": {
            "revenue_growth_pct": growth(previous.revenue, latest.revenue) if previous else None,
            "net_income_growth_pct": (
                growth(previous.net_income, latest.net_income) if previous else None
            ),
        },
    }
