"""Daily price history from the Yahoo Finance chart endpoint.

Stooq, which this module used previously, now gates its CSV download behind a
JavaScript proof-of-work challenge: scripted requests get an HTML page under an
HTTP 200 rather than a feed, so every price lookup failed and the memo lost its
valuation multiples. Yahoo's chart endpoint serves the same end-of-day series as
JSON with no key and no registration, which keeps the project runnable by anyone
who clones it. Prices are end-of-day, not live.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from .sources import SourceClient, SourceError

PRICE_URL = "https://{host}/v8/finance/chart/{symbol}?range={span}&interval=1d"
# The same endpoint is served from two hosts; one occasionally rate-limits.
PRICE_HOSTS = ("query1.finance.yahoo.com", "query2.finance.yahoo.com")
# Two years, so a trailing twelve-month return still has a full year behind it.
PRICE_SPAN = "2y"
TRADING_DAYS = 252


@dataclass(frozen=True)
class Bar:
    """One end-of-day price bar."""

    day: date
    close: float


def normalize_symbol(ticker: str) -> str:
    """Map a ticker to the form the price feed expects.

    Class shares are hyphenated (``BRK-B``), matching the SEC's own ticker
    index; a dotted ``BRK.B`` typed on the command line is accepted too.
    """
    return ticker.strip().upper().replace(".", "-")


def parse_chart(payload: Any) -> list[Bar]:
    """Parse a Yahoo chart document into bars ordered oldest first.

    Bars the feed leaves null — halted sessions, and the occasional gap — are
    skipped rather than treated as zero.
    """
    chart = payload.get("chart") if isinstance(payload, dict) else None
    if not isinstance(chart, dict):
        raise SourceError("price feed returned no usable rows")
    error = chart.get("error")
    if error:
        detail = error.get("description") if isinstance(error, dict) else error
        raise SourceError(f"price feed rejected the symbol: {detail}")

    results = chart.get("result") or []
    if not results:
        raise SourceError("price feed returned no usable rows")
    result = results[0]
    stamps = result.get("timestamp") or []
    quotes = result.get("indicators", {}).get("quote") or [{}]
    closes = quotes[0].get("close") or []

    bars: list[Bar] = []
    # strict=False: if the feed ever returns mismatched column lengths, take the
    # bars that do line up rather than raising.
    for stamp, close in zip(stamps, closes, strict=False):
        if stamp is None or close is None:
            continue
        try:
            day = datetime.fromtimestamp(int(stamp), tz=timezone.utc).date()
            bars.append(Bar(day=day, close=float(close)))
        except (OSError, OverflowError, TypeError, ValueError):
            continue
    if not bars:
        raise SourceError("price feed returned no usable rows")
    return sorted(bars, key=lambda bar: bar.day)


def _return_over(bars: list[Bar], sessions: int) -> float | None:
    if len(bars) <= sessions:
        return None
    start, end = bars[-1 - sessions].close, bars[-1].close
    if start <= 0:
        return None
    return (end - start) / start * 100.0


def annualized_volatility(bars: list[Bar], sessions: int = TRADING_DAYS) -> float | None:
    """Annualized standard deviation of daily returns, in percent."""
    window = bars[-(sessions + 1) :]
    if len(window) < 30:
        return None
    daily = [
        (window[i].close - window[i - 1].close) / window[i - 1].close
        for i in range(1, len(window))
        if window[i - 1].close > 0
    ]
    if len(daily) < 30:
        return None
    mean = sum(daily) / len(daily)
    variance = sum((value - mean) ** 2 for value in daily) / (len(daily) - 1)
    return math.sqrt(variance) * math.sqrt(TRADING_DAYS) * 100.0


def summarize(bars: list[Bar]) -> dict[str, object]:
    """Reduce a price series to the handful of figures a memo cites."""
    window = bars[-TRADING_DAYS:]
    closes = [bar.close for bar in window]
    latest = bars[-1]
    high, low = max(closes), min(closes)
    return {
        "as_of": latest.day.isoformat(),
        "last_close": latest.close,
        "fifty_two_week_high": high,
        "fifty_two_week_low": low,
        "pct_below_52w_high": (high - latest.close) / high * 100.0 if high > 0 else None,
        "return_1m_pct": _return_over(bars, 21),
        "return_3m_pct": _return_over(bars, 63),
        "return_12m_pct": _return_over(bars, TRADING_DAYS),
        "annualized_volatility_pct": annualized_volatility(bars),
        "sessions_available": len(bars),
    }


def price_history(client: SourceClient, ticker: str) -> dict[str, object]:
    """Fetch and summarize end-of-day prices for a US-listed ticker."""
    symbol = normalize_symbol(ticker)
    label = f"Yahoo Finance end-of-day price history for {symbol}"
    last_error: SourceError | None = None

    for host in PRICE_HOSTS:
        url = PRICE_URL.format(host=host, symbol=symbol, span=PRICE_SPAN)
        try:
            payload = client.get_json(url, label)
            summary = summarize(parse_chart(payload))
        except SourceError as exc:
            last_error = exc
            continue
        summary["ticker"] = symbol
        summary["currency"] = chart_currency(payload)
        summary["source"] = url
        return summary

    raise last_error or SourceError(f"could not retrieve prices for {symbol}")


def chart_currency(payload: Any) -> str | None:
    """The currency the feed quoted in, so a multiple is not silently cross-currency."""
    try:
        currency = payload["chart"]["result"][0]["meta"]["currency"]
    except (KeyError, IndexError, TypeError):
        return None
    return str(currency).upper() if currency else None
