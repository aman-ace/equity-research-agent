from datetime import date, timedelta

import pytest
from conftest import PRICE_CHART, build_price_chart, mock_source_client

from equity_agent import market
from equity_agent.sources import SourceError

START = date(2024, 1, 1)


def series(closes: list[float]) -> list[market.Bar]:
    return [
        market.Bar(day=START + timedelta(days=i), close=close) for i, close in enumerate(closes)
    ]


class TestParsing:
    def test_bars_are_read_from_the_chart(self):
        bars = market.parse_chart(build_price_chart())
        assert len(bars) == 31
        assert bars[0].day == date(2024, 1, 1)
        assert bars[-1].close == 131.0

    def test_bars_are_sorted_oldest_first(self):
        payload = build_price_chart([3.0, 1.0])
        stamps = payload["chart"]["result"][0]["timestamp"]
        stamps.reverse()
        assert [bar.close for bar in market.parse_chart(payload)] == [1.0, 3.0]

    def test_null_closes_are_skipped(self):
        bars = market.parse_chart(build_price_chart([10.0, None, 12.0]))
        assert [bar.close for bar in bars] == [10.0, 12.0]

    def test_unparseable_closes_are_skipped(self):
        bars = market.parse_chart(build_price_chart([10.0, "notanumber", 12.0]))
        assert [bar.close for bar in bars] == [10.0, 12.0]

    def test_empty_feed_raises(self):
        with pytest.raises(SourceError, match="no usable rows"):
            market.parse_chart(build_price_chart([]))

    def test_missing_result_raises(self):
        with pytest.raises(SourceError, match="no usable rows"):
            market.parse_chart({"chart": {"result": None, "error": None}})

    def test_unexpected_payload_raises(self):
        with pytest.raises(SourceError, match="no usable rows"):
            market.parse_chart("<html>a browser check</html>")

    def test_symbol_error_is_surfaced(self):
        payload = {"chart": {"result": None, "error": {"description": "symbol may be delisted"}}}
        with pytest.raises(SourceError, match="may be delisted"):
            market.parse_chart(payload)

    def test_currency_is_read_from_the_meta_block(self):
        assert market.chart_currency(build_price_chart(currency="usd")) == "USD"
        assert market.chart_currency({"chart": {}}) is None


class TestSymbols:
    def test_class_shares_are_hyphenated(self):
        assert market.normalize_symbol("brk.b") == "BRK-B"

    def test_plain_tickers_are_upper_cased(self):
        assert market.normalize_symbol(" cost ") == "COST"


class TestStatistics:
    def test_summary_reports_the_range(self):
        summary = market.summarize(series([100.0, 120.0, 90.0, 110.0]))
        assert summary["last_close"] == 110.0
        assert summary["fifty_two_week_high"] == 120.0
        assert summary["fifty_two_week_low"] == 90.0
        assert summary["pct_below_52w_high"] == pytest.approx(8.3333, abs=1e-3)

    def test_trailing_return_needs_enough_history(self):
        summary = market.summarize(series([100.0] * 10))
        assert summary["return_1m_pct"] is None

    def test_trailing_return_is_computed_when_history_allows(self):
        summary = market.summarize(series([100.0] * 21 + [110.0]))
        assert summary["return_1m_pct"] == pytest.approx(10.0)

    def test_volatility_needs_thirty_observations(self):
        assert market.annualized_volatility(series([100.0] * 20)) is None

    def test_volatility_of_a_flat_series_is_zero(self):
        assert market.annualized_volatility(series([100.0] * 60)) == pytest.approx(0.0)

    def test_volatility_is_positive_when_prices_move(self):
        closes = [100.0 + (5 if i % 2 else 0) for i in range(60)]
        assert market.annualized_volatility(series(closes)) > 0


class TestFetch:
    def test_price_history_summarizes_and_cites(self):
        client = mock_source_client({"finance.yahoo.com": (200, PRICE_CHART)})
        result = market.price_history(client, "cost")
        assert result["ticker"] == "COST"
        assert result["last_close"] == 131.0
        assert result["currency"] == "USD"
        assert "finance.yahoo.com" in str(result["source"])
        assert len(client.citations) == 1

    def test_upstream_failure_raises(self):
        client = mock_source_client({"nothing": (200, "")})
        with pytest.raises(SourceError):
            market.price_history(client, "cost")

    def test_the_second_host_is_tried_when_the_first_fails(self):
        client = mock_source_client({"query1": (429, "slow down"), "query2": (200, PRICE_CHART)})
        result = market.price_history(client, "cost")
        assert result["last_close"] == 131.0
        assert "query2" in str(result["source"])
        # Only the host that answered is cited.
        assert len(client.citations) == 1
