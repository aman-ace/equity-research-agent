from equity_agent.valuation import (
    FundamentalSnapshot,
    analyze,
    cagr,
    growth,
    leverage,
    multiples,
    percent,
    profitability,
    ratio,
    trend,
)


def approx(value, expected, tol=1e-6):
    assert value is not None
    assert abs(value - expected) < tol


class TestPrimitives:
    def test_ratio(self):
        approx(ratio(1, 4), 0.25)

    def test_ratio_guards_zero_denominator(self):
        assert ratio(1, 0) is None

    def test_ratio_guards_missing_input(self):
        assert ratio(None, 4) is None
        assert ratio(4, None) is None

    def test_percent(self):
        approx(percent(25, 200), 12.5)

    def test_growth(self):
        approx(growth(100, 125), 25.0)

    def test_growth_is_undefined_from_a_nonpositive_base(self):
        assert growth(0, 50) is None
        assert growth(-10, 50) is None

    def test_cagr(self):
        approx(cagr(100, 121, 2), 10.0)

    def test_cagr_rejects_nonpositive_endpoints(self):
        assert cagr(-100, 121, 2) is None
        assert cagr(100, 0, 2) is None

    def test_cagr_rejects_zero_span(self):
        assert cagr(100, 121, 0) is None


def snapshot(year, **overrides):
    base = {
        "revenue": 1000.0,
        "net_income": 100.0,
        "operating_income": 150.0,
        "total_assets": 2000.0,
        "total_liabilities": 1200.0,
        "stockholders_equity": 800.0,
        "operating_cash_flow": 130.0,
        "diluted_eps": 2.0,
        "shares_diluted": 50.0,
    }
    base.update(overrides)
    return FundamentalSnapshot(fiscal_year=year, **base)


class TestMetrics:
    def test_profitability(self):
        result = profitability(snapshot(2024))
        approx(result["net_margin_pct"], 10.0)
        approx(result["operating_margin_pct"], 15.0)
        approx(result["return_on_equity_pct"], 12.5)
        approx(result["return_on_assets_pct"], 5.0)
        approx(result["cash_conversion_pct"], 130.0)

    def test_leverage(self):
        result = leverage(snapshot(2024))
        approx(result["debt_to_equity"], 1.5)
        approx(result["equity_to_assets"], 0.4)

    def test_multiples(self):
        result = multiples(20.0, snapshot(2024))
        approx(result["market_cap"], 1000.0)
        approx(result["trailing_pe"], 10.0)
        approx(result["price_to_sales"], 1.0)
        approx(result["price_to_book"], 1.25)

    def test_multiples_without_a_price(self):
        result = multiples(None, snapshot(2024))
        assert result["market_cap"] is None
        assert result["trailing_pe"] is None

    def test_untagged_fields_do_not_crash_the_metric_set(self):
        result = profitability(snapshot(2024, operating_income=None))
        assert result["operating_margin_pct"] is None
        approx(result["net_margin_pct"], 10.0)


class TestSeries:
    def test_trend(self):
        series = [snapshot(2022, revenue=100.0), snapshot(2024, revenue=121.0)]
        approx(trend(series)["revenue_cagr_pct"], 10.0)

    def test_trend_needs_two_years(self):
        assert trend([snapshot(2024)])["revenue_cagr_pct"] is None

    def test_analyze_reports_year_over_year(self):
        series = [snapshot(2023, revenue=800.0, net_income=80.0), snapshot(2024)]
        result = analyze(series, price=20.0)
        assert result["latest_fiscal_year"] == 2024
        approx(result["year_over_year"]["revenue_growth_pct"], 25.0)
        approx(result["year_over_year"]["net_income_growth_pct"], 25.0)

    def test_analyze_requires_data(self):
        try:
            analyze([])
        except ValueError:
            return
        raise AssertionError("expected ValueError on an empty series")
