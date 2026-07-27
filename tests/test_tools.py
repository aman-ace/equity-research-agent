import json

import pytest
from conftest import PRICE_CHART, SUBMISSIONS, TICKER_INDEX, build_company_facts, mock_source_client

from equity_agent import tools


@pytest.fixture(autouse=True)
def configured(source_client):
    """Bind the tool module to a mocked context for every test here."""
    context = tools.ResearchContext(client=source_client)
    tools.configure(context)
    return context


def spec(name: str):
    return tools.TOOLS_BY_NAME[name]


def payload(name: str, **kwargs):
    return json.loads(spec(name).call(kwargs))


class TestSchemas:
    def test_every_tool_declares_a_name_and_description(self):
        for tool in tools.ALL_TOOLS:
            assert tool.name
            assert tool.description
            assert tool.input_schema["type"] == "object"

    def test_only_genuinely_required_arguments_are_required(self):
        assert spec("lookup_company").input_schema["required"] == ["query"]
        assert spec("get_fundamentals").input_schema["required"] == ["ticker"]
        assert "years" not in spec("get_fundamentals").input_schema["required"]

    def test_every_property_is_documented(self):
        for tool in tools.ALL_TOOLS:
            for name, prop in tool.input_schema["properties"].items():
                assert prop.get("description"), f"{tool.name}.{name} has no description"

    def test_schema_properties_match_the_callable(self):
        import inspect

        for tool in tools.ALL_TOOLS:
            parameters = set(inspect.signature(tool.run).parameters)
            assert set(tool.input_schema["properties"]) <= parameters, tool.name

    def test_unexpected_arguments_are_reported_not_raised(self):
        result = json.loads(spec("lookup_company").call({"nonsense": 1}))
        assert "invalid arguments" in result["error"]


class TestLookup:
    def test_resolves_a_ticker(self):
        assert payload("lookup_company", query="COST")["cik"] == "0000909832"

    def test_unknown_ticker_returns_an_error_rather_than_raising(self):
        assert "error" in payload("lookup_company", query="ZZZZ")


class TestFundamentals:
    def test_returns_a_year_series(self):
        result = payload("get_fundamentals", ticker="COST")
        assert [row["fiscal_year"] for row in result["fiscal_years"]] == [2022, 2023, 2024]
        assert result["fiscal_years"][-1]["revenue"] == 1100e9

    def test_years_argument_narrows_the_series(self):
        result = payload("get_fundamentals", ticker="COST", years=2)
        assert len(result["fiscal_years"]) == 2

    def test_company_facts_are_fetched_once_per_run(self, configured, source_client):
        payload("get_fundamentals", ticker="COST")
        payload("get_fundamentals", ticker="COST")
        facts_calls = [c for c in source_client.citations if "companyfacts" in c.url]
        assert len(facts_calls) == 1


class TestValuation:
    def test_metrics_are_computed_with_a_price(self):
        result = payload("get_valuation_metrics", ticker="COST")
        assert result["metrics"]["profitability"]["net_margin_pct"] == pytest.approx(11.0)
        assert result["metrics"]["multiples"]["trailing_pe"] is not None
        assert "last close" in result["price_basis"]

    def test_multiples_are_omitted_when_the_price_feed_fails(self):
        client = mock_source_client(
            {
                "company_tickers.json": (200, TICKER_INDEX),
                "companyfacts": (200, json.dumps(build_company_facts())),
            }
        )
        tools.configure(tools.ResearchContext(client=client))
        result = payload("get_valuation_metrics", ticker="COST")
        assert result["price_basis"].startswith("share price unavailable")
        assert result["metrics"]["multiples"]["trailing_pe"] is None
        assert result["metrics"]["profitability"]["net_margin_pct"] is not None

    def test_unknown_ticker_returns_an_error(self):
        assert "error" in payload("get_valuation_metrics", ticker="ZZZZ")


class TestPrices:
    def test_price_history_is_summarized(self):
        result = payload("get_price_history", ticker="COST")
        assert result["last_close"] == 131.0


class TestFilings:
    def test_filings_are_listed_with_links(self):
        result = payload("list_filings", ticker="COST", form="10-K")
        assert result["filings"][0]["url"].startswith("https://www.sec.gov/Archives/")

    def test_reading_a_filing_returns_text(self):
        text = spec("read_filing").call(
            {"url": "https://www.sec.gov/Archives/edgar/data/909832/x.htm"}
        )
        assert "Risk Factors" in text

    def test_non_sec_urls_are_refused(self):
        result = json.loads(spec("read_filing").call({"url": "https://example.com/evil.htm"}))
        assert "error" in result


def test_submissions_fixture_is_valid_json():
    assert json.loads(SUBMISSIONS)["filings"]["recent"]["form"][0] == "10-K"


def test_price_fixture_is_parseable():
    assert json.loads(PRICE_CHART)["chart"]["result"][0]["meta"]["currency"] == "USD"
