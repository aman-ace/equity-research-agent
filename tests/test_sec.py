import json

import pytest
from conftest import TICKER_INDEX, build_company_facts, mock_source_client

from equity_agent import sec
from equity_agent.sources import SourceError


class TestTickerResolution:
    def test_exact_ticker_match(self, source_client):
        result = sec.resolve_ticker(source_client, "cost")
        assert result == {
            "ticker": "COST",
            "name": "COSTCO WHOLESALE CORP /NEW",
            "cik": "0000909832",
        }

    def test_name_substring_match(self, source_client):
        assert sec.resolve_ticker(source_client, "Apple")["ticker"] == "AAPL"

    def test_hyphenated_class_share_match(self, source_client):
        assert sec.resolve_ticker(source_client, "BRK-B")["ticker"] == "BRK-B"

    def test_dotted_class_share_is_accepted(self, source_client):
        # The SEC index hyphenates class shares; a dotted ticker still resolves.
        assert sec.resolve_ticker(source_client, "brk.b")["cik"] == "0001067983"

    def test_unknown_registrant_raises(self, source_client):
        with pytest.raises(SourceError, match="no SEC registrant"):
            sec.resolve_ticker(source_client, "NOTAREALTICKER")

    def test_cik_is_zero_padded(self):
        assert sec.pad_cik(320193) == "0000320193"
        assert sec.pad_cik("0000320193") == "0000320193"


class TestFundamentals:
    def test_series_is_ordered_oldest_first(self, company_facts):
        series = sec.fundamentals(company_facts)
        assert [item.fiscal_year for item in series] == [2022, 2023, 2024]

    def test_values_are_extracted_from_frames(self, company_facts):
        series = sec.fundamentals(company_facts)
        assert series[-1].revenue == 1100e9
        assert series[-1].net_income == 121e9
        assert series[-1].stockholders_equity == 800e9
        assert series[-1].diluted_eps == 6.05

    def test_years_limit_keeps_the_most_recent(self, company_facts):
        series = sec.fundamentals(company_facts, years=2)
        assert [item.fiscal_year for item in series] == [2023, 2024]

    def test_falls_back_to_ten_k_entries_without_frames(self):
        series = sec.fundamentals(build_company_facts(frames=False))
        assert [item.fiscal_year for item in series] == [2022, 2023, 2024]
        assert series[-1].revenue == 1100e9

    def test_concept_fallback_order(self, company_facts):
        # Move revenue to a lower-priority concept; extraction should still find it.
        facts = json.loads(json.dumps(company_facts))
        facts["facts"]["us-gaap"]["SalesRevenueNet"] = facts["facts"]["us-gaap"].pop("Revenues")
        assert sec.fundamentals(facts)[-1].revenue == 1100e9

    def test_a_tag_switch_mid_history_is_stitched_together(self, company_facts):
        """A filer that changed tags must not read as untagged for recent years.

        NVIDIA tags old years under RevenueFromContractWithCustomerExcludingAssessedTax
        and recent ones under Revenues. Taking the first concept with any data at
        all would return only the stale years and null out the reported ones.
        """
        facts = json.loads(json.dumps(company_facts))
        gaap = facts["facts"]["us-gaap"]
        entries = gaap["Revenues"]["units"]["USD"]
        old, recent = entries[:1], entries[1:]
        # Highest-priority concept covers 2022 only; Revenues covers 2023-2024.
        gaap["RevenueFromContractWithCustomerExcludingAssessedTax"] = {"units": {"USD": old}}
        gaap["Revenues"]["units"]["USD"] = recent

        revenues = {item.fiscal_year: item.revenue for item in sec.fundamentals(facts)}
        assert revenues == {2022: 900e9, 2023: 1000e9, 2024: 1100e9}

    def test_higher_priority_concept_wins_a_contested_year(self, company_facts):
        facts = json.loads(json.dumps(company_facts))
        gaap = facts["facts"]["us-gaap"]
        preferred = json.loads(json.dumps(gaap["Revenues"]["units"]["USD"]))
        for entry in preferred:
            entry["val"] = 1.0
        gaap["RevenueFromContractWithCustomerExcludingAssessedTax"] = {"units": {"USD": preferred}}
        assert sec.fundamentals(facts)[-1].revenue == 1.0

    def test_untagged_concepts_come_back_as_none(self, company_facts):
        facts = json.loads(json.dumps(company_facts))
        facts["facts"]["us-gaap"].pop("OperatingIncomeLoss")
        assert sec.fundamentals(facts)[-1].operating_income is None

    def test_empty_facts_raise(self):
        with pytest.raises(SourceError, match="no annual"):
            sec.fundamentals({"facts": {"us-gaap": {}}})

    def test_partial_year_durations_are_ignored(self):
        facts = {
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {
                                    "start": "2024-01-01",
                                    "end": "2024-03-31",
                                    "val": 250e9,
                                    "form": "10-K",
                                    "fp": "FY",
                                    "filed": "2025-02-01",
                                },
                                {
                                    "start": "2024-01-01",
                                    "end": "2024-12-31",
                                    "val": 1100e9,
                                    "form": "10-K",
                                    "fp": "FY",
                                    "filed": "2025-02-01",
                                },
                            ]
                        }
                    }
                }
            }
        }
        assert sec.fundamentals(facts)[-1].revenue == 1100e9


class TestFilings:
    def test_filings_are_filtered_by_form(self, source_client):
        filings = sec.recent_filings(source_client, "0000909832", forms=("10-K",))
        assert len(filings) == 1
        assert filings[0]["form"] == "10-K"

    def test_document_url_is_built_from_the_accession_number(self, source_client):
        filings = sec.recent_filings(source_client, "0000909832", forms=("10-K",))
        assert filings[0]["url"] == (
            "https://www.sec.gov/Archives/edgar/data/909832/000090983224000012/cost-20240901.htm"
        )

    def test_limit_is_respected(self, source_client):
        filings = sec.recent_filings(
            source_client, "0000909832", forms=("10-K", "10-Q", "8-K"), limit=2
        )
        assert len(filings) == 2


class TestFilingText:
    def test_markup_is_stripped(self):
        client = mock_source_client(
            {
                "sec.gov": (
                    200,
                    "<html><head><style>p{color:red}</style></head>"
                    "<body><p>Item&nbsp;1A. Risk   Factors</p>"
                    "<script>ignore()</script></body></html>",
                )
            }
        )
        text = sec.filing_text(client, "https://www.sec.gov/Archives/x.htm")
        assert "Item 1A. Risk Factors" in text
        assert "color:red" not in text
        assert "ignore()" not in text

    def test_long_filings_are_truncated(self):
        client = mock_source_client({"sec.gov": (200, "word " * 50_000)})
        text = sec.filing_text(client, "https://www.sec.gov/Archives/x.htm", max_chars=500)
        assert text.endswith("[truncated]")
        assert len(text) < 600


def test_ticker_index_fixture_is_valid_json():
    assert "COST" in json.loads(TICKER_INDEX)["0"]["ticker"]
