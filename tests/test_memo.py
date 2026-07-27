from equity_agent.memo import DISCLAIMER, Memo
from equity_agent.sources import Citation


def build(
    citations=None,
    body="## Summary\n\nCostco sells things in bulk.",
    provider="claude",
    model="claude-opus-5",
) -> Memo:
    return Memo(
        subject="COST",
        body=body,
        citations=citations if citations is not None else [],
        provider=provider,
        model=model,
        effort="high",
        input_tokens=1200,
        output_tokens=900,
    )


class TestRendering:
    def test_header_names_the_subject_and_run_settings(self):
        rendered = build().to_markdown()
        assert rendered.startswith("# Equity Research Memo — COST")
        assert "`claude-opus-5`" in rendered
        assert "`high`" in rendered

    def test_header_records_which_provider_produced_it(self):
        rendered = build(provider="gemini", model="gemini-2.5-pro").to_markdown()
        assert "gemini `gemini-2.5-pro`" in rendered

    def test_body_is_included(self):
        assert "Costco sells things in bulk." in build().to_markdown()

    def test_sources_are_numbered(self):
        citations = [
            Citation("SEC EDGAR company facts", "https://sec.gov/a", "2026-07-27"),
            Citation("Yahoo Finance price history", "https://finance.yahoo.com/b", "2026-07-27"),
        ]
        rendered = build(citations).to_markdown()
        assert "1. SEC EDGAR company facts" in rendered
        assert "2. Yahoo Finance price history" in rendered

    def test_no_sources_is_stated_explicitly(self):
        assert "No external sources were retrieved" in build([]).to_markdown()

    def test_disclaimer_is_always_appended(self):
        assert DISCLAIMER in build().to_markdown()

    def test_sections_appear_in_order(self):
        rendered = build([Citation("SEC", "https://sec.gov/a", "2026-07-27")]).to_markdown()
        assert rendered.index("## Sources") < rendered.index("## Disclaimer")
        assert rendered.index("Costco sells things") < rendered.index("## Sources")

    def test_token_accounting_is_carried(self):
        memo = build()
        assert (memo.input_tokens, memo.output_tokens) == (1200, 900)
