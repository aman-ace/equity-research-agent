import json

import pytest
from conftest import PRICE_CHART, SUBMISSIONS, TICKER_INDEX, build_company_facts, mock_source_client

from equity_agent import providers, tools
from equity_agent.agent import SYSTEM_PROMPT, AgentConfig, build_prompt, research
from equity_agent.providers.base import RunResult


class TestConfig:
    def test_defaults(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        config = AgentConfig()
        assert config.provider == "claude"
        assert config.model == providers.default_model("claude")
        assert config.effort == "high"
        assert config.years == 5

    def test_model_defaults_to_the_provider(self):
        assert AgentConfig(provider="gemini").model == providers.default_model("gemini")

    def test_explicit_model_is_kept(self):
        assert AgentConfig(provider="gemini", model="gemini-3-custom").model == "gemini-3-custom"

    def test_invalid_provider_is_rejected(self):
        with pytest.raises(ValueError, match="provider must be one of"):
            AgentConfig(provider="llama")

    def test_invalid_effort_is_rejected(self):
        with pytest.raises(ValueError, match="effort must be one of"):
            AgentConfig(effort="turbo")

    @pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
    def test_supported_effort_levels(self, effort):
        assert AgentConfig(effort=effort).effort == effort

    def test_user_agent_comes_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("SEC_USER_AGENT", "Aman Chandaliya a@example.com")
        assert AgentConfig().user_agent == "Aman Chandaliya a@example.com"


class TestPrompt:
    def test_subject_is_included(self):
        assert "COST" in build_prompt("COST", None)

    def test_specific_question_is_appended(self):
        assert "membership income" in build_prompt("COST", "How durable is membership income?")

    def test_system_prompt_forbids_unsourced_figures(self):
        assert "must come from a tool result" in SYSTEM_PROMPT

    def test_system_prompt_delegates_arithmetic_to_the_tools(self):
        assert "Do not do arithmetic yourself" in SYSTEM_PROMPT

    def test_system_prompt_reserves_sources_and_disclaimer_for_the_renderer(self):
        assert "Do not add a sources section or a disclaimer" in SYSTEM_PROMPT

    def test_every_required_section_is_specified(self):
        for section in [
            "## Summary",
            "## Business",
            "## Financial performance",
            "## Valuation",
            "## Risks",
            "## What would change this view",
        ]:
            assert section in SYSTEM_PROMPT


class StubProvider:
    """A backend that runs the real tools, then returns a fixed memo body."""

    name = "stub"
    model = "stub-1"

    def __init__(self, body: str = "## Summary\n\nA memo.", use_tools: bool = True):
        self.body = body
        self.use_tools = use_tools
        self.seen: dict = {}

    def run(self, *, system, prompt, tools, max_tokens, max_turns) -> RunResult:
        self.seen = {"system": system, "prompt": prompt, "tools": [t.name for t in tools]}
        if self.use_tools:
            by_name = {tool.name: tool for tool in tools}
            by_name["get_fundamentals"].call({"ticker": "COST"})
            by_name["get_price_history"].call({"ticker": "COST"})
        return RunResult(self.body, input_tokens=100, output_tokens=50, turns=2)


class TestResearch:
    def test_memo_is_assembled_from_the_provider_result(self, source_client):
        tools.configure(tools.ResearchContext(client=source_client))
        provider = StubProvider(use_tools=False)
        memo = research("cost", provider=provider)
        assert memo.subject == "COST"
        assert memo.body == "## Summary\n\nA memo."
        assert memo.provider == "stub"
        assert memo.model == "stub-1"
        assert (memo.input_tokens, memo.output_tokens) == (100, 50)

    def test_citations_come_from_what_the_tools_fetched(self, monkeypatch, source_client):
        # research() builds its own SourceClient; point it at the mocked one.
        monkeypatch.setattr("equity_agent.agent.SourceClient", lambda **_: source_client)
        memo = research("COST", provider=StubProvider())
        urls = [citation.url for citation in memo.citations]
        assert any("companyfacts" in url for url in urls)
        assert any("finance.yahoo.com" in url for url in urls)

    def test_the_full_tool_surface_is_offered(self, monkeypatch, source_client):
        monkeypatch.setattr("equity_agent.agent.SourceClient", lambda **_: source_client)
        provider = StubProvider(use_tools=False)
        research("COST", provider=provider)
        assert provider.seen["tools"] == [tool.name for tool in tools.ALL_TOOLS]

    def test_empty_output_produces_a_visible_placeholder(self, monkeypatch, source_client):
        monkeypatch.setattr("equity_agent.agent.SourceClient", lambda **_: source_client)
        memo = research("COST", provider=StubProvider(body="   ", use_tools=False))
        assert "stopped before writing a memo" in memo.body

    def test_question_reaches_the_prompt(self, monkeypatch, source_client):
        monkeypatch.setattr("equity_agent.agent.SourceClient", lambda **_: source_client)
        provider = StubProvider(use_tools=False)
        research("COST", question="Is the membership model durable?", provider=provider)
        assert "membership model" in provider.seen["prompt"]


def test_tool_surface_is_registered():
    assert {tool.name for tool in tools.ALL_TOOLS} == {
        "lookup_company",
        "get_fundamentals",
        "get_price_history",
        "get_valuation_metrics",
        "list_filings",
        "read_filing",
    }


def test_fixtures_are_well_formed():
    assert json.loads(SUBMISSIONS)["filings"]["recent"]["form"][0] == "10-K"
    assert json.loads(TICKER_INDEX)["0"]["ticker"] == "COST"
    assert json.loads(PRICE_CHART)["chart"]["result"][0]["timestamp"]
    assert build_company_facts()["cik"] == 909832
    assert mock_source_client({}) is not None
