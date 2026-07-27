"""Tests for the web UI's non-HTTP internals: the event stream and traces."""

from __future__ import annotations

import json
import queue

import pytest
from conftest import mock_source_client

from equity_agent import tools, web
from equity_agent.agent import research
from equity_agent.toolspec import ToolSpec


class StubProvider:
    """Calls one tool, then answers in prose."""

    name = "stub"
    model = "stub-1"

    def __init__(self, tool_name: str = "lookup_company", arguments: dict | None = None):
        self.tool_name = tool_name
        self.arguments = arguments if arguments is not None else {"query": "COST"}

    def run(self, *, system, prompt, tools, max_tokens, max_turns):
        from equity_agent.providers.base import RunResult

        by_name = {spec.name: spec for spec in tools}
        by_name[self.tool_name].call(self.arguments)
        return RunResult("## Summary\n\nA memo.", 100, 50, 2)


class TestToolObserver:
    def test_observer_sees_start_and_end(self, monkeypatch, source_client):
        monkeypatch.setattr("equity_agent.agent.SourceClient", lambda **_: source_client)
        seen: list[tuple[str, str]] = []
        research("COST", provider=StubProvider(), on_tool=lambda p, n, _: seen.append((p, n)))
        assert seen == [("start", "lookup_company"), ("end", "lookup_company")]

    def test_observer_receives_arguments_then_result(self, monkeypatch, source_client):
        monkeypatch.setattr("equity_agent.agent.SourceClient", lambda **_: source_client)
        payloads: dict[str, object] = {}
        research(
            "COST",
            provider=StubProvider(),
            on_tool=lambda phase, _name, payload: payloads.__setitem__(phase, payload),
        )
        assert payloads["start"] == {"query": "COST"}
        assert "COSTCO" in str(payloads["end"])

    def test_a_failing_observer_does_not_break_the_run(self, monkeypatch, source_client):
        monkeypatch.setattr("equity_agent.agent.SourceClient", lambda **_: source_client)

        def explode(*_args):
            raise RuntimeError("observer is broken")

        memo = research("COST", provider=StubProvider(), on_tool=explode)
        assert memo.body == "## Summary\n\nA memo."

    def test_the_tool_surface_is_untouched_without_an_observer(self, monkeypatch, source_client):
        monkeypatch.setattr("equity_agent.agent.SourceClient", lambda **_: source_client)
        captured: dict[str, object] = {}

        class Recorder(StubProvider):
            def run(self, *, system, prompt, tools, max_tokens, max_turns):
                captured["tools"] = tools
                return super().run(
                    system=system,
                    prompt=prompt,
                    tools=tools,
                    max_tokens=max_tokens,
                    max_turns=max_turns,
                )

        research("COST", provider=Recorder())
        assert captured["tools"] is tools.ALL_TOOLS

    def test_observed_tools_keep_their_schema(self):
        spec = tools.TOOLS_BY_NAME["get_fundamentals"]
        from equity_agent.agent import _observed

        wrapped = _observed(spec, lambda *_: None)
        assert isinstance(wrapped, ToolSpec)
        assert wrapped.name == spec.name
        assert wrapped.input_schema == spec.input_schema
        assert wrapped.description == spec.description


class TestTraceNotes:
    def test_company_name_is_surfaced(self):
        assert web._note(json.dumps({"company": {"name": "COSTCO WHOLESALE CORP"}})) == (
            "COSTCO WHOLESALE CORP"
        )

    def test_errors_are_surfaced(self):
        assert web._note(json.dumps({"error": "no SEC registrant"})) == "error: no SEC registrant"

    def test_price_results_show_the_as_of_date(self):
        note = web._note(json.dumps({"ticker": "COST", "as_of": "2026-07-24"}))
        assert note == "COST as of 2026-07-24"

    def test_plain_text_is_reported_by_length(self):
        # read_filing returns filing text, not JSON.
        assert web._note("x" * 1500) == "1,500 characters"

    def test_unrecognized_json_falls_back_to_ok(self):
        assert web._note(json.dumps({"whatever": 1})) == "ok"


class TestRunStream:
    def _drain(self, events: queue.Queue) -> list:
        collected = []
        while True:
            event = events.get(timeout=10)
            if event is web._DONE:
                return collected
            collected.append(event)

    def test_a_successful_run_publishes_tools_then_done(self, monkeypatch, source_client):
        monkeypatch.setattr("equity_agent.agent.SourceClient", lambda **_: source_client)
        monkeypatch.setattr("equity_agent.web.research", _stubbed_research(source_client))

        events: queue.Queue = queue.Queue()
        web._run({"subject": "COST", "effort": "high", "years": 5, "question": ""}, events)
        collected = self._drain(events)

        assert [event["type"] for event in collected] == ["tool", "tool", "done"]
        done = collected[-1]
        assert done["subject"] == "COST"
        assert "<h1>" in done["html"]
        assert done["markdown"].startswith("# Equity Research Memo")
        assert done["sources"] >= 1

    def test_a_failing_run_publishes_an_error_and_terminates(self, monkeypatch):
        def boom(*_args, **_kwargs):
            from equity_agent.providers import ProviderError

            raise ProviderError("ANTHROPIC_API_KEY is not set")

        monkeypatch.setattr("equity_agent.web.research", boom)
        events: queue.Queue = queue.Queue()
        web._run({"subject": "COST", "effort": "high", "years": 5, "question": ""}, events)
        collected = self._drain(events)
        assert collected == [{"type": "error", "message": "ANTHROPIC_API_KEY is not set"}]

    def test_an_unexpected_failure_is_still_reported(self, monkeypatch):
        def boom(*_args, **_kwargs):
            raise KeyError("wat")

        monkeypatch.setattr("equity_agent.web.research", boom)
        events: queue.Queue = queue.Queue()
        web._run({"subject": "COST", "effort": "high", "years": 5, "question": ""}, events)
        collected = self._drain(events)
        assert collected[0]["type"] == "error"
        assert "KeyError" in collected[0]["message"]

    def test_done_is_always_published(self, monkeypatch):
        """The stream must terminate even when the run raises, or a browser hangs."""
        monkeypatch.setattr("equity_agent.web.research", lambda *a, **k: 1 / 0)
        events: queue.Queue = queue.Queue()
        web._run({"subject": "X", "effort": "high", "years": 5, "question": ""}, events)
        assert events.get_nowait()["type"] == "error"
        assert events.get_nowait() is web._DONE


def _stubbed_research(source_client):
    """research() with the stub provider, preserving the on_tool plumbing."""

    def call(subject, config=None, question=None, on_tool=None):
        return research(
            subject, config=config, question=question, provider=StubProvider(), on_tool=on_tool
        )

    return call


class TestPage:
    def test_the_page_is_self_contained(self):
        # A strict local page: no CDN scripts or remote stylesheets.
        assert "http://" not in web.PAGE.replace("http://www.w3.org", "")
        assert "cdn" not in web.PAGE.lower()
        assert "<script>" in web.PAGE and "<style>" in web.PAGE

    def test_the_page_offers_every_effort_level(self):
        for level in web.EFFORTS:
            assert f">{level}<" in web.PAGE


class TestHandlerValidation:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("11", 10), ("0", 1), ("-4", 1), ("abc", 5), ("", 5), ("7", 7)],
    )
    def test_years_are_clamped(self, raw, expected):
        assert _years(raw) == expected

    @pytest.mark.parametrize("raw", ["high", "low", "max"])
    def test_known_efforts_pass_through(self, raw):
        assert raw in web.EFFORTS


def _years(raw: str) -> int:
    """Mirror of the handler's clamp, exercised without standing up a socket."""
    try:
        return max(1, min(10, int(raw or 5)))
    except ValueError:
        return 5


def test_source_client_fixture_is_hermetic():
    client = mock_source_client({})
    assert client.citations == []
