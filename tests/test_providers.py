"""Provider loops, driven by stub clients.

No network and no API key: each stub returns a scripted sequence of responses so
the loop's behaviour — tool dispatch, turn limits, refusals, token accounting —
is exercised without calling a model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from equity_agent import providers
from equity_agent.providers import ProviderError, RefusalError
from equity_agent.providers.claude import ClaudeProvider
from equity_agent.providers.gemini import GeminiProvider
from equity_agent.toolspec import ToolSpec, schema, string

CALLS: list[tuple[str, dict]] = []


def echo(ticker: str) -> str:
    CALLS.append(("echo", {"ticker": ticker}))
    return json.dumps({"ticker": ticker.upper()})


def explode() -> str:
    raise RuntimeError("upstream is down")


TOOLS = [
    ToolSpec(
        name="echo",
        description="Echo a ticker.",
        input_schema=schema({"ticker": string("A ticker.")}, ["ticker"]),
        run=echo,
    ),
    ToolSpec(
        name="explode",
        description="Always fails.",
        input_schema=schema({}, []),
        run=explode,
    ),
]


@pytest.fixture(autouse=True)
def clear_calls():
    CALLS.clear()


# --------------------------------------------------------------------------
# Claude stubs
# --------------------------------------------------------------------------


@dataclass
class Block:
    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class Usage:
    input_tokens: int = 10
    output_tokens: int = 5


@dataclass
class ClaudeResponse:
    content: list[Block]
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)


class FakeClaudeClient:
    def __init__(self, script: list[ClaudeResponse]):
        self.script = list(script)
        self.requests: list[dict] = []
        self.messages = self

    def create(self, **kwargs: Any) -> ClaudeResponse:
        # The provider reuses one messages list across turns, so snapshot it.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.script.pop(0)


def claude_with(script: list[ClaudeResponse]) -> tuple[ClaudeProvider, FakeClaudeClient]:
    client = FakeClaudeClient(script)
    return ClaudeProvider(client=client), client


def run(provider, max_turns: int = 5):
    return provider.run(
        system="be an analyst",
        prompt="research COST",
        tools=TOOLS,
        max_tokens=1000,
        max_turns=max_turns,
    )


class TestClaudeLoop:
    def test_plain_answer_returns_text(self):
        provider, _ = claude_with([ClaudeResponse([Block("text", text="the memo")])])
        result = run(provider)
        assert result.text == "the memo"
        assert result.turns == 1

    def test_tool_call_is_executed_and_fed_back(self):
        provider, client = claude_with(
            [
                ClaudeResponse(
                    [Block("tool_use", id="t1", name="echo", input={"ticker": "cost"})],
                    stop_reason="tool_use",
                ),
                ClaudeResponse([Block("text", text="done")]),
            ]
        )
        result = run(provider)
        assert CALLS == [("echo", {"ticker": "cost"})]
        assert result.text == "done"
        # Second request carries the assistant turn plus the tool result.
        follow_up = client.requests[1]["messages"]
        assert follow_up[-1]["content"][0]["tool_use_id"] == "t1"
        assert "COST" in follow_up[-1]["content"][0]["content"]

    def test_tokens_accumulate_across_turns(self):
        provider, _ = claude_with(
            [
                ClaudeResponse(
                    [Block("tool_use", id="t1", name="echo", input={"ticker": "x"})],
                    stop_reason="tool_use",
                ),
                ClaudeResponse([Block("text", text="done")]),
            ]
        )
        result = run(provider)
        assert (result.input_tokens, result.output_tokens) == (20, 10)

    def test_refusal_raises(self):
        provider, _ = claude_with([ClaudeResponse([], stop_reason="refusal")])
        with pytest.raises(RefusalError):
            run(provider)

    def test_pause_turn_resumes(self):
        provider, _ = claude_with(
            [
                ClaudeResponse([Block("text", text="partial")], stop_reason="pause_turn"),
                ClaudeResponse([Block("text", text="complete")]),
            ]
        )
        assert run(provider).text == "complete"

    def test_turn_limit_stops_the_loop(self):
        script = [
            ClaudeResponse(
                [Block("tool_use", id=f"t{i}", name="echo", input={"ticker": "x"})],
                stop_reason="tool_use",
            )
            for i in range(6)
        ]
        provider, _ = claude_with(script)
        result = run(provider, max_turns=3)
        assert result.text == ""
        assert result.turns == 3

    def test_unknown_tool_returns_an_error_result(self):
        provider, client = claude_with(
            [
                ClaudeResponse(
                    [Block("tool_use", id="t1", name="nope", input={})],
                    stop_reason="tool_use",
                ),
                ClaudeResponse([Block("text", text="ok")]),
            ]
        )
        run(provider)
        assert "unknown tool nope" in client.requests[1]["messages"][-1]["content"][0]["content"]

    def test_tool_exceptions_are_reported_not_raised(self):
        provider, client = claude_with(
            [
                ClaudeResponse(
                    [Block("tool_use", id="t1", name="explode", input={})],
                    stop_reason="tool_use",
                ),
                ClaudeResponse([Block("text", text="recovered")]),
            ]
        )
        assert run(provider).text == "recovered"
        payload = client.requests[1]["messages"][-1]["content"][0]["content"]
        assert "upstream is down" in payload

    def test_request_carries_tool_schemas_and_effort(self):
        provider, client = claude_with([ClaudeResponse([Block("text", text="x")])])
        run(provider)
        request = client.requests[0]
        assert request["output_config"] == {"effort": "high"}
        assert [tool["name"] for tool in request["tools"]] == ["echo", "explode"]
        assert request["tools"][0]["input_schema"]["required"] == ["ticker"]

    def test_invalid_effort_is_rejected(self):
        with pytest.raises(ValueError, match="effort must be one of"):
            ClaudeProvider(client=FakeClaudeClient([]), effort="turbo")


# --------------------------------------------------------------------------
# Gemini stubs
# --------------------------------------------------------------------------


@dataclass
class GeminiUsage:
    prompt_token_count: int = 10
    candidates_token_count: int = 5


@dataclass
class FunctionCall:
    name: str
    args: dict


@dataclass
class Part:
    text: str | None = None


@dataclass
class Content:
    parts: list[Part]
    role: str = "model"


@dataclass
class Candidate:
    content: Content | None
    finish_reason: str = "STOP"


@dataclass
class GeminiResponse:
    candidates: list[Candidate]
    function_calls: list[FunctionCall] = field(default_factory=list)
    usage_metadata: GeminiUsage = field(default_factory=GeminiUsage)


def text_response(text: str) -> GeminiResponse:
    return GeminiResponse([Candidate(Content([Part(text=text)]))])


def call_response(name: str, args: dict) -> GeminiResponse:
    return GeminiResponse([Candidate(Content([]))], function_calls=[FunctionCall(name, args)])


class FakeGeminiModels:
    def __init__(self, script: list[GeminiResponse]):
        self.script = list(script)
        self.requests: list[dict] = []

    def generate_content(self, **kwargs: Any) -> GeminiResponse:
        # The provider reuses one contents list across turns, so snapshot it.
        self.requests.append({**kwargs, "contents": list(kwargs["contents"])})
        return self.script.pop(0)


class FakeGeminiClient:
    def __init__(self, script: list[GeminiResponse]):
        self.models = FakeGeminiModels(script)


def gemini_with(script: list[GeminiResponse]) -> tuple[GeminiProvider, FakeGeminiClient]:
    client = FakeGeminiClient(script)
    return GeminiProvider(client=client), client


class TestGeminiLoop:
    def test_plain_answer_returns_text(self):
        provider, _ = gemini_with([text_response("the memo")])
        result = run(provider)
        assert result.text == "the memo"
        assert result.turns == 1

    def test_function_call_is_executed_and_fed_back(self):
        provider, client = gemini_with(
            [call_response("echo", {"ticker": "cost"}), text_response("done")]
        )
        result = run(provider)
        assert CALLS == [("echo", {"ticker": "cost"})]
        assert result.text == "done"
        follow_up = client.models.requests[1]["contents"][-1]
        assert follow_up.role == "user"
        assert "COST" in follow_up.parts[0].function_response.response["result"]

    def test_tokens_accumulate_across_turns(self):
        provider, _ = gemini_with([call_response("echo", {"ticker": "x"}), text_response("done")])
        result = run(provider)
        assert (result.input_tokens, result.output_tokens) == (20, 10)

    def test_safety_finish_reason_raises(self):
        provider, _ = gemini_with([GeminiResponse([Candidate(None, finish_reason="SAFETY")])])
        with pytest.raises(RefusalError, match="declined"):
            run(provider)

    def test_turn_limit_stops_the_loop(self):
        provider, _ = gemini_with([call_response("echo", {"ticker": "x"}) for _ in range(6)])
        result = run(provider, max_turns=3)
        assert result.text == ""
        assert result.turns == 3

    def test_unknown_tool_returns_an_error_result(self):
        provider, client = gemini_with([call_response("nope", {}), text_response("ok")])
        run(provider)
        payload = client.models.requests[1]["contents"][-1].parts[0]
        assert "unknown tool nope" in payload.function_response.response["result"]

    def test_tool_exceptions_are_reported_not_raised(self):
        provider, client = gemini_with([call_response("explode", {}), text_response("recovered")])
        assert run(provider).text == "recovered"
        payload = client.models.requests[1]["contents"][-1].parts[0]
        assert "upstream is down" in payload.function_response.response["result"]

    def test_request_declares_the_tools(self):
        provider, client = gemini_with([text_response("x")])
        run(provider)
        config = client.models.requests[0]["config"]
        declared = config.tools[0].function_declarations
        assert [d.name for d in declared] == ["echo", "explode"]
        assert config.automatic_function_calling.disable is True
        assert config.system_instruction == "be an analyst"

    def test_multiple_text_parts_are_joined(self):
        response = GeminiResponse([Candidate(Content([Part(text="one"), Part(text="two")]))])
        provider, _ = gemini_with([response])
        assert run(provider).text == "one\n\ntwo"


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


class TestFactory:
    def test_unknown_provider_is_rejected(self):
        with pytest.raises(ProviderError, match="unknown provider"):
            providers.build("llama")

    def test_default_models_are_defined(self):
        assert providers.default_model("claude").startswith("claude")
        assert providers.default_model("gemini").startswith("gemini")

    def test_default_provider_follows_the_available_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        assert providers.default_provider() == "gemini"

    def test_claude_wins_when_both_keys_are_set(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "y")
        assert providers.default_provider() == "claude"

    def test_missing_key_is_a_provider_error(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
            providers.build("claude")


class TestApiErrors:
    """SDK exceptions become a clean ProviderError rather than a traceback."""

    class Boom:
        def __init__(self):
            self.messages = self
            self.models = self

        def create(self, **_):
            raise RuntimeError("400 INVALID_ARGUMENT\n{'error': {'code': 400}}")

        def generate_content(self, **_):
            raise RuntimeError("400 INVALID_ARGUMENT\n{'error': {'code': 400}}")

    def test_claude_api_error_is_wrapped(self):
        provider = ClaudeProvider(client=self.Boom())
        with pytest.raises(ProviderError, match="Claude request failed"):
            run(provider)

    def test_gemini_api_error_is_wrapped(self):
        provider = GeminiProvider(client=self.Boom())
        with pytest.raises(ProviderError, match="Gemini request failed"):
            run(provider)

    def test_only_the_first_line_is_surfaced(self):
        provider = ClaudeProvider(client=self.Boom())
        with pytest.raises(ProviderError) as caught:
            run(provider)
        assert "\n" not in str(caught.value)
