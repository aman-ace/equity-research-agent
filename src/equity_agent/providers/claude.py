"""Claude backend, built on the Anthropic Messages API.

The loop is written out rather than delegated to the SDK's tool runner so that
both providers share one control flow: same turn limit, same tool dispatch, same
accounting. That makes a Claude run and a Gemini run comparable.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..toolspec import ToolSpec
from .base import ProviderError, RefusalError, RunResult

DEFAULT_MODEL = "claude-opus-5"
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


@dataclass
class ClaudeProvider:
    """Runs the agent on Anthropic's API."""

    model: str = DEFAULT_MODEL
    effort: str = "high"
    client: Any = None
    name: str = "claude"

    def __post_init__(self) -> None:
        if self.effort not in EFFORT_LEVELS:
            raise ValueError(f"effort must be one of {list(EFFORT_LEVELS)}")
        if self.client is None:
            self.client = _build_client()

    def run(
        self,
        *,
        system: str,
        prompt: str,
        tools: Sequence[ToolSpec],
        max_tokens: int,
        max_turns: int,
    ) -> RunResult:
        by_name = {tool.name: tool for tool in tools}
        definitions = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in tools
        ]
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        input_tokens = output_tokens = 0

        for turn in range(1, max_turns + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    tools=definitions,
                    output_config={"effort": self.effort},
                    messages=messages,
                )
            except Exception as exc:  # noqa: BLE001 - surfaced as a clean CLI error
                raise ProviderError(f"Claude request failed: {_brief(exc)}") from exc
            input_tokens += response.usage.input_tokens
            output_tokens += response.usage.output_tokens

            if response.stop_reason == "refusal":
                raise RefusalError("Claude declined this request")

            messages.append({"role": "assistant", "content": response.content})

            # A server-side tool ran out of iterations; re-send to resume.
            if response.stop_reason == "pause_turn":
                continue

            calls = [block for block in response.content if block.type == "tool_use"]
            if not calls:
                text = "\n\n".join(block.text for block in response.content if block.type == "text")
                return RunResult(text.strip(), input_tokens, output_tokens, turn)

            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call.id,
                            "content": _dispatch(by_name, call.name, call.input),
                        }
                        for call in calls
                    ],
                }
            )

        return RunResult("", input_tokens, output_tokens, max_turns)


def _brief(exc: Exception) -> str:
    """A one-line form of an SDK exception, for a CLI message."""
    text = str(exc).strip().splitlines()
    first = text[0] if text else exc.__class__.__name__
    return first if len(first) <= 300 else first[:297] + "..."


def _dispatch(by_name: dict[str, ToolSpec], name: str, arguments: Any) -> str:
    tool = by_name.get(name)
    if tool is None:
        return f'{{"error": "unknown tool {name}"}}'
    return tool.call(arguments)


def _build_client() -> Any:
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on the install extra
        raise ProviderError(
            'the Claude backend needs the anthropic package: pip install -e ".[claude]"'
        ) from exc
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ProviderError("ANTHROPIC_API_KEY is not set")
    return anthropic.Anthropic()
