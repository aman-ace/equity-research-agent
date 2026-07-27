"""Gemini backend, built on the google-genai SDK.

Same loop shape as the Claude backend: send, execute whatever tools come back,
send the results, repeat until the model answers in prose. The differences are
all encoding — Gemini carries tool calls as ``function_call`` parts and expects
results as ``function_response`` parts in a following user turn.

Automatic function calling is switched off deliberately. It would run the tools
for us, but then the turn limit, the dispatch, and the accounting would differ
between the two providers, and comparing a Claude run to a Gemini run would mean
comparing two different control flows.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..toolspec import ToolSpec
from .base import ProviderError, RefusalError, RunResult

DEFAULT_MODEL = "gemini-2.5-pro"

# Gemini stops for reasons other than "finished"; these mean the request was
# declined rather than answered.
_REFUSAL_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"}


@dataclass
class GeminiProvider:
    """Runs the agent on Google's Gemini API.

    ``effort`` is accepted for interface symmetry but is not sent: Gemini's
    thinking controls are model-specific and an unsupported value is rejected
    outright, so the model's own default is used. Tune with ``--model`` instead.
    """

    model: str = DEFAULT_MODEL
    effort: str = "high"
    client: Any = None
    name: str = "gemini"

    def __post_init__(self) -> None:
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
        types = _types()
        by_name = {tool.name: tool for tool in tools}

        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=tool.name,
                            description=tool.description,
                            parameters_json_schema=tool.input_schema,
                        )
                        for tool in tools
                    ]
                )
            ],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        contents: list[Any] = [types.Content(role="user", parts=[types.Part(text=prompt)])]
        input_tokens = output_tokens = 0

        for turn in range(1, max_turns + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
            except Exception as exc:  # noqa: BLE001 - surfaced as a clean CLI error
                raise ProviderError(f"Gemini request failed: {_brief(exc)}") from exc
            usage = getattr(response, "usage_metadata", None)
            if usage is not None:
                input_tokens += usage.prompt_token_count or 0
                output_tokens += usage.candidates_token_count or 0

            candidate = _first_candidate(response)
            _check_refusal(candidate)

            if candidate is not None and candidate.content is not None:
                contents.append(candidate.content)

            calls = list(response.function_calls or [])
            if not calls:
                return RunResult(_text_of(candidate), input_tokens, output_tokens, turn)

            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name=call.name,
                            response={"result": _dispatch(by_name, call.name, call.args)},
                        )
                        for call in calls
                    ],
                )
            )

        return RunResult("", input_tokens, output_tokens, max_turns)


def _brief(exc: Exception) -> str:
    """A one-line form of an SDK exception, for a CLI message.

    Provider SDKs raise exceptions whose text runs to several hundred characters
    of nested JSON; the first line carries the part a user can act on.
    """
    text = str(exc).strip().splitlines()
    first = text[0] if text else exc.__class__.__name__
    return first if len(first) <= 300 else first[:297] + "..."


def _first_candidate(response: Any) -> Any:
    candidates = getattr(response, "candidates", None) or []
    return candidates[0] if candidates else None


def _check_refusal(candidate: Any) -> None:
    reason = getattr(candidate, "finish_reason", None)
    label = getattr(reason, "name", None) or (str(reason) if reason else "")
    if label.upper() in _REFUSAL_REASONS:
        raise RefusalError(f"Gemini declined this request ({label})")


def _text_of(candidate: Any) -> str:
    """Join the text parts of a response.

    ``response.text`` is avoided: it raises when a candidate holds anything other
    than plain text, which is exactly the case in a tool-calling loop.
    """
    if candidate is None or candidate.content is None:
        return ""
    parts = candidate.content.parts or []
    return "\n\n".join(part.text.strip() for part in parts if getattr(part, "text", None)).strip()


def _dispatch(by_name: dict[str, ToolSpec], name: str, arguments: Any) -> str:
    tool = by_name.get(name)
    if tool is None:
        return json.dumps({"error": f"unknown tool {name}"})
    return tool.call(arguments)


def _types() -> Any:
    try:
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - depends on the install extra
        raise ProviderError(
            'the Gemini backend needs google-genai: pip install -e ".[gemini]"'
        ) from exc
    return types


def _build_client() -> Any:
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - depends on the install extra
        raise ProviderError(
            'the Gemini backend needs google-genai: pip install -e ".[gemini]"'
        ) from exc
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ProviderError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=key)


def list_models(client: Any = None) -> list[str]:
    """Model names available to this key.

    Model identifiers move faster than this README does, so the CLI exposes this
    rather than hardcoding a list that will go stale.
    """
    client = client or _build_client()
    names: list[str] = []
    for model in client.models.list():
        actions = getattr(model, "supported_actions", None)
        if actions and "generateContent" not in actions:
            continue
        name = getattr(model, "name", "") or ""
        names.append(name.removeprefix("models/"))
    return sorted(names)
