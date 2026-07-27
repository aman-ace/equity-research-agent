"""What a model provider has to supply.

The agent needs one thing from a model: run a conversation, calling the tools it
asks for, and hand back the final text. Everything provider-specific — request
shape, tool-call encoding, token accounting — lives behind this interface.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ..toolspec import ToolSpec


@dataclass(frozen=True)
class RunResult:
    """The outcome of one agent run."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    turns: int = 0


class RefusalError(RuntimeError):
    """Raised when a model declines the request."""


class ProviderError(RuntimeError):
    """Raised when a provider is unavailable or misconfigured."""


class Provider(Protocol):
    """A model backend the agent can drive."""

    name: str
    model: str

    def run(
        self,
        *,
        system: str,
        prompt: str,
        tools: Sequence[ToolSpec],
        max_tokens: int,
        max_turns: int,
    ) -> RunResult:
        """Run the conversation to completion and return the final text."""
        ...
