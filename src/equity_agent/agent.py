"""The agent loop.

A single research run: give the model a ticker and a tool surface, let it decide
what to pull and read, and collect the memo it writes. The turn-by-turn mechanics
live in :mod:`equity_agent.providers`; this module owns the prompt, the run
configuration, and the assembly of the result.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from . import providers, tools
from .memo import Memo
from .providers import Provider, RefusalError
from .sources import SourceClient
from .toolspec import ToolSpec

# Called as on_tool(phase, tool_name, payload) where phase is "start" or "end".
# On "start" the payload is the arguments; on "end" it is the result string.
ToolObserver = Callable[[str, str, Any], None]

DEFAULT_EFFORT = "high"
DEFAULT_MAX_TOKENS = 16_000
MAX_TURNS = 24

SYSTEM_PROMPT = """\
You are an equity research analyst. You write short, sourced memos on public \
companies for a reader who is financially literate and short on time.

How you work:

- Every figure you state must come from a tool result in this conversation. If \
  you did not retrieve it, you do not know it. Never fill a gap from memory, and \
  never estimate a number that looks like a reported one.
- Do not do arithmetic yourself. get_valuation_metrics computes margins, \
  returns, growth, leverage, and trailing multiples; use its output rather than \
  deriving your own.
- When a field comes back null, say the issuer did not tag it rather than \
  substituting a proxy without saying so.
- Read the issuer's own words when the question is qualitative. Risk factors and \
  management's discussion in the latest 10-K are usually worth one read_filing \
  call; skimming many filings rarely is.
- Distinguish what the filings show from what you infer. Attribute claims about \
  strategy or outlook to the filing that makes them.
- Trailing multiples are trailing. Do not present them as forward estimates, and \
  do not compare them to a peer multiple you have not retrieved.

Structure the memo with these sections, as Markdown, starting at heading level 2:

## Summary — three or four sentences: what the company does, how the last \
reported year went, and where the valuation sits.
## Business — segments and revenue drivers, from the filings.
## Financial performance — multi-year revenue, margin, and cash-flow trend, with \
the figures that support it. A small table is usually clearer than prose here.
## Valuation — trailing multiples and what they are being paid for. State the \
price and date the multiples rest on.
## Risks — the two or three risks that would most change the picture, drawn from \
the issuer's own risk factors rather than generic market commentary.
## What would change this view — the specific, observable developments that \
would move the conclusion.

Write in plain declarative prose. No preamble, no restating the question, no \
recommendation to buy or sell. Do not add a sources section or a disclaimer — \
those are appended for you. Aim for roughly 700 to 1,100 words.
"""


@dataclass
class AgentConfig:
    """Configuration for one research run."""

    provider: str = field(default_factory=providers.default_provider)
    model: str | None = None
    effort: str = DEFAULT_EFFORT
    max_tokens: int = DEFAULT_MAX_TOKENS
    years: int = 5
    user_agent: str = field(
        default_factory=lambda: os.environ.get(
            "SEC_USER_AGENT", "equity-research-agent (contact: set SEC_USER_AGENT)"
        )
    )

    def __post_init__(self) -> None:
        if self.provider not in providers.PROVIDERS:
            raise ValueError(f"provider must be one of {list(providers.PROVIDERS)}")
        allowed = {"low", "medium", "high", "xhigh", "max"}
        if self.effort not in allowed:
            raise ValueError(f"effort must be one of {sorted(allowed)}")
        if self.model is None:
            self.model = providers.default_model(self.provider)


def build_prompt(subject: str, question: str | None) -> str:
    """The opening user turn for a run."""
    ask = (
        f"Research {subject} and write the memo.\n\n"
        "Start by resolving the company, then pull its fundamentals and "
        "valuation metrics before reading anything qualitative."
    )
    if question:
        ask += f"\n\nThe reader specifically wants to know: {question}"
    return ask


def _observed(spec: ToolSpec, on_tool: ToolObserver) -> ToolSpec:
    """Wrap a tool so a caller can watch it run.

    Only ``run`` is swapped, so :meth:`ToolSpec.call` keeps its own error
    handling: a failing tool still returns a message to the model rather than
    ending the run. The observer is never allowed to break a run either — a
    callback that raises would otherwise turn a working tool into a failed one.
    """

    def observed(**arguments: Any) -> str:
        _safely(on_tool, "start", spec.name, arguments)
        result = spec.run(**arguments)
        _safely(on_tool, "end", spec.name, result)
        return result

    return replace(spec, run=observed)


def _safely(on_tool: ToolObserver, phase: str, name: str, payload: Any) -> None:
    # Reporting must never fail the research run.
    with contextlib.suppress(Exception):
        on_tool(phase, name, payload)


def research(
    subject: str,
    config: AgentConfig | None = None,
    question: str | None = None,
    provider: Provider | None = None,
    on_tool: ToolObserver | None = None,
) -> Memo:
    """Run one research pass and return the finished memo.

    Args:
        subject: Ticker symbol or company name.
        config: Model and run configuration; defaults are used when omitted.
        question: An optional specific question to answer alongside the memo.
        provider: A pre-built backend. One is constructed from ``config`` when
            not supplied; the tests inject a stub here.
        on_tool: Optional callback invoked around every tool call, for progress
            reporting. Omitted, the tool surface is passed through untouched.

    Raises:
        RefusalError: If the model declines the request.
        ProviderError: If the chosen backend is missing or unconfigured.
    """
    config = config or AgentConfig()
    backend = provider or providers.build(config.provider, config.model, config.effort)

    source_client = SourceClient(user_agent=config.user_agent)
    tools.configure(tools.ResearchContext(client=source_client, default_years=config.years))

    surface = tools.ALL_TOOLS
    if on_tool is not None:
        surface = [_observed(spec, on_tool) for spec in surface]

    result = backend.run(
        system=SYSTEM_PROMPT,
        prompt=build_prompt(subject, question),
        tools=surface,
        max_tokens=config.max_tokens,
        max_turns=MAX_TURNS,
    )

    body = result.text.strip()
    if not body:
        body = (
            "_The agent stopped before writing a memo — it may have hit the "
            f"{MAX_TURNS}-turn limit. Re-run, or raise MAX_TURNS._"
        )

    return Memo(
        subject=subject.upper(),
        body=body,
        citations=list(source_client.citations),
        provider=backend.name,
        model=backend.model,
        effort=config.effort,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


__all__ = [
    "DEFAULT_EFFORT",
    "DEFAULT_MAX_TOKENS",
    "MAX_TURNS",
    "SYSTEM_PROMPT",
    "AgentConfig",
    "RefusalError",
    "build_prompt",
    "research",
]
