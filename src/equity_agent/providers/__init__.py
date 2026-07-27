"""Model backends.

``build`` is the only entry point the rest of the project uses; it maps a
provider name and model to a ready-to-run backend.
"""

from __future__ import annotations

import os

from .base import Provider, ProviderError, RefusalError, RunResult

PROVIDERS = ("claude", "gemini")


def default_provider() -> str:
    """Pick a provider from whichever API key is present.

    Claude wins when both are set, since that is the tested path.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    return "claude"


def default_model(provider: str) -> str:
    """The model used when none is given."""
    if provider == "claude":
        from .claude import DEFAULT_MODEL

        return DEFAULT_MODEL
    if provider == "gemini":
        from .gemini import DEFAULT_MODEL

        return DEFAULT_MODEL
    raise ProviderError(f"unknown provider {provider!r}; choose from {list(PROVIDERS)}")


def build(provider: str, model: str | None = None, effort: str = "high") -> Provider:
    """Construct a backend.

    Args:
        provider: ``"claude"`` or ``"gemini"``.
        model: Model identifier; the provider's default is used when omitted.
        effort: Reasoning effort. Applies to Claude; Gemini uses its model default.
    """
    if provider == "claude":
        from .claude import ClaudeProvider

        return ClaudeProvider(model=model or default_model("claude"), effort=effort)
    if provider == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider(model=model or default_model("gemini"), effort=effort)
    raise ProviderError(f"unknown provider {provider!r}; choose from {list(PROVIDERS)}")


__all__ = [
    "PROVIDERS",
    "Provider",
    "ProviderError",
    "RefusalError",
    "RunResult",
    "build",
    "default_model",
    "default_provider",
]
