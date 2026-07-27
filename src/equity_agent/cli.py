"""Command-line entry point."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import providers
from .agent import DEFAULT_EFFORT, DEFAULT_MAX_TOKENS, AgentConfig, research
from .providers import ProviderError, RefusalError

KEY_ENV = {"claude": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="equity-research",
        description="Research a public company from SEC filings and write a sourced memo.",
    )
    parser.add_argument(
        "subject", nargs="?", help='ticker or company name, e.g. "COST" or "Costco"'
    )
    parser.add_argument("-o", "--out", type=Path, help="write the memo here instead of stdout")
    parser.add_argument("-q", "--question", help="a specific question to answer in the memo")
    parser.add_argument(
        "--provider",
        choices=list(providers.PROVIDERS),
        help="model backend (default: whichever API key is set)",
    )
    parser.add_argument("--model", help="model id (default: the provider's own default)")
    parser.add_argument(
        "--effort",
        default=DEFAULT_EFFORT,
        choices=["low", "medium", "high", "xhigh", "max"],
        help=f"reasoning effort, Claude only (default: {DEFAULT_EFFORT})",
    )
    parser.add_argument("--years", type=int, default=5, help="fiscal years of history (default: 5)")
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="output token ceiling"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="list the Gemini models available to your key, then exit",
    )
    parser.add_argument(
        "--serve", action="store_true", help="run the local web UI instead of writing one memo"
    )
    parser.add_argument("--port", type=int, default=8000, help="port for --serve (default: 8000)")
    parser.add_argument(
        "--no-browser", action="store_true", help="with --serve, do not open a browser"
    )
    return parser


def _list_models() -> int:
    from .providers.gemini import list_models

    try:
        for name in list_models():
            print(name)
    except ProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _check_environment(provider: str) -> int | None:
    """Verify the API key is present. Returns an exit code, or None when fine."""
    key = KEY_ENV[provider]
    if not os.environ.get(key) and not (provider == "gemini" and os.environ.get("GOOGLE_API_KEY")):
        print(f"{key} is not set (needed for --provider {provider}).", file=sys.stderr)
        return 2
    if not os.environ.get("SEC_USER_AGENT"):
        print(
            "warning: SEC_USER_AGENT is not set. EDGAR asks for a descriptive "
            'user agent, e.g. SEC_USER_AGENT="Your Name your@email.com"',
            file=sys.stderr,
        )
    return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_models:
        return _list_models()

    provider = args.provider or providers.default_provider()

    if args.serve:
        # Checked up front: a missing key would otherwise only surface as an
        # error inside the page, after the user had typed a ticker.
        failure = _check_environment(provider)
        if failure is not None:
            return failure
        from .web import serve

        serve(port=args.port, open_browser=not args.no_browser)
        return 0

    if not args.subject:
        print("error: a ticker or company name is required", file=sys.stderr)
        return 2

    failure = _check_environment(provider)
    if failure is not None:
        return failure

    config = AgentConfig(
        provider=provider,
        model=args.model,
        effort=args.effort,
        max_tokens=args.max_tokens,
        years=args.years,
    )

    try:
        memo = research(args.subject, config=config, question=args.question)
    except (RefusalError, ProviderError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("interrupted", file=sys.stderr)
        return 130

    rendered = memo.to_markdown()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(rendered)

    print(
        f"{memo.provider} {memo.model} · tokens in/out: "
        f"{memo.input_tokens}/{memo.output_tokens} · sources: {len(memo.citations)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
