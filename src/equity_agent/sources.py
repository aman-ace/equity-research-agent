"""HTTP access with a citation ledger.

Every network read the agent performs goes through :class:`SourceClient`, which
records the URL and a short label. The memo writer then renders those records as
a sources section, so each claim in a memo can be traced back to a primary
document rather than to the model's recollection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_TIMEOUT = 30.0
MAX_RETRIES = 3


class SourceError(RuntimeError):
    """Raised when a source cannot be retrieved."""


@dataclass(frozen=True)
class Citation:
    """A single retrieved document."""

    label: str
    url: str
    retrieved_at: str

    def as_markdown(self) -> str:
        return f"{self.label}. Accessed {self.retrieved_at}. <{self.url}>"


@dataclass
class SourceClient:
    """Thin HTTP wrapper that remembers what it fetched.

    Args:
        user_agent: Sent on every request. The SEC requires a descriptive
            ``User-Agent`` that identifies the requester by name and email.
        client: Optional pre-built httpx client. Injected by the tests.
    """

    user_agent: str
    client: httpx.Client | None = None
    citations: list[Citation] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = httpx.Client(timeout=DEFAULT_TIMEOUT, follow_redirects=True)

    def get_json(self, url: str, label: str) -> Any:
        """Fetch and parse a JSON document, recording it as a citation."""
        response = self._get(url)
        self._record(label, url)
        try:
            return response.json()
        except ValueError as exc:  # pragma: no cover - malformed upstream payload
            raise SourceError(f"{url} did not return JSON") from exc

    def get_text(self, url: str, label: str) -> str:
        """Fetch a text document, recording it as a citation."""
        response = self._get(url)
        self._record(label, url)
        return response.text

    def _get(self, url: str) -> httpx.Response:
        assert self.client is not None
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.get(url, headers={"User-Agent": self.user_agent})
            except httpx.HTTPError as exc:
                last_error = exc
            else:
                if response.status_code == 200:
                    return response
                # 429 and 5xx are worth another try; 4xx are not.
                if response.status_code < 500 and response.status_code != 429:
                    raise SourceError(f"{url} returned HTTP {response.status_code}")
                last_error = SourceError(f"{url} returned HTTP {response.status_code}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(0.5 * 2**attempt)
        raise SourceError(f"could not retrieve {url}: {last_error}")

    def _record(self, label: str, url: str) -> None:
        if any(existing.url == url for existing in self.citations):
            return
        stamp = time.strftime("%Y-%m-%d", time.gmtime())
        self.citations.append(Citation(label=label, url=url, retrieved_at=stamp))
