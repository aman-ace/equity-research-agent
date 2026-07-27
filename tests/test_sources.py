import httpx
import pytest
from conftest import mock_source_client

from equity_agent.sources import Citation, SourceClient, SourceError


class TestFetching:
    def test_get_json_parses_the_body(self):
        client = mock_source_client({"example.com": (200, '{"ok": true}')})
        assert client.get_json("https://example.com/data", "Example") == {"ok": True}

    def test_get_text_returns_the_body(self):
        client = mock_source_client({"example.com": (200, "hello")})
        assert client.get_text("https://example.com/doc", "Example") == "hello"

    def test_client_errors_are_not_retried(self):
        calls = []

        def handler(request):
            calls.append(request.url)
            return httpx.Response(404, text="missing")

        client = SourceClient(
            user_agent="test", client=httpx.Client(transport=httpx.MockTransport(handler))
        )
        with pytest.raises(SourceError, match="404"):
            client.get_json("https://example.com/missing", "Example")
        assert len(calls) == 1

    def test_server_errors_are_retried_then_surface(self, monkeypatch):
        monkeypatch.setattr("equity_agent.sources.time.sleep", lambda _: None)
        calls = []

        def handler(request):
            calls.append(request.url)
            return httpx.Response(503, text="unavailable")

        client = SourceClient(
            user_agent="test", client=httpx.Client(transport=httpx.MockTransport(handler))
        )
        with pytest.raises(SourceError):
            client.get_text("https://example.com/flaky", "Example")
        assert len(calls) == 3

    def test_user_agent_is_sent(self):
        seen = {}

        def handler(request):
            seen["ua"] = request.headers.get("User-Agent")
            return httpx.Response(200, text="{}")

        client = SourceClient(
            user_agent="Aman test@example.com",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        client.get_json("https://example.com/x", "Example")
        assert seen["ua"] == "Aman test@example.com"


class TestLedger:
    def test_each_fetch_is_recorded(self):
        client = mock_source_client({"example.com": (200, "{}")})
        client.get_json("https://example.com/a", "Doc A")
        client.get_json("https://example.com/b", "Doc B")
        assert [item.label for item in client.citations] == ["Doc A", "Doc B"]

    def test_repeat_fetches_are_recorded_once(self):
        client = mock_source_client({"example.com": (200, "{}")})
        client.get_json("https://example.com/a", "Doc A")
        client.get_json("https://example.com/a", "Doc A")
        assert len(client.citations) == 1

    def test_failed_fetches_are_not_recorded(self):
        client = mock_source_client({"example.com/ok": (200, "{}")})
        with pytest.raises(SourceError):
            client.get_json("https://example.com/bad", "Doc")
        assert client.citations == []

    def test_citation_renders_as_markdown(self):
        citation = Citation(label="SEC EDGAR", url="https://sec.gov/x", retrieved_at="2026-07-27")
        rendered = citation.as_markdown()
        assert "SEC EDGAR" in rendered
        assert "<https://sec.gov/x>" in rendered
        assert "2026-07-27" in rendered
