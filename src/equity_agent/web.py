"""A local web UI for the agent.

Built on the standard library's ``ThreadingHTTPServer`` rather than an ASGI
framework: this is a single-user tool bound to localhost, and the project's
premise is that a fresh clone runs with only ``httpx`` installed.

A research run takes a minute or two and makes six to ten tool calls, so the
run streams its progress over Server-Sent Events instead of leaving the page
blank until the memo lands. ``research()`` executes on a worker thread and
publishes events to a queue that the request thread drains.
"""

from __future__ import annotations

import json
import queue
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import render
from .agent import DEFAULT_EFFORT, DEFAULT_MAX_TOKENS, AgentConfig, research
from .providers import ProviderError, RefusalError
from .sources import SourceError

EFFORTS = ("low", "medium", "high", "xhigh", "max")
_DONE = object()  # sentinel: the worker has finished publishing


def _run(params: dict[str, Any], events: queue.Queue[Any]) -> None:
    """Execute one research run, publishing progress to ``events``."""

    def on_tool(phase: str, name: str, payload: Any) -> None:
        if phase == "start":
            events.put({"type": "tool", "phase": "start", "name": name, "args": payload})
        else:
            events.put({"type": "tool", "phase": "end", "name": name, "note": _note(payload)})

    try:
        config = AgentConfig(
            effort=params["effort"],
            years=params["years"],
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        memo = research(
            params["subject"],
            config=config,
            question=params["question"] or None,
            on_tool=on_tool,
        )
    except (RefusalError, ProviderError, SourceError, ValueError) as exc:
        events.put({"type": "error", "message": str(exc)})
    except Exception as exc:  # noqa: BLE001 - surface anything else in the page
        events.put({"type": "error", "message": f"{exc.__class__.__name__}: {exc}"})
    else:
        events.put(
            {
                "type": "done",
                "markdown": memo.to_markdown(),
                "html": render.to_html(memo.to_markdown()),
                "subject": memo.subject,
                "provider": memo.provider,
                "model": memo.model,
                "input_tokens": memo.input_tokens,
                "output_tokens": memo.output_tokens,
                "sources": len(memo.citations),
            }
        )
    finally:
        events.put(_DONE)


def _note(result: Any) -> str:
    """A short, human-readable trace of what a tool returned."""
    text = result if isinstance(result, str) else str(result)
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return f"{len(text):,} characters"  # read_filing returns plain text
    if isinstance(parsed, dict):
        if "error" in parsed:
            return f"error: {parsed['error']}"
        company = parsed.get("company")
        if isinstance(company, dict) and company.get("name"):
            return str(company["name"])
        if parsed.get("ticker") and parsed.get("as_of"):
            return f"{parsed['ticker']} as of {parsed['as_of']}"
        if parsed.get("name"):
            return str(parsed["name"])
    return "ok"


class Handler(BaseHTTPRequestHandler):
    server_version = "equity-research"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's interface
        route = urlparse(self.path)
        if route.path == "/":
            self._html(PAGE)
        elif route.path == "/api/research":
            self._stream(parse_qs(route.query))
        else:
            self.send_error(404, "not found")

    def log_message(self, fmt: str, *args: Any) -> None:
        """Silence per-request logging; the page is the interesting output."""

    def _html(self, body: str) -> None:
        payload = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _stream(self, query: dict[str, list[str]]) -> None:
        def first(key: str, default: str = "") -> str:
            return query.get(key, [default])[0].strip()

        subject = first("subject")
        if not subject:
            self.send_error(400, "a ticker or company name is required")
            return

        effort = first("effort", DEFAULT_EFFORT)
        if effort not in EFFORTS:
            effort = DEFAULT_EFFORT
        try:
            years = max(1, min(10, int(first("years", "5") or 5)))
        except ValueError:
            years = 5

        params = {
            "subject": subject,
            "effort": effort,
            "years": years,
            "question": first("question"),
        }

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        events: queue.Queue[Any] = queue.Queue()
        worker = threading.Thread(target=_run, args=(params, events), daemon=True)
        worker.start()

        while True:
            event = events.get()
            if event is _DONE:
                break
            try:
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # The browser navigated away. The worker is a daemon thread and
                # will not outlive the process, so just stop writing.
                return


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Equity Research Agent</title>
<style>
  :root {
    --bg: #fbfbfa; --panel: #fff; --ink: #1a1a19; --muted: #6b6b66;
    --line: #e4e4e0; --accent: #1f5f4f; --error: #a3341f;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #17181a; --panel: #1e2022; --ink: #e9e9e6; --muted: #9a9a94;
      --line: #2e3134; --accent: #74c4ad; --error: #e8836b;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  }
  .wrap { max-width: 62rem; margin: 0 auto; padding: 2rem 1.25rem 4rem; }
  h1 { font-size: 1.35rem; margin: 0 0 .25rem; letter-spacing: -.01em; }
  .sub { color: var(--muted); font-size: .875rem; margin: 0 0 1.5rem; }
  .card {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 10px; padding: 1.25rem; margin-bottom: 1.25rem;
  }
  .row { display: flex; gap: .75rem; flex-wrap: wrap; }
  .row > * { flex: 1 1 8rem; }
  label { display: block; font-size: .8rem; color: var(--muted); margin-bottom: .3rem; }
  input, select, button {
    width: 100%; padding: .6rem .7rem; font: inherit; font-size: .9rem;
    border: 1px solid var(--line); border-radius: 7px;
    background: var(--bg); color: var(--ink);
  }
  button {
    margin-top: 1rem; background: var(--accent); color: #fff;
    border-color: transparent; font-weight: 600; cursor: pointer;
  }
  button:disabled { opacity: .55; cursor: not-allowed; }
  #trace { list-style: none; margin: 0; padding: 0; font-size: .875rem; }
  #trace li {
    display: flex; gap: .6rem; padding: .3rem 0;
    border-bottom: 1px solid var(--line); align-items: baseline;
  }
  #trace li:last-child { border-bottom: 0; }
  .mark { width: 1rem; flex: none; color: var(--accent); }
  .tname { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; flex: none; }
  .tnote { color: var(--muted); overflow-wrap: anywhere; }
  .spin { display: inline-block; animation: spin 1s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .err { color: var(--error); }
  .meta { color: var(--muted); font-size: .8rem; display: flex; justify-content: space-between;
          gap: 1rem; flex-wrap: wrap; align-items: center; }
  #memo :is(h1,h2,h3) { letter-spacing: -.01em; line-height: 1.3; }
  #memo h1 { font-size: 1.3rem; }
  #memo h2 { font-size: 1.05rem; margin-top: 1.75rem; }
  #memo table { border-collapse: collapse; width: 100%; font-size: .85rem; margin: 1rem 0; }
  #memo :is(th,td) { border: 1px solid var(--line); padding: .4rem .55rem; text-align: left; }
  #memo th { background: var(--bg); font-weight: 600; }
  #memo a { color: var(--accent); overflow-wrap: anywhere; }
  #memo hr { border: 0; border-top: 1px solid var(--line); margin: 1.75rem 0; }
  .scroll { overflow-x: auto; }
  .hide { display: none; }
  a.dl { color: var(--accent); font-size: .8rem; font-weight: 600; text-decoration: none; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Equity Research Agent</h1>
  <p class="sub">Researches a public company from SEC filings and writes a sourced memo.</p>

  <form class="card" id="form">
    <div class="row">
      <div style="flex: 2 1 14rem">
        <label for="subject">Ticker or company name</label>
        <input id="subject" name="subject" placeholder="COST" required autofocus>
      </div>
      <div>
        <label for="effort">Effort</label>
        <select id="effort" name="effort">
          <option>low</option><option>medium</option>
          <option selected>high</option><option>xhigh</option><option>max</option>
        </select>
      </div>
      <div>
        <label for="years">Years</label>
        <input id="years" name="years" type="number" value="5" min="1" max="10">
      </div>
    </div>
    <div style="margin-top:.75rem">
      <label for="question">Specific question (optional)</label>
      <input id="question" name="question" placeholder="Is the membership model durable?">
    </div>
    <button id="go" type="submit">Research</button>
  </form>

  <div class="card hide" id="progress">
    <ul id="trace"></ul>
  </div>

  <div class="card hide" id="result">
    <div class="meta">
      <span id="stats"></span>
      <a class="dl" id="download" download>Download .md</a>
    </div>
    <div class="scroll"><div id="memo"></div></div>
  </div>
</div>

<script>
const form = document.getElementById('form');
const go = document.getElementById('go');
const trace = document.getElementById('trace');
const progress = document.getElementById('progress');
const result = document.getElementById('result');
const memo = document.getElementById('memo');
const stats = document.getElementById('stats');
const download = document.getElementById('download');
let stream = null;
let pending = null;

function row(name, note, done) {
  const li = document.createElement('li');
  li.innerHTML = `<span class="mark"></span><span class="tname"></span><span class="tnote"></span>`;
  li.querySelector('.mark').innerHTML = done ? '&check;' : '<span class="spin">&#8635;</span>';
  li.querySelector('.tname').textContent = name;
  li.querySelector('.tnote').textContent = note || '';
  return li;
}

function summarize(args) {
  if (!args) return '';
  return Object.entries(args).map(([k, v]) => `${k}=${v}`).join('  ');
}

form.addEventListener('submit', (event) => {
  event.preventDefault();
  if (stream) stream.close();

  trace.innerHTML = '';
  memo.innerHTML = '';
  progress.classList.remove('hide');
  result.classList.add('hide');
  go.disabled = true;
  go.textContent = 'Researching…';

  const query = new URLSearchParams(new FormData(form)).toString();
  stream = new EventSource('/api/research?' + query);

  stream.onmessage = (message) => {
    const event = JSON.parse(message.data);

    if (event.type === 'tool') {
      if (event.phase === 'start') {
        pending = row(event.name, summarize(event.args), false);
        trace.appendChild(pending);
      } else if (pending) {
        pending.replaceWith(row(event.name, event.note, true));
        pending = null;
      }
      return;
    }

    if (event.type === 'error') {
      const li = document.createElement('li');
      li.className = 'err';
      li.textContent = 'error: ' + event.message;
      trace.appendChild(li);
      finish();
      return;
    }

    if (event.type === 'done') {
      memo.innerHTML = event.html;
      stats.textContent =
        `${event.provider} ${event.model} · tokens in/out ` +
        `${event.input_tokens.toLocaleString()}/${event.output_tokens.toLocaleString()} ` +
        `· ${event.sources} sources · est. $` +
        (event.input_tokens / 1e6 * 5 + event.output_tokens / 1e6 * 25).toFixed(2);
      download.href = URL.createObjectURL(new Blob([event.markdown], {type: 'text/markdown'}));
      download.setAttribute('download', event.subject.toLowerCase() + '-memo.md');
      result.classList.remove('hide');
      finish();
    }
  };

  stream.onerror = () => { if (go.disabled) finish(); };
});

function finish() {
  if (stream) { stream.close(); stream = null; }
  if (pending) { pending.querySelector('.mark').textContent = '·'; pending = null; }
  go.disabled = false;
  go.textContent = 'Research';
}
</script>
</body>
</html>
"""


def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    """Run the UI until interrupted."""
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"Equity Research Agent UI on {url}  (ctrl-c to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
