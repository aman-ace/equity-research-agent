"""Markdown to HTML, scoped to what a memo actually contains.

The web UI needs to display a memo, and a memo uses a small, known subset of
Markdown: headings, a table of figures, numbered sources, bold and italic runs,
autolinked URLs, and horizontal rules. Rendering that subset here keeps the
project's install to one dependency rather than pulling in a full Markdown
implementation for six block types.

Every text run is HTML-escaped before any tag is emitted, so a filing quote that
happens to contain angle brackets cannot inject markup into the page.
"""

from __future__ import annotations

import html
import re

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_RULE = re.compile(r"^(-{3,}|\*{3,}|_{3,})$")
_ORDERED = re.compile(r"^\d+\.\s+")
_BULLET = re.compile(r"^[-*+]\s+")
_TABLE_SEPARATOR = re.compile(r"^\|[\s:|-]+\|$")


def to_html(markdown: str) -> str:
    """Render a memo's Markdown body as an HTML fragment."""
    lines = markdown.splitlines()
    blocks: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index].strip()

        if not line:
            index += 1
            continue

        if _RULE.match(line):
            blocks.append("<hr>")
            index += 1
            continue

        heading = _HEADING.match(line)
        if heading:
            level = len(heading.group(1))
            blocks.append(f"<h{level}>{inline(heading.group(2))}</h{level}>")
            index += 1
            continue

        if line.startswith("|") and _is_table(lines, index):
            block, index = _table(lines, index)
            blocks.append(block)
            continue

        if _ORDERED.match(line):
            block, index = _list(lines, index, _ORDERED, "ol")
            blocks.append(block)
            continue

        if _BULLET.match(line):
            block, index = _list(lines, index, _BULLET, "ul")
            blocks.append(block)
            continue

        block, index = _paragraph(lines, index)
        blocks.append(block)

    return "\n".join(blocks)


def inline(text: str) -> str:
    """Render inline spans. Escapes first, so input markup cannot survive."""
    rendered = html.escape(text, quote=False)
    rendered = re.sub(r"`([^`]+)`", r"<code>\1</code>", rendered)
    rendered = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", rendered)
    rendered = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", rendered)
    # [label](url) before bare autolinks, so a labelled link keeps its label.
    rendered = re.sub(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
        r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>',
        rendered,
    )
    # Escaping turned <https://…> into &lt;https://…&gt;.
    rendered = re.sub(
        r"&lt;(https?://[^&\s]+)&gt;",
        r'<a href="\1" target="_blank" rel="noopener noreferrer">\1</a>',
        rendered,
    )
    return rendered


def _is_table(lines: list[str], index: int) -> bool:
    """A table needs a header row followed by a |---|---| separator."""
    return index + 1 < len(lines) and bool(_TABLE_SEPARATOR.match(lines[index + 1].strip()))


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _table(lines: list[str], index: int) -> tuple[str, int]:
    header = _cells(lines[index])
    index += 2  # skip the header and its separator

    rows: list[list[str]] = []
    while index < len(lines) and lines[index].strip().startswith("|"):
        rows.append(_cells(lines[index]))
        index += 1

    head = "".join(f"<th>{inline(cell)}</th>" for cell in header)
    body = "".join(
        "<tr>" + "".join(f"<td>{inline(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>", index


def _list(lines: list[str], index: int, marker: re.Pattern[str], tag: str) -> tuple[str, int]:
    items: list[str] = []
    while index < len(lines) and marker.match(lines[index].strip()):
        items.append(inline(marker.sub("", lines[index].strip())))
        index += 1
    body = "".join(f"<li>{item}</li>" for item in items)
    return f"<{tag}>{body}</{tag}>", index


def _paragraph(lines: list[str], index: int) -> tuple[str, int]:
    """Gather wrapped lines until a blank line or the start of another block.

    The first line is always consumed, even if it looks like the start of some
    other block. Text can reach here that the block dispatcher declined — a
    pipe-prefixed row with no separator beneath it, or a ``#`` with no space
    after it — and returning without advancing would spin the caller's loop
    forever.
    """
    parts: list[str] = [lines[index].strip()]
    index += 1
    while index < len(lines):
        line = lines[index].strip()
        if not line or line.startswith(("#", "|")) or _RULE.match(line):
            break
        if _ORDERED.match(line) or _BULLET.match(line):
            break
        parts.append(line)
        index += 1
    return f"<p>{inline(' '.join(parts))}</p>", index
