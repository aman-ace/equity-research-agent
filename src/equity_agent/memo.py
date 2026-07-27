"""Assembly of the final research memo.

The model writes the analysis; this module wraps it in a header, appends the
sources the tools actually retrieved, and adds the disclaimer. Keeping the
sources section out of the model's hands means the citation list reflects what
was read, not what the model remembers reading.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .sources import Citation

DISCLAIMER = (
    "This memo was produced by an automated research agent from public filings "
    "and end-of-day market data. It is for research and educational purposes "
    "only and is not investment advice, an offer, or a solicitation. Figures are "
    "as reported by the issuer and have not been independently audited. Verify "
    "against the primary sources listed above before relying on any figure."
)


@dataclass(frozen=True)
class Memo:
    """A finished memo and the metadata describing how it was produced."""

    subject: str
    body: str
    citations: list[Citation]
    model: str
    effort: str
    provider: str = "claude"
    input_tokens: int = 0
    output_tokens: int = 0

    def to_markdown(self) -> str:
        stamp = time.strftime("%d %B %Y", time.gmtime())
        lines = [
            f"# Equity Research Memo — {self.subject}",
            "",
            f"*Prepared {stamp} · {self.provider} `{self.model}` · Effort: `{self.effort}`*",
            "",
            "---",
            "",
            self.body.strip(),
            "",
            "---",
            "",
            "## Sources",
            "",
        ]
        if self.citations:
            lines.extend(
                f"{index}. {citation.as_markdown()}"
                for index, citation in enumerate(self.citations, start=1)
            )
        else:
            lines.append("No external sources were retrieved for this memo.")
        lines.extend(["", "## Disclaimer", "", DISCLAIMER, ""])
        return "\n".join(lines)
