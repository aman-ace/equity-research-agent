"""A provider-neutral description of a tool.

Claude and Gemini both want the same three things — a name, a description, and a
JSON Schema for the arguments — but they want them in different shapes. Declaring
tools once, here, keeps the provider adapters thin and keeps the tool code itself
free of any vendor's SDK.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    """One capability offered to the model.

    Attributes:
        name: The identifier the model calls.
        description: What the tool does and when to reach for it. Models lean
            heavily on this, so it is written for the model, not for a reader.
        input_schema: JSON Schema for the arguments object.
        run: The Python callable, taking the schema's properties as keyword
            arguments and returning a string.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    run: Callable[..., str]

    def call(self, arguments: Mapping[str, Any] | None) -> str:
        """Invoke the tool, turning any failure into a result the model can read.

        A tool that raises would end the run. Returning the error instead lets
        the agent recover — retry with a different ticker, or say in the memo
        that a figure could not be retrieved.
        """
        try:
            return self.run(**dict(arguments or {}))
        except TypeError as exc:
            return json.dumps({"error": f"invalid arguments for {self.name}: {exc}"})
        except Exception as exc:  # noqa: BLE001 - the model handles the message
            return json.dumps({"error": f"{self.name} failed: {exc}"})


def string(description: str) -> dict[str, Any]:
    """A required string property."""
    return {"type": "string", "description": description}


def integer(description: str, default: int) -> dict[str, Any]:
    """An optional integer property with a documented default."""
    return {"type": "integer", "description": f"{description} Defaults to {default}."}


def schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    """An arguments object schema."""
    return {"type": "object", "properties": properties, "required": required}
