"""An AI agent that researches public companies from primary sources."""

from .agent import AgentConfig, research
from .memo import Memo
from .providers import Provider, ProviderError, RefusalError, RunResult
from .sources import Citation, SourceClient, SourceError
from .toolspec import ToolSpec

__version__ = "0.2.0"

__all__ = [
    "AgentConfig",
    "Citation",
    "Memo",
    "Provider",
    "ProviderError",
    "RefusalError",
    "RunResult",
    "SourceClient",
    "SourceError",
    "ToolSpec",
    "research",
    "__version__",
]
