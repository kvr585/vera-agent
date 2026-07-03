"""Interface for memory management components."""

from typing import Any, Protocol

from vera_engine.core.entities import SessionMemory, WorkingMemory


class MemoryManager(Protocol):
    """Port contract defining access and mutations to Agent memories."""

    @property
    def working_memory(self) -> WorkingMemory:
        """Retrieves the short-term working memory scratchpad."""
        ...

    @property
    def session_memory(self) -> SessionMemory:
        """Retrieves the full session context memory."""
        ...

    def add_history_entry(self, role: str, content: str, **metadata: Any) -> None:
        """Appends a new interaction entry to the session memory.

        Args:
            role: The actor role (e.g. 'user', 'assistant', 'system', 'tool').
            content: The text message or payload content.
            metadata: Additional key-value pairs (timestamps, token usage, etc.).
        """
        ...

    def clear_working_memory(self) -> None:
        """Clears short-term operational memory states."""
        ...
