"""Concrete in-memory implementation of the MemoryManager interface."""

from typing import Any

from vera_engine.core.entities import SessionMemory, WorkingMemory
from vera_engine.core.interfaces.memory import MemoryManager


class LocalMemoryManager(MemoryManager):
    """Manages working scratchpads and active interaction histories in memory."""

    def __init__(self) -> None:
        self._working = WorkingMemory()
        self._session = SessionMemory()

    @property
    def working_memory(self) -> WorkingMemory:
        """Retrieves the short-term working memory scratchpad."""
        return self._working

    @property
    def session_memory(self) -> SessionMemory:
        """Retrieves the full session context memory."""
        return self._session

    def add_history_entry(self, role: str, content: str, **metadata: Any) -> None:
        """Appends an interaction entry to the history list."""
        entry = {"role": role, "content": content}
        if metadata:
            entry.update(metadata)
        self._session.history.append(entry)

    def clear_working_memory(self) -> None:
        """Resets the short-term working memory scratchpad."""
        self._working.clear()
