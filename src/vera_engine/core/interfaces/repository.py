"""Interface for session persistence adapters."""

from collections.abc import Sequence
from typing import Protocol

from vera_engine.core.entities import AgentState


class SessionRepository(Protocol):
    """Port contract defining database persistence capabilities for agent states."""

    def save_state(self, state: AgentState) -> None:
        """Saves or updates the complete agent session state to database storage.

        Args:
            state: The current AgentState object to persist.
        """
        ...

    def get_state(self, session_id: str) -> AgentState | None:
        """Retrieves the latest persisted agent state snapshot for a session.

        Args:
            session_id: The unique identifier of the target session.

        Returns:
            The reconstructed AgentState, or None if the session does not exist.
        """
        ...

    def list_sessions(self) -> Sequence[AgentState]:
        """Lists historical agent sessions saved in storage.

        Returns:
            A sequence of historical AgentState snapshots.
        """
        ...
