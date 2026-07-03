"""Base interface contract for VERA workflows."""

from typing import Protocol

from vera_engine.runtime.state import StateManager


class Workflow(Protocol):
    """Port contract defining executable workflows within the Runtime Kernel."""

    @property
    def name(self) -> str:
        """The logical name identifying this workflow (e.g. 'default', 'coding')."""
        ...

    def execute(self, state_manager: StateManager) -> None:
        """Runs the workflow graph, mutating states through the state manager.

        Args:
            state_manager: The active StateManager containing the session state.
        """
        ...
