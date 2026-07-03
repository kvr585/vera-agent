"""Manager class to register and retrieve pluggable workflows."""

from vera_engine.runtime.workflows.base import Workflow


class WorkflowManager:
    """Registry managing available workflow implementations inside the Kernel."""

    def __init__(self) -> None:
        self._workflows: dict[str, Workflow] = {}

    def register(self, workflow: Workflow) -> None:
        """Registers a workflow implementation.

        Args:
            workflow: The Workflow instance.
        """
        self._workflows[workflow.name] = workflow

    def get_workflow(self, name: str) -> Workflow | None:
        """Retrieves a workflow by name.

        Args:
            name: The workflow key.

        Returns:
            The Workflow instance if registered, otherwise None.
        """
        return self._workflows.get(name)
