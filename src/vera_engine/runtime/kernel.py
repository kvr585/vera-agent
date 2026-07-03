"""The VERA Runtime Kernel coordinating orchestration services and workflows."""

from vera_engine.core.entities import Capability, Event
from vera_engine.core.interfaces.llm import LLMProvider
from vera_engine.core.interfaces.repository import SessionRepository
from vera_engine.core.interfaces.tool import ToolRegistry
from vera_engine.runtime import logging
from vera_engine.runtime.events import EventDispatcher
from vera_engine.runtime.prompt import PromptManager
from vera_engine.runtime.state import StateManager
from vera_engine.runtime.workflows.manager import WorkflowManager


class RuntimeKernel:
    """The central execution engine (Kernel) of the VERA AI platform."""

    def __init__(
        self,
        repository: SessionRepository,
        llm: LLMProvider,
        tool_registry: ToolRegistry,
        event_dispatcher: EventDispatcher,
        prompt_manager: PromptManager,
        workflow_manager: WorkflowManager,
        capabilities: list[Capability] | None = None,
    ) -> None:
        """Initializes the kernel, wiring interfaces and registering logging listeners.

        Args:
            repository: Port for persisting states.
            llm: Port for language model queries.
            tool_registry: Port for tool registrations.
            event_dispatcher: Event dispatcher bus.
            prompt_manager: Prompt manager templates loader.
            workflow_manager: Workflow registration manager.
            capabilities: Active capabilities list.
        """
        self._repository = repository
        self._llm = llm
        self._tool_registry = tool_registry
        self._dispatcher = event_dispatcher
        self._prompt_manager = prompt_manager
        self._workflow_manager = workflow_manager
        self._capabilities = capabilities or []

        # Register unified logger listener for global event tracking
        self._register_event_logger()

    def run(self, session_id: str, goal: str, workflow_name: str) -> StateManager:
        """Bootstraps state and executes the specified workflow.

        Args:
            session_id: Unique execution identifier.
            goal: User goal description.
            workflow_name: Name of the workflow to run.

        Returns:
            The finalized StateManager object.
        """
        logging.info(f"Kernel: Bootstrapping workflow '{workflow_name}'...")

        # Initialize the StateManager for this run
        state_manager = StateManager(
            session_id=session_id,
            goal=goal,
            workflow=workflow_name,
            repository=self._repository,
            dispatcher=self._dispatcher,
            capabilities=self._capabilities,
        )

        # Retrieve the workflow from the workflow manager
        workflow = self._workflow_manager.get_workflow(workflow_name)
        if not workflow:
            err = f"Workflow '{workflow_name}' is not registered with the kernel."
            logging.error(err)
            state_manager.finalize_session(summary=err, success=False)
            return state_manager

        try:
            # Execute workflow logic
            workflow.execute(state_manager)
            logging.info(
                f"Kernel: Workflow '{workflow_name}' completed. "
                f"Final Success: {state_manager.state.success}"
            )
        except Exception as err:
            logging.exception(
                f"Kernel: Fatal exception crashed workflow '{workflow_name}': {err}"
            )
            state_manager.finalize_session(
                summary=(f"Crashed due to exception: {type(err).__name__}: {err}"),
                success=False,
            )

        return state_manager

    def _register_event_logger(self) -> None:
        """Registers a global listener on the event bus to log all system events."""

        def log_event(event: Event) -> None:
            logging.info(f"[EVENT] {event.name} | Payload: {event.payload}")

        self._dispatcher.subscribe("*", log_event)
