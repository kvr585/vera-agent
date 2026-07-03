"""Unit tests for the StateManager."""

from collections.abc import Sequence

from vera_engine.core.entities import AgentState, Capability, Event, Task, TaskStatus
from vera_engine.runtime.events import EventDispatcher
from vera_engine.runtime.state import StateManager


class InMemoryRepository:
    """Mock repository double for unit testing state persistence."""

    def __init__(self) -> None:
        self.states: dict[str, AgentState] = {}

    def save_state(self, state: AgentState) -> None:
        self.states[state.session_id] = state.model_copy(deep=True)

    def get_state(self, session_id: str) -> AgentState | None:
        return self.states.get(session_id)

    def list_sessions(self) -> Sequence[AgentState]:
        return list(self.states.values())


def test_state_manager_initialization() -> None:
    """Verifies that the StateManager initializes state, persists, and dispatches."""
    repo = InMemoryRepository()
    dispatcher = EventDispatcher()
    received_events: list[Event] = []

    def listener(event: Event) -> None:
        received_events.append(event)

    dispatcher.subscribe("SessionStarted", listener)

    cap = Capability(name="Filesystem", description="Disk read write")
    manager = StateManager(
        session_id="session-123",
        goal="Test VERA state",
        workflow="default",
        repository=repo,
        dispatcher=dispatcher,
        capabilities=[cap],
    )

    # Assert local state and repository state
    assert manager.state.session_id == "session-123"
    assert manager.state.goal == "Test VERA state"
    assert len(manager.state.capabilities) == 1
    assert manager.state.capabilities[0].name == "Filesystem"

    persisted_state = repo.get_state("session-123")
    assert persisted_state is not None
    assert persisted_state.goal == "Test VERA state"

    # Assert event was fired
    assert len(received_events) == 1
    assert received_events[0].name == "SessionStarted"
    assert received_events[0].payload["session_id"] == "session-123"


def test_state_manager_update_tasks() -> None:
    """Verifies task list updates persist and dispatch events."""
    repo = InMemoryRepository()
    dispatcher = EventDispatcher()
    received_events: list[Event] = []

    manager = StateManager(
        session_id="session-123",
        goal="Goal",
        workflow="default",
        repository=repo,
        dispatcher=dispatcher,
    )
    dispatcher.subscribe("PlanCreated", lambda e: received_events.append(e))

    tasks = [
        Task(id="task-1", description="Task 1"),
        Task(id="task-2", description="Task 2"),
    ]
    manager.update_tasks(tasks)

    # Verification
    assert len(manager.state.tasks) == 2
    assert manager.state.tasks[0].id == "task-1"

    persisted = repo.get_state("session-123")
    assert persisted is not None
    assert len(persisted.tasks) == 2

    assert len(received_events) == 1
    assert received_events[0].name == "PlanCreated"
    assert received_events[0].payload["tasks_count"] == 2


def test_state_manager_task_lifecycle() -> None:
    """Verifies starting, completing, and failing tasks mutate state properly."""
    repo = InMemoryRepository()
    dispatcher = EventDispatcher()
    received_events: list[Event] = []

    manager = StateManager(
        session_id="session-123",
        goal="Goal",
        workflow="default",
        repository=repo,
        dispatcher=dispatcher,
    )

    dispatcher.subscribe("*", lambda e: received_events.append(e))

    tasks = [
        Task(id="task-1", description="Task 1"),
        Task(id="task-2", description="Task 2"),
    ]
    manager.update_tasks(tasks)
    received_events.clear()

    # 1. Start Task 1
    manager.start_task("task-1")
    assert manager.state.current_task_id == "task-1"
    assert manager.state.tasks[0].status == TaskStatus.IN_PROGRESS
    assert repo.get_state("session-123").tasks[0].status == TaskStatus.IN_PROGRESS
    assert len(received_events) == 1
    assert received_events[0].name == "TaskStarted"
    assert received_events[0].payload["task_id"] == "task-1"
    received_events.clear()

    # 2. Complete Task 1
    manager.complete_task("task-1", "Task 1 output")
    assert manager.state.current_task_id is None
    assert manager.state.tasks[0].status == TaskStatus.COMPLETED
    assert manager.state.tasks[0].execution_result == "Task 1 output"
    assert len(received_events) == 1
    assert received_events[0].name == "TaskCompleted"
    received_events.clear()

    # 3. Start and Fail Task 2
    manager.start_task("task-2")
    manager.fail_task("task-2", "Fatal error occurred")
    assert manager.state.current_task_id is None
    assert manager.state.tasks[1].status == TaskStatus.FAILED
    assert manager.state.tasks[1].execution_result == "Fatal error occurred"
    assert received_events[-1].name == "TaskFailed"
    assert received_events[-1].payload["error"] == "Fatal error occurred"


def test_state_manager_tool_and_finalize() -> None:
    """Verifies that active tools and finalizing the session update states correctly."""
    repo = InMemoryRepository()
    dispatcher = EventDispatcher()
    manager = StateManager(
        session_id="session-123",
        goal="Goal",
        workflow="default",
        repository=repo,
        dispatcher=dispatcher,
    )

    # Tool setting
    manager.set_running_tool("read_file")
    assert manager.state.running_tool == "read_file"

    manager.set_running_tool(None)
    assert manager.state.running_tool is None

    # Finalization
    manager.finalize_session(summary="Workflow completed successfully", success=True)
    assert manager.state.is_completed is True
    assert manager.state.success is True
    assert manager.state.summary == "Workflow completed successfully"
