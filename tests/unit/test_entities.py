"""Unit tests for core domain entities."""

from datetime import datetime

from vera_engine.core.entities import (
    AgentState,
    Capability,
    Event,
    Task,
    TaskStatus,
    WorkingMemory,
)


def test_task_status_enum() -> None:
    """Verifies that TaskStatus Enum behaves as a string and has correct values."""
    assert TaskStatus.PENDING == "pending"
    assert TaskStatus.IN_PROGRESS == "in_progress"
    assert TaskStatus.COMPLETED == "completed"
    assert TaskStatus.FAILED == "failed"


def test_capability_creation() -> None:
    """Verifies Capability initialization and default values."""
    cap = Capability(
        name="Filesystem Access",
        description="Allows reading and writing files",
        associated_tools=["read_file", "write_file"],
    )
    assert cap.name == "Filesystem Access"
    assert cap.description == "Allows reading and writing files"
    assert cap.associated_tools == ["read_file", "write_file"]


def test_task_default_values() -> None:
    """Verifies Task model initialization, schemas, and default values."""
    task = Task(id="task-1", description="Read configuration file")
    assert task.id == "task-1"
    assert task.description == "Read configuration file"
    assert task.status == TaskStatus.PENDING
    assert task.dependencies == []
    assert task.assigned_tool is None
    assert task.execution_result is None
    assert task.retry_count == 0


def test_agent_state_nested_validation() -> None:
    """Verifies AgentState correctly structures and nests Tasks and Capabilities."""
    task = Task(id="task-1", description="Read workspace config")
    cap = Capability(name="Filesystem", description="Disk access")
    state = AgentState(
        session_id="session-xyz",
        goal="Initialize the project",
        current_workflow="default",
        tasks=[task],
        capabilities=[cap],
    )

    assert state.session_id == "session-xyz"
    assert len(state.tasks) == 1
    assert state.tasks[0].id == "task-1"
    assert len(state.capabilities) == 1
    assert state.capabilities[0].name == "Filesystem"
    assert isinstance(state.created_at, datetime)
    assert isinstance(state.updated_at, datetime)
    assert not state.is_completed
    assert not state.success


def test_working_memory_reset() -> None:
    """Verifies that WorkingMemory correctly adds and clears operational contexts."""
    mem = WorkingMemory()
    mem.notes.append("Found local config file at workspace/config.yaml")
    mem.active_errors.append("Connection timed out to Ollama")

    assert len(mem.notes) == 1
    assert len(mem.active_errors) == 1

    mem.clear()
    assert len(mem.notes) == 0
    assert len(mem.active_errors) == 0


def test_event_initialization() -> None:
    """Verifies that Event structure holds names and payloads correctly."""
    evt = Event(name="StateUpdated", payload={"session_id": "123", "step": 4})
    assert evt.name == "StateUpdated"
    assert evt.payload == {"session_id": "123", "step": 4}
    assert isinstance(evt.timestamp, datetime)
