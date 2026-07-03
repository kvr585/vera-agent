"""Core domain entities for the VERA Agent Engine."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(StrEnum):
    """Execution status of a specific task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Capability(BaseModel):
    """High-level abstraction mapping logical abilities to one or more system tools."""

    name: str
    description: str
    associated_tools: list[str] = Field(default_factory=list)


class Task(BaseModel):
    """A granular unit of work decomposed from the main goal."""

    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    dependencies: list[str] = Field(default_factory=list)
    assigned_tool: str | None = None
    execution_result: str | None = None
    retry_count: int = 0


class ProjectMetadata(BaseModel):
    """Structural facts discovered about the analyzed codebase/workspace."""

    root_path: str
    directory_tree: str
    detected_languages: list[str] = Field(default_factory=list)
    detected_frameworks: list[str] = Field(default_factory=list)
    important_files: list[str] = Field(default_factory=list)
    entry_points: list[str] = Field(default_factory=list)
    documentation_files: list[str] = Field(default_factory=list)
    file_statistics: dict[str, Any] = Field(default_factory=dict)


class AgentState(BaseModel):
    """The central state registry representing the active status of an agent session."""

    session_id: str
    goal: str
    current_workflow: str
    tasks: list[Task] = Field(default_factory=list)
    capabilities: list[Capability] = Field(default_factory=list)
    current_task_id: str | None = None
    running_tool: str | None = None
    project_metadata: ProjectMetadata | None = None
    is_completed: bool = False
    success: bool = False
    summary: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class WorkingMemory(BaseModel):
    """Short-term operational memory used inside task execution iterations."""

    notes: list[str] = Field(default_factory=list)
    active_errors: list[str] = Field(default_factory=list)

    def clear(self) -> None:
        """Resets the working memory notes and active errors."""
        self.notes.clear()
        self.active_errors.clear()


class SessionMemory(BaseModel):
    """Interaction history and context accumulated throughout the active session."""

    history: list[dict[str, Any]] = Field(default_factory=list)


class Event(BaseModel):
    """A system event broadcasted to notify about internal state changes."""

    name: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
