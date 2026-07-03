"""Unit tests for VERA orchestration services.

Covers Planning, Reasoning, Execution, and Observation.
"""

from typing import Any, TypeVar

from pydantic import BaseModel

from vera_engine.adapters.tools.registry import LocalToolRegistry
from vera_engine.core.entities import Capability, Task, TaskStatus
from vera_engine.core.interfaces.tool import ToolResult
from vera_engine.runtime.prompt import PromptManager, RenderedPrompt
from vera_engine.runtime.services.execution import ExecutionService
from vera_engine.runtime.services.observation import (
    ObservationEvaluation,
    ObservationService,
)
from vera_engine.runtime.services.planning import (
    PlanningService,
    TaskPlan,
    TaskPlanItem,
)
from vera_engine.runtime.services.reasoning import ActionDecision, ReasoningService

T = TypeVar("T", bound=BaseModel)


# --- Mock Implementations ---


class MockLLMProvider:
    """Mock LLMProvider implementing generation ports for testing."""

    def __init__(self) -> None:
        self.last_prompt: str | None = None
        self.last_system_prompt: str | None = None
        self.json_mode: bool = False
        self.raw_response = "Mock raw response"
        self.structured_response: BaseModel | None = None

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_mode: bool = False,
    ) -> str:
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        self.json_mode = json_mode
        return self.raw_response

    def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
    ) -> T:
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        if self.structured_response is None:
            # Fallback if no specific stubbed response
            return response_model.model_validate({})
        return self.structured_response  # type: ignore[return-value]


class MockTool:
    """Mock tool class implementing Tool interface."""

    def __init__(self, name: str = "mock_tool", success: bool = True) -> None:
        self._name = name
        self._success = success
        self.executed = False
        self.last_args: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Mock tool description"

    @property
    def input_schema(self) -> type[BaseModel]:
        class Schema(BaseModel):
            input_val: str

        return Schema

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        self.executed = True
        self.last_args = arguments
        if self._success:
            return ToolResult(success=True, output="Tool executed successfully")
        return ToolResult(success=False, output="", error="Tool execution failed")


class DummyPromptManager(PromptManager):
    """Dummy prompt manager returning preset prompts to bypass file reads."""

    def __init__(self) -> None:
        pass

    def render(self, workflow: str, category: str, **variables: Any) -> RenderedPrompt:
        return RenderedPrompt(
            system=f"System template for {category}",
            user=f"User template for {category} with variables {variables}",
        )


# --- Test Cases ---


def test_planning_service() -> None:
    """Verifies that PlanningService generates Tasks from structured outputs."""
    llm = MockLLMProvider()
    prompts = DummyPromptManager()

    # Stub planning response
    llm.structured_response = TaskPlan(
        tasks=[
            TaskPlanItem(id="task-1", description="Read code file", dependencies=[]),
            TaskPlanItem(
                id="task-2", description="Analyze logic", dependencies=["task-1"]
            ),
        ]
    )

    service = PlanningService(llm, prompts)
    tasks = service.generate_plan(
        goal="Analyze repo",
        capabilities=[
            Capability(
                name="FileAccess", description="Read disks", associated_tools=["read"]
            )
        ],
    )

    assert len(tasks) == 2
    assert tasks[0].id == "task-1"
    assert tasks[0].status == TaskStatus.PENDING
    assert tasks[1].dependencies == ["task-1"]
    assert "Analyze repo" in llm.last_prompt


def test_reasoning_service() -> None:
    """Verifies that ReasoningService decides actions correctly."""
    llm = MockLLMProvider()
    prompts = DummyPromptManager()

    llm.structured_response = ActionDecision(
        thought="I should execute read_file",
        action_type="execute_tool",
        tool_name="read_file",
        tool_args={"path": "workspace/test.txt"},
    )

    service = ReasoningService(llm, prompts)

    from vera_engine.core.entities import AgentState

    state = AgentState(
        session_id="s1", goal="Test reasoning", current_workflow="default"
    )

    decision = service.decide_action(
        state=state,
        history=[{"role": "user", "content": "Query"}],
        available_tools=[MockTool("read_file")],
    )

    assert decision.action_type == "execute_tool"
    assert decision.tool_name == "read_file"
    assert decision.tool_args == {"path": "workspace/test.txt"}
    assert "Test reasoning" in llm.last_prompt


def test_execution_service() -> None:
    """Verifies that ExecutionService resolves and executes registry tools safely."""
    registry = LocalToolRegistry()
    mock_tool = MockTool(name="my_tool", success=True)
    registry.register(mock_tool)

    service = ExecutionService(registry)
    result = service.execute_tool("my_tool", {"input_val": "test"})

    assert result.success is True
    assert result.output == "Tool executed successfully"
    assert mock_tool.executed is True
    assert mock_tool.last_args == {"input_val": "test"}

    # Verification of non-existent tool error handling
    result_missing = service.execute_tool("missing_tool", {})
    assert result_missing.success is False
    assert "not registered" in result_missing.error


def test_observation_service() -> None:
    """Verifies that ObservationService evaluates task outcomes correctly."""
    llm = MockLLMProvider()
    prompts = DummyPromptManager()

    llm.structured_response = ObservationEvaluation(
        thought="The file was read successfully, content matches expectations.",
        task_completed=True,
        observation_summary="Configuration contains correct database parameters.",
        new_notes=["Database URL matches local settings."],
    )

    service = ObservationService(llm, prompts)
    task = Task(id="t-1", description="Read configuration")
    result = ToolResult(success=True, output="db_url: sqlite:///logs/vera.db")

    eval_result = service.evaluate_result(
        goal="Verify configs",
        task=task,
        tool_name="read_file",
        tool_args={"path": "config.yaml"},
        result=result,
    )

    assert eval_result.task_completed is True
    assert "Verify configs" in llm.last_prompt
    assert (
        eval_result.observation_summary
        == "Configuration contains correct database parameters."
    )
    assert eval_result.new_notes == ["Database URL matches local settings."]


def test_execution_service_tool_exception() -> None:
    """Verifies that ExecutionService captures and wraps exceptions raised by tools."""

    class CrashingTool:
        @property
        def name(self) -> str:
            return "crashing_tool"

        @property
        def description(self) -> str:
            return "Crashes on execution"

        @property
        def input_schema(self) -> type[BaseModel]:
            class EmptySchema(BaseModel):
                pass

            return EmptySchema

        def execute(self, arguments: dict[str, Any]) -> ToolResult:
            raise RuntimeError("Tool logic encountered fatal exception")

    registry = LocalToolRegistry()
    registry.register(CrashingTool())

    service = ExecutionService(registry)
    result = service.execute_tool("crashing_tool", {})
    assert result.success is False
    assert "RuntimeError" in result.error


def test_execution_service_tool_returning_failure() -> None:
    """Verifies that ExecutionService handles tool failures correctly."""
    registry = LocalToolRegistry()
    failed_tool = MockTool(name="failed_tool", success=False)
    registry.register(failed_tool)

    service = ExecutionService(registry)
    result = service.execute_tool("failed_tool", {"input_val": "test"})
    assert result.success is False
    assert result.error == "Tool execution failed"


def test_reasoning_service_with_active_task() -> None:
    """Verifies ReasoningService renders active task details."""
    llm = MockLLMProvider()
    prompts = DummyPromptManager()

    llm.structured_response = ActionDecision(
        thought="Continuing with active task",
        action_type="finish",
        summary="Done",
    )

    service = ReasoningService(llm, prompts)

    from vera_engine.core.entities import AgentState

    state = AgentState(
        session_id="s1",
        goal="Task Goal",
        current_workflow="default",
        tasks=[
            Task(id="task-1", description="Main work"),
        ],
        current_task_id="task-1",
    )

    service.decide_action(
        state=state,
        history=[],
        available_tools=[],
    )

    # Active task should be formatted in the prompt user variables
    assert "ID: task-1 - Main work" in llm.last_prompt
