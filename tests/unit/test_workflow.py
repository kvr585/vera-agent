"""Unit tests for the VERA DefaultWorkflow."""

from pydantic import BaseModel

from vera_engine.adapters.tools.registry import LocalToolRegistry
from vera_engine.core.entities import AgentState, TaskStatus
from vera_engine.core.interfaces.tool import ToolResult
from vera_engine.runtime.events import EventDispatcher
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
from vera_engine.runtime.state import StateManager
from vera_engine.runtime.workflows.default import DefaultWorkflow

# --- Doubles and Mocking helpers ---


class InMemoryRepository:
    def __init__(self) -> None:
        self.state: AgentState | None = None

    def save_state(self, state: AgentState) -> None:
        self.state = state.model_copy(deep=True)

    def get_state(self, session_id: str) -> AgentState | None:
        return self.state

    def list_sessions(self) -> list[AgentState]:
        return [self.state] if self.state else []


class MockLLM:
    def __init__(self) -> None:
        self.raw_output = "Mock Summary"
        self.structured_outputs: list[BaseModel] = []
        self.call_count = 0

    def generate(
        self, prompt: str, system_prompt: str | None = None, json_mode: bool = False
    ) -> str:
        return self.raw_output

    def generate_structured(
        self, prompt: str, response_model: type, system_prompt: str | None = None
    ) -> BaseModel:
        self.call_count += 1
        if self.structured_outputs:
            return self.structured_outputs.pop(0)
        return response_model.model_validate({})


class MockTool:
    @property
    def name(self) -> str:
        return "mock_tool"

    @property
    def description(self) -> str:
        return "Test tool"

    @property
    def input_schema(self) -> type[BaseModel]:
        class Schema(BaseModel):
            pass

        return Schema

    def execute(self, arguments: dict) -> ToolResult:
        return ToolResult(success=True, output="Executed")


class QuickPrompts:
    def render(self, workflow: str, category: str, **variables: any) -> any:
        class DummyPrompt:
            system = "Sys"
            user = "User"

        return DummyPrompt()


# --- Test Cases ---


def test_default_workflow_successful_run() -> None:
    """Verifies complete, successful execution of VERA's default workflow loop."""
    repo = InMemoryRepository()
    dispatcher = EventDispatcher()
    llm = MockLLM()
    prompts = QuickPrompts()

    # 1. Stub LLM structured outputs sequentially:
    # First: Decompose goal into 1 task
    # Second: Decides to run tool mock_tool for active task
    # Third: Observes tool execution and concludes task complete
    llm.structured_outputs = [
        TaskPlan(tasks=[TaskPlanItem(id="t1", description="Work", dependencies=[])]),
        ActionDecision(
            thought="Run tool",
            action_type="execute_tool",
            tool_name="mock_tool",
            tool_args={},
        ),
        ObservationEvaluation(
            thought="Success",
            task_completed=True,
            observation_summary="Job done",
        ),
    ]

    registry = LocalToolRegistry()
    registry.register(MockTool())

    planner = PlanningService(llm, prompts)
    reasoner = ReasoningService(llm, prompts)
    executor = ExecutionService(registry)
    observer = ObservationService(llm, prompts)

    workflow = DefaultWorkflow(
        planning_service=planner,
        reasoning_service=reasoner,
        execution_service=executor,
        observation_service=observer,
        llm=llm,
        prompt_manager=prompts,
        tool_registry=registry,
    )

    state_mgr = StateManager(
        session_id="s1",
        goal="Run a test",
        workflow="default",
        repository=repo,
        dispatcher=dispatcher,
    )

    workflow.execute(state_mgr)

    # Verifications
    assert workflow.name == "default"
    assert state_mgr.state.is_completed is True
    assert state_mgr.state.success is True
    assert state_mgr.state.summary == "Mock Summary"
    assert state_mgr.state.tasks[0].status == TaskStatus.COMPLETED


def test_default_workflow_planning_failure() -> None:
    """Verifies that planning phase exceptions abort execution immediately."""
    repo = InMemoryRepository()
    dispatcher = EventDispatcher()
    llm = MockLLM()
    prompts = QuickPrompts()

    # Stub Planning to raise exception (LLM fails structured output)
    def fail_call(*args, **kwargs):
        raise RuntimeError("LLM Unavailable")

    llm.generate_structured = fail_call

    planner = PlanningService(llm, prompts)
    reasoner = ReasoningService(llm, prompts)
    executor = ExecutionService(LocalToolRegistry())
    observer = ObservationService(llm, prompts)

    workflow = DefaultWorkflow(
        planner, reasoner, executor, observer, llm, prompts, LocalToolRegistry()
    )
    state_mgr = StateManager("s2", "Goal", "default", repo, dispatcher)

    workflow.execute(state_mgr)

    assert state_mgr.state.is_completed is True
    assert state_mgr.state.success is False
    assert "Planning phase failed" in state_mgr.state.summary


def test_default_workflow_reasoning_failure() -> None:
    """Verifies reasoning failure aborts execution and marks task status as failed."""
    repo = InMemoryRepository()
    dispatcher = EventDispatcher()
    llm = MockLLM()
    prompts = QuickPrompts()

    llm.structured_outputs = [
        TaskPlan(tasks=[TaskPlanItem(id="t1", description="Work")]),
    ]
    planner = PlanningService(llm, prompts)
    reasoner = ReasoningService(llm, prompts)
    executor = ExecutionService(LocalToolRegistry())
    observer = ObservationService(llm, prompts)

    # Force reasoning exception
    def fail_reasoning(*args, **kwargs):
        raise ValueError("Reasoning syntax error")

    llm.generate_structured = lambda prompt, response_model, **kwargs: (
        llm.structured_outputs.pop(0) if llm.structured_outputs else fail_reasoning()
    )

    workflow = DefaultWorkflow(
        planner, reasoner, executor, observer, llm, prompts, LocalToolRegistry()
    )
    state_mgr = StateManager("s3", "Goal", "default", repo, dispatcher)

    workflow.execute(state_mgr)

    assert state_mgr.state.is_completed is True
    assert state_mgr.state.success is False
    assert state_mgr.state.tasks[0].status == TaskStatus.FAILED
    assert "Reasoning loop exception" in state_mgr.state.summary


def test_default_workflow_finish_and_clarification() -> None:
    """Verifies that direct finishes and clarification requests are handled."""
    repo = InMemoryRepository()
    dispatcher = EventDispatcher()

    # Test case 1: Finish action
    llm = MockLLM()
    prompts = QuickPrompts()
    llm.structured_outputs = [
        TaskPlan(tasks=[TaskPlanItem(id="t1", description="Work")]),
        ActionDecision(thought="Done", action_type="finish", summary="Task resolved"),
    ]
    planner = PlanningService(llm, prompts)
    reasoner = ReasoningService(llm, prompts)
    executor = ExecutionService(LocalToolRegistry())
    observer = ObservationService(llm, prompts)

    workflow = DefaultWorkflow(
        planner, reasoner, executor, observer, llm, prompts, LocalToolRegistry()
    )
    state_mgr = StateManager("s4", "Goal", "default", repo, dispatcher)
    workflow.execute(state_mgr)

    assert state_mgr.state.success is True
    assert state_mgr.state.summary == "Task resolved"

    # Test case 2: Clarification request
    llm2 = MockLLM()
    llm2.structured_outputs = [
        TaskPlan(tasks=[TaskPlanItem(id="t1", description="Work")]),
        ActionDecision(
            thought="Need help",
            action_type="request_clarification",
            clarification_query="What database port?",
        ),
    ]
    planner2 = PlanningService(llm2, prompts)
    reasoner2 = ReasoningService(llm2, prompts)
    workflow2 = DefaultWorkflow(
        planner2, reasoner2, executor, observer, llm2, prompts, LocalToolRegistry()
    )
    state_mgr2 = StateManager("s5", "Goal", "default", repo, dispatcher)
    workflow2.execute(state_mgr2)

    assert state_mgr2.state.success is False
    assert "What database port?" in state_mgr2.state.summary


def test_default_workflow_max_steps() -> None:
    """Verifies safety limit triggers when execution loop step threshold is hit."""
    repo = InMemoryRepository()
    dispatcher = EventDispatcher()
    llm = MockLLM()
    prompts = QuickPrompts()

    # Infinite loop planning: 1 task. Reasoning constantly returns
    # execute_tool, but observer returns task_completed = False.
    llm.structured_outputs = [
        TaskPlan(tasks=[TaskPlanItem(id="t1", description="Work")]),
    ]

    # Set default generator to loop execute_tool and incomplete observations
    def infinite_loop_gen(prompt, response_model, **kwargs):
        if llm.structured_outputs:
            return llm.structured_outputs.pop(0)
        if response_model == ActionDecision:
            return ActionDecision(
                thought="Looping",
                action_type="execute_tool",
                tool_name="mock_tool",
            )
        return ObservationEvaluation(
            thought="Not done",
            task_completed=False,
            observation_summary="Ongoing",
        )

    llm.generate_structured = infinite_loop_gen

    registry = LocalToolRegistry()
    registry.register(MockTool())

    planner = PlanningService(llm, prompts)
    reasoner = ReasoningService(llm, prompts)
    executor = ExecutionService(registry)
    observer = ObservationService(llm, prompts)

    # Run with max_steps = 3
    workflow = DefaultWorkflow(
        planner,
        reasoner,
        executor,
        observer,
        llm,
        prompts,
        registry,
        max_steps=3,
    )
    state_mgr = StateManager("s6", "Goal", "default", repo, dispatcher)

    workflow.execute(state_mgr)

    assert state_mgr.state.is_completed is True
    assert state_mgr.state.success is False
    assert "Exceeded maximum safety steps" in state_mgr.state.summary


def test_default_workflow_observation_crashed() -> None:
    """Verifies observation crash terminates workflow with failure."""
    repo = InMemoryRepository()
    dispatcher = EventDispatcher()
    llm = MockLLM()
    prompts = QuickPrompts()

    llm.structured_outputs = [
        TaskPlan(tasks=[TaskPlanItem(id="t1", description="Work")]),
        ActionDecision(
            thought="Run tool",
            action_type="execute_tool",
            tool_name="mock_tool",
        ),
    ]
    planner = PlanningService(llm, prompts)
    reasoner = ReasoningService(llm, prompts)
    executor = ExecutionService(LocalToolRegistry())
    registry = LocalToolRegistry()
    registry.register(MockTool())

    # Force observation exception
    observer = ObservationService(llm, prompts)

    def fail_obs(*args, **kwargs):
        raise RuntimeError("Observation logic crashed")

    observer.evaluate_result = fail_obs

    workflow = DefaultWorkflow(
        planner, reasoner, executor, observer, llm, prompts, registry
    )
    state_mgr = StateManager("s7", "Goal", "default", repo, dispatcher)
    workflow.execute(state_mgr)

    assert state_mgr.state.is_completed is True
    assert state_mgr.state.success is False
    assert "Observation logic crashed" in state_mgr.state.summary


def test_default_workflow_tool_missing_name() -> None:
    """Verifies execute_tool without tool_name terminates with failure."""
    repo = InMemoryRepository()
    dispatcher = EventDispatcher()
    llm = MockLLM()
    prompts = QuickPrompts()

    llm.structured_outputs = [
        TaskPlan(tasks=[TaskPlanItem(id="t1", description="Work")]),
        ActionDecision(thought="Run tool", action_type="execute_tool", tool_name=None),
    ]
    planner = PlanningService(llm, prompts)
    reasoner = ReasoningService(llm, prompts)
    executor = ExecutionService(LocalToolRegistry())
    observer = ObservationService(llm, prompts)

    workflow = DefaultWorkflow(
        planner, reasoner, executor, observer, llm, prompts, LocalToolRegistry()
    )
    state_mgr = StateManager("s8", "Goal", "default", repo, dispatcher)
    workflow.execute(state_mgr)

    assert state_mgr.state.is_completed is True
    assert state_mgr.state.success is False
    assert "tool_name was missing" in state_mgr.state.summary


def test_default_workflow_task_fails_on_observation_error() -> None:
    """Verifies that if observation evaluates execution error, task status is failed."""
    repo = InMemoryRepository()
    dispatcher = EventDispatcher()
    llm = MockLLM()
    prompts = QuickPrompts()

    llm.structured_outputs = [
        TaskPlan(tasks=[TaskPlanItem(id="t1", description="Work")]),
        ActionDecision(
            thought="Run tool",
            action_type="execute_tool",
            tool_name="mock_tool",
        ),
        ObservationEvaluation(
            thought="Fail",
            task_completed=False,
            observation_summary="Error",
            error_messages=["File not found"],
        ),
    ]
    planner = PlanningService(llm, prompts)
    reasoner = ReasoningService(llm, prompts)
    registry = LocalToolRegistry()
    registry.register(MockTool())
    executor = ExecutionService(registry)
    observer = ObservationService(llm, prompts)

    workflow = DefaultWorkflow(
        planner, reasoner, executor, observer, llm, prompts, registry
    )
    state_mgr = StateManager("s9", "Goal", "default", repo, dispatcher)
    workflow.execute(state_mgr)

    assert state_mgr.state.is_completed is True
    assert state_mgr.state.success is False
    assert state_mgr.state.tasks[0].status == TaskStatus.FAILED
    assert "Task execution errors" in state_mgr.state.summary


def test_default_workflow_final_summary_crashed() -> None:
    """Verifies that final summary LLM crash still terminates workflow with success."""
    repo = InMemoryRepository()
    dispatcher = EventDispatcher()
    llm = MockLLM()
    prompts = QuickPrompts()

    llm.structured_outputs = [
        TaskPlan(tasks=[]),  # Empty plan leads straight to final summary
    ]

    # Force raw generate call to fail
    def fail_gen(*args, **kwargs):
        raise RuntimeError("Summarizer network error")

    llm.generate = fail_gen

    planner = PlanningService(llm, prompts)
    reasoner = ReasoningService(llm, prompts)
    executor = ExecutionService(LocalToolRegistry())
    observer = ObservationService(llm, prompts)

    workflow = DefaultWorkflow(
        planner, reasoner, executor, observer, llm, prompts, LocalToolRegistry()
    )
    state_mgr = StateManager("s10", "Goal", "default", repo, dispatcher)
    workflow.execute(state_mgr)

    assert state_mgr.state.is_completed is True
    assert state_mgr.state.success is True
    assert "final summary generation failed" in state_mgr.state.summary.lower()


def test_local_memory_manager_all_methods() -> None:
    """Verifies memory getter and clear actions on LocalMemoryManager."""
    from vera_engine.runtime.memory import LocalMemoryManager

    mem = LocalMemoryManager()

    # 1. working memory
    mem.working_memory.notes.append("note")
    assert len(mem.working_memory.notes) == 1
    mem.clear_working_memory()
    assert len(mem.working_memory.notes) == 0

    # 2. session memory
    mem.add_history_entry("user", "Hello", timestamp="2026")
    assert len(mem.session_memory.history) == 1
    assert mem.session_memory.history[0]["timestamp"] == "2026"


def test_default_workflow_loop_break_when_completed() -> None:
    """Verifies that workflow loop breaks if state becomes completed mid-workflow."""
    repo = InMemoryRepository()
    dispatcher = EventDispatcher()
    llm = MockLLM()
    prompts = QuickPrompts()

    llm.structured_outputs = [
        TaskPlan(tasks=[TaskPlanItem(id="t1", description="Work")]),
    ]

    call_count = [0]

    def loop_gen(prompt, response_model, **kwargs):
        if llm.structured_outputs:
            return llm.structured_outputs.pop(0)
        call_count[0] += 1
        if call_count[0] == 2:
            state_mgr.finalize_session("Finished mid-loop", True)
        if response_model == ActionDecision:
            return ActionDecision(
                thought="Looping",
                action_type="execute_tool",
                tool_name="mock_tool",
            )
        return ObservationEvaluation(
            thought="Not done",
            task_completed=False,
            observation_summary="Ongoing",
        )

    llm.generate_structured = loop_gen
    registry = LocalToolRegistry()
    registry.register(MockTool())

    planner = PlanningService(llm, prompts)
    reasoner = ReasoningService(llm, prompts)
    executor = ExecutionService(registry)
    observer = ObservationService(llm, prompts)

    workflow = DefaultWorkflow(
        planner, reasoner, executor, observer, llm, prompts, registry
    )
    state_mgr = StateManager("s11", "Goal", "default", repo, dispatcher)
    workflow.execute(state_mgr)

    assert state_mgr.state.is_completed is True
    assert state_mgr.state.success is True
    assert state_mgr.state.summary == "Finished mid-loop"
