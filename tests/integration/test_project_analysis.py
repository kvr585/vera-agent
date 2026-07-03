"""Integration tests for the VERA Project Analysis capabilities."""

from pathlib import Path

from pydantic import BaseModel

from vera_engine.adapters.tools.local.filesystem import (
    ListDirectoryTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
)
from vera_engine.adapters.tools.registry import LocalToolRegistry
from vera_engine.core.entities import AgentState
from vera_engine.runtime.events import EventDispatcher
from vera_engine.runtime.services.discovery import DiscoveryService
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


class InMemoryRepository:
    """Mock repository in memory for integration testing."""

    def __init__(self) -> None:
        self.state: AgentState | None = None

    def save_state(self, state: AgentState) -> None:
        self.state = state.model_copy(deep=True)

    def get_state(self, session_id: str) -> AgentState | None:
        return self.state

    def list_sessions(self) -> list[AgentState]:
        return [self.state] if self.state else []


class MockLLM:
    """Mock LLM provider returning pre-determined structured outputs."""

    def __init__(self) -> None:
        self.raw_output = "Mock Summary"
        self.structured_outputs: list[BaseModel] = []
        self.call_count = 0

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_mode: bool = False,
    ) -> str:
        return self.raw_output

    def generate_structured(
        self,
        prompt: str,
        response_model: type,
        system_prompt: str | None = None,
    ) -> BaseModel:
        self.call_count += 1
        if self.structured_outputs:
            return self.structured_outputs.pop(0)
        return response_model.model_validate({})


class QuickPrompts:
    """Mock prompts manager returning dummy structures."""

    def render(self, workflow: str, category: str, **variables: any) -> any:
        class DummyPrompt:
            system = "Sys"
            user = "User"

        return DummyPrompt()


def test_integration_project_analysis_workflow(tmp_path: Path) -> None:
    """Verifies end-to-end integration workflow with discovery and filesystem tools."""
    # 1. Setup workspace structure
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "main.py").write_text(
        "import sys\n# Main entry point", encoding="utf-8"
    )
    (workspace / "README.md").write_text("# Project VERA", encoding="utf-8")
    (workspace / "package.json").write_text('{"name": "test"}', encoding="utf-8")
    (workspace / "config.yaml").write_text("debug: true", encoding="utf-8")

    # 2. Setup tools
    list_tool = ListDirectoryTool(workspace_dir=str(workspace))
    search_tool = SearchFilesTool(workspace_dir=str(workspace))
    read_tool = ReadFileTool(workspace_dir=str(workspace))
    write_tool = WriteFileTool(workspace_dir=str(workspace))

    registry = LocalToolRegistry()
    registry.register(list_tool)
    registry.register(search_tool)
    registry.register(read_tool)
    registry.register(write_tool)

    # 3. Setup LLM, Prompts, Services
    llm = MockLLM()
    prompts = QuickPrompts()

    # Prepopulate LLM structured outputs
    tasks = [
        TaskPlanItem(id="t1", description="Discover layout"),
        TaskPlanItem(id="t2", description="Analyze files"),
        TaskPlanItem(id="t3", description="Write summary report"),
    ]
    # Task 1: Execute tool -> write status
    action_1 = ActionDecision(
        thought="Use write_file to save initial status",
        action_type="execute_tool",
        tool_name="write_file",
        tool_args={"path": "status.txt", "content": "Analyzing..."},
    )
    obs_1 = ObservationEvaluation(
        thought="Task 1 tool executed successfully",
        task_completed=False,
        observation_summary="Wrote status.txt successfully",
    )
    action_1_finish = ActionDecision(
        thought="Task 1 finished",
        action_type="finish",
        summary="Finished Task 1",
    )
    # Task 2: Execute tool -> read config
    action_2 = ActionDecision(
        thought="Read configuration",
        action_type="execute_tool",
        tool_name="read_file",
        tool_args={"path": "config.yaml"},
    )
    obs_2 = ObservationEvaluation(
        thought="Task 2 tool executed successfully",
        task_completed=False,
        observation_summary="Read config successfully",
    )
    action_2_finish = ActionDecision(
        thought="Task 2 finished",
        action_type="finish",
        summary="Finished Task 2",
    )
    # Task 3: Execute tool -> write report
    action_3 = ActionDecision(
        thought="Write final analysis report",
        action_type="execute_tool",
        tool_name="write_file",
        tool_args={
            "path": "reports/analysis.md",
            "content": "# Project Analysis Report",
        },
    )
    obs_3 = ObservationEvaluation(
        thought="Task 3 tool executed successfully",
        task_completed=False,
        observation_summary="Report saved successfully",
    )
    action_3_finish = ActionDecision(
        thought="All tasks completed, finishing goal",
        action_type="finish",
        summary="Workflow successfully completed and report written",
    )

    llm.structured_outputs = [
        TaskPlan(tasks=tasks),
        action_1,
        obs_1,
        action_1_finish,
        action_2,
        obs_2,
        action_2_finish,
        action_3,
        obs_3,
        action_3_finish,
    ]

    planner = PlanningService(llm, prompts)
    reasoner = ReasoningService(llm, prompts)
    executor = ExecutionService(registry)
    observer = ObservationService(llm, prompts)
    discovery = DiscoveryService(list_tool, search_tool, read_tool)

    workflow = DefaultWorkflow(
        planning_service=planner,
        reasoning_service=reasoner,
        execution_service=executor,
        observation_service=observer,
        llm=llm,
        prompt_manager=prompts,
        tool_registry=registry,
        discovery_service=discovery,
        max_steps=20,
    )

    # 4. Execute workflow
    repo = InMemoryRepository()
    dispatcher = EventDispatcher()
    state_mgr = StateManager(
        "session-analysis-1",
        "Analyze workspace and write report",
        "default",
        repo,
        dispatcher,
    )

    workflow.execute(state_mgr)

    # 5. Verify results
    assert state_mgr.state.is_completed is True
    assert state_mgr.state.success is True
    assert (
        state_mgr.state.summary == "Workflow successfully completed and report written"
    )

    # Verify discovery metadata was populated
    meta = state_mgr.state.project_metadata
    assert meta is not None
    assert meta.root_path == str(workspace)
    assert "Python" in meta.detected_languages
    assert "Node.js NPM" in meta.detected_frameworks
    assert "main.py" in meta.entry_points
    assert "package.json" in meta.important_files

    # Verify files created in the workspace
    assert (workspace / "status.txt").exists()
    assert (workspace / "status.txt").read_text(encoding="utf-8") == "Analyzing..."
    assert (workspace / "reports" / "analysis.md").exists()
    assert (workspace / "reports" / "analysis.md").read_text(
        encoding="utf-8"
    ) == "# Project Analysis Report"
