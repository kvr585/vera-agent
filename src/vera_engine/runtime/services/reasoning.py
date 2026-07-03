"""Reasoning service to analyze states and decide next action steps."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from vera_engine.core.entities import AgentState
from vera_engine.core.interfaces.llm import LLMProvider
from vera_engine.core.interfaces.tool import Tool
from vera_engine.runtime.prompt import PromptManager


def format_project_metadata(metadata: Any | None) -> str:
    """Formats ProjectMetadata into a human-readable prompt string."""
    if not metadata:
        return "No project metadata available."

    languages = ", ".join(metadata.detected_languages) or "None"
    frameworks = ", ".join(metadata.detected_frameworks) or "None"
    important = "\n".join(f"- {f}" for f in metadata.important_files) or "None"
    entries = "\n".join(f"- {f}" for f in metadata.entry_points) or "None"
    docs = "\n".join(f"- {f}" for f in metadata.documentation_files) or "None"

    return (
        f"Root Path: {metadata.root_path}\n"
        f"Detected Languages: {languages}\n"
        f"Detected Frameworks: {frameworks}\n"
        f"Important Configuration Files:\n{important}\n"
        f"Entry Points:\n{entries}\n"
        f"Documentation Files:\n{docs}\n"
        f"File Statistics: {metadata.file_statistics}\n"
        f"Workspace Directory Tree:\n{metadata.directory_tree}"
    )


class ActionDecision(BaseModel):
    """Pydantic model representing the LLM's next action decision."""

    thought: str = Field(description="Step-by-step reasoning thought process")
    action_type: Literal["execute_tool", "finish", "request_clarification"] = Field(
        description="The logical next step action type."
    )
    tool_name: str | None = Field(
        None, description="Name of the tool to run, if action_type is 'execute_tool'"
    )
    tool_args: dict[str, Any] | None = Field(
        None,
        description="JSON dictionary of tool inputs, if action_type is 'execute_tool'",
    )
    clarification_query: str | None = Field(
        None,
        description=(
            "Question to ask the user, if action_type is 'request_clarification'"
        ),
    )
    summary: str | None = Field(
        None, description="Final summary text, if action_type is 'finish'"
    )


class ReasoningService:
    """Invokes LLM reasoning to determine tool selection, parameters, or termination."""

    def __init__(self, llm: LLMProvider, prompt_manager: PromptManager) -> None:
        """Initializes the reasoning service.

        Args:
            llm: The LLM model provider.
            prompt_manager: The templates rendering manager.
        """
        self._llm = llm
        self._prompt_manager = prompt_manager

    def decide_action(
        self,
        state: AgentState,
        history: list[dict[str, Any]],
        available_tools: list[Tool],
    ) -> ActionDecision:
        """Analyzes active agent states and returns a structured action decision.

        Args:
            state: The current AgentState snapshot.
            history: Thread history list of inputs, actions, and observations.
            available_tools: The list of tools available in the registry.

        Returns:
            An ActionDecision model.
        """
        # Format task list for prompt context
        tasks_list = []
        for task in state.tasks:
            tasks_list.append(
                f"- [{task.status.value.upper()}] ID: {task.id} - "
                f"{task.description} (Result: {task.execution_result})"
            )
        tasks_context = "\n".join(tasks_list)

        # Format tool descriptions and schemas
        tools_list = []
        for tool in available_tools:
            schema_json = tool.input_schema.model_json_schema()
            tools_list.append(
                f"- Name: {tool.name}\n"
                f"  Description: {tool.description}\n"
                f"  Arguments Schema: {schema_json}"
            )
        tools_context = "\n\n".join(tools_list)

        # Format history context
        history_list = []
        for entry in history:
            role = entry.get("role", "system").upper()
            content = entry.get("content", "")
            history_list.append(f"[{role}]: {content}")
        history_context = "\n".join(history_list)

        # Retrieve active task description
        active_task_desc = "None (No active task started)"
        if state.current_task_id:
            for task in state.tasks:
                if task.id == state.current_task_id:
                    active_task_desc = f"ID: {task.id} - {task.description}"
                    break

        # Render prompt template
        prompt = self._prompt_manager.render(
            workflow="default",
            category="reasoning",
            goal=state.goal,
            active_task=active_task_desc,
            tasks=tasks_context,
            tools=tools_context,
            history=history_context,
            project_metadata=format_project_metadata(state.project_metadata),
        )

        # Generate structured decision
        return self._llm.generate_structured(
            prompt=prompt.user,
            response_model=ActionDecision,
            system_prompt=prompt.system,
        )
