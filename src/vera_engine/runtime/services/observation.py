"""Observation service to evaluate tool results and check task progress."""

from typing import Any

from pydantic import BaseModel, Field, field_validator

from vera_engine.core.entities import Task
from vera_engine.core.interfaces.llm import LLMProvider
from vera_engine.core.interfaces.tool import ToolResult
from vera_engine.runtime.prompt import PromptManager


class ObservationEvaluation(BaseModel):
    """Pydantic model representing the LLM's evaluation of a tool's execution result."""

    thought: str = Field(
        description="Analysis of the tool output against the task goal"
    )
    task_completed: bool = Field(
        description="True if the tool output indicates the task is fully completed."
    )
    observation_summary: str = Field(
        description="Clean, concise summary of what was observed from the tool result"
    )
    new_notes: list[str] = Field(
        default_factory=list,
        description="Key findings to store in working memory.",
    )
    error_messages: list[str] = Field(
        default_factory=list,
        description="Fatal errors or warning messages detected in the execution.",
    )

    @field_validator("new_notes", "error_messages", mode="before")
    @classmethod
    def convert_null_to_list(cls, v: Any) -> list[str]:
        """Converts null/None inputs to empty lists."""
        if v is None:
            return []
        return v


class ObservationService:
    """Orchestrates LLM evaluation of tool results to verify task progress."""

    def __init__(self, llm: LLMProvider, prompt_manager: PromptManager) -> None:
        """Initializes the observation service.

        Args:
            llm: The LLM model provider.
            prompt_manager: The templates rendering manager.
        """
        self._llm = llm
        self._prompt_manager = prompt_manager

    def evaluate_result(
        self, goal: str, task: Task, tool_name: str, tool_args: dict, result: ToolResult
    ) -> ObservationEvaluation:
        """Invokes LLM analysis to evaluate the outcome of a tool execution.

        Args:
            goal: The main objective.
            task: The current Task being executed.
            tool_name: The name of the tool that ran.
            tool_args: The arguments passed to the tool.
            result: The ToolResult output.

        Returns:
            An ObservationEvaluation model.
        """
        # Render prompt template
        prompt = self._prompt_manager.render(
            workflow="default",
            category="observation",
            goal=goal,
            task_description=task.description,
            tool_name=tool_name,
            tool_args=str(tool_args),
            tool_success=str(result.success),
            tool_output=result.output,
            tool_error=result.error or "None",
        )

        # Generate structured evaluation
        return self._llm.generate_structured(
            prompt=prompt.user,
            response_model=ObservationEvaluation,
            system_prompt=prompt.system,
        )
