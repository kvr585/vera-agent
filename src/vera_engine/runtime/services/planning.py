"""Planning service to decompose goals into structured task lists."""

from typing import Any
from pydantic import BaseModel, Field, model_validator

from vera_engine.core.entities import Capability, Task, TaskStatus
from vera_engine.core.interfaces.llm import LLMProvider
from vera_engine.runtime.prompt import PromptManager


class TaskPlanItem(BaseModel):
    """Pydantic model representing a single task item in an LLM plan."""

    id: str = Field(description="Unique identifier for the task, e.g. 'task-1'")
    description: str = Field(description="Actionable task details")
    dependencies: list[str] = Field(
        default_factory=list,
        description="IDs of tasks this task depends on.",
    )


class TaskPlan(BaseModel):
    """Pydantic model representing the overall structured plan returned by the LLM."""

    tasks: list[TaskPlanItem] = Field(description="Ordered list of plan tasks")

    @model_validator(mode="before")
    @classmethod
    def normalize_tasks(cls, data: Any) -> Any:
        """Normalizes plan responses if LLM returned a single task item at the root."""
        if isinstance(data, dict):
            if "description" in data and "tasks" not in data:
                task_item = {
                    "id": data.get("id", "task-1"),
                    "description": data["description"],
                    "dependencies": data.get("dependencies", []),
                }
                return {"tasks": [task_item]}
        return data


class PlanningService:
    """Orchestrates goal deconstruction into tasks.

    Deconstructs goals into a typed set of dependencies-mapped Tasks.
    """

    def __init__(self, llm: LLMProvider, prompt_manager: PromptManager) -> None:
        """Initializes the planning service.

        Args:
            llm: The LLM model provider.
            prompt_manager: The templates rendering manager.
        """
        self._llm = llm
        self._prompt_manager = prompt_manager

    def generate_plan(self, goal: str, capabilities: list[Capability]) -> list[Task]:
        """Decomposes a high-level goal into a list of executable tasks.

        Args:
            goal: The main user objective.
            capabilities: The list of active capabilities the agent possesses.

        Returns:
            A list of validated Task domain entities.
        """
        # Format capability details for prompt rendering
        capabilities_list = [
            f"- {cap.name}: {cap.description} "
            f"(Tools: {', '.join(cap.associated_tools)})"
            for cap in capabilities
        ]
        capabilities_str = "\n".join(capabilities_list)

        # Render prompt template
        prompt = self._prompt_manager.render(
            workflow="default",
            category="planner",
            goal=goal,
            capabilities=capabilities_str,
        )

        # Query structured LLM
        plan_output = self._llm.generate_structured(
            prompt=prompt.user,
            response_model=TaskPlan,
            system_prompt=prompt.system,
        )

        # Map to domain entities
        domain_tasks: list[Task] = []
        for item in plan_output.tasks:
            task = Task(
                id=item.id,
                description=item.description,
                status=TaskStatus.PENDING,
                dependencies=item.dependencies,
            )
            domain_tasks.append(task)

        return domain_tasks
