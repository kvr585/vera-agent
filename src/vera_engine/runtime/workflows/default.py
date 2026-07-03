from pathlib import Path

from vera_engine.core.entities import TaskStatus
from vera_engine.core.interfaces.llm import LLMProvider
from vera_engine.core.interfaces.tool import ToolRegistry
from vera_engine.runtime import logging
from vera_engine.runtime.memory import LocalMemoryManager
from vera_engine.runtime.prompt import PromptManager
from vera_engine.runtime.services.discovery import DiscoveryService
from vera_engine.runtime.services.execution import ExecutionService
from vera_engine.runtime.services.observation import ObservationService
from vera_engine.runtime.services.planning import PlanningService
from vera_engine.runtime.services.reasoning import (
    ReasoningService,
    format_project_metadata,
)
from vera_engine.runtime.state import StateManager
from vera_engine.runtime.workflows.base import Workflow


class DefaultWorkflow(Workflow):
    """The standard Goal -> Plan -> Reason -> Execute -> Observe -> Summarize loop."""

    def __init__(
        self,
        planning_service: PlanningService,
        reasoning_service: ReasoningService,
        execution_service: ExecutionService,
        observation_service: ObservationService,
        llm: LLMProvider,
        prompt_manager: PromptManager,
        tool_registry: ToolRegistry,
        max_steps: int = 15,
        discovery_service: DiscoveryService | None = None,
    ) -> None:
        """Initializes the default workflow.

        Args:
            planning_service: Service to generate plans.
            reasoning_service: Service to decide next steps.
            execution_service: Service to run tools.
            observation_service: Service to evaluate outcomes.
            llm: The model provider for summarization.
            prompt_manager: Prompt templates renderer.
            tool_registry: Tool registry repository.
            max_steps: Safeguard limit on reasoning loops.
            discovery_service: Service to scan project metadata.
        """
        self._planner = planning_service
        self._reasoner = reasoning_service
        self._executor = execution_service
        self._observer = observation_service
        self._llm = llm
        self._prompt_manager = prompt_manager
        self._registry = tool_registry
        self._max_steps = max_steps
        self._discovery_service = discovery_service

    @property
    def name(self) -> str:
        """The distinct workflow name."""
        return "default"

    def execute(self, state_manager: StateManager) -> None:
        """Executes the complete VERA reasoning loop."""
        logging.info("Default Workflow: Commencing planning phase...")
        memory = LocalMemoryManager()

        state = state_manager.state
        memory.add_history_entry(
            role="system", content=f"Initializing session for goal: {state.goal}"
        )

        # 1. Generate Task List Plan
        try:
            tasks = self._planner.generate_plan(state.goal, state.capabilities)
            state_manager.update_tasks(tasks)
            logging.info(f"Default Workflow: Plan created with {len(tasks)} tasks.")
        except Exception as err:
            err_msg = f"Planning phase failed: {err}"
            logging.error(err_msg)
            state_manager.finalize_session(summary=err_msg, success=False)
            return

        # 2. Discovery Phase
        if self._discovery_service:
            logging.info("Default Workflow: Commencing discovery phase...")
            try:
                workspace_path = "workspace"
                list_tool = self._registry.get_tool("list_directory")
                if list_tool and hasattr(list_tool, "_workspace"):
                    workspace_path = str(list_tool._workspace)

                metadata = self._discovery_service.discover(workspace_path)
                state_manager.update_project_metadata(metadata)
                logging.info(
                    f"Default Workflow: Discovery complete. "
                    f"Detected languages: {metadata.detected_languages}"
                )
            except Exception as err:
                logging.warning(f"Default Workflow: Discovery phase failed: {err}")

        # 3. Main reasoning loop
        step_count = 0
        while step_count < self._max_steps:
            state = state_manager.state
            if state.is_completed:
                return

            # Find the first incomplete task
            active_task = None
            for task in state.tasks:
                if task.status != TaskStatus.COMPLETED:
                    active_task = task
                    break

            if not active_task:
                # All planned tasks are completed, generate final summary
                logging.info(
                    "Default Workflow: All tasks completed. Concluding goal..."
                )
                self._generate_final_summary(state_manager, memory)
                return

            # Set current task in state if not already set
            if state.current_task_id != active_task.id:
                state_manager.start_task(active_task.id)
                memory.add_history_entry(
                    role="system",
                    content=(
                        f"Starting task '{active_task.id}': {active_task.description}"
                    ),
                )

            # Query reasoning service for next action
            logging.info(
                f"Default Workflow (Step {step_count}): Deciding next action..."
            )
            try:
                decision = self._reasoner.decide_action(
                    state=state_manager.state,
                    history=memory.session_memory.history,
                    available_tools=list(self._registry.list_tools()),
                )
            except Exception as err:
                err_msg = f"Reasoning loop exception on step {step_count}: {err}"
                logging.exception(err_msg)
                state_manager.fail_task(active_task.id, err_msg)
                state_manager.finalize_session(summary=err_msg, success=False)
                return

            # Record thought block in history
            memory.add_history_entry(role="assistant", content=decision.thought)
            logging.info(f"Reasoning thought: {decision.thought}")

            # Act on decision
            if decision.action_type == "execute_tool":
                if not decision.tool_name:
                    err_msg = (
                        "LLM returned execute_tool action but tool_name was missing."
                    )
                    state_manager.fail_task(active_task.id, err_msg)
                    state_manager.finalize_session(summary=err_msg, success=False)
                    return

                tool_args = decision.tool_args or {}

                # Execute tool
                state_manager.set_running_tool(decision.tool_name)
                result = self._executor.execute_tool(decision.tool_name, tool_args)
                state_manager.set_running_tool(None)

                # Record tool outcome in history
                memory.add_history_entry(
                    role="tool",
                    content=(
                        f"Executed tool '{decision.tool_name}' with args {tool_args}. "
                        f"Success: {result.success}. "
                        f"Output: {result.output}. "
                        f"Error: {result.error or 'None'}."
                    ),
                )

                # Evaluate observation result
                try:
                    eval_result = self._observer.evaluate_result(
                        goal=state.goal,
                        task=active_task,
                        tool_name=decision.tool_name,
                        tool_args=tool_args,
                        result=result,
                    )
                except Exception as err:
                    err_msg = f"Observation evaluation crashed: {err}"
                    logging.exception(err_msg)
                    state_manager.fail_task(active_task.id, err_msg)
                    state_manager.finalize_session(summary=err_msg, success=False)
                    return

                # Record observation in history
                memory.add_history_entry(
                    role="system",
                    content=f"Observation: {eval_result.observation_summary}",
                )
                logging.info(f"Observation: {eval_result.observation_summary}")

                # Update task status based on observation
                if eval_result.task_completed:
                    state_manager.complete_task(
                        active_task.id, eval_result.observation_summary
                    )
                elif eval_result.error_messages:
                    err_msg = (
                        f"Task execution errors: "
                        f"{', '.join(eval_result.error_messages)}"
                    )
                    state_manager.fail_task(active_task.id, err_msg)
                    state_manager.finalize_session(summary=err_msg, success=False)
                    return

            elif decision.action_type == "finish":
                # Mark current active task as completed
                summary = decision.summary or "Task completed successfully."
                state_manager.complete_task(active_task.id, summary)
                memory.add_history_entry(
                    role="system",
                    content=(
                        f"Task '{active_task.id}' marked as COMPLETED. "
                        f"Summary: {summary}"
                    ),
                )

                # Check if there are other incomplete tasks
                remaining_tasks = [
                    t
                    for t in state_manager.state.tasks
                    if t.status != TaskStatus.COMPLETED
                ]
                if not remaining_tasks:
                    logging.info(
                        "Default Workflow: All tasks completed. Concluding goal..."
                    )
                    self._save_workflow_report(state_manager, summary)
                    state_manager.finalize_session(summary=summary, success=True)
                    return

                # Clear working memory for the next task
                memory.working_memory.clear()

            elif decision.action_type == "request_clarification":
                # Pause/Fail V0.1 session requesting human interaction
                summary = f"Clarification requested: {decision.clarification_query}"
                state_manager.finalize_session(summary=summary, success=False)
                return

            step_count += 1

        if not state_manager.state.is_completed:
            # Max steps safety trigger
            err_msg = (
                f"Workflow aborted: Exceeded maximum safety steps ({self._max_steps})."
            )
            logging.warning(err_msg)
            state_manager.finalize_session(summary=err_msg, success=False)

    def _generate_final_summary(
        self, state_manager: StateManager, memory: LocalMemoryManager
    ) -> None:
        """Invokes LLM summarization step over execution history.

        Produces the final summary using the LLM.
        """
        state = state_manager.state

        # Format history string
        history_lines = []
        for entry in memory.session_memory.history:
            role = entry.get("role", "system").upper()
            content = entry.get("content", "")
            history_lines.append(f"[{role}]: {content}")
        history_str = "\n".join(history_lines)

        try:
            prompt = self._prompt_manager.render(
                workflow="default",
                category="summarizer",
                goal=state.goal,
                history=history_str,
                project_metadata=format_project_metadata(state.project_metadata),
            )
            summary = self._llm.generate(
                prompt=prompt.user, system_prompt=prompt.system
            )
            self._save_workflow_report(state_manager, summary.strip())
            state_manager.finalize_session(summary=summary.strip(), success=True)
        except Exception as err:
            logging.error(f"Failed to generate final LLM summary: {err}")
            fallback_sum = "All tasks completed. Final summary generation failed."
            self._save_workflow_report(state_manager, fallback_sum)
            state_manager.finalize_session(
                summary=fallback_sum,
                success=True,
            )

    def _save_workflow_report(
        self, state_manager: StateManager, summary: str
    ) -> None:
        """Saves a structured codebase analysis report based on real project metadata."""
        try:
            workspace_path = "workspace"
            list_tool = self._registry.get_tool("list_directory")
            if list_tool and hasattr(list_tool, "_workspace"):
                workspace_path = str(list_tool._workspace)

            report_dir = Path(workspace_path) / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_file = report_dir / "project_summary.md"

            meta = state_manager.state.project_metadata
            if meta:
                entry_str = ", ".join(f"`{f}`" for f in meta.entry_points)
                config_str = ", ".join(f"`{f}`" for f in meta.important_files)
                docs_str = ", ".join(f"`{f}`" for f in meta.documentation_files)
                langs_str = ", ".join(meta.detected_languages)
                frams_str = ", ".join(meta.detected_frameworks)

                report_content = (
                    f"# VERA Architecture & Codebase Report\n\n"
                    f"## Goal\n{state_manager.state.goal}\n\n"
                    f"## Discovered Project Metadata\n"
                    f"- **Root Path**: `{meta.root_path}`\n"
                    f"- **Detected Languages**: {langs_str or 'None'}\n"
                    f"- **Detected Frameworks**: {frams_str or 'None'}\n\n"
                    f"### Key Files\n"
                    f"- **Entry Points**: {entry_str or 'None'}\n"
                    f"- **Configuration Files**: {config_str or 'None'}\n"
                    f"- **Documentation**: {docs_str or 'None'}\n\n"
                    f"### File Statistics\n"
                    f"- **Total Files**: {meta.file_statistics.get('total_files', 0)}\n"
                    f"- **Total Directories**: {meta.file_statistics.get('total_directories', 0)}\n"
                    f"- **Extensions**: `{meta.file_statistics.get('extensions', {})}`\n\n"
                    f"### Directory Structure\n"
                    f"```\n{meta.directory_tree}\n```\n\n"
                    f"## Execution Summary\n"
                    f"{summary}\n"
                )
                report_file.write_text(report_content, encoding="utf-8")
                logging.info(
                    f"Default Workflow: Successfully saved architecture report "
                    f"to '{report_file}'"
                )
        except Exception as err:
            logging.warning(
                f"Default Workflow: Failed to automatically save report: {err}"
            )
