"""Command Line Interface for running the VERA Agent Engine."""

import uuid
from pathlib import Path

import typer

from vera_engine.adapters.providers.llm.ollama import OllamaProvider
from vera_engine.adapters.repositories.sqlite import SQLiteSessionRepository
from vera_engine.adapters.tools.local.filesystem import (
    ListDirectoryTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
)
from vera_engine.adapters.tools.registry import LocalToolRegistry
from vera_engine.config.manager import load_settings
from vera_engine.runtime import logging
from vera_engine.runtime.events import EventDispatcher
from vera_engine.runtime.kernel import RuntimeKernel
from vera_engine.runtime.services.execution import ExecutionService
from vera_engine.runtime.services.observation import ObservationService
from vera_engine.runtime.services.planning import PlanningService
from vera_engine.runtime.services.reasoning import ReasoningService
from vera_engine.runtime.workflows.default import DefaultWorkflow
from vera_engine.runtime.workflows.manager import WorkflowManager

app = typer.Typer(help="VERA Agent Engine Command Line Interface.")


@app.command()
def run(
    goal: str = typer.Option(
        ..., "--goal", "-g", help="The user objective goal for the session."
    ),
    session_id: str | None = typer.Option(
        None, "--session-id", "-s", help="Optional custom session ID."
    ),
    config: str | None = typer.Option(
        None, "--config", "-c", help="Path to config YAML file."
    ),
) -> None:
    """Executes a goals-oriented agent session end-to-end."""
    config_path = Path(config) if config else None
    if not config_path:
        default_config = Path("config/config.yaml")
        if default_config.exists():
            config_path = default_config
    settings = load_settings(config_path)
    typer.echo(f"Loaded model: {settings.provider.model}")

    # Initialize log files under workspace/logs/
    log_dir = Path(settings.workspace_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "vera_run.log"

    logging.configure_logging(log_file=log_file, debug_mode=settings.debug)
    logging.info(f"Starting execution kernel for session. Goal: {goal}")

    # Bootstrapping components
    repository = SQLiteSessionRepository(settings.database_url)
    llm = OllamaProvider(
        base_url=settings.provider.base_url,
        model_name=settings.provider.model,
        timeout_seconds=settings.provider.timeout_seconds,
    )

    from vera_engine.runtime.prompt import PromptManager

    prompts_dir = Path(settings.prompts_dir)
    prompt_manager = PromptManager(prompts_dir=prompts_dir)

    list_dir_tool = ListDirectoryTool(settings.workspace_dir)
    search_files_tool = SearchFilesTool(settings.workspace_dir)
    read_file_tool = ReadFileTool(settings.workspace_dir)
    write_file_tool = WriteFileTool(settings.workspace_dir)

    tool_registry = LocalToolRegistry()
    tool_registry.register(list_dir_tool)
    tool_registry.register(search_files_tool)
    tool_registry.register(read_file_tool)
    tool_registry.register(write_file_tool)

    from vera_engine.runtime.services.discovery import DiscoveryService

    discovery_service = DiscoveryService(
        list_dir_tool=list_dir_tool,
        search_files_tool=search_files_tool,
        read_file_tool=read_file_tool,
    )

    planning_service = PlanningService(llm, prompt_manager)
    reasoning_service = ReasoningService(llm, prompt_manager)
    execution_service = ExecutionService(tool_registry)
    observation_service = ObservationService(llm, prompt_manager)

    default_workflow = DefaultWorkflow(
        planning_service=planning_service,
        reasoning_service=reasoning_service,
        execution_service=execution_service,
        observation_service=observation_service,
        llm=llm,
        prompt_manager=prompt_manager,
        tool_registry=tool_registry,
        discovery_service=discovery_service,
    )

    workflow_manager = WorkflowManager()
    workflow_manager.register(default_workflow)

    event_dispatcher = EventDispatcher()

    kernel = RuntimeKernel(
        repository=repository,
        llm=llm,
        tool_registry=tool_registry,
        event_dispatcher=event_dispatcher,
        prompt_manager=prompt_manager,
        workflow_manager=workflow_manager,
    )

    s_id = session_id or str(uuid.uuid4())
    typer.echo(f"Bootstrapping VERA Session ID: {s_id}")

    try:
        state_manager = kernel.run(session_id=s_id, goal=goal, workflow_name="default")
        state = state_manager.state

        typer.echo("\n--- Session Concluded ---")
        typer.echo(f"Status: {'SUCCESS' if state.success else 'FAILED'}")
        typer.echo(f"Summary:\n{state.summary}")
    except Exception as err:
        typer.echo(f"\nExecution crashed with fatal kernel exception: {err}")
        raise typer.Exit(code=1) from err


@app.command()
def list_sessions(
    config: str | None = typer.Option(
        None, "--config", "-c", help="Path to config YAML file."
    ),
) -> None:
    """Lists historical sessions and execution outcomes."""
    config_path = Path(config) if config else None
    if not config_path:
        default_config = Path("config/config.yaml")
        if default_config.exists():
            config_path = default_config
    settings = load_settings(config_path)

    repository = SQLiteSessionRepository(settings.database_url)
    sessions = repository.list_sessions()

    if not sessions:
        typer.echo("No sessions found.")
        return

    typer.echo(f"{'Session ID':<38} | {'Success':<8} | Goal")
    typer.echo("-" * 80)
    for s in sessions:
        success_str = (
            "PENDING" if not s.is_completed else ("SUCCESS" if s.success else "FAILED")
        )
        typer.echo(f"{s.session_id:<38} | {success_str:<8} | {s.goal}")


if __name__ == "__main__":
    app()
