"""Unit tests for the VERA CLI interface."""

from pathlib import Path

from typer.testing import CliRunner

from vera_engine.cli import app
from vera_engine.config.manager import ProviderSettings, Settings
from vera_engine.core.entities import AgentState

runner = CliRunner()


def test_cli_list_sessions(tmp_path: Path, mocker: any) -> None:
    """Verifies list-sessions displays empty and populated session databases."""
    db_file = tmp_path / "test_sessions.db"
    db_url = f"sqlite:///{db_file}"

    # Mock load_settings to point to temp db
    mock_settings = Settings(
        database_url=db_url,
        workspace_dir=str(tmp_path / "workspace"),
        prompts_dir=str(tmp_path / "prompts"),
    )
    mocker.patch("vera_engine.cli.load_settings", return_value=mock_settings)

    # 1. Verify empty database print
    result = runner.invoke(app, ["list-sessions"])
    assert result.exit_code == 0
    assert "No sessions found" in result.output

    # 2. Save a completed state in db
    from vera_engine.adapters.repositories.sqlite import SQLiteSessionRepository

    repo = SQLiteSessionRepository(str(db_file))
    state = AgentState(
        session_id="mock-session-123",
        goal="Do clean list",
        current_workflow="default",
    )
    state.is_completed = True
    state.success = True
    repo.save_state(state)

    # 3. Verify formatted table print
    result2 = runner.invoke(app, ["list-sessions"])
    assert result2.exit_code == 0
    assert "mock-session-123" in result2.output
    assert "SUCCESS" in result2.output
    assert "Do clean list" in result2.output


def test_cli_run_workflow(tmp_path: Path, mocker: any) -> None:
    """Verifies cli run command executes workflow successfully with mocked LLM."""
    db_file = tmp_path / "test_run.db"
    db_url = f"sqlite:///{db_file}"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    default_prompts = prompts / "default"
    default_prompts.mkdir()
    (default_prompts / "planner.md").write_text(
        "# System\nPlanner sys\n# User\nPlanner usr", encoding="utf-8"
    )
    (default_prompts / "reasoner.md").write_text(
        "# System\nReasoner sys\n# User\nReasoner usr", encoding="utf-8"
    )
    (default_prompts / "observer.md").write_text(
        "# System\nObserver sys\n# User\nObserver usr", encoding="utf-8"
    )
    (default_prompts / "summarizer.md").write_text(
        "# System\nSummarizer sys\n# User\nSummarizer usr", encoding="utf-8"
    )

    # Mock settings
    mock_settings = Settings(
        database_url=db_url,
        workspace_dir=str(workspace),
        prompts_dir=str(prompts),
        provider=ProviderSettings(
            name="ollama",
            base_url="http://localhost:11434",
            model="llama3",
        ),
    )
    mocker.patch("vera_engine.cli.load_settings", return_value=mock_settings)

    # Mock Ollama HTTP client generate calls
    mock_post = mocker.patch("httpx.Client.post")

    # 1. Planning returns empty task list
    mock_response_plan = mocker.Mock()
    mock_response_plan.json.return_value = {"response": '{"tasks": []}'}

    # 2. Summarizer returns final statement
    mock_response_summary = mocker.Mock()
    mock_response_summary.json.return_value = {
        "response": "All steps executed beautifully."
    }

    mock_post.side_effect = [mock_response_plan, mock_response_summary]

    # Run the CLI
    result = runner.invoke(
        app,
        [
            "run",
            "--goal",
            "Say hello",
            "--session-id",
            "session-test-run",
        ],
    )

    assert result.exit_code == 0
    assert "Bootstrapping VERA Session" in result.output
    assert "SUCCESS" in result.output
    assert "All steps executed beautifully" in result.output


def test_cli_run_workflow_crashed(tmp_path: Path, mocker: any) -> None:
    """Verifies that kernel run failures raise CLI exits correctly."""
    mock_settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'fail.db'}",
        workspace_dir=str(tmp_path / "workspace"),
        prompts_dir=str(tmp_path / "prompts"),
    )
    mocker.patch("vera_engine.cli.load_settings", return_value=mock_settings)

    # Force crash
    mocker.patch(
        "vera_engine.runtime.kernel.RuntimeKernel.run",
        side_effect=ValueError("Kernel blew up"),
    )

    result = runner.invoke(
        app,
        [
            "run",
            "--goal",
            "Explode",
        ],
    )
    assert result.exit_code == 1
    assert "Kernel blew up" in result.output
