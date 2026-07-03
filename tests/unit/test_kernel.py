"""Unit tests for the VERA RuntimeKernel, Logging Wrapper, and Ollama Provider."""

from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

from vera_engine.adapters.providers.llm.ollama import OllamaProvider
from vera_engine.adapters.tools.registry import LocalToolRegistry
from vera_engine.core.entities import AgentState
from vera_engine.core.interfaces.repository import SessionRepository
from vera_engine.runtime import logging
from vera_engine.runtime.events import EventDispatcher
from vera_engine.runtime.kernel import RuntimeKernel
from vera_engine.runtime.prompt import PromptManager
from vera_engine.runtime.workflows.base import Workflow
from vera_engine.runtime.workflows.manager import WorkflowManager

# --- Dummies and Test Doubles ---


class DummyRepository(SessionRepository):
    """Mock repository that stores state snapshots in memory."""

    def __init__(self) -> None:
        self.state: AgentState | None = None

    def save_state(self, state: AgentState) -> None:
        self.state = state

    def get_state(self, session_id: str) -> AgentState | None:
        return self.state

    def list_sessions(self) -> Sequence[AgentState]:
        return [self.state] if self.state else []


class DummyLLM:
    """Mock LLMProvider double."""

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_mode: bool = False,
    ) -> str:
        return "Dummy output"

    def generate_structured(
        self,
        prompt: str,
        response_model: type,
        system_prompt: str | None = None,
    ) -> type:
        return response_model.model_validate({})


class MockWorkflow(Workflow):
    """Stub workflow to verify kernel execution triggers."""

    def __init__(self) -> None:
        self.executed = False

    @property
    def name(self) -> str:
        return "mock_workflow"

    def execute(self, state_manager: any) -> None:
        self.executed = True
        state_manager.finalize_session(summary="Mock workflow finished", success=True)


# --- Kernel and Logging Tests ---


def test_logging_wrapper_runs_without_crashes(tmp_path: Path) -> None:
    """Verifies that our custom logging wrapper configures outputs cleanly."""
    log_file = tmp_path / "test_run.log"
    logging.configure_logging(log_file=log_file, debug_mode=True)

    # Run log statements
    logging.info("Test info message")
    logging.debug("Test debug message")
    logging.warning("Test warning message")
    logging.error("Test error message")
    logging.exception("Test exception message")

    assert log_file.exists()
    log_content = log_file.read_text(encoding="utf-8")
    assert "Test info message" in log_content
    assert "Test debug message" in log_content


def test_kernel_workflow_execution() -> None:
    """Verifies that the RuntimeKernel bootstraps and runs workflows."""
    repo = DummyRepository()
    llm = DummyLLM()
    registry = LocalToolRegistry()
    dispatcher = EventDispatcher()

    # Prompt manager returning basic templates
    class QuickPrompts(PromptManager):
        def __init__(self) -> None:
            pass

    prompts = QuickPrompts()

    # Workflow catalog
    workflow_mgr = WorkflowManager()
    mock_workflow = MockWorkflow()
    workflow_mgr.register(mock_workflow)

    # Initialize kernel
    kernel = RuntimeKernel(
        repository=repo,
        llm=llm,
        tool_registry=registry,
        event_dispatcher=dispatcher,
        prompt_manager=prompts,
        workflow_manager=workflow_mgr,
    )

    # Dispatcher should contain a subscriber after kernel init
    assert len(dispatcher._global_listeners) == 1

    # Run workflow
    state_mgr = kernel.run(
        session_id="run-1", goal="Accomplish task", workflow_name="mock_workflow"
    )

    assert mock_workflow.executed is True
    assert state_mgr.state.is_completed is True
    assert state_mgr.state.success is True
    assert state_mgr.state.summary == "Mock workflow finished"


def test_kernel_unregistered_workflow_handling() -> None:
    """Verifies that executing an unregistered workflow terminates gracefully."""
    repo = DummyRepository()
    llm = DummyLLM()
    registry = LocalToolRegistry()
    dispatcher = EventDispatcher()
    workflow_mgr = WorkflowManager()

    class QuickPrompts(PromptManager):
        def __init__(self) -> None:
            pass

    prompts = QuickPrompts()

    kernel = RuntimeKernel(
        repository=repo,
        llm=llm,
        tool_registry=registry,
        event_dispatcher=dispatcher,
        prompt_manager=prompts,
        workflow_manager=workflow_mgr,
    )

    state_mgr = kernel.run(
        session_id="run-2", goal="Accomplish task", workflow_name="missing_workflow"
    )

    assert state_mgr.state.is_completed is True
    assert state_mgr.state.success is False
    assert "missing_workflow" in state_mgr.state.summary


# --- Provider Adapter Integration Tests ---


def test_ollama_provider_generate(mocker: any) -> None:
    """Verifies OllamaProvider generate requests parameters."""
    mock_post = mocker.patch("httpx.Client.post")
    mock_response = mocker.Mock()
    mock_response.json.return_value = {"response": "Llama response text"}
    mock_post.return_value = mock_response

    provider = OllamaProvider(base_url="http://localhost:11434", model_name="llama3")
    result = provider.generate(
        "Write a shell script", system_prompt="Be concise", json_mode=True
    )

    assert result == "Llama response text"

    # Assert parameters passed
    _, kwargs = mock_post.call_args
    payload = kwargs["json"]
    assert payload["model"] == "llama3"
    assert payload["prompt"] == "Write a shell script"
    assert payload["system"] == "Be concise"
    assert payload["format"] == "json"


class SimpleResponse(BaseModel):
    """Pydantic model for structured testing."""

    name: str


def test_ollama_provider_generate_structured_success(mocker: any) -> None:
    """Verifies OllamaProvider structured generation returns parsed model."""
    mock_post = mocker.patch("httpx.Client.post")
    mock_response = mocker.Mock()
    mock_response.json.return_value = {"response": '{"name": "VERA"}'}
    mock_post.return_value = mock_response

    provider = OllamaProvider(base_url="http://localhost:11434", model_name="llama3")
    result = provider.generate_structured(
        "Get name", SimpleResponse, system_prompt="Test System"
    )

    assert result.name == "VERA"
    _, kwargs = mock_post.call_args
    payload = kwargs["json"]
    assert payload["format"] == "json"
    assert "You MUST respond strictly in valid JSON" in payload["prompt"]
    assert payload["system"] == "Test System"


def test_ollama_provider_generate_structured_http_error(mocker: any) -> None:
    """Verifies that generate_structured propagates httpx HTTP errors."""
    mock_post = mocker.patch("httpx.Client.post")
    mock_response = mocker.Mock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Internal Server Error",
        request=mocker.Mock(),
        response=mock_response,
    )
    mock_post.return_value = mock_response

    provider = OllamaProvider(base_url="http://localhost:11434", model_name="llama3")
    with pytest.raises(httpx.HTTPStatusError):
        provider.generate_structured("Get name", SimpleResponse)


def test_load_settings(tmp_path: Path) -> None:
    """Verifies config manager loads defaults, merges YAML, and respects overrides."""
    import os

    from vera_engine.config.manager import load_settings

    # 1. Defaults
    settings = load_settings()
    assert settings.debug is False
    assert settings.provider.name == "ollama"

    # 2. YAML file config merge
    yaml_content = """
database_url: "sqlite:///custom.db"
debug: true
provider:
  model: "custom-llama"
"""
    config_file = tmp_path / "custom_config.yaml"
    config_file.write_text(yaml_content, encoding="utf-8")

    settings_yaml = load_settings(config_file)
    assert settings_yaml.database_url == "sqlite:///custom.db"
    assert settings_yaml.debug is True
    assert settings_yaml.provider.model == "custom-llama"

    # 3. Environment overrides
    os.environ["VERA_DEBUG"] = "True"
    os.environ["VERA_PROVIDER__TIMEOUT_SECONDS"] = "120.0"

    settings_env = load_settings()
    assert settings_env.debug is True
    assert settings_env.provider.timeout_seconds == 120.0

    # Clean up env
    del os.environ["VERA_DEBUG"]
    del os.environ["VERA_PROVIDER__TIMEOUT_SECONDS"]


def test_kernel_crashed_workflow_handling() -> None:
    """Verifies that kernel handles workflow execution exceptions gracefully."""

    class CrashingWorkflow(Workflow):
        @property
        def name(self) -> str:
            return "crash"

        def execute(self, state_manager: any) -> None:
            raise RuntimeError("Workflow logic crashed")

    repo = DummyRepository()
    llm = DummyLLM()
    registry = LocalToolRegistry()
    dispatcher = EventDispatcher()
    workflow_mgr = WorkflowManager()

    class QuickPrompts(PromptManager):
        def __init__(self) -> None:
            pass

    prompts = QuickPrompts()

    workflow_mgr.register(CrashingWorkflow())
    kernel = RuntimeKernel(
        repository=repo,
        llm=llm,
        tool_registry=registry,
        event_dispatcher=dispatcher,
        prompt_manager=prompts,
        workflow_manager=workflow_mgr,
    )

    state_mgr = kernel.run(
        session_id="run-3", goal="Verify robustness", workflow_name="crash"
    )
    assert state_mgr.state.is_completed is True
    assert state_mgr.state.success is False
    assert "RuntimeError: Workflow logic crashed" in state_mgr.state.summary
