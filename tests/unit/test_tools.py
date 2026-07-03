"""Unit tests for the Tool Registry and Sandboxing utility."""

from pathlib import Path

import pytest
from pydantic import BaseModel

from vera_engine.adapters.tools.registry import LocalToolRegistry
from vera_engine.adapters.tools.sandbox import validate_path_confinement
from vera_engine.core.interfaces.tool import ToolResult

# --- Test Tool Double ---


class MockInputSchema(BaseModel):
    arg: str


class MockTool:
    """Mock implementation of the Tool protocol for testing."""

    @property
    def name(self) -> str:
        return "mock_tool"

    @property
    def description(self) -> str:
        return "A mock tool for testing"

    @property
    def input_schema(self) -> type[BaseModel]:
        return MockInputSchema

    def execute(self, arguments: dict[str, object]) -> ToolResult:
        return ToolResult(success=True, output="Mock executed")


# --- Registry Tests ---


def test_tool_registry_registration_and_lookup() -> None:
    """Verifies that the LocalToolRegistry registers and retrieves tools correctly."""
    registry = LocalToolRegistry()
    mock_tool = MockTool()

    # Register
    registry.register(mock_tool)

    # Retrieval
    tool = registry.get_tool("mock_tool")
    assert tool is not None
    assert tool.name == "mock_tool"
    assert tool.description == "A mock tool for testing"
    assert tool.input_schema == MockInputSchema

    # Non-existent tool
    assert registry.get_tool("missing_tool") is None


def test_tool_registry_listing() -> None:
    """Verifies that list_tools returns a sequence of all registered tools."""
    registry = LocalToolRegistry()
    mock_tool = MockTool()

    assert len(registry.list_tools()) == 0
    registry.register(mock_tool)
    assert len(registry.list_tools()) == 1
    assert registry.list_tools()[0].name == "mock_tool"


# --- Sandboxing Tests ---


def test_sandbox_allows_valid_paths(tmp_path: Path) -> None:
    """Verifies that paths within the workspace resolve correctly."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Relative path within workspace
    target_rel = "input.txt"
    resolved_rel = validate_path_confinement(target_rel, workspace)
    assert resolved_rel == workspace / "input.txt"

    # Absolute path within workspace
    target_abs = workspace / "subfolder" / "data.csv"
    resolved_abs = validate_path_confinement(target_abs, workspace)
    assert resolved_abs == workspace / "subfolder" / "data.csv"

    # Workspace directory itself
    resolved_ws = validate_path_confinement(".", workspace)
    assert resolved_ws == workspace


def test_sandbox_blocks_escaping_paths(tmp_path: Path) -> None:
    """Verifies that relative or absolute path traversal escapes raise ValueError."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Escape via relative navigation (..)
    with pytest.raises(ValueError) as excinfo:
        validate_path_confinement("../outside_file.txt", workspace)
    assert "escaping workspace sandbox" in str(excinfo.value)

    # Escape via absolute path outside workspace
    outside_abs = tmp_path / "other_folder" / "secret.key"
    with pytest.raises(ValueError) as excinfo:
        validate_path_confinement(outside_abs, workspace)
    assert "escaping workspace sandbox" in str(excinfo.value)
