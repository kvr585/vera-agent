"""Unit tests for the PromptManager."""

from pathlib import Path

import pytest

from vera_engine.runtime.prompt import PromptManager


def test_prompt_manager_structured_parsing(tmp_path: Path) -> None:
    """Verifies that PromptManager correctly parses '# System' and '# User' sections."""
    # Create temporary workflow prompt
    workflow_dir = tmp_path / "default"
    workflow_dir.mkdir()
    prompt_file = workflow_dir / "planner.md"

    prompt_content = """# System Instructions
You are a helpful planner agent. Goal is: {{ target_goal }}.

# User Prompt
Please decompose {{ task_count }} steps for {{ target_goal }}.
"""
    prompt_file.write_text(prompt_content, encoding="utf-8")

    manager = PromptManager(tmp_path)
    rendered = manager.render(
        workflow="default",
        category="planner",
        target_goal="Build VERA",
        task_count=3,
    )

    assert rendered.system is not None
    assert "You are a helpful planner agent. Goal is: Build VERA." in rendered.system
    assert "Please decompose 3 steps for Build VERA." in rendered.user


def test_prompt_manager_flat_file_fallback(tmp_path: Path) -> None:
    """Verifies that a markdown file without section headers defaults to User prompt."""
    workflow_dir = tmp_path / "default"
    workflow_dir.mkdir()
    prompt_file = workflow_dir / "flat_prompt.md"
    prompt_file.write_text("Hello {{ name }}!", encoding="utf-8")

    manager = PromptManager(tmp_path)
    rendered = manager.render(workflow="default", category="flat_prompt", name="VERA")

    assert rendered.system is None
    assert rendered.user == "Hello VERA!"


def test_prompt_manager_missing_file_raises_error(tmp_path: Path) -> None:
    """Verifies that requesting a non-existent template raises FileNotFoundError."""
    manager = PromptManager(tmp_path)
    with pytest.raises(FileNotFoundError):
        manager.render(workflow="default", category="non_existent")
