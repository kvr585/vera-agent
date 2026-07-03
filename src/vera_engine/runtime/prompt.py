"""Prompt manager to load, parse, and render markdown prompt templates."""

from pathlib import Path
from typing import Any

from jinja2 import Template


class RenderedPrompt:
    """Represents a compiled system and user prompt pair ready for LLM consumption."""

    def __init__(self, system: str | None, user: str) -> None:
        self.system = system
        self.user = user


class PromptManager:
    """Loads, compiles, and renders markdown prompt templates from the filesystem."""

    def __init__(self, prompts_dir: Path) -> None:
        """Initializes the prompt manager.

        Args:
            prompts_dir: Path to the directory housing workflow prompts.
        """
        self._prompts_dir = Path(prompts_dir)

    def render(self, workflow: str, category: str, **variables: Any) -> RenderedPrompt:
        """Loads and compiles prompt templates for a workflow category with Jinja2.

        Args:
            workflow: The workflow name (e.g. 'default').
            category: The prompt category (e.g. 'planner', 'reasoning').
            variables: Keyword arguments representing variables to inject.

        Returns:
            A RenderedPrompt instance.
        """
        path = self._prompts_dir / workflow / f"{category}.md"
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found at: {path}")

        content = path.read_text(encoding="utf-8")
        return self._parse_and_compile(content, **variables)

    def _parse_and_compile(self, content: str, **variables: Any) -> RenderedPrompt:
        """Parses sections in a markdown file and compiles them with Jinja2.

        Recognizes '# System' and '# User' markdown headers as section dividers.
        If no headers are present, the whole file is compiled as a User prompt.
        """
        system_lines: list[str] = []
        user_lines: list[str] = []
        current_section: str | None = None

        for line in content.splitlines():
            stripped = line.strip()
            # Identify section headers
            if stripped.startswith("# System"):
                current_section = "system"
                continue
            elif stripped.startswith("# User") or stripped.startswith("# Prompt"):
                current_section = "user"
                continue

            # Route line to correct section list
            if current_section == "system":
                system_lines.append(line)
            elif current_section == "user":
                user_lines.append(line)
            elif current_section is None:
                # Default to user if text exists before headers
                if stripped:
                    current_section = "user"
                    user_lines.append(line)

        system_tmpl = "\n".join(system_lines).strip()
        user_tmpl = "\n".join(user_lines).strip()

        # Compile and render
        rendered_system = (
            Template(system_tmpl).render(**variables) if system_tmpl else None
        )
        rendered_user = Template(user_tmpl).render(**variables)

        return RenderedPrompt(system=rendered_system, user=rendered_user)
