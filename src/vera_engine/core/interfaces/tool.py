"""Interfaces and schema definitions for agent tools."""

from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import BaseModel


class ToolResult(BaseModel):
    """The outcome of a tool execution."""

    success: bool
    output: str
    error: str | None = None


class Tool(Protocol):
    """Port contract defining executable agent tools."""

    @property
    def name(self) -> str:
        """The distinct name of the tool (e.g. 'read_file')."""
        ...

    @property
    def description(self) -> str:
        """A detailed explanation of what the tool does, used by LLM selection."""
        ...

    @property
    def input_schema(self) -> type[BaseModel]:
        """A Pydantic model class defining the expected argument inputs."""
        ...

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Executes the tool with the validated arguments dictionary.

        Args:
            arguments: A key-value dictionary of arguments conforming to input_schema.

        Returns:
            A ToolResult object containing execution status and outputs.
        """
        ...


class ToolRegistry(Protocol):
    """Port contract defining tool collection management."""

    def register(self, tool: Tool) -> None:
        """Adds a tool to the registry.

        Args:
            tool: An object implementing the Tool protocol.
        """
        ...

    def get_tool(self, name: str) -> Tool | None:
        """Retrieves a registered tool by its name.

        Args:
            name: The name of the target tool.

        Returns:
            The Tool instance if registered, otherwise None.
        """
        ...

    def list_tools(self) -> Sequence[Tool]:
        """Lists all registered tools.

        Returns:
            A sequence of registered Tool instances.
        """
        ...
