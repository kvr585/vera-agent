"""Concrete implementation of the tool registry."""

from collections.abc import Sequence

from vera_engine.core.interfaces.tool import Tool, ToolRegistry


class LocalToolRegistry(ToolRegistry):
    """In-memory registry to register and lookup tools by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Registers a tool with the registry.

        Args:
            tool: The Tool instance to register.
        """
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Tool | None:
        """Retrieves a registered tool by its name.

        Args:
            name: Name of the tool.

        Returns:
            The Tool instance if found, otherwise None.
        """
        return self._tools.get(name)

    def list_tools(self) -> Sequence[Tool]:
        """Lists all registered tools.

        Returns:
            A sequence of registered Tool instances.
        """
        return list(self._tools.values())
