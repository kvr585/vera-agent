"""Execution service to execute registered agent tools safely."""

from typing import Any

from vera_engine.core.interfaces.tool import ToolRegistry, ToolResult
from vera_engine.runtime import logging


class ExecutionService:
    """Invokes registered tools and formats standard outputs and error results."""

    def __init__(self, registry: ToolRegistry) -> None:
        """Initializes the execution service.

        Args:
            registry: The tool registry holding all system tools.
        """
        self._registry = registry

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Invokes a registered tool by name with arguments.

        Args:
            name: The name of the tool to execute.
            arguments: The inputs dictionary to supply.

        Returns:
            A ToolResult object containing execution logs and outcomes.
        """
        logging.info(f"Execution Service: Invoking tool '{name}'...")

        tool = self._registry.get_tool(name)
        if not tool:
            err_msg = f"Tool '{name}' is not registered with the engine."
            logging.error(err_msg)
            return ToolResult(success=False, output="", error=err_msg)

        try:
            # Execute tool logic
            result = tool.execute(arguments)
            if result.success:
                logging.info(f"Tool '{name}' executed successfully.")
            else:
                logging.warning(
                    f"Tool '{name}' returned error: {result.error or 'Unknown error'}"
                )
            return result
        except Exception as err:
            err_msg = (
                f"Exception occurred during execution of '{name}': "
                f"{type(err).__name__}: {err}"
            )
            logging.exception(err_msg)
            return ToolResult(success=False, output="", error=err_msg)
