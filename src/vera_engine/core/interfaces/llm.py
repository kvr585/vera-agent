"""Interface for model provider adapters."""

from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(Protocol):
    """Port contract defining LLM generation and structured parsing capabilities."""

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_mode: bool = False,
    ) -> str:
        """Generates a text completion or raw JSON string from the LLM.

        Args:
            prompt: The main user text prompt.
            system_prompt: Optional system instructions guiding the model's behavior.
            json_mode: If True, instructs the backend to return valid JSON.

        Returns:
            The raw text completion or JSON string response.
        """
        ...

    def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
    ) -> T:
        """Generates output structured according to a target Pydantic model.

        Args:
            prompt: The user text prompt.
            response_model: The Pydantic model type to parse the output into.
            system_prompt: Optional system instructions.

        Returns:
            An instance of the response_model containing parsed data.
        """
        ...
