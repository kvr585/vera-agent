"""Ollama LLM provider implementation."""

from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from vera_engine.core.interfaces.llm import LLMProvider
from vera_engine.runtime import logging

T = TypeVar("T", bound=BaseModel)


class OllamaProvider(LLMProvider):
    """Concrete adapter connecting VERA to a local Ollama service."""

    def __init__(
        self, base_url: str, model_name: str, timeout_seconds: float = 60.0
    ) -> None:
        """Initializes the Ollama HTTP client provider.

        Args:
            base_url: Base endpoint (e.g. 'http://localhost:11434').
            model_name: The target model name (e.g. 'llama3').
            timeout_seconds: Read/write HTTP request timeout limit.
        """
        if not base_url.endswith("/"):
            base_url += "/"
        self._client = httpx.Client(base_url=base_url, timeout=timeout_seconds)
        self._model_name = model_name

    def _send_request(self, payload: dict[str, Any]) -> httpx.Response:
        """Sends a request to the Ollama generate endpoint.

        Performs robust error handling.
        """
        try:
            response = self._client.post("api/generate", json=payload)
            if response.status_code == 404:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", "")
                    if "not found" in error_msg.lower():
                        raise ValueError(
                            f"Model '{self._model_name}' is not pulled. "
                            f"Please run 'ollama pull {self._model_name}'."
                        ) from None
                except (ValueError, KeyError):
                    pass
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as err:
            logging.error(
                f"Ollama request failed: {err.response.status_code} "
                f"- {err.response.text}"
            )
            raise

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_mode: bool = False,
    ) -> str:
        """Generates a text completion from the local Ollama instance."""
        logging.debug(
            f"Ollama generate request on '{self._model_name}' "
            f"(system_prompt={bool(system_prompt)}, json={json_mode})"
        )

        payload: dict[str, Any] = {
            "model": self._model_name,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if json_mode:
            payload["format"] = "json"

        response = self._send_request(payload)
        data = response.json()
        return str(data["response"])

    def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
    ) -> T:
        """Generates Pydantic-structured output from the local Ollama instance.

        Uses standard format='json' with prompt-injected schema for maximum compatibility.
        """
        schema = response_model.model_json_schema()
        logging.debug(
            f"Ollama structured request on '{self._model_name}' "
            f"for model '{response_model.__name__}'"
        )

        payload: dict[str, Any] = {
            "model": self._model_name,
            "prompt": (
                f"{prompt}\n\nYou MUST respond strictly in valid JSON matching "
                f"this JSON Schema:\n{schema}"
            ),
            "stream": False,
            "format": "json",
        }
        if system_prompt:
            payload["system"] = system_prompt

        response = self._send_request(payload)
        raw_text = response.json()["response"]
        return response_model.model_validate_json(raw_text)
