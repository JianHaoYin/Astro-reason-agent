"""Base client abstraction and Ollama HTTP client implementation."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Generic, Mapping, TypeVar, TypedDict
from urllib import error as urllib_error
from urllib import request as urllib_request

RequestT = TypeVar("RequestT", bound=Mapping[str, Any])
ResponseT = TypeVar("ResponseT")


class ClientBase(ABC, Generic[RequestT, ResponseT]):
    """Generic base class for HTTP-backed model clients."""

    def __init__(self, *args: Any, base_url: str, **kwargs: Any) -> None:
        """Store the normalized API base URL."""
        self.args = args
        self.kwargs = kwargs
        self.base_url = base_url.rstrip("/")

    @abstractmethod
    def __call__(self, data: RequestT, timeout: int = 300) -> ResponseT:
        """Send a request payload and return the parsed response."""

    def _post_json(self, path: str, payload: Mapping[str, Any], timeout: int) -> dict[str, Any]:
        """Send a JSON POST request and decode the JSON response body."""
        request = urllib_request.Request(
            f"{self.base_url}/{path.lstrip('/')}",
            data=json.dumps(dict(payload)).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib_request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Request failed with status {exc.code}: {body}") from exc


class OllamaRequest(TypedDict, total=False):
    """Request payload accepted by the Ollama chat completions endpoint."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    temperature: float
    model: str
    tool_choice: str
    stream: bool


class Ollama(ClientBase[OllamaRequest, dict[str, Any]]):
    """Wrapper around Ollama's OpenAI-compatible chat completions API."""

    def __init__(
        self,
        base_url: str,
    ) -> None:
        """Initialize the Ollama client with the server base URL."""
        super().__init__(base_url=base_url)

    def __call__(self, data: OllamaRequest, timeout: int = 300) -> dict[str, Any]:
        """Build an Ollama chat payload from `data` and send the request."""
        payload: OllamaRequest = {
            "model": data["model"],
            "messages": data["messages"],
            "tools": data.get("tools", []),
            "tool_choice": data.get("tool_choice", "auto"),
            "temperature": data.get("temperature", 0.0),
            "stream": data.get("stream", False),
        }
        return self._post_json("/chat/completions", payload=payload, timeout=timeout)
