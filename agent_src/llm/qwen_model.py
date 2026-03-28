"""Qwen model adapter that translates managed messages into Ollama requests."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from .ollama import Ollama
from ..message_manger.message_manager import ManagedMessage, MessageManager
from ..tools.tool_manager import ToolManager


@dataclass(slots=True)
class ToolCall:
    """Normalized tool call returned by the model."""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    raw_arguments: str


@dataclass(slots=True)
class ModelResponse:
    """Parsed model response including the recorded managed message."""

    text: str
    message: ManagedMessage
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_response: dict[str, Any] = field(default_factory=dict)


class BaseModel(ABC):
    """Abstract base class for model adapters."""

    @abstractmethod
    def generate(
        self,
        messages: Sequence[ManagedMessage],
        *,
        tools: Sequence[str] | None = None,
        session_id: str | None = None,
    ) -> ModelResponse:
        """Generate one response from already-managed chat messages."""


class QwenModel(BaseModel):
    """Adapter for local Qwen models exposed through Ollama's chat API.

    The adapter expects message tracking to be handled by `MessageManager`.
    During generation it only needs a sequence of managed messages plus a plain
    list of tool names. Tool schemas are expanded via `ToolManager`.
    """

    def __init__(
        self,
        ollama_client: Ollama,
        *,
        variant: Literal["9b", "27b"] = "9b",
        temperature: float = 0.0,
        tool_manager: ToolManager | None = None,
        message_manager: MessageManager | None = None,
        model_prefix: str = "qwen3.5",
    ) -> None:
        """Initialize the adapter with shared message/tool managers."""
        self.ollama_client = ollama_client
        self.variant = variant
        self.temperature = temperature
        self.tool_manager = tool_manager or ToolManager()
        self.message_manager = message_manager or MessageManager()
        self.model_name = f"{model_prefix}:{variant}"

    def generate(
        self,
        messages: Sequence[ManagedMessage],
        *,
        tools: Sequence[str] | None = None,
        session_id: str | None = None,
    ) -> ModelResponse:
        """Pack managed messages and tool names, call Ollama, and parse the reply.

        Args:
            messages: Fully composed request history created by `MessageManager`.
            tools: Plain list of tool names. Detailed schemas are resolved by
                `ToolManager` before the Ollama request is sent.
            session_id: Logical session identifier used for recording the reply.

        Returns:
            A normalized response containing assistant text, parsed tool calls,
            the recorded managed message, and the raw Ollama payload.
        """
        payload = {
            "model": self.model_name,
            "messages": self._build_qwen_messages(messages),
            "tools": self._build_tool_schemas(list(tools or [])),
            "temperature": self.temperature,
            "tool_choice": "auto",
            "stream": False,
        }
        raw_response = self.ollama_client(payload)
        response_message = self._extract_response_message(raw_response)
        text = str(response_message.get("content") or "")
        parsed_tool_calls = self._parse_tool_calls(response_message)
        managed_message = self.message_manager.build_response(
            session_id=session_id,
            role="assistant",
            components=[text] if text else [""],
            source="model",
            exported_from_session=session_id,
            tool_calls=[self._tool_call_to_payload(tool_call) for tool_call in parsed_tool_calls],
            raw_payload=raw_response,
            metadata={"model": self.model_name},
        )
        return ModelResponse(
            text=text,
            message=managed_message,
            tool_calls=parsed_tool_calls,
            raw_response=raw_response,
        )

    def _build_tool_schemas(self, tools: list[str]) -> list[dict[str, Any]]:
        """Resolve tool names into Ollama/OpenAI-style function schemas."""
        return self.tool_manager.build_tool_schemas(tools)

    def _build_qwen_messages(self, messages: Sequence[ManagedMessage]) -> list[dict[str, Any]]:
        """Translate internal managed messages into Qwen/Ollama chat payloads."""
        return [self._translate_message(message) for message in messages]

    def _translate_message(self, message: ManagedMessage) -> dict[str, Any]:
        """Translate one internal managed message into the Qwen wire format."""
        payload: dict[str, Any] = {
            "role": message.role,
            "content": message.content,
        }
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            payload["tool_calls"] = [self._translate_tool_call(tool_call) for tool_call in message.tool_calls]
        return payload

    def _extract_response_message(self, response: Mapping[str, Any]) -> dict[str, Any]:
        """Return the first assistant message from an Ollama response."""
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("Ollama response did not include any choices.")
        first_choice = choices[0]
        if not isinstance(first_choice, Mapping):
            raise RuntimeError("Ollama response choice is not a mapping.")
        message = first_choice.get("message")
        if not isinstance(message, Mapping):
            raise RuntimeError("Ollama response choice did not include a message payload.")
        return dict(message)

    def _parse_tool_calls(self, response_message: Mapping[str, Any]) -> list[ToolCall]:
        """Parse structured tool calls from one assistant message."""
        parsed: list[ToolCall] = []
        tool_calls = response_message.get("tool_calls")
        if not isinstance(tool_calls, list):
            return parsed

        for raw_tool_call in tool_calls:
            if not isinstance(raw_tool_call, Mapping):
                continue
            function_payload = raw_tool_call.get("function")
            if not isinstance(function_payload, Mapping):
                continue
            name = function_payload.get("name")
            if not isinstance(name, str) or not name.strip():
                continue

            raw_arguments = function_payload.get("arguments") or "{}"
            arguments = self._coerce_tool_arguments(raw_arguments)
            parsed.append(
                ToolCall(
                    tool_call_id=str(raw_tool_call.get("id") or ""),
                    tool_name=name,
                    arguments=arguments,
                    raw_arguments=raw_arguments if isinstance(raw_arguments, str) else json.dumps(raw_arguments, ensure_ascii=False),
                )
            )
        return parsed

    def _coerce_tool_arguments(self, raw_arguments: Any) -> dict[str, Any]:
        """Normalize tool arguments into a dictionary."""
        if isinstance(raw_arguments, Mapping):
            return dict(raw_arguments)
        if isinstance(raw_arguments, str):
            stripped = raw_arguments.strip()
            if not stripped:
                return {}
            try:
                loaded = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Tool call arguments are not valid JSON: {stripped}") from exc
            if not isinstance(loaded, dict):
                raise ValueError("Tool call arguments must decode to a JSON object.")
            return loaded
        raise ValueError(f"Unsupported tool argument payload type: {type(raw_arguments).__name__}")

    def _tool_call_to_payload(self, tool_call: ToolCall) -> dict[str, Any]:
        """Convert a parsed tool call into the internal message-manager shape."""
        return {
            "tool_call_id": tool_call.tool_call_id,
            "tool_name": tool_call.tool_name,
            "arguments": dict(tool_call.arguments),
        }

    def _translate_tool_call(self, tool_call: Mapping[str, Any]) -> dict[str, Any]:
        """Translate one internal tool-call object into the Qwen wire shape."""
        return {
            "id": str(tool_call.get("tool_call_id") or ""),
            "type": "function",
            "function": {
                "name": str(tool_call.get("tool_name") or ""),
                "arguments": json.dumps(tool_call.get("arguments") or {}, ensure_ascii=False),
            },
        }
