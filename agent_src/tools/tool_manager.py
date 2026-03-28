"""Tool profile loading and tool-call execution for the refactored agent stack."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

ToolDomain = Literal["benchmark", "satnet"]


@dataclass(slots=True)
class ToolDefinition:
    """Definition of one callable tool exposed to a chat model.

    Args:
        name: Tool name used in model tool calls.
        description: Human-readable summary for the model.
        parameters: JSON schema describing accepted arguments.
        metadata: Optional out-of-band information for the runtime.
    """

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_ollama_schema(self) -> dict[str, Any]:
        """Convert the tool definition into Ollama/OpenAI tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(slots=True)
class ToolCallRequest:
    """Normalized tool call request originating from a model response.

    Args:
        tool_call_id: Model-generated identifier for the tool call.
        tool_name: Name of the tool to execute.
        arguments: Parsed JSON object passed to the tool.
        raw_arguments: Original arguments payload before JSON normalization.
    """

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    raw_arguments: str = "{}"


@dataclass(slots=True)
class ToolExecutionResult:
    """Result of one executed tool call.

    Args:
        tool_call_id: ID of the originating tool call.
        tool_name: Executed tool name.
        arguments: Parsed arguments used for the invocation.
        result: Raw tool output.
        success: Whether the execution completed without an error payload.
    """

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    result: Any
    success: bool

    def to_tool_message(self) -> dict[str, Any]:
        """Convert the execution result into an Ollama/OpenAI tool message."""
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": json.dumps(self.result, ensure_ascii=False),
        }


class ToolManager:
    """Manage available tools, select task-specific tool lists, and execute tool calls.

    This class is intentionally thin. It reuses the existing benchmark and SatNet
    tool profile definitions plus their planner registries, while presenting one
    consistent interface to the refactored `agent_src` stack.
    """

    def __init__(
        self,
        tools: Sequence[ToolDefinition | Mapping[str, Any]] | None = None,
        *,
        domain: ToolDomain = "benchmark",
        task: str | None = None,
        sandbox_dir: str | Path | None = None,
    ) -> None:
        """Initialize the manager and optionally preload a runtime profile."""
        self.domain: ToolDomain = domain
        self.task: str | None = None
        self.sandbox_dir: Path | None = None
        self._registry: Any | None = None
        self._tools: dict[str, ToolDefinition] = {}
        self._allowed_tool_names: list[str] = []

        if tools:
            self.register_tools(tools)

        if task is not None or sandbox_dir is not None:
            self.configure_runtime(domain=domain, task=task, sandbox_dir=sandbox_dir)

    def configure_runtime(
        self,
        *,
        domain: ToolDomain | None = None,
        task: str | None = None,
        sandbox_dir: str | Path | None = None,
    ) -> None:
        """Load task-specific tools and initialize the underlying execution registry.

        Args:
            domain: Tool domain to use, either `benchmark` or `satnet`.
            task: Benchmark type or task profile name. For SatNet this defaults to
                the single built-in profile if omitted.
            sandbox_dir: Sandbox directory passed to the existing planner registry.
        """
        if domain is not None:
            self.domain = domain

        if sandbox_dir is not None:
            self.sandbox_dir = Path(sandbox_dir)

        definitions, default_task, allowed_names, registry_cls = self._load_domain_profile(
            self.domain,
            task=task,
        )
        self.task = task or default_task
        self._allowed_tool_names = list(allowed_names)
        self._tools = {}
        for tool_name in self._allowed_tool_names:
            definition = definitions[tool_name]
            self.register_tool(
                {
                    "name": tool_name,
                    "description": definition["description"],
                    "parameters": definition["parameters"],
                    "metadata": {"domain": self.domain, "task": self.task},
                }
            )

        self._registry = None
        if self.sandbox_dir is not None:
            self._registry = registry_cls(self.sandbox_dir, allowed_tools=self._allowed_tool_names)

    def register_tool(self, tool: ToolDefinition | Mapping[str, Any]) -> ToolDefinition:
        """Register one tool definition and return the normalized object."""
        definition = self._normalize_tool(tool)
        self._tools[definition.name] = definition
        if definition.name not in self._allowed_tool_names:
            self._allowed_tool_names.append(definition.name)
        return definition

    def register_tools(self, tools: Sequence[ToolDefinition | Mapping[str, Any]]) -> None:
        """Register multiple tools in order."""
        for tool in tools:
            self.register_tool(tool)

    def get_tool(self, name: str) -> ToolDefinition:
        """Fetch one tool definition by name."""
        try:
            return self._tools[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._tools))
            raise KeyError(f"Unknown tool '{name}'. Registered tools: {known or '<none>'}") from exc

    def get_available_tool_names(self, task: str | None = None) -> list[str]:
        """Return the tool-name allowlist for the current or requested task."""
        if task is not None and task != self.task:
            self.configure_runtime(domain=self.domain, task=task, sandbox_dir=self.sandbox_dir)
        return list(self._allowed_tool_names)

    def list_tool_names(self) -> list[str]:
        """Return all currently registered tool names."""
        return list(self._tools)

    def build_tool_schemas(self, tool_names: Sequence[str] | None) -> list[dict[str, Any]]:
        """Expand a plain tool-name list into Ollama/OpenAI schemas."""
        names = list(tool_names) if tool_names is not None else self.get_available_tool_names()
        return [self.get_tool(name).to_ollama_schema() for name in names]

    def execute_tool_call(self, tool_call: ToolCallRequest | Mapping[str, Any]) -> ToolExecutionResult:
        """Execute one tool call returned by the model and wrap the result.

        Args:
            tool_call: Either a normalized `ToolCallRequest` or an OpenAI-style
                mapping with `id` and `function.{name,arguments}`.

        Returns:
            A structured execution result containing the raw tool output.
        """
        if self._registry is None:
            raise RuntimeError("Tool runtime is not configured. Call configure_runtime() with a sandbox_dir first.")

        request = self._normalize_tool_call(tool_call)
        result = self._registry.invoke(request.tool_name, request.arguments)
        success = not (isinstance(result, Mapping) and "error" in result)
        return ToolExecutionResult(
            tool_call_id=request.tool_call_id,
            tool_name=request.tool_name,
            arguments=request.arguments,
            result=result,
            success=success,
        )

    def execute_tool_calls(
        self,
        tool_calls: Sequence[ToolCallRequest | Mapping[str, Any]],
    ) -> list[ToolExecutionResult]:
        """Execute multiple tool calls in order."""
        return [self.execute_tool_call(tool_call) for tool_call in tool_calls]

    def _load_domain_profile(
        self,
        domain: ToolDomain,
        *,
        task: str | None,
    ) -> tuple[dict[str, dict[str, Any]], str, list[str], type[Any]]:
        """Load tool definitions and registry class for the selected domain."""
        if domain == "benchmark":
            from benchmark.custom_agent.tool_profiles import (  # type: ignore[import-not-found]
                DEFAULT_BENCHMARK_TYPE,
                TOOL_DEFINITIONS,
                get_allowed_tool_names,
            )
            from benchmark.custom_agent.tools import PlannerToolRegistry  # type: ignore[import-not-found]

            selected_task = task or DEFAULT_BENCHMARK_TYPE
            return TOOL_DEFINITIONS, DEFAULT_BENCHMARK_TYPE, get_allowed_tool_names(selected_task), PlannerToolRegistry

        from satnet_agent.custom_agent.tool_profiles import (  # type: ignore[import-not-found]
            DEFAULT_BENCHMARK_TYPE,
            TOOL_DEFINITIONS,
            get_allowed_tool_names,
        )
        from satnet_agent.custom_agent.tools import PlannerToolRegistry  # type: ignore[import-not-found]

        selected_task = task or DEFAULT_BENCHMARK_TYPE
        return TOOL_DEFINITIONS, DEFAULT_BENCHMARK_TYPE, get_allowed_tool_names(selected_task), PlannerToolRegistry

    def _normalize_tool(self, tool: ToolDefinition | Mapping[str, Any]) -> ToolDefinition:
        """Accept both dataclass and dict-like tool definitions."""
        if isinstance(tool, ToolDefinition):
            return tool
        return ToolDefinition(
            name=str(tool["name"]),
            description=str(tool.get("description", "")),
            parameters=dict(tool.get("parameters") or tool.get("input_schema") or {"type": "object", "properties": {}}),
            metadata=dict(tool.get("metadata") or {}),
        )

    def _normalize_tool_call(self, tool_call: ToolCallRequest | Mapping[str, Any]) -> ToolCallRequest:
        """Convert supported tool-call payloads into `ToolCallRequest`."""
        if isinstance(tool_call, ToolCallRequest):
            return tool_call

        if "tool_name" in tool_call:
            arguments = self._coerce_arguments(tool_call.get("arguments") or {})
            raw_arguments = tool_call.get("raw_arguments")
            if not isinstance(raw_arguments, str):
                raw_arguments = json.dumps(arguments, ensure_ascii=False)
            return ToolCallRequest(
                tool_call_id=str(tool_call.get("tool_call_id") or ""),
                tool_name=str(tool_call["tool_name"]),
                arguments=arguments,
                raw_arguments=raw_arguments,
            )

        function_payload = tool_call.get("function")
        if not isinstance(function_payload, Mapping):
            raise ValueError("Tool call payload must include a 'function' mapping.")

        tool_name = function_payload.get("name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("Tool call payload is missing a valid function name.")

        raw_arguments = function_payload.get("arguments") or "{}"
        arguments = self._coerce_arguments(raw_arguments)
        return ToolCallRequest(
            tool_call_id=str(tool_call.get("id") or ""),
            tool_name=tool_name,
            arguments=arguments,
            raw_arguments=raw_arguments if isinstance(raw_arguments, str) else json.dumps(raw_arguments, ensure_ascii=False),
        )

    def _coerce_arguments(self, raw_arguments: Any) -> dict[str, Any]:
        """Normalize tool arguments into a dictionary."""
        if isinstance(raw_arguments, Mapping):
            return dict(raw_arguments)
        if isinstance(raw_arguments, str):
            stripped = raw_arguments.strip()
            if not stripped:
                return {}
            loaded = json.loads(stripped)
            if not isinstance(loaded, dict):
                raise ValueError("Tool call arguments must decode to a JSON object.")
            return loaded
        raise ValueError(f"Unsupported tool argument payload type: {type(raw_arguments).__name__}")
