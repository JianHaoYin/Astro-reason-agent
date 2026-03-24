from __future__ import annotations

import json
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    FastAPI = None
    HTTPException = None
    Request = Any
    JSONResponse = Any
    StreamingResponse = Any

try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    OpenAI = None

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_OLLAMA_API_KEY = "ollama"
DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"

app = FastAPI(title="Anthropic to Ollama Proxy") if FastAPI is not None else None


def get_ollama_client() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("openai package is required to use the Ollama proxy")
    return OpenAI(
        base_url=os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        api_key=os.environ.get("OLLAMA_API_KEY", DEFAULT_OLLAMA_API_KEY),
    )


def choose_model(requested_model: str | None) -> str:
    override = os.environ.get("OLLAMA_MODEL")
    return override or requested_model or DEFAULT_OLLAMA_MODEL


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                else:
                    parts.append(_json_dumps(item))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        if value.get("type") == "text":
            return value.get("text", "")
        return _json_dumps(value)
    return str(value)


def _system_to_text(system: Any) -> str:
    return _coerce_text(system).strip()


def anthropic_messages_to_openai(messages: list[dict[str, Any]], system: Any = None) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    system_text = _system_to_text(system)
    if system_text:
        converted.append({"role": "system", "content": system_text})

    for message in messages:
        role = message["role"]
        content = message.get("content", "")

        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []

        for block in content:
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id") or f"toolu_{uuid.uuid4().hex[:12]}",
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": _json_dumps(block.get("input", {})),
                        },
                    }
                )
            elif block_type == "tool_result":
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": _coerce_text(block.get("content", "")),
                    }
                )
            else:
                text_parts.append(_json_dumps(block))

        text_content = "\n\n".join(part for part in text_parts if part).strip()
        if role == "assistant":
            assistant_message: dict[str, Any] = {"role": "assistant", "content": text_content}
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            converted.append(assistant_message)
        else:
            if text_content:
                converted.append({"role": role, "content": text_content})
            converted.extend(tool_results)

    return converted


def anthropic_tools_to_openai(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None

    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for tool in tools
    ]


def anthropic_tool_choice_to_openai(tool_choice: dict[str, Any] | None) -> Any:
    if not tool_choice:
        return None

    choice_type = tool_choice.get("type")
    if choice_type in {"auto", "none", "required"}:
        return choice_type
    if choice_type == "tool":
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    return None


def _parse_tool_arguments(raw_arguments: str | None) -> dict[str, Any]:
    if not raw_arguments:
        return {}
    try:
        parsed = json.loads(raw_arguments)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        return {"raw": raw_arguments}


def openai_message_to_anthropic(message: Any, model: str, usage: Any = None) -> dict[str, Any]:
    content: list[dict[str, Any]] = []

    if getattr(message, "content", None):
        content.append({"type": "text", "text": message.content})

    tool_calls = getattr(message, "tool_calls", None) or []
    for index, tool_call in enumerate(tool_calls):
        tool_id = getattr(tool_call, "id", None) or f"toolu_{index}"
        function = getattr(tool_call, "function", None)
        name = getattr(function, "name", None) or f"tool_{index}"
        arguments = getattr(function, "arguments", None)
        content.append(
            {
                "type": "tool_use",
                "id": tool_id,
                "name": name,
                "input": _parse_tool_arguments(arguments),
            }
        )

    stop_reason = "tool_use" if tool_calls else "end_turn"
    input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


def estimate_tokens(payload: dict[str, Any]) -> int:
    chunks: list[str] = []
    chunks.append(_system_to_text(payload.get("system")))
    for message in payload.get("messages", []):
        chunks.append(_coerce_text(message.get("content", "")))
    text = "\n".join(chunk for chunk in chunks if chunk)
    return max(1, len(text) // 4) if text else 1


async def stream_openai_to_anthropic(stream: Any, model: str) -> AsyncIterator[str]:
    message_id = f"msg_{uuid.uuid4().hex}"
    yield _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )

    text_started = False
    tool_state: dict[int, dict[str, Any]] = {}

    for chunk in stream:
        choice = chunk.choices[0] if chunk.choices else None
        if choice is None:
            continue
        delta = choice.delta

        text = getattr(delta, "content", None)
        if text:
            if not text_started:
                yield _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                text_started = True
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": text},
                },
            )

        for tool_delta in getattr(delta, "tool_calls", None) or []:
            index = getattr(tool_delta, "index", 0) + (1 if text_started else 0)
            state = tool_state.setdefault(
                index,
                {
                    "started": False,
                    "id": getattr(tool_delta, "id", None) or f"toolu_{uuid.uuid4().hex[:12]}",
                    "name": None,
                },
            )
            function = getattr(tool_delta, "function", None)
            if function is not None and getattr(function, "name", None):
                state["name"] = function.name
            if getattr(tool_delta, "id", None):
                state["id"] = tool_delta.id
            if not state["started"]:
                yield _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {
                            "type": "tool_use",
                            "id": state["id"],
                            "name": state["name"] or f"tool_{index}",
                            "input": {},
                        },
                    },
                )
                state["started"] = True

            arguments_delta = getattr(function, "arguments", None) if function is not None else None
            if arguments_delta:
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "input_json_delta", "partial_json": arguments_delta},
                    },
                )

    if text_started:
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    for index in sorted(tool_state):
        if tool_state[index]["started"]:
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": index})

    stop_reason = "tool_use" if tool_state else "end_turn"
    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": 0},
        },
    )
    yield _sse("message_stop", {"type": "message_stop"})


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {_json_dumps(data)}\n\n"


if app is not None:

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}


    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        model = choose_model(None)
        return {
            "data": [
                {
                    "id": model,
                    "type": "model",
                    "display_name": model,
                }
            ]
        }


    @app.post("/v1/messages/count_tokens")
    @app.post("/messages/count_tokens")
    async def count_tokens(request: Request) -> dict[str, int]:
        payload = await request.json()
        return {"input_tokens": estimate_tokens(payload)}


    @app.post("/v1/messages")
    @app.post("/messages")
    async def create_message(request: Request) -> JSONResponse | StreamingResponse:
        payload = await request.json()
        model = choose_model(payload.get("model"))

        openai_messages = anthropic_messages_to_openai(payload.get("messages", []), system=payload.get("system"))
        openai_tools = anthropic_tools_to_openai(payload.get("tools"))
        openai_tool_choice = anthropic_tool_choice_to_openai(payload.get("tool_choice"))

        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": openai_messages,
            "stream": bool(payload.get("stream", False)),
        }
        if payload.get("max_tokens") is not None:
            create_kwargs["max_tokens"] = payload["max_tokens"]
        if payload.get("temperature") is not None:
            create_kwargs["temperature"] = payload["temperature"]
        if openai_tools:
            create_kwargs["tools"] = openai_tools
        if openai_tool_choice is not None:
            create_kwargs["tool_choice"] = openai_tool_choice

        try:
            response = get_ollama_client().chat.completions.create(**create_kwargs)
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            raise HTTPException(status_code=502, detail=f"Ollama request failed: {exc}") from exc

        if payload.get("stream"):
            return StreamingResponse(stream_openai_to_anthropic(response, model), media_type="text/event-stream")

        choice = response.choices[0] if response.choices else None
        if choice is None:
            raise HTTPException(status_code=502, detail="Ollama returned no choices")

        anthropic_response = openai_message_to_anthropic(choice.message, model=model, usage=response.usage)
        return JSONResponse(anthropic_response)
