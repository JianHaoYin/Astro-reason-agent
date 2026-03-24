from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ollama_anthropic_proxy import (
    anthropic_messages_to_openai,
    anthropic_tool_choice_to_openai,
    anthropic_tools_to_openai,
    estimate_tokens,
    openai_message_to_anthropic,
)


def test_anthropic_messages_to_openai_handles_system_tools_and_tool_results():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "find windows"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "calling tool"},
                {"type": "tool_use", "id": "toolu_1", "name": "query_windows", "input": {"limit": 5}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "[]"}]},
                {"type": "text", "text": "continue"},
            ],
        },
    ]

    converted = anthropic_messages_to_openai(messages, system=[{"type": "text", "text": "You are helpful."}])

    assert converted[0] == {"role": "system", "content": "You are helpful."}
    assert converted[1] == {"role": "user", "content": "find windows"}
    assert converted[2]["role"] == "assistant"
    assert converted[2]["tool_calls"][0]["function"]["name"] == "query_windows"
    assert converted[3] == {"role": "user", "content": "continue"}
    assert converted[4] == {"role": "tool", "tool_call_id": "toolu_1", "content": "[]"}


def test_openai_message_to_anthropic_handles_text_and_tool_call():
    message = SimpleNamespace(
        content="I need a tool.",
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(name="query_actions", arguments='{"satellite_id": "SAT-1"}'),
            )
        ],
    )
    usage = SimpleNamespace(prompt_tokens=12, completion_tokens=7)

    converted = openai_message_to_anthropic(message, model="qwen3.5:9b", usage=usage)

    assert converted["model"] == "qwen3.5:9b"
    assert converted["stop_reason"] == "tool_use"
    assert converted["content"][0] == {"type": "text", "text": "I need a tool."}
    assert converted["content"][1] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "query_actions",
        "input": {"satellite_id": "SAT-1"},
    }
    assert converted["usage"] == {"input_tokens": 12, "output_tokens": 7}


def test_tool_definitions_and_choices_are_mapped():
    tools = anthropic_tools_to_openai(
        [
            {
                "name": "query_actions",
                "description": "List actions",
                "input_schema": {"type": "object", "properties": {"limit": {"type": "integer"}}},
            }
        ]
    )

    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "query_actions",
                "description": "List actions",
                "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}},
            },
        }
    ]
    assert anthropic_tool_choice_to_openai({"type": "tool", "name": "query_actions"}) == {
        "type": "function",
        "function": {"name": "query_actions"},
    }


def test_estimate_tokens_returns_positive_count():
    payload = {
        "system": [{"type": "text", "text": "system prompt"}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello world"}]}],
    }

    assert estimate_tokens(payload) > 0
