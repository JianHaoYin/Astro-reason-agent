"""Core agent loop for the local SatNet runner."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .ollama_client import OllamaChatClient
from .prompts import build_user_prompt, load_system_prompt
from .tool_profiles import DEFAULT_BENCHMARK_TYPE, render_tool_summary
from .tools import PlannerToolRegistry, build_tool_specs, get_allowed_tool_names


class LocalPlanningAgent:
    """Very small tool-calling agent loop for SatNet."""

    def __init__(self, base_url: str, model: str, timeout: int = 300, max_turns: int = 16):
        self.client = OllamaChatClient(base_url=base_url, model=model, timeout=timeout)
        self.max_turns = max_turns

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def _append_parsed_log(self, path: Path, role: str, body: str) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(f"[{self._timestamp()}] {role}:\n")
            f.write("----------------------------------------\n")
            f.write((body.rstrip() if body.strip() else "<empty>") + "\n")
            f.write("============================================================\n\n")

    def _format_json(self, payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _load_benchmark_type(self, sandbox_dir: Path) -> str:
        manifest_path = sandbox_dir / "data" / "manifest.json"
        if not manifest_path.exists():
            return DEFAULT_BENCHMARK_TYPE
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return DEFAULT_BENCHMARK_TYPE
        benchmark_type = manifest.get("benchmark_type")
        if isinstance(benchmark_type, str) and benchmark_type.strip():
            return benchmark_type
        return DEFAULT_BENCHMARK_TYPE

    def _message_text(self, message: dict[str, Any]) -> str:
        chunks: list[str] = []
        content = message.get("content") or ""
        reasoning = message.get("reasoning") or ""
        if isinstance(content, str) and content.strip():
            chunks.append(content)
        if isinstance(reasoning, str) and reasoning.strip():
            chunks.append(reasoning)
        return "\n".join(chunks)

    def _tool_call_repair_needed(self, message: dict[str, Any], tool_names: set[str]) -> bool:
        text = self._message_text(message)
        if not text.strip():
            return False
        if "<tool_call>" in text or "<function=" in text:
            return True
        return any(re.search(rf"\b{re.escape(name)}\s*\(", text) for name in tool_names)

    def _build_repair_message(self, mode: str) -> str:
        if mode == "pseudo_tool_call":
            return (
                "Your previous response described a tool call in plain text instead of returning structured tool_calls. "
                "Re-issue the exact next step using actual tool_calls only. Do not use XML tags, markdown code blocks, "
                "or pseudo-tool syntax."
            )
        return (
            "Your previous response returned neither usable content nor structured tool_calls, and the task is not "
            "finished yet. Continue from the current SatNet state and return either a real tool call or a short status "
            "update plus the next real tool call. Do not stop until commit_plan succeeds."
        )

    def _run_schedule_phase(
        self,
        messages: list[dict[str, Any]],
        model_requests_path: Path,
        model_responses_path: Path,
        transcript: list[dict[str, Any]],
        parsed_log_path: Path,
    ) -> tuple[str, list[dict[str, Any]]]:
        planning_messages = deepcopy(messages)
        planning_messages.append(
            {
                "role": "user",
                "content": (
                    "Before executing any actions, produce a concrete scheduling plan in Markdown. "
                    "Do not call tools in this step. Include a short objective summary and a numbered list of intended "
                    "request allocations, likely antennas, and why those choices should reduce unmet demand fairly."
                ),
            }
        )
        self._append_parsed_log(parsed_log_path, "USER", planning_messages[-1]["content"])

        request_payload = {
            "phase": "schedule_planning",
            "messages": planning_messages,
            "tools": [],
        }
        self._append_jsonl(model_requests_path, request_payload)

        response = self.client.create_chat_completion(messages=planning_messages, tools=[])
        self._append_jsonl(model_responses_path, {"phase": "schedule_planning", "response": response})

        message = response["choices"][0]["message"]
        transcript.append({"phase": "schedule_planning", "assistant": message})

        reasoning = message.get("reasoning") or ""
        if isinstance(reasoning, str) and reasoning.strip():
            self._append_parsed_log(parsed_log_path, "ASSISTANT", f"[Reasoning]\n{reasoning.strip()}")

        schedule_text = (message.get("content") or "").strip()
        self._append_parsed_log(parsed_log_path, "ASSISTANT", schedule_text or "No schedule proposed.")
        return schedule_text, planning_messages

    def run(self, sandbox_dir: Path, output_dir: Path) -> dict[str, Any]:
        mission_brief = (sandbox_dir / "workspace" / "mission_brief.md").read_text(encoding="utf-8")
        benchmark_type = self._load_benchmark_type(sandbox_dir)
        allowed_tool_names = get_allowed_tool_names(benchmark_type)
        registry = PlannerToolRegistry(sandbox_dir, allowed_tools=allowed_tool_names)
        tools = build_tool_specs(benchmark_type)
        tool_name_set = set(allowed_tool_names)

        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        model_requests_path = logs_dir / "model_requests.jsonl"
        model_responses_path = logs_dir / "model_responses.jsonl"
        tool_calls_path = logs_dir / "tool_calls.jsonl"
        conversation_path = logs_dir / "conversation.json"
        parsed_log_path = logs_dir / "agent_parsed_log.txt"
        parsed_log_path.write_text(
            "============================================================\n"
            "SatNet Local Agent Log\n"
            "============================================================\n\n",
            encoding="utf-8",
        )

        system_prompt = load_system_prompt()
        tool_summary = render_tool_summary(benchmark_type)
        user_prompt = build_user_prompt(mission_brief, benchmark_type, tool_summary)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        self._append_parsed_log(parsed_log_path, "USER", user_prompt)

        transcript: list[dict[str, Any]] = []
        final_text = ""
        committed = False
        stop_reason = "max_turns_exhausted"

        proposed_schedule, planning_messages = self._run_schedule_phase(
            messages=messages,
            model_requests_path=model_requests_path,
            model_responses_path=model_responses_path,
            transcript=transcript,
            parsed_log_path=parsed_log_path,
        )
        proposed_schedule_path = output_dir / "proposed_schedule.md"
        proposed_schedule_path.write_text(proposed_schedule or "No schedule proposed.", encoding="utf-8")

        messages = planning_messages
        messages.append({"role": "assistant", "content": proposed_schedule})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Execute the scheduling plan above using the tools. Follow it closely. If a planned allocation is "
                    "infeasible, explain the adjustment briefly, then continue. Finish only after commit_plan succeeds."
                ),
            }
        )
        self._append_parsed_log(parsed_log_path, "USER", messages[-1]["content"])

        repair_attempts = 0
        max_repair_attempts = 3

        with tqdm(total=self.max_turns, desc="Agent turns", unit="turn") as turn_progress:
            for turn in range(1, self.max_turns + 1):
                turn_progress.set_postfix(turn=turn)

                request_payload = {
                    "turn": turn,
                    "messages": messages,
                    "tools": tools,
                }
                self._append_jsonl(model_requests_path, request_payload)

                response = self.client.create_chat_completion(messages=messages, tools=tools)
                self._append_jsonl(model_responses_path, {"turn": turn, "response": response})

                message = response["choices"][0]["message"]
                transcript.append({"turn": turn, "assistant": message})

                reasoning = message.get("reasoning") or ""
                if isinstance(reasoning, str) and reasoning.strip():
                    self._append_parsed_log(parsed_log_path, "ASSISTANT", f"[Reasoning]\n{reasoning.strip()}")

                content = message.get("content") or ""
                if isinstance(content, str) and content.strip():
                    final_text = content.strip()
                    self._append_parsed_log(parsed_log_path, "ASSISTANT", final_text)

                messages.append(
                    {
                        "role": "assistant",
                        "content": message.get("content", ""),
                        "tool_calls": message.get("tool_calls", []),
                    }
                )

                tool_calls = message.get("tool_calls") or []
                if not tool_calls:
                    pseudo_tool_call = self._tool_call_repair_needed(message, tool_name_set)
                    empty_message = not self._message_text(message).strip()

                    if not committed and repair_attempts < max_repair_attempts and (pseudo_tool_call or empty_message):
                        repair_attempts += 1
                        repair_mode = "pseudo_tool_call" if pseudo_tool_call else "empty_response"
                        repair_message = self._build_repair_message(repair_mode)
                        messages.append({"role": "user", "content": repair_message})
                        self._append_parsed_log(parsed_log_path, "USER", repair_message)
                        stop_reason = repair_mode
                        turn_progress.update(1)
                        continue

                    if committed:
                        stop_reason = "committed"
                    elif pseudo_tool_call:
                        stop_reason = "pseudo_tool_call_repair_exhausted"
                    elif empty_message:
                        stop_reason = "empty_response_repair_exhausted"
                    else:
                        stop_reason = "no_tool_calls"
                    turn_progress.update(1)
                    break

                repair_attempts = 0

                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    raw_arguments = tool_call["function"].get("arguments") or "{}"
                    arguments = json.loads(raw_arguments)
                    self._append_parsed_log(
                        parsed_log_path,
                        "ASSISTANT",
                        f"[Tool -> {function_name}]\nArguments: {self._format_json(arguments)}",
                    )

                    result = registry.invoke(function_name, arguments)

                    tool_event = {
                        "turn": turn,
                        "tool_call_id": tool_call["id"],
                        "tool_name": function_name,
                        "tool_arguments": arguments,
                        "tool_result": result,
                    }
                    self._append_jsonl(tool_calls_path, tool_event)
                    transcript.append(tool_event)
                    self._append_parsed_log(parsed_log_path, "TOOL", self._format_json(result))

                    if function_name == "commit_plan" and isinstance(result, dict) and result.get("valid"):
                        committed = True
                        stop_reason = "committed"

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )

                turn_progress.update(1)
                if committed:
                    break

        transcript_path = output_dir / "agent_transcript.json"
        transcript_path.write_text(json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8")

        summary_path = output_dir / "agent_summary.md"
        summary_parts = [
            "# Proposed Schedule",
            proposed_schedule or "No schedule proposed.",
            "",
            "# Execution Result",
            final_text or "No final summary produced.",
        ]
        summary_path.write_text("\n".join(summary_parts), encoding="utf-8")

        conversation_path.write_text(
            json.dumps(
                {
                    "benchmark_type": benchmark_type,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "available_tools": allowed_tool_names,
                    "proposed_schedule": proposed_schedule,
                    "final_messages": messages,
                    "committed": committed,
                    "stop_reason": stop_reason,
                    "final_text": final_text,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return {
            "benchmark_type": benchmark_type,
            "committed": committed,
            "stop_reason": stop_reason,
            "plan_path": str(registry.output_path),
            "proposed_schedule_path": str(proposed_schedule_path),
            "summary_path": str(summary_path),
            "transcript_path": str(transcript_path),
            "logs_dir": str(logs_dir),
            "model_requests_path": str(model_requests_path),
            "model_responses_path": str(model_responses_path),
            "tool_calls_path": str(tool_calls_path),
            "conversation_path": str(conversation_path),
            "parsed_log_path": str(parsed_log_path),
            "final_text": final_text,
        }
