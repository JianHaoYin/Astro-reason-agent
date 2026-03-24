"""Core agent loop for the local benchmark runner."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .ollama_client import OllamaChatClient
from .prompts import build_user_prompt, load_system_prompt
from .tools import PlannerToolRegistry, build_tool_specs


class LocalPlanningAgent:
    """Very small tool-calling agent loop.

    The design goal here is simplicity:
    - one model
    - one conversation
    - local tools only
    - stop after a valid commit or when turns are exhausted
    """

    def __init__(self, base_url: str, model: str, timeout: int = 300, max_turns: int = 16):
        self.client = OllamaChatClient(base_url=base_url, model=model, timeout=timeout)
        self.max_turns = max_turns

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        """Append one JSON object per line for easy incremental debugging."""
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

    def _tool_call_repair_needed(self, message: dict[str, Any]) -> bool:
        content = message.get("content") or ""
        if not isinstance(content, str) or not content.strip():
            return False
        if "<tool_call>" in content or "<function=" in content:
            return True
        known_tools = {
            "query_satellites",
            "query_targets",
            "query_stations",
            "compute_access_windows",
            "query_windows",
            "stage_action",
            "get_plan_status",
            "commit_plan",
            "evaluate_revisit_gaps",
            "wait",
        }
        return any(re.search(rf"\b{name}\s*\(", content) for name in known_tools)

    def _run_schedule_phase(
        self,
        messages: list[dict[str, Any]],
        model_requests_path: Path,
        model_responses_path: Path,
        transcript: list[dict[str, Any]],
        parsed_log_path: Path,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Ask the model to propose a concrete schedule before execution begins."""
        planning_messages = deepcopy(messages)
        planning_messages.append({
            "role": "user",
            "content": (
                "Before executing any actions, produce a concrete execution schedule in Markdown. "
                "Do not call tools in this step. Include a short objective summary and a numbered "
                "list of planned observations/downlinks with asset names, rough timing, and intent."
            ),
        })
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
            self._append_parsed_log(parsed_log_path, "ASSISTANT", f"🧠 [思考过程]:\n{reasoning.strip()}")

        schedule_text = (message.get("content") or "").strip()
        self._append_parsed_log(parsed_log_path, "ASSISTANT", schedule_text or "No schedule proposed.")
        return schedule_text, planning_messages

    def run(self, sandbox_dir: Path, output_dir: Path) -> dict[str, Any]:
        """Run the agent on one prepared benchmark sandbox."""
        mission_brief = (sandbox_dir / "workspace" / "mission_brief.md").read_text(encoding="utf-8")
        registry = PlannerToolRegistry(sandbox_dir)
        tools = build_tool_specs()

        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        model_requests_path = logs_dir / "model_requests.jsonl"
        model_responses_path = logs_dir / "model_responses.jsonl"
        tool_calls_path = logs_dir / "tool_calls.jsonl"
        conversation_path = logs_dir / "conversation.json"
        parsed_log_path = logs_dir / "agent_parsed_log.txt"
        parsed_log_path.write_text(
            "============================================================\n"
            "AstroReason-Bench 代理执行日志 (Agent Log)\n"
            "============================================================\n\n",
            encoding="utf-8",
        )

        system_prompt = load_system_prompt()
        user_prompt = build_user_prompt(mission_brief)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        self._append_parsed_log(parsed_log_path, "USER", user_prompt)

        transcript: list[dict[str, Any]] = []
        final_text = ""
        committed = False

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
        messages.append({
            "role": "assistant",
            "content": proposed_schedule,
        })
        messages.append({
            "role": "user",
            "content": (
                "Execute the schedule above using the tools. Follow it closely. "
                "If a planned step is infeasible, explain the adjustment briefly, then continue. "
                "Finish only after commit_plan succeeds."
            ),
        })
        self._append_parsed_log(parsed_log_path, "USER", messages[-1]["content"])

        repair_attempts = 0

        for turn in range(1, self.max_turns + 1):
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
                self._append_parsed_log(parsed_log_path, "ASSISTANT", f"🧠 [思考过程]:\n{reasoning.strip()}")

            content = message.get("content") or ""
            if isinstance(content, str) and content.strip():
                final_text = content.strip()
                self._append_parsed_log(parsed_log_path, "ASSISTANT", final_text)

            messages.append({
                "role": "assistant",
                "content": message.get("content", ""),
                "tool_calls": message.get("tool_calls", []),
            })

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                if not committed and repair_attempts < 2 and self._tool_call_repair_needed(message):
                    repair_attempts += 1
                    repair_message = (
                        "Your previous response described a tool invocation in plain text instead of returning "
                        "structured tool_calls. Re-issue the exact next step using actual tool calls only. "
                        "Do not use XML tags, markdown code blocks, or pseudo-tool syntax."
                    )
                    messages.append({"role": "user", "content": repair_message})
                    self._append_parsed_log(parsed_log_path, "USER", repair_message)
                    continue
                break

            repair_attempts = 0

            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                raw_arguments = tool_call["function"].get("arguments") or "{}"
                arguments = json.loads(raw_arguments)
                self._append_parsed_log(
                    parsed_log_path,
                    "ASSISTANT",
                    f"🛠️ [调用工具 -> {function_name}]:\n参数: {self._format_json(arguments)}",
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
                self._append_parsed_log(parsed_log_path, "USER", f"✅ [工具返回结果]:\n{self._format_json(result)}")

                if function_name == "commit_plan" and isinstance(result, dict) and result.get("valid"):
                    committed = True

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                })

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
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "proposed_schedule": proposed_schedule,
                    "final_messages": messages,
                    "committed": committed,
                    "final_text": final_text,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return {
            "committed": committed,
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
