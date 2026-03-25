"""Tool profiles for the SatNet local planning agent."""

from __future__ import annotations

from typing import Any

DEFAULT_BENCHMARK_TYPE = "satnet"

TOOL_DEFINITIONS: dict[str, dict[str, Any]] = {
    "list_unsatisfied_requests": {
        "description": "List requests that still need DSN antenna time.",
        "parameters": {
            "type": "object",
            "properties": {
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
        },
    },
    "get_antenna_status": {
        "description": "Inspect current DSN antenna availability and optional blocked ranges.",
        "parameters": {
            "type": "object",
            "properties": {
                "include_blocked_ranges": {"type": "boolean"},
            },
        },
    },
    "find_view_periods": {
        "description": "Find available scheduling windows for a request.",
        "parameters": {
            "type": "object",
            "required": ["request_id"],
            "properties": {
                "request_id": {"type": "string"},
                "min_duration_hours": {"type": "number"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
        },
    },
    "schedule_track": {
        "description": "Schedule a DSN track for a request on a specific antenna and time interval.",
        "parameters": {
            "type": "object",
            "required": ["request_id", "antenna", "trx_on", "trx_off"],
            "properties": {
                "request_id": {"type": "string"},
                "antenna": {"type": "string"},
                "trx_on": {"type": "integer"},
                "trx_off": {"type": "integer"},
                "dry_run": {"type": "boolean"},
            },
        },
    },
    "unschedule_track": {
        "description": "Remove a previously scheduled track.",
        "parameters": {
            "type": "object",
            "required": ["action_id"],
            "properties": {
                "action_id": {"type": "string"},
                "dry_run": {"type": "boolean"},
            },
        },
    },
    "get_plan_status": {
        "description": "Inspect the current schedule and fairness metrics.",
        "parameters": {"type": "object", "properties": {}},
    },
    "commit_plan": {
        "description": "Finalize the schedule, save plan.json, and return final metrics.",
        "parameters": {"type": "object", "properties": {}},
    },
    "reset": {
        "description": "Reset the schedule to its initial empty state.",
        "parameters": {"type": "object", "properties": {}},
    },
}

BENCHMARK_TOOL_PROFILES: dict[str, list[str]] = {
    DEFAULT_BENCHMARK_TYPE: list(TOOL_DEFINITIONS.keys()),
}


def get_allowed_tool_names(benchmark_type: str) -> list[str]:
    """Return the configured tool names for a benchmark type."""
    return list(BENCHMARK_TOOL_PROFILES.get(benchmark_type, BENCHMARK_TOOL_PROFILES[DEFAULT_BENCHMARK_TYPE]))



def build_tool_specs(benchmark_type: str) -> list[dict[str, Any]]:
    """Return OpenAI-style tool specs for the selected SatNet tool profile."""
    specs: list[dict[str, Any]] = []
    for name in get_allowed_tool_names(benchmark_type):
        definition = TOOL_DEFINITIONS[name]
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": definition["description"],
                    "parameters": definition["parameters"],
                },
            }
        )
    return specs



def render_tool_summary(benchmark_type: str) -> str:
    """Render a short markdown summary of the active SatNet tool set."""
    lines: list[str] = []
    for name in get_allowed_tool_names(benchmark_type):
        lines.append(f"- {name}: {TOOL_DEFINITIONS[name]['description']}")
    return "\n".join(lines)
