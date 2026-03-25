"""Task-specific tool profiles for benchmark planning agents."""

from __future__ import annotations

from typing import Any

TOOL_DEFINITIONS: dict[str, dict[str, Any]] = {
    "query_satellites": {
        "description": "Query available satellites with optional filters.",
        "parameters": {
            "type": "object",
            "properties": {
                "filters": {"type": "object"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
        },
    },
    "query_targets": {
        "description": "Query available targets with optional filters.",
        "parameters": {
            "type": "object",
            "properties": {
                "filters": {"type": "object"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
        },
    },
    "query_stations": {
        "description": "Query available ground stations with optional filters.",
        "parameters": {
            "type": "object",
            "properties": {
                "filters": {"type": "object"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
        },
    },
    "register_strips": {
        "description": "Register strip targets for mosaic or regional-coverage planning.",
        "parameters": {
            "type": "object",
            "required": ["strips"],
            "properties": {
                "strips": {"type": "array", "items": {"type": "object"}},
            },
        },
    },
    "unregister_strips": {
        "description": "Remove previously registered strips from the scenario.",
        "parameters": {
            "type": "object",
            "required": ["strip_ids"],
            "properties": {
                "strip_ids": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    "query_strips": {
        "description": "Inspect currently registered strips.",
        "parameters": {
            "type": "object",
            "properties": {
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
        },
    },
    "compute_strip_windows": {
        "description": "Compute and register strip visibility windows.",
        "parameters": {
            "type": "object",
            "required": ["sat_ids", "strip_ids", "start_time", "end_time"],
            "properties": {
                "sat_ids": {"type": "array", "items": {"type": "string"}},
                "strip_ids": {"type": "array", "items": {"type": "string"}},
                "start_time": {"type": "string"},
                "end_time": {"type": "string"},
                "constraints": {"type": "array", "items": {"type": "object"}},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
        },
    },
    "query_windows": {
        "description": "Query already registered access windows.",
        "parameters": {
            "type": "object",
            "properties": {
                "filters": {"type": "object"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
        },
    },
    "query_actions": {
        "description": "Query currently staged actions.",
        "parameters": {
            "type": "object",
            "properties": {
                "filters": {"type": "object"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
        },
    },
    "compute_lighting_windows": {
        "description": "Compute satellite lighting windows such as sunlight and eclipse.",
        "parameters": {
            "type": "object",
            "required": ["sat_ids", "start_time", "end_time"],
            "properties": {
                "sat_ids": {"type": "array", "items": {"type": "string"}},
                "start_time": {"type": "string"},
                "end_time": {"type": "string"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
        },
    },
    "get_ground_track": {
        "description": "Sample a satellite ground track over time, optionally clipped to a polygon.",
        "parameters": {
            "type": "object",
            "required": ["satellite_id", "start_time", "end_time"],
            "properties": {
                "satellite_id": {"type": "string"},
                "start_time": {"type": "string"},
                "end_time": {"type": "string"},
                "step_sec": {"type": "number"},
                "filter_polygon": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
        },
    },
    "evaluate_comms_latency": {
        "description": "Evaluate end-to-end communication latency between two ground stations.",
        "parameters": {
            "type": "object",
            "required": ["source_station_id", "dest_station_id", "start_time", "end_time"],
            "properties": {
                "source_station_id": {"type": "string"},
                "dest_station_id": {"type": "string"},
                "start_time": {"type": "string"},
                "end_time": {"type": "string"},
                "sample_step_sec": {"type": "number"},
            },
        },
    },
    "compute_access_windows": {
        "description": "Compute and register access windows for observation, downlink, or ISL planning.",
        "parameters": {
            "type": "object",
            "required": ["sat_ids", "start_time", "end_time"],
            "properties": {
                "sat_ids": {"type": "array", "items": {"type": "string"}},
                "target_ids": {"type": "array", "items": {"type": "string"}},
                "station_ids": {"type": "array", "items": {"type": "string"}},
                "peer_satellite_ids": {"type": "array", "items": {"type": "string"}},
                "start_time": {"type": "string"},
                "end_time": {"type": "string"},
                "constraints": {"type": "array", "items": {"type": "object"}},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
        },
    },
    "stage_action": {
        "description": "Stage an action into the plan. Use dry_run first if unsure.",
        "parameters": {
            "type": "object",
            "required": ["action"],
            "properties": {
                "action": {"type": "object"},
                "dry_run": {"type": "boolean"},
            },
        },
    },
    "unstage_action": {
        "description": "Remove a previously staged action. Use dry_run first if unsure.",
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
        "description": "Inspect the current staged plan and resource state.",
        "parameters": {"type": "object", "properties": {}},
    },
    "commit_plan": {
        "description": "Validate and export the current plan to plan.json.",
        "parameters": {"type": "object", "properties": {}},
    },
    "reset_plan": {
        "description": "Clear all staged actions and reset to the initial plan state.",
        "parameters": {"type": "object", "properties": {}},
    },
    "evaluate_revisit_gaps": {
        "description": "Evaluate revisit gaps for selected targets.",
        "parameters": {
            "type": "object",
            "required": ["target_ids"],
            "properties": {
                "target_ids": {"type": "array", "items": {"type": "string"}},
                "start_time": {"type": "string"},
                "end_time": {"type": "string"},
            },
        },
    },
    "evaluate_stereo_coverage": {
        "description": "Evaluate whether selected targets have sufficient stereo coverage.",
        "parameters": {
            "type": "object",
            "required": ["target_ids"],
            "properties": {
                "target_ids": {"type": "array", "items": {"type": "string"}},
                "min_separation_deg": {"type": "number"},
            },
        },
    },
    "evaluate_polygon_coverage": {
        "description": "Estimate how much of a polygon has been covered by strip observations.",
        "parameters": {
            "type": "object",
            "required": ["polygon"],
            "properties": {
                "polygon": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            },
        },
    },
    "wait": {
        "description": "Sleep briefly if needed.",
        "parameters": {
            "type": "object",
            "required": ["seconds"],
            "properties": {
                "seconds": {"type": "number"},
            },
        },
    },
}

COMMON_TOOLS = [
    "query_satellites",
    "query_stations",
    "query_windows",
    "query_actions",
    "stage_action",
    "unstage_action",
    "get_plan_status",
    "commit_plan",
    "reset_plan",
    "wait",
]

BENCHMARK_TOOL_PROFILES: dict[str, list[str]] = {
    "revisit-optimization": COMMON_TOOLS
    + [
        "query_targets",
        "compute_access_windows",
        "evaluate_revisit_gaps",
    ],
    "stereo-imaging": COMMON_TOOLS
    + [
        "query_targets",
        "compute_access_windows",
        "compute_lighting_windows",
        "evaluate_stereo_coverage",
    ],
    "latency-optimization": COMMON_TOOLS
    + [
        "query_targets",
        "compute_access_windows",
        "evaluate_comms_latency",
    ],
    "regional-coverage": COMMON_TOOLS
    + [
        "register_strips",
        "unregister_strips",
        "query_strips",
        "compute_strip_windows",
        "compute_lighting_windows",
        "get_ground_track",
        "evaluate_polygon_coverage",
    ],
}

DEFAULT_BENCHMARK_TYPE = "revisit-optimization"


def get_allowed_tool_names(benchmark_type: str) -> list[str]:
    """Return the configured tool names for a benchmark type."""
    return list(BENCHMARK_TOOL_PROFILES.get(benchmark_type, BENCHMARK_TOOL_PROFILES[DEFAULT_BENCHMARK_TYPE]))



def build_tool_specs(benchmark_type: str) -> list[dict[str, Any]]:
    """Return OpenAI-style tool specs for a benchmark-specific tool profile."""
    specs: list[dict[str, Any]] = []
    for tool_name in get_allowed_tool_names(benchmark_type):
        definition = TOOL_DEFINITIONS[tool_name]
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": definition["description"],
                    "parameters": definition["parameters"],
                },
            }
        )
    return specs



def render_tool_summary(benchmark_type: str) -> str:
    """Render a concise Markdown summary of available tools for prompt injection."""
    lines = []
    for tool_name in get_allowed_tool_names(benchmark_type):
        lines.append(f"- `{tool_name}`: {TOOL_DEFINITIONS[tool_name]['description']}")
    return "\n".join(lines)
