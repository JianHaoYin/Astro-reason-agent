"""Local tool registry for the benchmark planning agent.

This deliberately avoids spinning up a real MCP stdio server. The benchmark
already has a stable planner Scenario API, so for a lightweight local agent we
can expose a small in-process tool set with the same semantics.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

from planner.helpers import (
    format_plan_status,
    satellite_filter_key,
    satellite_summary_key,
    station_key,
    target_key,
    window_filter_key,
    window_summary_key,
)
from planner.helpers.mcp_helpers import filter_items, paginate, to_llm_dict
from planner.models import ConflictError, ResourceViolationError, ValidationError
from planner.scenario import Scenario
from planner.state import StateFile


class PlannerToolRegistry:
    """Minimal in-process tool surface for the local planning agent."""

    def __init__(self, sandbox_dir: Path):
        self.sandbox_dir = sandbox_dir
        self.case_dir = sandbox_dir / "data"
        self.workspace_dir = sandbox_dir / "workspace"
        self.state_path = sandbox_dir / "state" / "scenario.json"
        self.output_path = self.workspace_dir / "plan.json"
        self.state_file = StateFile(self.state_path)

        # Preserve the same environment contract used by the planner code.
        os.environ["CASE_PATH"] = str(self.case_dir)
        os.environ["ASTROX_CASE_PATH"] = str(self.case_dir)
        os.environ["ASTROX_STATE_PATH"] = str(self.state_path)
        os.environ["ASTROX_OUTPUT_PATH"] = str(self.output_path)

        self.scenario = self._new_scenario()
        self._persist()

        self._tools: dict[str, Callable[..., Any]] = {
            "query_satellites": self.query_satellites,
            "query_targets": self.query_targets,
            "query_stations": self.query_stations,
            "compute_access_windows": self.compute_access_windows,
            "query_windows": self.query_windows,
            "stage_action": self.stage_action,
            "get_plan_status": self.get_plan_status,
            "commit_plan": self.commit_plan,
            "evaluate_revisit_gaps": self.evaluate_revisit_gaps,
            "wait": self.wait,
        }

    def _new_scenario(self) -> Scenario:
        return Scenario(
            satellite_file=str(self.case_dir / "satellites.yaml"),
            target_file=str(self.case_dir / "targets.yaml"),
            station_file=str(self.case_dir / "stations.yaml"),
            plan_file=str(self.case_dir / "initial_plan.json"),
        )

    def _restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.scenario = Scenario.from_state(
            satellite_file=str(self.case_dir / "satellites.yaml"),
            target_file=str(self.case_dir / "targets.yaml"),
            station_file=str(self.case_dir / "stations.yaml"),
            plan_file=str(self.case_dir / "initial_plan.json"),
            state_dict=snapshot,
        )

    def _persist(self) -> None:
        self.state_file.write(self.scenario.export_to_state())

    def invoke(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self._tools:
            return {"error": f"Unknown tool: {name}"}
        return self._tools[name](**arguments)

    def query_satellites(self, filters: dict[str, Any] | None = None, offset: int = 0, limit: int = 10) -> Any:
        all_sats = self.scenario.query_satellites()
        filtered = filter_items(all_sats, filters or {}, satellite_filter_key)
        paged = paginate(filtered, offset, limit)
        return [satellite_summary_key(sat) for sat in paged]

    def query_targets(self, filters: dict[str, Any] | None = None, offset: int = 0, limit: int = 10) -> Any:
        all_targets = self.scenario.query_targets()
        filtered = filter_items(all_targets, filters or {}, target_key)
        paged = paginate(filtered, offset, limit)
        return [to_llm_dict(target) for target in paged]

    def query_stations(self, filters: dict[str, Any] | None = None, offset: int = 0, limit: int = 10) -> Any:
        all_stations = self.scenario.query_stations()
        filtered = filter_items(all_stations, filters or {}, station_key)
        paged = paginate(filtered, offset, limit)
        return [to_llm_dict(station) for station in paged]

    def compute_access_windows(
        self,
        sat_ids: list[str],
        start_time: str,
        end_time: str,
        target_ids: list[str] | None = None,
        station_ids: list[str] | None = None,
        peer_satellite_ids: list[str] | None = None,
        constraints: list[dict[str, Any]] | None = None,
        offset: int = 0,
        limit: int = 10,
    ) -> Any:
        windows = self.scenario.compute_access_windows(
            sat_ids,
            target_ids,
            station_ids,
            peer_satellite_ids,
            start_time,
            end_time,
            constraints=constraints,
        )
        registered = self.scenario.register_windows(windows)
        self._persist()
        paged = paginate(registered, offset, min(limit, 20))
        return [window_summary_key(window) for window in paged]

    def query_windows(self, filters: dict[str, Any] | None = None, offset: int = 0, limit: int = 10) -> Any:
        from planner.helpers import record_matches_filters

        windows = self.scenario.query_windows()
        if filters:
            filter_dicts = [window_filter_key(window) for window in windows]
            indices = [idx for idx, item in enumerate(filter_dicts) if record_matches_filters(item, filters)]
            windows = [windows[idx] for idx in indices]
        paged = paginate(windows, offset, min(limit, 20))
        return [window_summary_key(window) for window in paged]

    def stage_action(self, action: dict[str, Any], dry_run: bool = False) -> Any:
        try:
            if dry_run:
                snapshot = self.scenario.export_to_state()
                horizon_start = self.scenario.horizon_start
                horizon_end = self.scenario.horizon_end
                result = self.scenario.stage_action(action)
                status = self.scenario.get_plan_status()
                self._restore_snapshot(snapshot)
                return {
                    "action_id": result.action_id,
                    "status": "feasible",
                    "projected_status": to_llm_dict(format_plan_status(status, horizon_start, horizon_end)),
                }

            result = self.scenario.stage_action(action)
            self._persist()
            return {"action_id": result.action_id, "status": "staged"}
        except (ValidationError, ConflictError, ResourceViolationError) as exc:
            return {"feasible": False, "reason": str(exc)}

    def get_plan_status(self) -> Any:
        status = self.scenario.get_plan_status()
        return to_llm_dict(format_plan_status(status, self.scenario.horizon_start, self.scenario.horizon_end))

    def commit_plan(self) -> Any:
        result = self.scenario.commit_plan(path=str(self.output_path))
        self._persist()
        return {
            "valid": result.valid,
            "action_count": result.metrics.total_actions,
            "total_observations": result.metrics.total_observations,
            "total_downlinks": result.metrics.total_downlinks,
            "violations": [to_llm_dict(v) for v in result.violations],
            "plan_json_path": result.plan_json_path,
        }

    def evaluate_revisit_gaps(self, target_ids: list[str], start_time: str | None = None, end_time: str | None = None) -> Any:
        return [to_llm_dict(item) for item in self.scenario.evaluate_revisit_gaps(target_ids, start_time, end_time)]

    def wait(self, seconds: float) -> Any:
        actual = max(0.0, min(float(seconds), 30.0))
        time.sleep(actual)
        return {"waited": actual}


def build_tool_specs() -> list[dict[str, Any]]:
    """Return OpenAI-style tool specs for the minimal planner tool set."""
    return [
        {
            "type": "function",
            "function": {
                "name": "query_satellites",
                "description": "Query available satellites with optional filters.",
                "parameters": {"type": "object", "properties": {"filters": {"type": "object"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_targets",
                "description": "Query available targets with optional filters.",
                "parameters": {"type": "object", "properties": {"filters": {"type": "object"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_stations",
                "description": "Query available ground stations with optional filters.",
                "parameters": {"type": "object", "properties": {"filters": {"type": "object"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compute_access_windows",
                "description": "Compute and register access windows for planning.",
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
        },
        {
            "type": "function",
            "function": {
                "name": "query_windows",
                "description": "Query already registered access windows.",
                "parameters": {"type": "object", "properties": {"filters": {"type": "object"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "stage_action",
                "description": "Stage an action into the plan. Use dry_run if unsure.",
                "parameters": {
                    "type": "object",
                    "required": ["action"],
                    "properties": {
                        "action": {"type": "object"},
                        "dry_run": {"type": "boolean"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_plan_status",
                "description": "Inspect the current staged plan and resource state.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "commit_plan",
                "description": "Validate and export the current plan to plan.json.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "evaluate_revisit_gaps",
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
        },
        {
            "type": "function",
            "function": {
                "name": "wait",
                "description": "Sleep briefly if needed.",
                "parameters": {
                    "type": "object",
                    "required": ["seconds"],
                    "properties": {"seconds": {"type": "number"}},
                },
            },
        },
    ]
