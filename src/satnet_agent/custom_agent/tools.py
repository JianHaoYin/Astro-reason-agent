"""Local tool registry for the SatNet planning agent."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

from satnet_agent import (
    SatNetConflictError,
    SatNetNotFoundError,
    SatNetScenario,
    SatNetValidationError,
)
from satnet_agent.state import SatNetStateFile


class SatNetToolRegistry:
    """Minimal in-process tool surface for the local SatNet planning agent."""

    def __init__(self, sandbox_dir: Path):
        self.sandbox_dir = sandbox_dir
        self.data_dir = sandbox_dir / "data"
        self.workspace_dir = sandbox_dir / "workspace"
        self.state_path = sandbox_dir / "state" / "scenario.json"
        self.output_path = self.workspace_dir / "plan.json"
        self.state_file = SatNetStateFile(self.state_path)

        self.week = int(os.environ.get("SATNET_WEEK", "40"))
        self.year = int(os.environ.get("SATNET_YEAR", "2018"))
        self.problems_path = str(self.data_dir / "problems.json")
        self.maintenance_path = str(self.data_dir / "maintenance.csv")

        os.environ["SATNET_STATE_PATH"] = str(self.state_path)
        os.environ["SATNET_PROBLEMS_PATH"] = self.problems_path
        os.environ["SATNET_MAINTENANCE_PATH"] = self.maintenance_path
        os.environ["SATNET_OUTPUT_PATH"] = str(self.output_path)
        os.environ["SATNET_WEEK"] = str(self.week)
        os.environ["SATNET_YEAR"] = str(self.year)

        if not self.state_file.exists():
            self.state_file.initialize(self.problems_path, self.maintenance_path, self.week, self.year)

        state = self.state_file.read()
        if state is None:
            raise RuntimeError("Failed to initialize SatNet state")
        self.scenario = SatNetScenario.from_state(state)

        self._tools: dict[str, Callable[..., Any]] = {
            "list_unsatisfied_requests": self.list_unsatisfied_requests,
            "get_antenna_status": self.get_antenna_status,
            "find_view_periods": self.find_view_periods,
            "schedule_track": self.schedule_track,
            "unschedule_track": self.unschedule_track,
            "get_plan_status": self.get_plan_status,
            "commit_plan": self.commit_plan,
            "reset": self.reset,
            "wait": self.wait,
        }

    def _persist(self) -> None:
        self.state_file.write(self.scenario.to_state())

    def invoke(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self._tools:
            return {"error": f"Unknown tool: {name}"}
        return self._tools[name](**arguments)

    def _format_request_summary(self, req: Any) -> str:
        return f"{req.request_id}: {req.remaining_hours:.1f}h remaining (min {req.min_duration_hours:.1f}h)"

    def _format_view_period(self, vp: Any) -> dict[str, Any]:
        return {
            "antenna": vp.antenna,
            "start_seconds": vp.start_seconds,
            "end_seconds": vp.end_seconds,
            "duration_hours": vp.duration_hours,
        }

    def _format_track(self, track: Any) -> dict[str, Any]:
        return {
            "action_id": track.action_id,
            "request_id": track.request_id,
            "mission_id": track.mission_id,
            "antenna": track.antenna,
            "trx_on": track.trx_on,
            "trx_off": track.trx_off,
            "setup_start": track.setup_start,
            "teardown_end": track.teardown_end,
            "duration_hours": round(track.duration_hours, 3),
        }

    def list_unsatisfied_requests(self, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        requests = self.scenario.list_unsatisfied_requests()
        page = requests[offset:offset + limit]
        items = []
        for req in page:
            items.append({
                "request_id": req.request_id,
                "mission_id": req.mission_id,
                "total_required_hours": req.total_required_hours,
                "remaining_hours": req.remaining_hours,
                "min_duration_hours": req.min_duration_hours,
                "setup_seconds": req.setup_seconds,
                "teardown_seconds": req.teardown_seconds,
                "summary": self._format_request_summary(req),
            })
        return {
            "total": len(requests),
            "offset": offset,
            "limit": limit,
            "items": items,
        }

    def get_antenna_status(self, include_blocked_ranges: bool = False) -> dict[str, Any]:
        status = self.scenario.get_antenna_status()
        result = {}
        for antenna, item in status.items():
            entry = {
                "hours_available": item.hours_available,
                "summary": f"{antenna}: {item.hours_available:.1f}h available",
            }
            if include_blocked_ranges:
                entry["blocked_ranges"] = item.blocked_ranges
            result[antenna] = entry
        return result

    def find_view_periods(
        self,
        request_id: str,
        min_duration_hours: float = 0,
        offset: int = 0,
        limit: int = 10,
    ) -> dict[str, Any]:
        try:
            vps = self.scenario.find_view_periods(request_id, min_duration_hours)
        except SatNetNotFoundError as exc:
            return {"error": str(exc), "status": 6523}

        page = vps[offset:offset + limit]
        return {
            "total": len(vps),
            "offset": offset,
            "limit": limit,
            "items": [self._format_view_period(vp) for vp in page],
            "hint": f"Found {len(vps)} scheduling opportunities. Windows are sorted by duration (longest first).",
        }

    def schedule_track(
        self,
        request_id: str,
        antenna: str,
        trx_on: int,
        trx_off: int,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        try:
            if dry_run:
                self.scenario._validate_track(request_id, antenna, trx_on, trx_off)
                return {
                    "status": 0,
                    "dry_run": True,
                    "message": "Validation successful - track would be accepted",
                }

            result = self.scenario.schedule_track(request_id, antenna, trx_on, trx_off)
            self._persist()
            duration_h = result.track.duration_hours
            buffer_h = (result.track.teardown_end - result.track.setup_start) / 3600 - duration_h
            return {
                "action_id": result.action_id,
                "track": self._format_track(result.track),
                "status": 0,
                "dry_run": False,
                "summary": f"Scheduled {duration_h:.2f}h track (+ {buffer_h:.2f}h setup/teardown buffer)",
            }
        except SatNetNotFoundError as exc:
            return {
                "error": str(exc),
                "status": 6523,
                "suggestion": "Check request_id from list_unsatisfied_requests()",
            }
        except SatNetValidationError as exc:
            return {
                "error": str(exc),
                "status": 8794,
                "suggestion": "Use find_view_periods() to get valid scheduling windows",
            }
        except SatNetConflictError as exc:
            return {
                "error": str(exc),
                "status": 8800,
                "suggestion": "Choose a different time or antenna with no conflicts",
            }

    def unschedule_track(self, action_id: str, dry_run: bool = False) -> dict[str, Any]:
        try:
            if dry_run:
                if action_id not in self.scenario.get_plan_status().tracks:
                    raise SatNetNotFoundError(f"Track not found: {action_id}")
                return {"status": 0, "dry_run": True, "message": "Track exists and can be unscheduled"}

            self.scenario.unschedule_track(action_id)
            self._persist()
            return {"status": 0, "dry_run": False}
        except SatNetNotFoundError as exc:
            return {"error": str(exc), "status": 6523}

    def get_plan_status(self) -> dict[str, Any]:
        status = self.scenario.get_plan_status()
        tracks = [self._format_track(track) for track in status.tracks.values()]
        metrics = None
        if status.metrics:
            metrics = {
                "total_allocated_hours": status.metrics.total_allocated_hours,
                "requests_satisfied": status.metrics.requests_satisfied,
                "requests_unsatisfied": status.metrics.requests_unsatisfied,
                "u_max": status.metrics.u_max,
                "u_rms": status.metrics.u_rms,
            }
        return {
            "num_tracks": len(tracks),
            "tracks": tracks,
            "metrics": metrics,
        }

    def commit_plan(self) -> dict[str, Any]:
        result = self.scenario.commit_plan(str(self.output_path))
        self._persist()
        return {
            "total_allocated_hours": result.metrics.total_allocated_hours,
            "requests_satisfied": result.metrics.requests_satisfied,
            "requests_unsatisfied": result.metrics.requests_unsatisfied,
            "u_max": result.metrics.u_max,
            "u_rms": result.metrics.u_rms,
            "plan_json_path": result.plan_json_path,
        }

    def reset(self) -> dict[str, Any]:
        self.scenario.reset()
        self._persist()
        return {"status": "reset"}

    def wait(self, seconds: float) -> dict[str, Any]:
        actual = max(0.0, min(float(seconds), 30.0))
        time.sleep(actual)
        return {"waited": actual}


def build_tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "list_unsatisfied_requests",
                "description": "List requests that still need DSN time allocation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "offset": {"type": "integer"},
                        "limit": {"type": "integer"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_antenna_status",
                "description": "Inspect DSN antenna availability and optionally blocked ranges.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "include_blocked_ranges": {"type": "boolean"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_view_periods",
                "description": "Find scheduleable view periods for a request.",
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
        },
        {
            "type": "function",
            "function": {
                "name": "schedule_track",
                "description": "Schedule a DSN track into the plan. Use dry_run first if unsure.",
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
        },
        {
            "type": "function",
            "function": {
                "name": "unschedule_track",
                "description": "Remove a previously scheduled DSN track.",
                "parameters": {
                    "type": "object",
                    "required": ["action_id"],
                    "properties": {
                        "action_id": {"type": "string"},
                        "dry_run": {"type": "boolean"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_plan_status",
                "description": "Inspect current SatNet schedule metrics and scheduled tracks.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "commit_plan",
                "description": "Finalize and export the current SatNet plan to plan.json.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "reset",
                "description": "Reset the current SatNet schedule to empty state.",
                "parameters": {"type": "object", "properties": {}},
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
