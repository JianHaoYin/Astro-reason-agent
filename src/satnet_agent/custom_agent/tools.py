"""Local tool registry for the SatNet planning agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from satnet_agent.models import (
    SatNetAntennaStatus,
    SatNetNotFoundError,
    SatNetRequest,
    SatNetTrack,
    SatNetValidationError,
    SatNetConflictError,
    SatNetViewPeriod,
)
from satnet_agent.scenario import SatNetScenario
from satnet_agent.state import SatNetStateFile

from .tool_profiles import build_tool_specs, get_allowed_tool_names


class PlannerToolRegistry:
    """SatNet-local tool registry with an MCP-like API surface."""

    def __init__(self, sandbox_dir: Path, allowed_tools: list[str] | None = None):
        self.sandbox_dir = sandbox_dir
        self.case_dir = sandbox_dir / "data"
        self.workspace_dir = sandbox_dir / "workspace"
        self.state_path = sandbox_dir / "state" / "scenario.json"
        self.output_path = self.workspace_dir / "plan.json"
        self.problems_path = self.case_dir / "problems.json"
        self.maintenance_path = self.case_dir / "maintenance.csv"
        self.week = int(os.environ.get("SATNET_WEEK", "40"))
        self.year = int(os.environ.get("SATNET_YEAR", "2018"))
        self.state_file = SatNetStateFile(self.state_path)

        os.environ["SATNET_STATE_PATH"] = str(self.state_path)
        os.environ["SATNET_PROBLEMS_PATH"] = str(self.problems_path)
        os.environ["SATNET_MAINTENANCE_PATH"] = str(self.maintenance_path)
        os.environ["SATNET_OUTPUT_PATH"] = str(self.output_path)
        os.environ["SATNET_WEEK"] = str(self.week)
        os.environ["SATNET_YEAR"] = str(self.year)

        self.scenario = self._load_or_initialize_scenario()
        self._persist()

        all_tools: dict[str, Callable[..., Any]] = {
            "list_unsatisfied_requests": self.list_unsatisfied_requests,
            "get_antenna_status": self.get_antenna_status,
            "find_view_periods": self.find_view_periods,
            "schedule_track": self.schedule_track,
            "unschedule_track": self.unschedule_track,
            "get_plan_status": self.get_plan_status,
            "commit_plan": self.commit_plan,
            "reset": self.reset,
        }
        self.allowed_tools = set(allowed_tools or all_tools.keys())
        self._tools = {name: tool for name, tool in all_tools.items() if name in self.allowed_tools}

    def _load_or_initialize_scenario(self) -> SatNetScenario:
        if not self.state_file.exists():
            self.state_file.initialize(
                str(self.problems_path),
                str(self.maintenance_path),
                self.week,
                self.year,
            )
        state = self.state_file.read()
        if state is None:
            raise RuntimeError("SatNet state file could not be initialized")
        return SatNetScenario.from_state(state)

    def _persist(self) -> None:
        self.state_file.write(self.scenario.to_state())

    def invoke(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self._tools:
            return {"error": f"Tool '{name}' is not enabled for this benchmark."}
        try:
            return self._tools[name](**arguments)
        except TypeError as exc:
            return {"error": f"Invalid arguments for {name}: {exc}"}
        except Exception as exc:  # pragma: no cover - last-resort safety for agent runs
            return {"error": f"{name} failed: {exc}"}

    def _format_request(self, req: SatNetRequest) -> dict[str, Any]:
        return {
            "request_id": req.request_id,
            "mission_id": req.mission_id,
            "total_required_hours": req.total_required_hours,
            "remaining_hours": req.remaining_hours,
            "min_duration_hours": req.min_duration_hours,
            "setup_seconds": req.setup_seconds,
            "teardown_seconds": req.teardown_seconds,
            "summary": f"{req.request_id}: {req.remaining_hours:.1f}h remaining (min {req.min_duration_hours:.1f}h)",
        }

    def _format_view_period(self, vp: SatNetViewPeriod) -> dict[str, Any]:
        return {
            "antenna": vp.antenna,
            "start_seconds": vp.start_seconds,
            "end_seconds": vp.end_seconds,
            "duration_hours": vp.duration_hours,
        }

    def _format_track(self, track: SatNetTrack) -> dict[str, Any]:
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

    def _format_antenna_status(self, status: SatNetAntennaStatus, include_blocked_ranges: bool) -> dict[str, Any]:
        result = {
            "hours_available": status.hours_available,
            "summary": f"{status.antenna}: {status.hours_available:.1f}h available",
        }
        if include_blocked_ranges:
            result["blocked_ranges"] = status.blocked_ranges
        return result

    def list_unsatisfied_requests(self, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        requests = self.scenario.list_unsatisfied_requests()
        page = requests[offset:offset + limit]
        return {
            "total": len(requests),
            "offset": offset,
            "limit": limit,
            "items": [self._format_request(req) for req in page],
        }

    def get_antenna_status(self, include_blocked_ranges: bool = False) -> dict[str, Any]:
        status = self.scenario.get_antenna_status()
        return {
            antenna: self._format_antenna_status(item, include_blocked_ranges)
            for antenna, item in status.items()
        }

    def find_view_periods(
        self,
        request_id: str,
        min_duration_hours: float = 0,
        offset: int = 0,
        limit: int = 10,
    ) -> dict[str, Any]:
        try:
            periods = self.scenario.find_view_periods(request_id, min_duration_hours)
        except SatNetNotFoundError as exc:
            return {"error": str(exc), "status": 6523}

        page = periods[offset:offset + limit]
        return {
            "total": len(periods),
            "offset": offset,
            "limit": limit,
            "items": [self._format_view_period(vp) for vp in page],
            "hint": f"Found {len(periods)} scheduling opportunities. Windows are sorted by duration (longest first).",
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
                return {
                    "status": 0,
                    "dry_run": True,
                    "message": "Track exists and can be unscheduled",
                }

            self.scenario.unschedule_track(action_id)
            self._persist()
            return {"status": 0, "dry_run": False}
        except SatNetNotFoundError as exc:
            return {"error": str(exc), "status": 6523}

    def get_plan_status(self) -> dict[str, Any]:
        status = self.scenario.get_plan_status()
        metrics = None
        if status.metrics is not None:
            metrics = {
                "total_allocated_hours": status.metrics.total_allocated_hours,
                "requests_satisfied": status.metrics.requests_satisfied,
                "requests_unsatisfied": status.metrics.requests_unsatisfied,
                "u_max": status.metrics.u_max,
                "u_rms": status.metrics.u_rms,
            }
        return {
            "num_tracks": len(status.tracks),
            "tracks": [self._format_track(track) for track in status.tracks.values()],
            "metrics": metrics,
        }

    def commit_plan(self) -> dict[str, Any]:
        result = self.scenario.commit_plan(str(self.output_path))
        self._persist()
        return {
            "valid": True,
            "total_allocated_hours": result.metrics.total_allocated_hours,
            "requests_satisfied": result.metrics.requests_satisfied,
            "requests_unsatisfied": result.metrics.requests_unsatisfied,
            "u_max": result.metrics.u_max,
            "u_rms": result.metrics.u_rms,
            "plan_json_path": result.plan_json_path,
        }

    def reset(self) -> dict[str, str]:
        self.scenario.reset()
        self._persist()
        return {"status": "reset"}


__all__ = ["PlannerToolRegistry", "build_tool_specs", "get_allowed_tool_names"]
