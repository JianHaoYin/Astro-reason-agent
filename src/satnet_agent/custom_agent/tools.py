"""Local tool registry for the benchmark planning agent."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

from planner.helpers import (
    action_key,
    filter_items,
    format_plan_status,
    format_satellite_summary,
    paginate,
    record_matches_filters,
    satellite_filter_key,
    satellite_summary_key,
    station_key,
    strip_key,
    target_key,
    to_llm_dict,
    window_filter_key,
    window_summary_key,
)
from planner.models import ConflictError, ResourceViolationError, ValidationError
from planner.scenario import Scenario
from planner.state import StateFile

from .tool_profiles import build_tool_specs, get_allowed_tool_names


class PlannerToolRegistry:
    """Benchmark-local planner tools with benchmark-specific allowlists."""

    def __init__(self, sandbox_dir: Path, allowed_tools: list[str] | None = None):
        self.sandbox_dir = sandbox_dir
        self.case_dir = sandbox_dir / "data"
        self.workspace_dir = sandbox_dir / "workspace"
        self.state_path = sandbox_dir / "state" / "scenario.json"
        self.output_path = self.workspace_dir / "plan.json"
        self.state_file = StateFile(self.state_path)

        os.environ["CASE_PATH"] = str(self.case_dir)
        os.environ["ASTROX_CASE_PATH"] = str(self.case_dir)
        os.environ["ASTROX_STATE_PATH"] = str(self.state_path)
        os.environ["ASTROX_OUTPUT_PATH"] = str(self.output_path)

        self.scenario = self._new_scenario()
        self._persist()

        all_tools: dict[str, Callable[..., Any]] = {
            "query_satellites": self.query_satellites,
            "query_targets": self.query_targets,
            "query_stations": self.query_stations,
            "register_strips": self.register_strips,
            "unregister_strips": self.unregister_strips,
            "query_strips": self.query_strips,
            "compute_strip_windows": self.compute_strip_windows,
            "query_windows": self.query_windows,
            "query_actions": self.query_actions,
            "compute_lighting_windows": self.compute_lighting_windows,
            "get_ground_track": self.get_ground_track,
            "evaluate_comms_latency": self.evaluate_comms_latency,
            "compute_access_windows": self.compute_access_windows,
            "stage_action": self.stage_action,
            "unstage_action": self.unstage_action,
            "get_plan_status": self.get_plan_status,
            "commit_plan": self.commit_plan,
            "reset_plan": self.reset_plan,
            "evaluate_revisit_gaps": self.evaluate_revisit_gaps,
            "evaluate_stereo_coverage": self.evaluate_stereo_coverage,
            "evaluate_polygon_coverage": self.evaluate_polygon_coverage,
            "wait": self.wait,
        }
        self.allowed_tools = set(allowed_tools or all_tools.keys())
        self._tools = {name: tool for name, tool in all_tools.items() if name in self.allowed_tools}

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
            return {"error": f"Tool '{name}' is not enabled for this benchmark."}
        return self._tools[name](**arguments)

    def query_satellites(self, filters: dict[str, Any] | None = None, offset: int = 0, limit: int = 10) -> Any:
        all_sats = self.scenario.query_satellites()
        filtered = filter_items(all_sats, filters or {}, satellite_filter_key)
        paged = paginate(filtered, offset, limit)
        results = [satellite_summary_key(sat) for sat in paged]
        if len(filtered) > offset + len(results):
            return {"satellites": results, "warning": f"Showing {len(results)} of {len(filtered)} matches."}
        return results

    def query_targets(self, filters: dict[str, Any] | None = None, offset: int = 0, limit: int = 10) -> Any:
        all_targets = self.scenario.query_targets()
        filtered = filter_items(all_targets, filters or {}, target_key)
        paged = paginate(filtered, offset, limit)
        results = [to_llm_dict(target) for target in paged]
        if len(filtered) > offset + len(results):
            return {"targets": results, "warning": f"Showing {len(results)} of {len(filtered)} matches."}
        return results

    def query_stations(self, filters: dict[str, Any] | None = None, offset: int = 0, limit: int = 10) -> Any:
        all_stations = self.scenario.query_stations()
        filtered = filter_items(all_stations, filters or {}, station_key)
        paged = paginate(filtered, offset, limit)
        results = [to_llm_dict(station) for station in paged]
        if len(filtered) > offset + len(results):
            return {"stations": results, "warning": f"Showing {len(results)} of {len(filtered)} matches."}
        return results

    def register_strips(self, strips: list[dict[str, Any]]) -> Any:
        registered = self.scenario.register_strips(strips)
        self._persist()
        return [strip_key(strip) for strip in registered]

    def unregister_strips(self, strip_ids: list[str]) -> Any:
        self.scenario.unregister_strips(strip_ids)
        self._persist()
        return {"status": "success", "removed": strip_ids}

    def query_strips(self, offset: int = 0, limit: int = 10) -> Any:
        strips = self.scenario.query_strips()
        paged = paginate(strips, offset, limit)
        results = [strip_key(strip) for strip in paged]
        if len(strips) > offset + len(results):
            return {"strips": results, "warning": f"Showing {len(results)} of {len(strips)} strips."}
        return results

    def compute_strip_windows(
        self,
        sat_ids: list[str],
        strip_ids: list[str],
        start_time: str,
        end_time: str,
        constraints: list[dict[str, Any]] | None = None,
        offset: int = 0,
        limit: int = 10,
    ) -> Any:
        warning = None
        if len(sat_ids) > 1 and len(strip_ids) > 1:
            warning = (
                f"Multiple satellites ({len(sat_ids)}) and multiple strips ({len(strip_ids)}) provided. "
                f"Only the first strip '{strip_ids[0]}' will be used."
            )
            strip_ids = [strip_ids[0]]

        windows = self.scenario.compute_strip_windows(sat_ids, strip_ids, start_time, end_time, constraints=constraints)
        registered = self.scenario.register_windows(windows)
        self._persist()
        paged = paginate(registered, offset, min(limit, 20))
        results = [window_summary_key(window) for window in paged]

        warnings = []
        if warning:
            warnings.append(warning)
        if len(registered) > offset + len(results):
            warnings.append(f"Showing {len(results)} of {len(registered)} windows.")
        if warnings:
            return {"warning": " | ".join(warnings), "windows": results}
        return results

    def query_windows(self, filters: dict[str, Any] | None = None, offset: int = 0, limit: int = 10) -> Any:
        windows = self.scenario.query_windows()
        if filters:
            filter_dicts = [window_filter_key(window) for window in windows]
            indices = [idx for idx, item in enumerate(filter_dicts) if record_matches_filters(item, filters)]
            windows = [windows[idx] for idx in indices]
        paged = paginate(windows, offset, min(limit, 20))
        results = [window_summary_key(window) for window in paged]
        if len(windows) > offset + len(results):
            return {"windows": results, "warning": f"Showing {len(results)} of {len(windows)} windows."}
        return results

    def query_actions(self, filters: dict[str, Any] | None = None, offset: int = 0, limit: int = 10) -> Any:
        action_dicts = [action_key(action) for action in self.scenario.query_actions()]
        if filters:
            action_dicts = [item for item in action_dicts if record_matches_filters(item, filters)]
        paged = paginate(action_dicts, offset, min(limit, 20))
        results = [to_llm_dict(item) for item in paged]
        if len(action_dicts) > offset + len(results):
            return {"actions": results, "warning": f"Showing {len(results)} of {len(action_dicts)} actions."}
        return results

    def compute_lighting_windows(
        self,
        sat_ids: list[str],
        start_time: str,
        end_time: str,
        offset: int = 0,
        limit: int = 10,
    ) -> Any:
        windows = self.scenario.compute_lighting_windows(sat_ids, start_time, end_time)
        paged = paginate(windows, offset, min(limit, 20))
        results = [to_llm_dict(window) for window in paged]
        if len(windows) > offset + len(results):
            return {"lighting_windows": results, "warning": f"Showing {len(results)} of {len(windows)} windows."}
        return results

    def get_ground_track(
        self,
        satellite_id: str,
        start_time: str,
        end_time: str,
        step_sec: float = 60.0,
        filter_polygon: list[list[float]] | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Any:
        polygon = None
        if filter_polygon:
            polygon = [(pt[0], pt[1]) for pt in filter_polygon]
        points = self.scenario.get_ground_track(
            satellite_id,
            start_time,
            end_time,
            step_sec=step_sec,
            filter_polygon=polygon,
        )
        total_count = len(points)
        paged = paginate(points, offset, min(limit, 200))
        results = [
            {
                "lat": round(point.lat, 4),
                "lon": round(point.lon, 4),
                "time": point.time.isoformat(),
            }
            for point in paged
        ]
        if total_count > offset + len(results):
            return {
                "points": results,
                "total_count": total_count,
                "returned_count": len(results),
                "warning": f"Showing {len(results)} of {total_count} points.",
            }
        return results

    def evaluate_comms_latency(
        self,
        source_station_id: str,
        dest_station_id: str,
        start_time: str,
        end_time: str,
        sample_step_sec: float = 60.0,
    ) -> Any:
        result = self.scenario.evaluate_comms_latency(
            source_station_id,
            dest_station_id,
            start_time,
            end_time,
            sample_step_sec,
        )
        formatted_windows = []
        total_established_sec = 0.0
        for window in result.windows:
            total_established_sec += window.duration_sec
            latencies = [sample.latency_ms for sample in window.latency_samples]
            formatted_windows.append(
                {
                    "path": " > ".join(window.path),
                    "start": window.start.isoformat(),
                    "end": window.end.isoformat(),
                    "duration_sec": round(window.duration_sec, 1),
                    "latency_min_ms": round(min(latencies), 2) if latencies else None,
                    "latency_max_ms": round(max(latencies), 2) if latencies else None,
                    "latency_mean_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
                    "sample_count": len(latencies),
                }
            )
        return {
            "window_count": len(formatted_windows),
            "total_duration_minutes": round(total_established_sec / 60.0, 2),
            "windows": formatted_windows,
        }

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
        results = [window_summary_key(window) for window in paged]
        if len(registered) > offset + len(results):
            return {"windows": results, "warning": f"Showing {len(results)} of {len(registered)} windows."}
        return results

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

    def unstage_action(self, action_id: str, dry_run: bool = False) -> Any:
        if action_id not in self.scenario.staged_actions:
            return {"status": "error", "reason": f"Action '{action_id}' not found"}
        if dry_run:
            snapshot = self.scenario.export_to_state()
            horizon_start = self.scenario.horizon_start
            horizon_end = self.scenario.horizon_end
            self.scenario.unstage_action(action_id)
            status = self.scenario.get_plan_status()
            self._restore_snapshot(snapshot)
            return {
                "action_id": action_id,
                "status": "can_unstage",
                "projected_status": to_llm_dict(format_plan_status(status, horizon_start, horizon_end)),
            }

        result = self.scenario.unstage_action(action_id)
        self._persist()
        return {"action_id": result.action_id, "status": "unstaged"}

    def get_plan_status(self) -> Any:
        status = self.scenario.get_plan_status()
        return to_llm_dict(format_plan_status(status, self.scenario.horizon_start, self.scenario.horizon_end))

    def commit_plan(self) -> Any:
        result = self.scenario.commit_plan(path=str(self.output_path))
        self._persist()
        satellites = [format_satellite_summary(metrics) for metrics in result.metrics.satellites.values()]
        violations = [
            {
                "action_id": violation.action_id,
                "type": violation.violation_type,
                "message": violation.message,
                "conflicting_action_ids": violation.conflicting_action_ids,
            }
            for violation in result.violations
        ]
        return {
            "valid": result.valid,
            "violations": violations if violations else None,
            "action_count": result.metrics.total_actions,
            "total_observations": result.metrics.total_observations,
            "total_downlinks": result.metrics.total_downlinks,
            "satellites": satellites,
            "plan_json_path": result.plan_json_path,
        }

    def reset_plan(self) -> Any:
        self.scenario.reset_plan()
        self._persist()
        return {"status": "reset", "message": "Plan reset to initial state"}

    def evaluate_revisit_gaps(self, target_ids: list[str], start_time: str | None = None, end_time: str | None = None) -> Any:
        return [to_llm_dict(item) for item in self.scenario.evaluate_revisit_gaps(target_ids, start_time, end_time)]

    def evaluate_stereo_coverage(self, target_ids: list[str], min_separation_deg: float = 10.0) -> Any:
        return [to_llm_dict(item) for item in self.scenario.evaluate_stereo_coverage(target_ids, min_separation_deg)]

    def evaluate_polygon_coverage(self, polygon: list[list[float]]) -> Any:
        poly_tuples = [(point[0], point[1]) for point in polygon]
        result = to_llm_dict(self.scenario.evaluate_polygon_coverage(poly_tuples))
        result.pop("coverage_grid", None)
        return result

    def wait(self, seconds: float) -> Any:
        actual = max(0.0, min(float(seconds), 30.0))
        time.sleep(actual)
        return {"waited": actual}


__all__ = ["PlannerToolRegistry", "build_tool_specs", "get_allowed_tool_names"]
