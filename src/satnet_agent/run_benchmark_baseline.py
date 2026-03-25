#!/usr/bin/env python3
"""Run SatNet benchmark weeks with a local Ollama-based planning agent."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from satnet_agent.custom_agent.agent_loop import LocalPlanningAgent  # noqa: E402
from satnet_agent.run_benchmark import score_result, setup_sandbox  # noqa: E402

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "benchmark_runs" / "local_agent_satnet"
DEFAULT_WEEK = 40
DEFAULT_YEAR = 2018
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "qwen3.5:27b"
DEFAULT_TIMEOUT = 300
DEFAULT_MAX_TURNS = 1000


def run_single_case(
    week: int,
    year: int,
    ollama_base_url: str,
    model: str,
    output_root: Path,
    timeout: int,
    max_turns: int,
    include_related_works: bool = False,
) -> dict[str, object]:
    output_dir = output_root / f"week_{week}_{year}" / model.replace(":", "_").replace("/", "__")
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    sandbox_dir = setup_sandbox(week, year, output_dir, include_related_works=include_related_works)

    os.environ["SATNET_WEEK"] = str(week)
    os.environ["SATNET_YEAR"] = str(year)
    os.environ["SATNET_OUTPUT_PATH"] = str(sandbox_dir / "workspace" / "plan.json")

    agent = LocalPlanningAgent(
        base_url=ollama_base_url,
        model=model,
        timeout=timeout,
        max_turns=max_turns,
    )
    agent_result = agent.run(sandbox_dir=sandbox_dir, output_dir=output_dir)

    score = score_result(sandbox_dir, week, year)

    result = {
        "week": week,
        "year": year,
        "ollama_base_url": ollama_base_url,
        "model": model,
        "elapsed_seconds": round(time.time() - start, 2),
        "output_dir": str(output_dir),
        "agent": agent_result,
        "score": score,
    }

    (output_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SatNet weeks with a local Ollama planning agent.")
    parser.add_argument("--week", type=int, default=DEFAULT_WEEK, help="ISO week number")
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR, help="Year")
    parser.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL, help="Ollama OpenAI-compatible base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Local Ollama model name")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output root inside the repository")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout in seconds for each Ollama request")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS, help="Maximum planning turns")
    parser.add_argument("--related-works", action="store_true", help="Copy related works into the sandbox workspace")

    args = parser.parse_args()
    result = run_single_case(
        week=args.week,
        year=args.year,
        ollama_base_url=args.ollama_base_url,
        model=args.model,
        output_root=args.output_dir,
        timeout=args.timeout,
        max_turns=args.max_turns,
        include_related_works=args.related_works,
    )
    #print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
