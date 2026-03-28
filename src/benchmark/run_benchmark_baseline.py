#!/usr/bin/env python3
"""Run benchmark cases with a simple local Ollama-based planning agent.

This runner reuses the existing benchmark infrastructure:
- case generation and mission brief templates
- sandbox setup
- verifier/scoring

The only part it replaces is the external Claude Code agent. Instead, it uses a
small in-process tool-calling loop backed by a local Ollama model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from runtime_requirements import require_python_312

require_python_312()

from benchmark.custom_agent.agent_loop import LocalPlanningAgent  # noqa: E402
from benchmark.run_benchmark import extract_key_metrics, score_result, setup_sandbox  # noqa: E402

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "benchmark_runs" / "local_agent"
DEFAULT_BENCHMARK = "latency-optimization"
DEFAULT_CASE_ID = "case_0001"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_TIMEOUT = 300
DEFAULT_MAX_TURNS = 16


def run_single_case(
    benchmark: str,
    case_id: str,
    ollama_base_url: str,
    model: str,
    output_root: Path,
    timeout: int,
    max_turns: int,
) -> dict[str, object]:
    """Run one benchmark case with the local planning agent."""
    output_dir = output_root / benchmark / case_id / model.replace(":", "_").replace("/", "__")
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    sandbox_dir = setup_sandbox(benchmark, case_id, output_dir)

    agent = LocalPlanningAgent(
        base_url=ollama_base_url,
        model=model,
        timeout=timeout,
        max_turns=max_turns,
    )
    agent_result = agent.run(sandbox_dir=sandbox_dir, output_dir=output_dir)

    score = score_result(benchmark, sandbox_dir, case_id)
    #TODO：返回是空
    key_metrics = extract_key_metrics(benchmark, score)

    result = {
        "benchmark": benchmark,
        "case_id": case_id,
        "ollama_base_url": ollama_base_url,
        "model": model,
        "elapsed_seconds": round(time.time() - start, 2),
        "output_dir": str(output_dir),
        "agent": agent_result,
        "score": score,
        "key_metrics": key_metrics,
    }

    (output_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark cases with a local Ollama planning agent.")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help="Benchmark name, for example revisit-optimization")
    parser.add_argument("--case", default=DEFAULT_CASE_ID, help="Case id, for example case_0001")
    parser.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL, help="Ollama OpenAI-compatible base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Local Ollama model name")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output root inside the repository")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout in seconds for each Ollama request")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS, help="Maximum planning turns")

    args = parser.parse_args()
    result = run_single_case(
        benchmark=args.benchmark,
        case_id=args.case,
        ollama_base_url=args.ollama_base_url,
        model=args.model,
        output_root=args.output_dir,
        timeout=args.timeout,
        max_turns=args.max_turns,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
