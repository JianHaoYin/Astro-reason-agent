#!/usr/bin/env python3
"""Wrapper for running the existing SatNet benchmark via a local Ollama model.

This script leaves the current SatNet runner unchanged. It starts a local
Anthropic-compatible proxy in front of Ollama, then delegates to the existing
`src/satnet_agent/run_benchmark.py` entrypoint.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUNNER_PATH = PROJECT_ROOT / "src" / "satnet_agent" / "run_benchmark.py"
PROXY_APP = "ollama_anthropic_proxy:app"
DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"


def wait_for_proxy(url: str, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=1.0) as response:
                if response.status == 200:
                    return
        except URLError:
            time.sleep(0.2)
    raise RuntimeError(f"Proxy did not become ready within {timeout:.1f}s: {url}")


def start_proxy(host: str, port: int, ollama_base_url: str, ollama_model: str) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["OLLAMA_BASE_URL"] = ollama_base_url.rstrip("/")
    env["OLLAMA_MODEL"] = ollama_model
    env.setdefault("OLLAMA_API_KEY", "ollama")

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        PROXY_APP,
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    return subprocess.Popen(cmd, env=env, cwd=PROJECT_ROOT, text=True)


def build_runner_command(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    for flag in ("--model", "--models"):
        if flag in passthrough:
            raise SystemExit(f"Do not pass {flag} to this wrapper. Use --anthropic-model or --ollama-model instead.")

    return [
        sys.executable,
        str(RUNNER_PATH),
        *passthrough,
        "--model",
        f"anthropic::{args.anthropic_model}",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SatNet with a local Ollama model via a compatibility proxy.")
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL, help="Local Ollama model name")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434/v1", help="Ollama OpenAI-compatible base URL")
    parser.add_argument("--proxy-host", default="127.0.0.1", help="Local host for the compatibility proxy")
    parser.add_argument("--proxy-port", type=int, default=8008, help="Local port for the compatibility proxy")
    parser.add_argument("--anthropic-model", default="local-ollama", help="Placeholder model id exposed to the existing runner")
    args, passthrough = parser.parse_known_args()

    proxy = start_proxy(args.proxy_host, args.proxy_port, args.ollama_base_url, args.ollama_model)
    proxy_url = f"http://{args.proxy_host}:{args.proxy_port}/health"

    try:
        wait_for_proxy(proxy_url)

        env = os.environ.copy()
        env["ANTHROPIC_BASE_URL"] = f"http://{args.proxy_host}:{args.proxy_port}/v1"
        env.setdefault("ANTHROPIC_AUTH_TOKEN", "ollama")

        runner_cmd = build_runner_command(args, passthrough)
        print(f"Proxy ready: {env['ANTHROPIC_BASE_URL']}")
        print(f"Ollama model: {args.ollama_model}")
        print(f"Runner command: {' '.join(runner_cmd)}")
        result = subprocess.run(runner_cmd, env=env, cwd=PROJECT_ROOT)
        return result.returncode
    finally:
        if proxy.poll() is None:
            proxy.send_signal(signal.SIGTERM)
            try:
                proxy.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proxy.kill()
                proxy.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
