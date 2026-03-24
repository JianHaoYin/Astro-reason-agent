"""Prompt loading helpers for the SatNet local benchmark agent."""

from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "prompts"


def load_system_prompt() -> str:
    return (PROMPT_DIR / "system_prompt.txt").read_text(encoding="utf-8")


def build_user_prompt(mission_brief: str) -> str:
    template = (PROMPT_DIR / "user_prompt.txt").read_text(encoding="utf-8")
    return template.format(mission_brief=mission_brief)
