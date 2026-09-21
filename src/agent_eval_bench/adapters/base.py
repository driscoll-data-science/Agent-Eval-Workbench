"""Adapter protocol: one method that turns (task, workspace, profile) into a RunRecord."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..models import Profile, RunRecord, Task


class AgentAdapter(Protocol):
    name: str

    def run(
        self,
        task: Task,
        rep: int,
        workspace: Path | None,
        profile: Profile,
        timeout_s: int,
    ) -> RunRecord: ...


def get_adapter(profile: Profile) -> AgentAdapter:
    if profile.adapter == "fake":
        from .fake import FakeAgent

        return FakeAgent()
    if profile.adapter == "claude_code":
        from .claude_code import ClaudeCodeCLIAdapter

        return ClaudeCodeCLIAdapter()
    if profile.adapter == "codex":
        from .codex import CodexCLIAdapter

        return CodexCLIAdapter()
    if profile.adapter == "claude_sdk":
        from .claude_sdk import ClaudeSDKAdapter

        return ClaudeSDKAdapter()
    raise ValueError(f"unknown adapter {profile.adapter!r}")


def summarize(value: object, limit: int = 300) -> str:
    """Compact, safe string for tool inputs/outputs stored in records and spans."""
    try:
        import json

        s = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        s = str(value)
    s = s.replace("\n", "\\n")
    return s if len(s) <= limit else s[: limit - 1] + "…"
