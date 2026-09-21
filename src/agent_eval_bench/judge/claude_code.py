"""Judge backend that runs Claude Code headless on the user's subscription.

Recipe verified 2026-09-17: ``--strict-mcp-config --setting-sources ""`` keeps the keychain
login while dropping CLAUDE.md, plugins, and MCP tools; ``--system-prompt`` replaces the
default system prompt; ``--tools ""`` removes all tools; ``--json-schema`` returns the parsed
verdict in ``structured_output``. Measured at about 1.2K prompt tokens per verdict.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from ..models import Criterion, RunRecord, Task, Verdict
from .prompts import SYSTEM_TEMPLATE, VERDICT_SCHEMA, build_user_message


class ClaudeCodeJudge:
    name = "claude_code"

    def __init__(
        self,
        model: str,
        timeout_s: int = 180,
        effort: str | None = "high",
        work_dir: Path | None = None,
        cli: str | None = None,
    ):
        self.model = model
        self.timeout_s = timeout_s
        self.effort = effort
        self.cli = cli or shutil.which("claude") or "claude"
        self.work_dir = work_dir or Path.home() / ".aeb" / "judge_cwd"
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def command(self) -> list[str]:
        cmd = [
            self.cli,
            "-p",
            "--model",
            self.model,
            "--strict-mcp-config",
            "--setting-sources",
            "",
            "--system-prompt",
            SYSTEM_TEMPLATE,
            "--tools",
            "",
            "--max-turns",
            "2",
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(VERDICT_SCHEMA),
        ]
        if self.effort:
            cmd += ["--effort", self.effort]
        return cmd

    def judge(self, criterion: Criterion, task: Task, record: RunRecord, version: str) -> Verdict:
        user_msg = build_user_message(criterion, task, record)
        base = Verdict(
            task_id=task.id,
            rep=record.rep,
            criterion=criterion.name,
            passed=None,
            judge_backend=self.name,
            judge_model=self.model,
            judge_version=version,
        )
        env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE_CODE_ENTRYPOINT")}
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                self.command(),
                input=user_msg,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                cwd=self.work_dir,
                env=env,
            )
        except subprocess.TimeoutExpired:
            base.error = f"judge timeout after {self.timeout_s}s"
            return base
        base.latency_ms = int((time.perf_counter() - t0) * 1000)
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            base.error = f"judge returned non-JSON (exit {proc.returncode}): {proc.stdout[:300]!r} {proc.stderr[:300]!r}"
            return base
        usage = data.get("usage") or {}
        base.input_tokens = int(usage.get("input_tokens") or 0)
        base.output_tokens = int(usage.get("output_tokens") or 0)
        base.cache_read_tokens = int(usage.get("cache_read_input_tokens") or 0)
        base.cache_write_tokens = int(usage.get("cache_creation_input_tokens") or 0)
        model_usage = data.get("modelUsage") or {}
        if model_usage:
            # Prefer the dated id the API actually served, e.g. claude-opus-5 vs claude-opus-5-2026xxxx.
            base.judge_model = sorted(model_usage.keys(), key=len)[-1]
        if data.get("is_error"):
            base.error = f"judge error: {str(data.get('result'))[:300]}"
            return base
        so = data.get("structured_output")
        if not isinstance(so, dict) or "passed" not in so:
            # Fall back to parsing the result text as JSON.
            try:
                so = json.loads(data.get("result") or "")
            except Exception:
                base.error = (
                    f"judge produced no structured verdict: {str(data.get('result'))[:300]!r}"
                )
                return base
        base.passed = bool(so.get("passed"))
        base.rationale = str(so.get("rationale") or "")
        return base
