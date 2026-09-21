"""Codex CLI adapter: ``codex exec --json`` as a subprocess.

Flags verified against Codex CLI 0.154.0 ``exec --help`` on 2026-09-17; event shapes follow
the documented ``--json`` JSONL schema (no authenticated run was made). Codex reports no USD
anywhere, so ``reported_cost_usd`` is always None and the kit prices the tokens.

Usage semantics (OpenAI Responses API, mirrored by Codex's TokenUsage): ``cached_input_tokens``
is a subset of ``input_tokens`` and ``reasoning_output_tokens`` is a subset of
``output_tokens``. The single ModelCall therefore carries uncached input (input minus cached),
cache_read = cached, and output = output_tokens unchanged; the reasoning count is kept in
``raw["reasoning_output_tokens"]``. Adding either subset on top would bill it twice.

Isolation: a throwaway ``CODEX_HOME`` holding only a copy of ``~/.codex/auth.json`` (if present),
plus ``--ephemeral`` (no session files), ``--ignore-user-config`` (no config.toml) and
``-c project_doc_max_bytes=0`` (no AGENTS.md). ``--full-auto`` is rejected by this version and
is never passed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..models import ModelCall, Profile, RunRecord, Task, ToolCall
from .base import summarize
from .claude_code import (
    EXIT_NOT_FOUND,
    EXIT_TIMEOUT,
    ParsedRun,
    _as_text,
    _bump,
    _int,
    _tail,
    refusal_heuristic,
    resolve_cwd,
)

DEFAULT_MODEL = "codex-default"
ITEM_EVENTS = ("item.started", "item.updated", "item.completed")


def _toml_value(v: Any) -> str:
    """Render a ``-c key=value`` value. Bare strings are accepted by Codex as literals."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int | float | str):
        return str(v)
    return json.dumps(v)


def parse_jsonl(lines: Iterable[str], model: str | None = None) -> ParsedRun:
    """Normalize a ``codex exec --json`` stream. Pure: no I/O.

    Items are tracked by id across started/updated/completed so a run that dies mid-command
    still yields the partial tool call. Usage comes from the last ``turn.completed`` (cumulative
    for the thread) as a single ModelCall.
    """
    parsed = ParsedRun()
    items: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    usage: dict[str, Any] | None = None
    turns = 0
    errors: list[str] = []
    counters: dict[str, int] = {}

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            _bump(counters, "bad_lines")
            continue
        if not isinstance(ev, dict):
            _bump(counters, "bad_lines")
            continue
        etype = ev.get("type")
        if etype == "thread.started":
            parsed.session_id = ev.get("thread_id") or parsed.session_id
        elif etype == "turn.started":
            turns += 1
        elif etype in ITEM_EVENTS:
            item = ev.get("item")
            if not isinstance(item, dict):
                continue
            iid = str(item.get("id") or f"anon-{len(order)}")
            if iid not in items:
                order.append(iid)
                items[iid] = {}
            items[iid].update(item)
            items[iid]["_event"] = etype
        elif etype == "turn.completed":
            if isinstance(ev.get("usage"), dict):
                usage = ev["usage"]
        elif etype == "turn.failed":
            err = ev.get("error")
            errors.append(str(err.get("message") if isinstance(err, dict) else err))
        elif etype == "error":
            errors.append(str(ev.get("message")))
        else:
            _bump(counters, f"other_event:{etype or '?'}")

    texts: list[str] = []
    for iid in order:
        item = items[iid]
        itype = item.get("type")
        status = item.get("status")
        if itype == "agent_message":
            if item.get("text"):
                texts.append(str(item["text"]))
        elif itype == "reasoning":
            _bump(counters, "reasoning_items")
        elif itype == "command_execution":
            exit_code = item.get("exit_code")
            tc = _tool(parsed, "shell", {"command": item.get("command")})
            tc.output_full = item.get("aggregated_output")
            tc.output_summary = summarize(tc.output_full) if tc.output_full is not None else None
            tc.is_error = exit_code not in (0, None) or status == "failed"
        elif itype == "file_change":
            tc = _tool(parsed, "apply_patch", {"changes": item.get("changes") or []})
            tc.output_full = status
            tc.output_summary = str(status) if status else None
            tc.is_error = status == "failed"
        elif itype == "mcp_tool_call":
            name = f"mcp:{item.get('server') or '?'}.{item.get('tool') or '?'}"
            tc = _tool(parsed, name, item.get("arguments"))
            err = item.get("error")
            tc.output_full = err if err else item.get("result")
            tc.output_summary = summarize(tc.output_full) if tc.output_full is not None else None
            tc.is_error = bool(err) or status == "failed"
        elif itype == "web_search":
            _tool(parsed, "web_search", {"query": item.get("query")})
        elif itype == "todo_list":
            _bump(counters, "todo_items")
        elif itype == "error":
            errors.append(str(item.get("message")))
        else:
            _bump(counters, f"other_item:{itype or '?'}")

    parsed.final_text = "\n\n".join(texts)
    parsed.model = model or DEFAULT_MODEL
    parsed.num_turns = turns or None
    if usage is not None:
        inp = _int(usage.get("input_tokens"))
        cached = _int(usage.get("cached_input_tokens"))
        parsed.model_calls.append(
            ModelCall(
                index=0,
                model=parsed.model,
                input_tokens=max(0, inp - cached),
                output_tokens=_int(usage.get("output_tokens")),
                cache_read_tokens=cached,
                cache_write_tokens=_int(usage.get("cache_write_input_tokens")),
            )
        )
        parsed.extras["reasoning_output_tokens"] = _int(usage.get("reasoning_output_tokens"))
        parsed.extras["usage_raw"] = usage
    if errors:
        parsed.is_error = True
        parsed.error = "codex: " + "; ".join(errors)[:500]
    parsed.extras.update(counters)
    return parsed


def _tool(parsed: ParsedRun, name: str, input_full: Any) -> ToolCall:
    tc = ToolCall(
        index=len(parsed.tool_calls),
        name=name,
        input_summary=summarize(input_full) if input_full is not None else "",
        input_full=input_full,
    )
    parsed.tool_calls.append(tc)
    return tc


class CodexCLIAdapter:
    name = "codex"

    def __init__(self, cli: str | None = None):
        self.cli = cli or shutil.which("codex") or "codex"

    def build_command(
        self,
        task: Task,
        workspace: Path | None,
        profile: Profile,
        output_file: Path | None = None,
    ) -> list[str]:
        """Pure. ``-`` at the end reads the prompt from stdin."""
        p = profile.params
        cmd = [self.cli, "exec", "--json", "--skip-git-repo-check", "--ephemeral"]
        if workspace is not None:
            cmd += ["-C", str(workspace)]
        if profile.model:
            cmd += ["-m", str(profile.model)]
        if p.get("sandbox"):
            cmd += ["-s", str(p["sandbox"])]
        if p.get("ignore_user_config", True):
            cmd.append("--ignore-user-config")
        if p.get("disable_agents_md", True):
            cmd += ["-c", "project_doc_max_bytes=0"]
        if p.get("reasoning_effort"):
            cmd += ["-c", f"model_reasoning_effort={_toml_value(p['reasoning_effort'])}"]
        for k, v in (p.get("config_overrides") or {}).items():
            cmd += ["-c", f"{k}={_toml_value(v)}"]
        if output_file is not None:
            cmd += ["-o", str(output_file)]
        cmd.append("-")
        return cmd

    @staticmethod
    def prepare_home(base_tmp: Path) -> Path:
        """Create a throwaway CODEX_HOME under ``base_tmp`` holding only a copy of the user's
        ``~/.codex/auth.json`` (when present). Honors $HOME, so tests can redirect it."""
        home = Path(base_tmp) / "codex_home"
        home.mkdir(parents=True, exist_ok=True)
        src = Path.home() / ".codex" / "auth.json"
        if src.is_file():
            dst = home / "auth.json"
            shutil.copyfile(src, dst)
            dst.chmod(0o600)
        return home

    def run(
        self, task: Task, rep: int, workspace: Path | None, profile: Profile, timeout_s: int
    ) -> RunRecord:
        import tempfile

        t0 = time.perf_counter()
        p = profile.params
        cwd, tmp_ws = resolve_cwd(workspace, "aeb-codex-qa-")
        tmp = Path(tempfile.mkdtemp(prefix="aeb-codex-"))
        env = dict(os.environ)
        isolated = bool(p.get("isolate_home", True))
        stdout = stderr = ""
        exit_status = 0
        error: str | None = None
        last_message: str | None = None
        out_file = tmp / "last_message.txt"
        cmd = self.build_command(task, cwd, profile, output_file=out_file)
        try:
            if isolated:
                env["CODEX_HOME"] = str(self.prepare_home(tmp))
            try:
                proc = subprocess.run(
                    cmd,
                    input=task.prompt,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout_s,
                    cwd=cwd,
                    env=env,
                )
                stdout, stderr, exit_status = proc.stdout, proc.stderr, proc.returncode
            except subprocess.TimeoutExpired as e:
                stdout, stderr = _as_text(e.stdout), _as_text(e.stderr)
                exit_status = EXIT_TIMEOUT
                error = f"timeout after {timeout_s}s"
            except FileNotFoundError:
                exit_status = EXIT_NOT_FOUND
                error = f"codex CLI not found: {self.cli!r}"
            if out_file.is_file():
                last_message = out_file.read_text(encoding="utf-8", errors="replace")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            if tmp_ws is not None:
                shutil.rmtree(tmp_ws, ignore_errors=True)
        wall_ms = int((time.perf_counter() - t0) * 1000)

        parsed = parse_jsonl(stdout.splitlines(), model=profile.model)
        final_text = parsed.final_text or (last_message or "").strip()
        if error is None:
            error = parsed.error
        if error is None and exit_status != 0:
            error = f"codex exited {exit_status}: {_tail(stderr, 500)}"
        raw = parsed.raw()
        raw.update(
            {
                "command": cmd,
                "cwd": str(cwd),
                "codex_home_isolated": isolated,
                "last_message_file": last_message is not None,
            }
        )
        if stderr.strip():
            raw["stderr_tail"] = _tail(stderr)
        return RunRecord(
            task_id=task.id,
            rep=rep,
            agent=self.name,
            profile=profile.name,
            model=parsed.model,
            final_text=final_text,
            model_calls=parsed.model_calls,
            tool_calls=parsed.tool_calls,
            wall_ms=wall_ms,
            exit_status=exit_status,
            error=error,
            refused=refusal_heuristic(final_text) and not parsed.tool_calls,
            workspace_path=str(workspace) if workspace else None,
            reported_cost_usd=None,
            raw=raw,
        )
