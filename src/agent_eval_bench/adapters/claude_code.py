"""Claude Code CLI adapter: ``claude -p --output-format stream-json`` as a subprocess.

Verified against Claude Code 2.1.275 on 2026-09-17:
- ``--output-format stream-json`` in print mode refuses to start without ``--verbose``
  ("Error: When using --print, --output-format=stream-json requires --verbose").
- The prompt is read from stdin when no positional prompt is given.
- One ``assistant`` event is emitted per content block. Events belonging to the same API
  message share ``message.id`` and carry identical, message-level ``usage``. A ModelCall is
  therefore one message id, not one event.
- Per-message ``usage.output_tokens`` is the count from the first stream frame and undercounts
  badly (4 and 1 observed against a true total of 285). The final ``result.usage`` holds the
  accurate totals. The parser keeps per-message input and cache counts (which reconcile exactly)
  and attributes any shortfall against the result totals to the last model call, recording the
  delta in ``raw["usage_reconciliation"]``.
- ``result.total_cost_usd`` is a client-side estimate that also covers auxiliary calls not in
  ``result.usage`` (seen in ``modelUsage``). It is stored as ``reported_cost_usd`` only; the kit
  prices the conversation tokens itself.

This module also hosts ``ParsedRun`` and ``refusal_heuristic``, shared by the other adapters.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..models import ModelCall, Profile, RunRecord, Task, ToolCall
from .base import summarize

EXIT_TIMEOUT = 124
EXIT_NOT_FOUND = 127
STDERR_TAIL = 2000

# Environment set by an enclosing Claude Code session. A nested headless run must not inherit
# it: CLAUDE_EFFORT would silently override --effort, and the session and messaging variables
# mark the child as part of the parent's session. CLAUDE_CODE_OAUTH_TOKEN and CLAUDE_CONFIG_DIR
# are deliberately kept so remote mode keeps working.
STRIP_ENV = frozenset(
    {
        "CLAUDECODE",
        "CLAUDE_CODE_ENTRYPOINT",
        "CLAUDE_CODE_SESSION_ID",
        "CLAUDE_CODE_CHILD_SESSION",
        "CLAUDE_CODE_SESSION_ATTENDED",
        "CLAUDE_CODE_MESSAGING_SOCKET",
        "CLAUDE_CODE_MESSAGING_TOKEN",
        "CLAUDE_CODE_EXECPATH",
        "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS",
        "CLAUDE_PID",
        "CLAUDE_EFFORT",
    }
)


def clean_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """A copy of the environment without the enclosing-session variables in STRIP_ENV."""
    src = os.environ if base is None else base
    return {k: v for k, v in src.items() if k not in STRIP_ENV}


# --- refusal heuristic ----------------------------------------------------------------------

REFUSAL_WINDOW = 400

_REFUSAL_RE = re.compile(
    r"(?:"
    r"\bI\s+(?:can(?:no|')t|am\s+not\s+able\s+to|'m\s+not\s+able\s+to|am\s+unable\s+to"
    r"|'m\s+unable\s+to|won'?t|will\s+not)\s+"
    r"(?:help|assist|provide|comply|fulfil?l|do\s+th(?:at|is)|support|engage|proceed|write"
    r"|create|generate|produce|answer|participate|continue)\b"
    r"|\bI\s+(?:must|have\s+to|need\s+to)\s+(?:respectfully\s+)?decline\b"
    r"|\b(?:not\s+able|unable)\s+to\s+(?:help|assist)\s+with\b"
    r"|\bagainst\s+my\s+(?:guidelines|policies|principles|programming)\b"
    r"|\bI\s+(?:don'?t|do\s+not)\s+feel\s+comfortable\b"
    r"|\bI\s+(?:won'?t|will\s+not|cannot|can'?t)\s+be\s+able\s+to\s+(?:help|assist)\b"
    r"|\b(?:this|that|your)\s+request\s+(?:violates|goes\s+against)\b"
    r")",
    re.IGNORECASE,
)


def refusal_heuristic(text: str | None) -> bool:
    """True when the opening of ``text`` reads like a refusal.

    A heuristic only. Callers should also require that the agent made no tool calls before
    marking a run refused, so "I can't create the file because the directory is read-only"
    after a failed Write does not count.
    """
    if not text:
        return False
    head = text.strip()[:REFUSAL_WINDOW]
    return _REFUSAL_RE.search(head) is not None


# --- parsed run -----------------------------------------------------------------------------


@dataclass
class ParsedRun:
    """Adapter-neutral result of parsing one agent's raw output stream."""

    final_text: str = ""
    model: str | None = None
    model_calls: list[ModelCall] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    reported_cost_usd: float | None = None
    session_id: str | None = None
    is_error: bool = False
    error: str | None = None
    num_turns: int | None = None
    raw_result: dict[str, Any] | None = None
    # Small, JSON-serializable extras that go into RunRecord.raw. Never the whole payload.
    extras: dict[str, Any] = field(default_factory=dict)

    def raw(self) -> dict[str, Any]:
        out: dict[str, Any] = {"session_id": self.session_id, "num_turns": self.num_turns}
        out.update(self.extras)
        return {k: v for k, v in out.items() if v is not None}


def _int(v: Any) -> int:
    return int(v) if isinstance(v, int | float) and not isinstance(v, bool) else 0


def _tail(s: str | None, n: int = STDERR_TAIL) -> str:
    return (s or "")[-n:].strip()


def _as_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return str(v)


def _bump(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _new_tool_call(parsed: ParsedRun, name: str, input_full: Any) -> ToolCall:
    tc = ToolCall(
        index=len(parsed.tool_calls),
        name=name,
        input_summary=summarize(input_full) if input_full is not None else "",
        input_full=input_full,
    )
    parsed.tool_calls.append(tc)
    return tc


def _attach_result(tc: ToolCall, block: dict[str, Any]) -> None:
    content = block.get("content")
    tc.output_full = content
    tc.output_summary = summarize(content) if content is not None else None
    tc.is_error = bool(block.get("is_error"))


def parse_stream_json(lines: Iterable[str | dict[str, Any]]) -> ParsedRun:
    """Normalize a ``--output-format stream-json`` stream. Pure: no I/O.

    Accepts raw JSONL lines or already-decoded event dicts (the SDK adapter feeds dicts).
    Malformed lines are counted, never fatal.
    """
    parsed = ParsedRun()
    messages: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    pending: dict[str, ToolCall] = {}
    counters: dict[str, int] = {}
    init_model: str | None = None
    anon = 0

    for item in lines:
        if isinstance(item, dict):
            ev: Any = item
        else:
            line = item.strip()
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

        if etype == "system":
            if ev.get("subtype") == "init":
                parsed.session_id = ev.get("session_id") or parsed.session_id
                init_model = ev.get("model") or init_model
                servers = ev.get("mcp_servers") or []
                parsed.extras["init"] = {
                    "claude_code_version": ev.get("claude_code_version"),
                    "permission_mode": ev.get("permissionMode"),
                    "api_key_source": ev.get("apiKeySource"),
                    "n_tools": len(ev.get("tools") or []),
                    "mcp_servers": [
                        s.get("name") for s in servers if isinstance(s, dict) and s.get("name")
                    ],
                }
            continue

        if etype == "assistant":
            msg = ev.get("message") or {}
            mid = msg.get("id")
            if not mid:
                anon += 1
                mid = f"anon-{anon}"
            acc = messages.get(mid)
            if acc is None:
                acc = {"model": None, "usage": {}, "stop_reason": None, "texts": [], "tools": 0}
                messages[mid] = acc
                order.append(mid)
            acc["model"] = msg.get("model") or acc["model"]
            if isinstance(msg.get("usage"), dict):
                acc["usage"] = msg["usage"]
            acc["stop_reason"] = msg.get("stop_reason") or acc["stop_reason"]
            for block in msg.get("content") or []:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    acc["texts"].append(str(block.get("text") or ""))
                elif btype in ("tool_use", "server_tool_use"):
                    acc["tools"] += 1
                    tc = _new_tool_call(
                        parsed, str(block.get("name") or "unknown"), block.get("input")
                    )
                    if block.get("id"):
                        pending[str(block["id"])] = tc
                elif btype == "thinking":
                    _bump(counters, "thinking_blocks")
            continue

        if etype == "user":
            content = (ev.get("message") or {}).get("content")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        continue
                    tc = pending.pop(str(block.get("tool_use_id")), None)
                    if tc is None:
                        _bump(counters, "orphan_tool_results")
                        continue
                    _attach_result(tc, block)
            continue

        if etype == "result":
            parsed.raw_result = ev
            continue

        _bump(counters, f"other_event:{etype or '?'}")

    for i, mid in enumerate(order):
        acc = messages[mid]
        u = acc["usage"] or {}
        parsed.model_calls.append(
            ModelCall(
                index=i,
                model=acc["model"],
                input_tokens=_int(u.get("input_tokens")),
                output_tokens=_int(u.get("output_tokens")),
                cache_read_tokens=_int(u.get("cache_read_input_tokens")),
                cache_write_tokens=_int(u.get("cache_creation_input_tokens")),
                stop_reason=acc["stop_reason"] or ("tool_use" if acc["tools"] else None),
            )
        )
        if acc["model"] and not parsed.model:
            parsed.model = acc["model"]

    last_text = next(
        ("\n".join(messages[m]["texts"]) for m in reversed(order) if messages[m]["texts"]), ""
    )
    if parsed.raw_result is not None:
        _apply_result(parsed, parsed.raw_result, init_model, last_text)
    else:
        parsed.final_text = last_text
        parsed.model = parsed.model or init_model
        parsed.is_error = True
        parsed.error = "stream ended without a result event"

    if pending:
        counters["unanswered_tool_calls"] = len(pending)
    parsed.extras.update(counters)
    return parsed


def _apply_result(
    parsed: ParsedRun, res: dict[str, Any], init_model: str | None, last_text: str
) -> None:
    parsed.is_error = bool(res.get("is_error"))
    parsed.num_turns = res.get("num_turns") if isinstance(res.get("num_turns"), int) else None
    parsed.session_id = res.get("session_id") or parsed.session_id
    cost = res.get("total_cost_usd")
    parsed.reported_cost_usd = float(cost) if isinstance(cost, int | float) else None
    result_text = res.get("result")
    has_text = isinstance(result_text, str) and result_text.strip() != ""
    parsed.final_text = result_text if has_text else last_text
    subtype = str(res.get("subtype") or "")
    if parsed.is_error or subtype.startswith("error"):
        parsed.is_error = True
        detail = result_text if has_text else subtype or "error"
        errors = res.get("errors")
        if isinstance(errors, list) and errors:
            detail = f"{detail}: {'; '.join(str(e) for e in errors)}"
        parsed.error = f"claude_code {subtype or 'error'}: {str(detail)[:500]}"

    model_usage = res.get("modelUsage") if isinstance(res.get("modelUsage"), dict) else {}
    if not parsed.model:
        # Prefer the longest key: the dated id the API actually served, as the judge does.
        parsed.model = sorted(model_usage, key=len)[-1] if model_usage else init_model

    usage = res.get("usage") if isinstance(res.get("usage"), dict) else {}
    totals = {
        "input_tokens": _int(usage.get("input_tokens")),
        "output_tokens": _int(usage.get("output_tokens")),
        "cache_read_tokens": _int(usage.get("cache_read_input_tokens")),
        "cache_write_tokens": _int(usage.get("cache_creation_input_tokens")),
    }
    if not parsed.model_calls:
        if any(totals.values()):
            parsed.model_calls.append(
                ModelCall(index=0, model=parsed.model, stop_reason=res.get("stop_reason"), **totals)
            )
    else:
        deltas: dict[str, int] = {}
        last = parsed.model_calls[-1]
        for name, total in totals.items():
            have = sum(getattr(c, name) for c in parsed.model_calls)
            if total > have:
                setattr(last, name, getattr(last, name) + (total - have))
                deltas[name] = total - have
        if deltas:
            parsed.extras["usage_reconciliation"] = deltas
        if last.stop_reason is None and res.get("stop_reason"):
            last.stop_reason = str(res["stop_reason"])

    # Auxiliary calls (title generation, classifiers) appear in modelUsage but not in usage.
    aux = {
        "input_tokens": sum(_int(m.get("inputTokens")) for m in model_usage.values()),
        "output_tokens": sum(_int(m.get("outputTokens")) for m in model_usage.values()),
    }
    aux_delta = {k: aux[k] - totals[k] for k in aux if aux[k] > totals[k]}
    denials = res.get("permission_denials")
    parsed.extras.update(
        {
            "subtype": subtype or None,
            "stop_reason": res.get("stop_reason"),
            "duration_ms": res.get("duration_ms"),
            "duration_api_ms": res.get("duration_api_ms"),
            "permission_denials": denials if isinstance(denials, list) and denials else None,
            "model_usage": {
                m: {
                    k: v
                    for k, v in (row or {}).items()
                    if k
                    in (
                        "inputTokens",
                        "outputTokens",
                        "cacheReadInputTokens",
                        "cacheCreationInputTokens",
                        "costUSD",
                    )
                }
                for m, row in model_usage.items()
                if isinstance(row, dict)
            }
            or None,
            "auxiliary_tokens": aux_delta or None,
        }
    )


# --- adapter --------------------------------------------------------------------------------


def _csv(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list | tuple):
        return ",".join(str(v) for v in value)
    return str(value)


def resolve_cwd(workspace: Path | None, prefix: str) -> tuple[Path, Path | None]:
    """Workspace tasks run in their workspace; qa tasks get a neutral temp dir (returned second
    so the caller removes it)."""
    if workspace is not None:
        return Path(workspace), None
    tmp = Path(tempfile.mkdtemp(prefix=prefix))
    return tmp, tmp


class ClaudeCodeCLIAdapter:
    name = "claude_code"

    def __init__(self, cli: str | None = None):
        self.cli = cli or shutil.which("claude") or "claude"

    def build_command(self, task: Task, workspace: Path | None, profile: Profile) -> list[str]:
        """Pure. The prompt goes on stdin and the workspace becomes the process cwd, so neither
        appears here. ``--verbose`` is mandatory with stream-json in print mode."""
        p = profile.params
        cmd = [self.cli, "-p", "--verbose", "--output-format", "stream-json"]
        if p.get("no_session_persistence", True):
            cmd.append("--no-session-persistence")
        if profile.model:
            cmd += ["--model", str(profile.model)]
        if p.get("clean_slate", True):
            cmd += ["--strict-mcp-config", "--setting-sources", ""]
        if p.get("permission_mode"):
            cmd += ["--permission-mode", str(p["permission_mode"])]
        if p.get("allowed_tools"):
            cmd += ["--allowedTools", _csv(p["allowed_tools"])]
        if p.get("tools") is not None:
            cmd += ["--tools", _csv(p["tools"])]
        if p.get("system_prompt_file"):
            cmd += ["--append-system-prompt-file", str(p["system_prompt_file"])]
        if p.get("max_turns") is not None:
            cmd += ["--max-turns", str(int(p["max_turns"]))]
        if p.get("max_budget_usd") is not None:
            cmd += ["--max-budget-usd", str(p["max_budget_usd"])]
        if p.get("effort"):
            cmd += ["--effort", str(p["effort"])]
        cmd += [str(a) for a in (p.get("extra_args") or [])]
        return cmd

    def run(
        self, task: Task, rep: int, workspace: Path | None, profile: Profile, timeout_s: int
    ) -> RunRecord:
        t0 = time.perf_counter()
        cmd = self.build_command(task, workspace, profile)
        cwd, tmp = resolve_cwd(workspace, "aeb-claude-qa-")
        stdout = stderr = ""
        exit_status = 0
        error: str | None = None
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
                env=clean_env(),
            )
            stdout, stderr, exit_status = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as e:
            stdout, stderr = _as_text(e.stdout), _as_text(e.stderr)
            exit_status = EXIT_TIMEOUT
            error = f"timeout after {timeout_s}s"
        except FileNotFoundError:
            exit_status = EXIT_NOT_FOUND
            error = f"claude CLI not found: {self.cli!r}"
        finally:
            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)
        wall_ms = int((time.perf_counter() - t0) * 1000)

        parsed = parse_stream_json(stdout.splitlines())
        if error is None:
            error = parsed.error
        if error is None and exit_status != 0:
            error = f"claude exited {exit_status}: {_tail(stderr, 500)}"
        raw = parsed.raw()
        raw.update({"command": cmd, "cwd": str(cwd)})
        if stderr.strip():
            raw["stderr_tail"] = _tail(stderr)
        return RunRecord(
            task_id=task.id,
            rep=rep,
            agent=self.name,
            profile=profile.name,
            model=parsed.model or profile.model,
            final_text=parsed.final_text,
            model_calls=parsed.model_calls,
            tool_calls=parsed.tool_calls,
            wall_ms=wall_ms,
            exit_status=exit_status,
            error=error,
            refused=refusal_heuristic(parsed.final_text) and not parsed.tool_calls,
            workspace_path=str(workspace) if workspace else None,
            reported_cost_usd=parsed.reported_cost_usd,
            raw=raw,
        )
