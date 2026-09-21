"""Claude Agent SDK adapter: drives an in-process agent entrypoint and normalizes its messages.

The entrypoint is ``params.entrypoint`` as ``module:function`` (default: the reference agent in
``agent_eval_bench.agents.reference_sdk_agent``) with the signature
``async def run(prompt: str, options: ClaudeAgentOptions) -> AsyncIterator[Message]``.

SDK messages are converted to the dict shape the CLI's stream-json emits and fed through
``parse_stream_json`` so both Claude adapters share one normalization path: message-id
deduplication, tool_use/tool_result pairing, and usage reconciliation against the result.
Built against claude-agent-sdk 0.2.154.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import shutil
import time
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from ..models import Profile, RunRecord, Task
from .claude_code import (
    EXIT_TIMEOUT,
    STRIP_ENV,
    ParsedRun,
    parse_stream_json,
    refusal_heuristic,
    resolve_cwd,
)

DEFAULT_ENTRYPOINT = "agent_eval_bench.agents.reference_sdk_agent:run"
Entrypoint = Callable[[str, ClaudeAgentOptions], AsyncIterator[Any]]


def resolve_entrypoint(spec: str) -> Entrypoint:
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(f"entrypoint must be 'module:function', got {spec!r}")
    fn = getattr(importlib.import_module(module_name), attr, None)
    if not callable(fn):
        raise ValueError(f"entrypoint {spec!r} is not callable")
    return fn


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [s.strip() for s in value.split(",") if s.strip()]
    return [str(v) for v in value]


def scrub_nested_env() -> None:
    """Drop enclosing-Claude-Code-session variables from this process. The SDK spawns the CLI
    with the process environment and its ``env`` option can only add, never remove."""
    for key in STRIP_ENV:
        os.environ.pop(key, None)


def _block(b: Any) -> dict[str, Any]:
    if isinstance(b, TextBlock):
        return {"type": "text", "text": b.text}
    if isinstance(b, ThinkingBlock):
        return {"type": "thinking"}
    if isinstance(b, ToolUseBlock):
        return {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
    if isinstance(b, ToolResultBlock):
        return {
            "type": "tool_result",
            "tool_use_id": b.tool_use_id,
            "content": b.content,
            "is_error": b.is_error,
        }
    if hasattr(b, "id") and hasattr(b, "name") and hasattr(b, "input"):
        # ServerToolUseBlock and friends: a tool call executed server-side.
        return {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
    if hasattr(b, "tool_use_id") and hasattr(b, "content"):
        return {
            "type": "tool_result",
            "tool_use_id": b.tool_use_id,
            "content": b.content,
            "is_error": getattr(b, "is_error", None),
        }
    return {"type": type(b).__name__}


def _plain(v: Any) -> Any:
    if is_dataclass(v) and not isinstance(v, type):
        return asdict(v)
    return v


def to_event(msg: Any) -> dict[str, Any] | None:
    """Convert one SDK message into the equivalent stream-json event dict; None to skip."""
    if isinstance(msg, AssistantMessage):
        return {
            "type": "assistant",
            "message": {
                "id": msg.message_id,
                "model": msg.model,
                "content": [_block(b) for b in msg.content],
                "usage": msg.usage,
                "stop_reason": msg.stop_reason,
            },
            "session_id": msg.session_id,
        }
    if isinstance(msg, UserMessage):
        content = msg.content
        blocks = [_block(b) for b in content] if isinstance(content, list) else content
        return {"type": "user", "message": {"content": blocks}}
    if isinstance(msg, ResultMessage):
        model_usage = msg.model_usage or {}
        return {
            "type": "result",
            "subtype": msg.subtype,
            "is_error": msg.is_error,
            "num_turns": msg.num_turns,
            "duration_ms": msg.duration_ms,
            "duration_api_ms": msg.duration_api_ms,
            "total_cost_usd": msg.total_cost_usd,
            "usage": msg.usage,
            "modelUsage": {k: _plain(v) for k, v in model_usage.items()},
            "result": msg.result,
            "session_id": msg.session_id,
            "permission_denials": msg.permission_denials,
            "stop_reason": msg.stop_reason,
            "errors": msg.errors,
        }
    if isinstance(msg, SystemMessage):
        return {"type": "system", "subtype": msg.subtype, **(msg.data or {})}
    return None


def collect(messages: Iterable[Any]) -> ParsedRun:
    """Normalize a sequence of SDK messages (live or hand-built). Pure: no I/O."""
    events = [ev for ev in (to_event(m) for m in messages) if ev is not None]
    return parse_stream_json(events)


async def _drive(
    entry: Entrypoint, prompt: str, options: ClaudeAgentOptions, sink: list[Any], timeout_s: int
) -> None:
    async def consume() -> None:
        async for message in entry(prompt, options):
            sink.append(message)

    await asyncio.wait_for(consume(), timeout=timeout_s)


class ClaudeSDKAdapter:
    name = "claude_sdk"

    def build_options(
        self, task: Task, workspace: Path | None, profile: Profile
    ) -> ClaudeAgentOptions:
        """Pure. Clean slate (default) means no settings files, no MCP servers, strict MCP."""
        p = profile.params
        clean = bool(p.get("clean_slate", True))
        kwargs: dict[str, Any] = {
            "model": profile.model,
            "cwd": str(workspace) if workspace is not None else None,
            "setting_sources": [] if clean else ["user", "project", "local"],
            "strict_mcp_config": clean,
            "mcp_servers": {},
            "permission_mode": p.get("permission_mode"),
            "allowed_tools": _as_list(p.get("allowed_tools")),
            "max_turns": int(p["max_turns"]) if p.get("max_turns") is not None else None,
            "max_budget_usd": (
                float(p["max_budget_usd"]) if p.get("max_budget_usd") is not None else None
            ),
            "effort": p.get("effort"),
        }
        if p.get("tools") is not None:
            kwargs["tools"] = _as_list(p["tools"])
        if p.get("system_prompt_file"):
            kwargs["system_prompt"] = {
                "type": "preset",
                "preset": "claude_code",
                "append": Path(p["system_prompt_file"]).read_text(encoding="utf-8"),
            }
        elif p.get("system_prompt"):
            kwargs["system_prompt"] = str(p["system_prompt"])
        if p.get("extra_args"):
            kwargs["extra_args"] = dict(p["extra_args"])
        return ClaudeAgentOptions(**kwargs)

    def run(
        self, task: Task, rep: int, workspace: Path | None, profile: Profile, timeout_s: int
    ) -> RunRecord:
        t0 = time.perf_counter()
        spec = str(profile.params.get("entrypoint") or DEFAULT_ENTRYPOINT)
        cwd, tmp = resolve_cwd(workspace, "aeb-sdk-qa-")
        messages: list[Any] = []
        exit_status = 0
        error: str | None = None
        try:
            entry = resolve_entrypoint(spec)
            options = self.build_options(task, cwd, profile)
            scrub_nested_env()
            asyncio.run(_drive(entry, task.prompt, options, messages, timeout_s))
        except TimeoutError:
            exit_status = EXIT_TIMEOUT
            error = f"timeout after {timeout_s}s"
        except Exception as e:  # noqa: BLE001 - any SDK/process failure becomes the record's error
            code = getattr(e, "exit_code", None)
            exit_status = int(code) if isinstance(code, int) and code != 0 else 1
            error = f"{type(e).__name__}: {e}"[:500]
        finally:
            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)
        wall_ms = int((time.perf_counter() - t0) * 1000)

        parsed = collect(messages)
        if error is None:
            error = parsed.error
        raw = parsed.raw()
        raw.update({"entrypoint": spec, "cwd": str(cwd), "n_messages": len(messages)})
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
