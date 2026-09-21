"""Deterministic fake agent for CI and for demonstrating the pipeline without credentials.

Behavior is a pure function of (task id, rep, profile params) so runs are reproducible.
Profile params let tests simulate regressions and drift:
  seed (int)           change the deterministic stream
  fail_rate (0..1)     fraction of qa cases answered wrongly
  refuse_rate (0..1)   fraction of cases refused
  extra_tool (str)     add an extra tool call of this name on every case (shifts tool mix)
  verbose (bool)       pad answers with filler (shifts output length, tokens)
  skip_tests (bool)    workspace: do not run the test command (shifts exec tool share)
  break_patch (bool)   workspace: write a wrong solution so tests fail
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from ..models import ModelCall, Profile, RunRecord, Task, ToolCall
from .base import summarize


def _unit(task_id: str, rep: int, seed: int, salt: str) -> float:
    h = hashlib.sha256(f"{seed}:{task_id}:{rep}:{salt}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


class FakeAgent:
    name = "fake"

    def run(
        self, task: Task, rep: int, workspace: Path | None, profile: Profile, timeout_s: int
    ) -> RunRecord:
        p = profile.params
        seed = int(p.get("seed", 0))
        t0 = time.perf_counter()
        model = profile.model or "fake-model-1"
        tools: list[ToolCall] = []
        calls: list[ModelCall] = []
        refused = _unit(task.id, rep, seed, "refuse") < float(p.get("refuse_rate", 0.0))
        wrong = _unit(task.id, rep, seed, "fail") < float(p.get("fail_rate", 0.0))

        prompt_tokens = max(20, len(task.prompt) // 4)
        if refused:
            text = "I can't help with that request."
            calls.append(
                ModelCall(
                    index=0,
                    model=model,
                    input_tokens=prompt_tokens,
                    output_tokens=12,
                    latency_ms=40,
                )
            )
        elif task.kind == "qa":
            text = self._qa_answer(task, wrong, bool(p.get("verbose", False)))
            if _unit(task.id, rep, seed, "search") < float(p.get("search_rate", 0.3)):
                tools.append(
                    ToolCall(
                        index=0,
                        name="WebSearch",
                        input_summary=summarize({"query": task.prompt[:60]}),
                        output_summary="3 results",
                        duration_ms=120,
                    )
                )
                calls.append(
                    ModelCall(
                        index=0,
                        model=model,
                        input_tokens=prompt_tokens,
                        output_tokens=30,
                        latency_ms=60,
                        stop_reason="tool_use",
                    )
                )
                calls.append(
                    ModelCall(
                        index=1,
                        model=model,
                        input_tokens=prompt_tokens + 400,
                        output_tokens=max(8, len(text) // 4),
                        cache_read_tokens=prompt_tokens,
                        latency_ms=90,
                        stop_reason="end_turn",
                    )
                )
            else:
                calls.append(
                    ModelCall(
                        index=0,
                        model=model,
                        input_tokens=prompt_tokens,
                        output_tokens=max(8, len(text) // 4),
                        latency_ms=80,
                        stop_reason="end_turn",
                    )
                )
        else:
            text = self._do_workspace(
                task,
                workspace,
                tools,
                calls,
                model,
                prompt_tokens,
                bool(p.get("skip_tests", False)),
                bool(p.get("break_patch", False)),
            )

        if p.get("extra_tool"):
            tools.append(
                ToolCall(
                    index=len(tools),
                    name=str(p["extra_tool"]),
                    input_summary="{}",
                    output_summary="ok",
                    duration_ms=10,
                )
            )

        return RunRecord(
            task_id=task.id,
            rep=rep,
            agent=self.name,
            profile=profile.name,
            model=model,
            final_text=text,
            model_calls=calls,
            tool_calls=tools,
            wall_ms=int((time.perf_counter() - t0) * 1000) + 50,
            exit_status=0,
            refused=refused,
            workspace_path=str(workspace) if workspace else None,
            reported_cost_usd=None,
        )

    @staticmethod
    def _qa_answer(task: Task, wrong: bool, verbose: bool) -> str:
        if wrong:
            text = "The answer is 42, as is well documented in the official specification."
        else:
            text = task.reference or task.expected or "I don't know."
        if verbose:
            text += " " + ("Additionally, it is worth noting several related considerations. " * 6)
        return text

    @staticmethod
    def _do_workspace(
        task: Task,
        workspace: Path | None,
        tools: list[ToolCall],
        calls: list[ModelCall],
        model: str,
        prompt_tokens: int,
        skip_tests: bool,
        break_patch: bool,
    ) -> str:
        assert workspace is not None
        tools.append(
            ToolCall(
                index=0,
                name="Read",
                input_summary=summarize({"file_path": "README.md"}),
                output_summary="…",
                duration_ms=5,
            )
        )
        calls.append(
            ModelCall(
                index=0,
                model=model,
                input_tokens=prompt_tokens,
                output_tokens=40,
                latency_ms=70,
                stop_reason="tool_use",
            )
        )
        patch = task.reference_patch or {}
        for rel, content in patch.items():
            target = workspace / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if break_patch:
                content = content + "\nraise RuntimeError('deliberately broken by fake agent')\n"
            target.write_text(content)
            tools.append(
                ToolCall(
                    index=len(tools),
                    name="Edit" if (workspace / rel).exists() else "Write",
                    input_summary=summarize({"file_path": rel}),
                    output_summary="ok",
                    duration_ms=8,
                )
            )
            calls.append(
                ModelCall(
                    index=len(calls),
                    model=model,
                    input_tokens=prompt_tokens + 300 * len(calls),
                    output_tokens=len(content) // 4 + 10,
                    cache_read_tokens=prompt_tokens,
                    latency_ms=90,
                    stop_reason="tool_use",
                )
            )
        if not skip_tests and task.checks and task.checks.command:
            tools.append(
                ToolCall(
                    index=len(tools),
                    name="Bash",
                    input_summary=summarize({"command": task.checks.command}),
                    output_summary="passed",
                    duration_ms=400,
                )
            )
            calls.append(
                ModelCall(
                    index=len(calls),
                    model=model,
                    input_tokens=prompt_tokens + 600,
                    output_tokens=25,
                    cache_read_tokens=prompt_tokens,
                    latency_ms=60,
                    stop_reason="tool_use",
                )
            )
        calls.append(
            ModelCall(
                index=len(calls),
                model=model,
                input_tokens=prompt_tokens + 700,
                output_tokens=60,
                cache_read_tokens=prompt_tokens,
                latency_ms=80,
                stop_reason="end_turn",
            )
        )
        return f"Applied the fix in {', '.join(patch) or 'no files'} and ran the tests."
