"""Write a RunRecord as an MLflow trace so every agent, in-process or subprocess, looks the
same in the trace UI: one AGENT root span, one LLM span per model call carrying token usage,
one TOOL span per tool call carrying full input and output.

Spans are reconstructed after the run, so their wall-clock durations are not meaningful; the
measured latency of each call is stored as an attribute instead.
"""

from __future__ import annotations

import contextlib

import mlflow
from mlflow.entities import SpanType

from .models import RunRecord, Task

CHAT_USAGE = "mlflow.chat.tokenUsage"
LLM_COST = "mlflow.llm.cost"


def log_record_trace(
    record: RunRecord,
    task: Task,
    suite: str,
    run_id: str | None,
    cost_split: tuple[float, float, float] | None = None,
) -> str | None:
    root_name = f"{record.agent}:{task.id}:rep{record.rep}"
    trace_id: str | None = None
    with mlflow.start_span(name=root_name, span_type=SpanType.AGENT, run_id=run_id) as root:
        root.set_inputs({"task_id": task.id, "kind": task.kind, "prompt": task.prompt})
        root.set_attributes(
            {
                "aeb.profile": record.profile,
                "aeb.model": record.model or "",
                "aeb.wall_ms": record.wall_ms,
                "aeb.exit_status": record.exit_status,
                "aeb.refused": record.refused,
            }
        )
        for call in record.model_calls:
            with mlflow.start_span(name=f"llm_call_{call.index}", span_type=SpanType.LLM) as s:
                s.set_inputs({"index": call.index, "model": call.model})
                s.set_outputs({"stop_reason": call.stop_reason})
                s.set_attribute(
                    CHAT_USAGE,
                    {
                        "input_tokens": call.input_tokens,
                        "output_tokens": call.output_tokens,
                        "total_tokens": call.input_tokens
                        + call.output_tokens
                        + call.cache_read_tokens
                        + call.cache_write_tokens,
                        "cache_read_input_tokens": call.cache_read_tokens,
                        "cache_creation_input_tokens": call.cache_write_tokens,
                    },
                )
                if call.latency_ms is not None:
                    s.set_attribute("aeb.latency_ms", call.latency_ms)
        for tool in record.tool_calls:
            with mlflow.start_span(name=f"tool:{tool.name}", span_type=SpanType.TOOL) as s:
                s.set_inputs(
                    {
                        "tool": tool.name,
                        "input": tool.input_full
                        if tool.input_full is not None
                        else tool.input_summary,
                    }
                )
                s.set_outputs(
                    {
                        "output": tool.output_full
                        if tool.output_full is not None
                        else tool.output_summary,
                        "is_error": tool.is_error,
                    }
                )
                if tool.duration_ms is not None:
                    s.set_attribute("aeb.duration_ms", tool.duration_ms)
        outputs = {"final_text": record.final_text, "exit_status": record.exit_status}
        if record.error:
            outputs["error"] = record.error
        if record.diff is not None:
            outputs["diff"] = record.diff
        root.set_outputs(outputs)
        if cost_split is not None:
            # MLflow coerces every value here with float(); never pass None.
            inp, out, total = cost_split
            root.set_attribute(
                LLM_COST,
                {"input_cost": float(inp), "output_cost": float(out), "total_cost": float(total)},
            )
        mlflow.update_current_trace(
            tags={
                "aeb.suite": suite,
                "aeb.task_id": task.id,
                "aeb.rep": str(record.rep),
                "aeb.profile": record.profile,
                "aeb.agent": record.agent,
            }
        )
        trace_id = getattr(root, "trace_id", None) or getattr(root, "request_id", None)
    return trace_id


def flush() -> None:
    with contextlib.suppress(Exception):
        mlflow.flush_trace_async_logging()
