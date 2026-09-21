"""Judge backend protocol and factory."""

from __future__ import annotations

from typing import Protocol

from ..config import Settings
from ..models import Criterion, RunRecord, Task, Verdict


class JudgeBackend(Protocol):
    name: str
    model: str | None

    def judge(
        self, criterion: Criterion, task: Task, record: RunRecord, version: str
    ) -> Verdict: ...


def get_judge(
    settings: Settings, backend: str | None = None, model: str | None = None
) -> JudgeBackend:
    kind = backend or settings.judge_backend
    model = model or settings.judge_model
    if kind == "fake":
        from .fake import FakeJudge

        return FakeJudge()
    if kind == "claude_code":
        from .claude_code import ClaudeCodeJudge

        return ClaudeCodeJudge(
            model=model,
            timeout_s=settings.judge_timeout_s,
            effort=settings.judge_effort,
            work_dir=settings.aeb_home / "judge_cwd",
        )
    if kind == "mlflow":
        from .mlflow_judge import MlflowJudge

        return MlflowJudge(model=model)
    raise ValueError(f"unknown judge backend {kind!r}")
