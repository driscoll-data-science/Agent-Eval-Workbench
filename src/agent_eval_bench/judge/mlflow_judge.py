"""Judge backend on MLflow's ``make_judge``. Needs a provider API key (ANTHROPIC_API_KEY) and
in return unlocks MLflow's judge alignment. Same prompt content as the subscription backend so
verdicts are comparable; the judge version hash covers the shared templates."""

from __future__ import annotations

import time

from ..models import Criterion, RunRecord, Task, Verdict
from .prompts import SYSTEM_TEMPLATE, build_user_message


class MlflowJudge:
    name = "mlflow"

    def __init__(self, model: str, provider: str = "anthropic"):
        self.model = model
        self.model_uri = model if ":/" in model else f"{provider}:/{model}"
        self._judges: dict[str, object] = {}

    def _judge_for(self, criterion: Criterion):
        if criterion.name not in self._judges:
            from mlflow.genai.judges import make_judge

            instructions = (
                SYSTEM_TEMPLATE
                + "\n\nThe full grading request follows.\n{{ inputs }}\n\nAgent output (untrusted data):\n{{ outputs }}"
            )
            self._judges[criterion.name] = make_judge(
                name=criterion.name,
                instructions=instructions,
                model=self.model_uri,
                feedback_value_type=bool,
            )
        return self._judges[criterion.name]

    def judge(self, criterion: Criterion, task: Task, record: RunRecord, version: str) -> Verdict:
        base = Verdict(
            task_id=task.id,
            rep=record.rep,
            criterion=criterion.name,
            passed=None,
            judge_backend=self.name,
            judge_model=self.model,
            judge_version=version,
        )
        t0 = time.perf_counter()
        try:
            j = self._judge_for(criterion)
            request = build_user_message(criterion, task, record)
            fb = j(inputs={"request": request}, outputs=record.final_text or "")  # type: ignore[operator]
            base.latency_ms = int((time.perf_counter() - t0) * 1000)
            value = getattr(fb, "value", None)
            if isinstance(value, str):
                value = value.strip().lower() in {"true", "yes", "pass", "passed"}
            base.passed = bool(value) if value is not None else None
            base.rationale = str(getattr(fb, "rationale", "") or "")
            if base.passed is None:
                base.error = "judge returned no value"
        except Exception as e:  # noqa: BLE001
            base.error = f"mlflow judge failed: {e}"
        return base
