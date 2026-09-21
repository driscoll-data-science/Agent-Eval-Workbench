"""Deterministic heuristic judge for CI. Mirrors what FakeAgent can do wrong so end-to-end
tests can assert regressions without a model call."""

from __future__ import annotations

from ..models import Criterion, RunRecord, Task, Verdict


class FakeJudge:
    name = "fake"
    model = "fake-judge"

    def judge(self, criterion: Criterion, task: Task, record: RunRecord, version: str) -> Verdict:
        text = (record.final_text or "").strip()
        passed, why = self._decide(criterion, task, record, text)
        return Verdict(
            task_id=task.id,
            rep=record.rep,
            criterion=criterion.name,
            passed=passed,
            rationale=why,
            judge_backend=self.name,
            judge_model=self.model,
            judge_version=version,
            input_tokens=max(50, len(text) // 4 + len(task.prompt) // 4),
            output_tokens=30,
            latency_ms=1,
        )

    @staticmethod
    def _decide(criterion: Criterion, task: Task, record: RunRecord, text: str) -> tuple[bool, str]:
        if record.refused or not text or text.lower().startswith("i don't know"):
            return False, "Empty answer, refusal, or non-answer."
        if record.error:
            return False, f"Agent run failed: {record.error}"
        wrong = "42, as is well documented" in text
        if criterion.name in {"correct", "grounded", "expected_match"}:
            if wrong:
                return False, "Confident answer that contradicts the reference."
            if task.reference and task.reference.strip().lower()[:40] in text.lower():
                return True, "Matches the reference."
            if task.expected and task.expected.lower() in text.lower():
                return True, "Contains the expected content."
            if task.kind == "workspace":
                return ("broken" not in (record.diff or "")), "Diff inspected."
            return False, "Does not match the reference."
        if criterion.name == "concise":
            return (len(text) <= 600), f"{len(text)} characters."
        if criterion.name == "minimal_diff":
            return (
                "deliberately broken" not in (record.diff or "")
            ), "Diff inspected for extraneous changes."
        if criterion.name in {"readable", "explanation_accurate", "follows_format"}:
            if wrong:
                return (
                    False,
                    "Confident wrong answer with filler; fails the heuristic for this criterion.",
                )
            return True, "Heuristic pass."
        if wrong:
            return False, "Confident wrong answer; fails the heuristic for this criterion."
        return True, "Heuristic pass."
