"""Core data models shared by every module.

Design notes
- Tasks come in two kinds. ``qa`` tasks are prompt-in, answer-out and are graded by the
  judge. ``workspace`` tasks copy a fixture directory into a temp dir, let the agent work
  there, and are graded primarily on the end state (programmatic checks), with the judge
  covering taste dimensions only.
- ``RunRecord`` is the adapter-neutral shape every agent run is normalized into. Everything
  downstream (cost, drift, judge, report) reads this, never raw agent output.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

TaskKind = Literal["qa", "workspace"]
CriterionKind = Literal["judge", "programmatic"]
AdapterName = Literal["fake", "claude_code", "codex", "claude_sdk"]

# Programmatic criteria the kit knows how to compute. Suites reference them by name.
PROGRAMMATIC_CRITERIA = (
    "tests_pass",
    "only_allowed_paths",
    "not_noop",
    "must_exist",
    "expected_match",
)


class Criterion(BaseModel):
    """One independently gradeable property.

    ``description`` must be a concrete, checkable claim ("the response does not fabricate
    API parameters"), never a vague scale ("rate helpfulness 1-5").
    """

    name: str
    description: str
    kind: CriterionKind = "judge"

    @model_validator(mode="after")
    def _programmatic_name_known(self) -> Criterion:
        if self.kind == "programmatic" and self.name not in PROGRAMMATIC_CRITERIA:
            raise ValueError(
                f"programmatic criterion {self.name!r} is not one of {PROGRAMMATIC_CRITERIA}"
            )
        return self


class WorkspaceChecks(BaseModel):
    """End-state checks for a workspace task. All paths are relative to the workspace root."""

    command: str | None = None
    timeout_s: int = 300
    must_exist: list[str] = Field(default_factory=list)
    must_not_change: list[str] = Field(default_factory=list)
    allowed_paths: list[str] | None = None
    no_op_fail: bool = True


class Task(BaseModel):
    id: str
    kind: TaskKind = "qa"
    prompt: str
    tags: list[str] = Field(default_factory=list)
    # Judge-only context. Never shown to the agent under test.
    reference: str | None = None
    rubric_notes: str | None = None
    # Optional programmatic check for qa: case-insensitive substring the answer must contain.
    expected: str | None = None
    # Override which suite criteria apply (by name). None means all suite criteria.
    criteria: list[str] | None = None
    # Workspace-only fields.
    fixture: str | None = None
    hidden_tests: str | None = None
    checks: WorkspaceChecks | None = None
    # A known-valid solution (path -> content). Consumed by FakeAgent and documentation only.
    reference_patch: dict[str, str] | None = None

    @model_validator(mode="after")
    def _workspace_fields(self) -> Task:
        if self.kind == "workspace":
            if not self.fixture:
                raise ValueError(f"workspace task {self.id!r} needs a fixture directory")
            if self.checks is None:
                self.checks = WorkspaceChecks()
        return self


class Suite(BaseModel):
    name: str
    description: str = ""
    version: str = "1"
    criteria: list[Criterion]
    tasks: list[Task]
    root: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _unique_ids(self) -> Suite:
        ids = [t.id for t in self.tasks]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate task ids: {sorted(dupes)}")
        names = [c.name for c in self.criteria]
        if len(set(names)) != len(names):
            raise ValueError("duplicate criterion names")
        return self

    def criteria_for(self, task: Task) -> list[Criterion]:
        if task.criteria is None:
            return list(self.criteria)
        by_name = {c.name: c for c in self.criteria}
        missing = [n for n in task.criteria if n not in by_name]
        if missing:
            raise ValueError(f"task {task.id!r} references unknown criteria {missing}")
        return [by_name[n] for n in task.criteria]

    def judge_criteria_for(self, task: Task) -> list[Criterion]:
        return [c for c in self.criteria_for(task) if c.kind == "judge"]

    def programmatic_criteria_for(self, task: Task) -> list[Criterion]:
        return [c for c in self.criteria_for(task) if c.kind == "programmatic"]

    def judge_criteria_names(self) -> list[str]:
        return [c.name for c in self.criteria if c.kind == "judge"]


class Profile(BaseModel):
    """A named agent configuration. Recorded verbatim as MLflow run params."""

    name: str
    adapter: AdapterName
    model: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    def flat_params(self) -> dict[str, str]:
        out = {"profile": self.name, "adapter": self.adapter, "model": str(self.model)}
        for k, v in sorted(self.params.items()):
            out[f"param.{k}"] = str(v)
        return out


class ModelCall(BaseModel):
    index: int
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_ms: int | None = None
    stop_reason: str | None = None


class ToolCall(BaseModel):
    index: int
    name: str
    input_summary: str = ""
    output_summary: str | None = None
    is_error: bool = False
    duration_ms: int | None = None
    # Full content, kept for the MLflow trace so spans are debuggable. May be large.
    input_full: Any = None
    output_full: Any = None


class RunRecord(BaseModel):
    """Adapter-neutral record of one agent run on one task."""

    task_id: str
    rep: int
    agent: str
    profile: str
    model: str | None = None
    final_text: str = ""
    model_calls: list[ModelCall] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    wall_ms: int = 0
    exit_status: int = 0
    error: str | None = None
    refused: bool = False
    workspace_path: str | None = None
    reported_cost_usd: float | None = None
    trace_id: str | None = None
    # Workspace tasks: unified diff of what the agent changed. Shown to the judge, never a count.
    diff: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.exit_status == 0 and self.error is None

    @property
    def input_tokens(self) -> int:
        return sum(m.input_tokens for m in self.model_calls)

    @property
    def output_tokens(self) -> int:
        return sum(m.output_tokens for m in self.model_calls)

    @property
    def cache_read_tokens(self) -> int:
        return sum(m.cache_read_tokens for m in self.model_calls)

    @property
    def cache_write_tokens(self) -> int:
        return sum(m.cache_write_tokens for m in self.model_calls)

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    def tool_names(self) -> list[str]:
        return [t.name for t in self.tool_calls]


class Verdict(BaseModel):
    task_id: str
    rep: int
    criterion: str
    passed: bool | None
    rationale: str = ""
    judge_backend: str = ""
    judge_model: str | None = None
    judge_version: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = None
    latency_ms: int | None = None
    error: str | None = None
    determinism_rerun: bool = False


class CaseResult(BaseModel):
    record: RunRecord
    programmatic: dict[str, bool] = Field(default_factory=dict)
    verdicts: list[Verdict] = Field(default_factory=list)
    agent_cost_usd: float | None = None
    agent_cost_known: bool = True
    judge_cost_usd: float | None = None

    @property
    def task_id(self) -> str:
        return self.record.task_id

    @property
    def rep(self) -> int:
        return self.record.rep

    def passed(self, criterion: str) -> bool | None:
        if criterion in self.programmatic:
            return self.programmatic[criterion]
        for v in self.verdicts:
            if v.criterion == criterion and not v.determinism_rerun:
                return v.passed
        return None

    def all_criteria(self) -> list[str]:
        names = list(self.programmatic)
        names += [v.criterion for v in self.verdicts if not v.determinism_rerun]
        seen: list[str] = []
        for n in names:
            if n not in seen:
                seen.append(n)
        return seen


class RunManifest(BaseModel):
    run_id: str
    experiment: str
    experiment_id: str = ""
    suite: str
    suite_version: str
    profile: Profile
    judge_backend: str
    judge_model: str | None
    judge_version: str
    reps: int
    n_tasks: int
    started_at: datetime
    finished_at: datetime | None = None
    kit_version: str = ""
    tracking_uri: str = ""
    baseline_run_id: str | None = None
    notes: dict[str, Any] = Field(default_factory=dict)
