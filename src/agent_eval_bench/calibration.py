"""Human calibration labels and human-vs-judge agreement.

Flow
  aeb labels sample <run>   stratified sample of cases -> <labels_dir>/<suite>.sample.<run8>.json
  (label in the MLflow UI: Feedback named after the criterion, source Human)
  aeb labels pull <run>     HUMAN feedback on the run's traces -> <labels_dir>/<suite>.yaml
  aeb labels tui <run>      terminal labeling loop writing the same YAML
  aeb calibrate <run>       per-criterion raw agreement + Cohen's kappa
                            -> <calibration_dir>/<suite>/<judge_version>.json
                            -> MLflow experiment aeb/judge-calibration

The YAML labels file is the source of truth; MLflow feedback is the primary way to author it.
Agreement compares the human label with the judge verdict on the same frozen output. When the
judge version has moved on since the run, the labeled records are re-judged with the current
judge so agreement compares like with like.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import yaml
from mlflow import MlflowClient
from mlflow.entities import AssessmentSourceType
from pydantic import BaseModel, Field

from .config import Settings
from .judge.base import get_judge
from .judge.prompts import judge_version
from .models import CaseResult, RunManifest, Suite, Task
from .storage import RunStore
from .suites import load_suite

Log = Callable[[str], None]
InputFn = Callable[[str], str]

CALIBRATION_EXPERIMENT = "aeb/judge-calibration"
MIN_LABELS_PER_CRITERION = 10
TRUE_WORDS = frozenset({"true", "yes", "pass", "passed", "1", "y", "t"})
FALSE_WORDS = frozenset({"false", "no", "fail", "failed", "0", "n", "f"})
TUI_OUTPUT_CHARS = 1500
TUI_DIFF_CHARS = 3000


# Labels file --------------------------------------------------------------------------------
class Label(BaseModel):
    run_id: str
    trace_id: str | None = None
    task_id: str
    rep: int
    criterion: str
    human: bool
    rationale: str | None = None
    labeled_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def key(self) -> tuple[str, str, int, str]:
        return (self.run_id, self.task_id, self.rep, self.criterion)


class LabelsFile(BaseModel):
    suite: str
    entries: list[Label] = Field(default_factory=list)

    def upsert(self, new: list[Label]) -> int:
        """Insert or replace by (run_id, task_id, rep, criterion). Returns the count merged."""
        by_key = {e.key: e for e in self.entries}
        for lab in new:
            by_key[lab.key] = lab
        self.entries = sorted(by_key.values(), key=lambda e: e.key)
        return len(new)

    def for_run(self, run_id: str) -> list[Label]:
        return [e for e in self.entries if e.run_id == run_id]


def labels_path(settings: Settings, suite: str) -> Path:
    return settings.labels_dir / f"{suite}.yaml"


def load_labels(settings: Settings, suite: str) -> LabelsFile:
    p = labels_path(settings, suite)
    if not p.is_file():
        return LabelsFile(suite=suite)
    data = yaml.safe_load(p.read_text()) or {}
    data.setdefault("suite", suite)
    data.setdefault("entries", [])
    return LabelsFile.model_validate(data)


def save_labels(settings: Settings, labels: LabelsFile) -> Path:
    p = labels_path(settings, labels.suite)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = labels.model_dump(mode="json")
    p.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
    return p


# Sampling -----------------------------------------------------------------------------------
class SampleEntry(BaseModel):
    task_id: str
    rep: int
    trace_id: str | None
    tags: list[str]
    criteria: list[str]
    judge_verdicts: dict[str, bool | None]
    trace_url: str | None


class SampleSheet(BaseModel):
    run_id: str
    suite: str
    entries: list[SampleEntry]
    path: Path
    instructions: str


def sample_path(settings: Settings, suite: str, run_id: str) -> Path:
    return settings.labels_dir / f"{suite}.sample.{run_id[:8]}.json"


def _ui_base(settings: Settings) -> str:
    """The browsable MLflow UI. A direct sqlite/file tracking URI is not a URL; `aeb mlflow up`
    serves the same store at the configured host and port."""
    uri = settings.effective_tracking_uri()
    return uri.rstrip("/") if uri.startswith("http") else settings.server_uri


def trace_url(settings: Settings, experiment_id: str, trace_id: str | None) -> str | None:
    if not trace_id:
        return None
    return f"{_ui_base(settings)}/#/experiments/{experiment_id}/traces?selectedTraceId={trace_id}"


def _first_tag(task: Task | None) -> str:
    return task.tags[0] if task and task.tags else "untagged"


def stratified_sample(
    results: list[CaseResult], suite: Suite, n: int, seed: int
) -> list[CaseResult]:
    """Up to n cases, round-robin across first tags, seeded. One case per task, rep 0 preferred."""
    by_id = {t.id: t for t in suite.tasks}
    per_task: dict[str, CaseResult] = {}
    for r in sorted(results, key=lambda r: (r.task_id, r.rep)):
        if r.task_id not in per_task or (r.rep == 0 and per_task[r.task_id].rep != 0):
            per_task[r.task_id] = r
    buckets: dict[str, list[CaseResult]] = {}
    for r in per_task.values():
        buckets.setdefault(_first_tag(by_id.get(r.task_id)), []).append(r)
    rng = random.Random(seed)
    order = sorted(buckets)
    rng.shuffle(order)
    for tag in order:
        rng.shuffle(buckets[tag])
    picked: list[CaseResult] = []
    while len(picked) < n and any(buckets.values()):
        for tag in order:
            if buckets[tag] and len(picked) < n:
                picked.append(buckets[tag].pop())
    return picked


def _instructions(settings: Settings, suite: str, run_id: str, path: Path) -> str:
    return "\n".join(
        [
            f"Label {suite} run {run_id[:8]} in the MLflow UI:",
            f"  1. Open each trace_url in {path} (if the UI is not running: `aeb mlflow up`).",
            "  2. In the trace's Assessments panel click Add Feedback.",
            '  3. Name = the criterion NAME exactly as listed (for example "correct").',
            "     Value = boolean true/false.  Source = Human.  Rationale is optional.",
            "  4. Repeat for every criterion listed for that case.",
            f"  5. Run `aeb labels pull {run_id[:8]}` to merge them into "
            f"{labels_path(settings, suite)}.",
            f"Terminal alternative: `aeb labels tui {run_id[:8]}` writes the same file.",
        ]
    )


def sample_for_labeling(settings: Settings, run_id: str, n: int = 40, seed: int = 0) -> SampleSheet:
    store = RunStore(settings, run_id)
    manifest = store.load_manifest()
    suite = load_suite(manifest.suite, settings)
    results = store.load_results()
    by_id = {t.id: t for t in suite.tasks}
    entries: list[SampleEntry] = []
    for r in stratified_sample(results, suite, n, seed):
        task = by_id.get(r.task_id)
        crits = [c.name for c in suite.judge_criteria_for(task)] if task else []
        entries.append(
            SampleEntry(
                task_id=r.task_id,
                rep=r.rep,
                trace_id=r.record.trace_id,
                tags=list(task.tags) if task else [],
                criteria=crits,
                judge_verdicts={c: r.passed(c) for c in crits},
                trace_url=trace_url(settings, manifest.experiment_id, r.record.trace_id),
            )
        )
    path = sample_path(settings, suite.name, run_id)
    instructions = _instructions(settings, suite.name, run_id, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "suite": suite.name,
                "run_id": run_id,
                "experiment_id": manifest.experiment_id,
                "judge_version": manifest.judge_version,
                "created_at": datetime.now(UTC).isoformat(),
                "instructions": instructions,
                "entries": [e.model_dump(mode="json") for e in entries],
            },
            indent=2,
        )
    )
    return SampleSheet(
        run_id=run_id, suite=suite.name, entries=entries, path=path, instructions=instructions
    )


# Pulling HUMAN feedback from traces ---------------------------------------------------------
def coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in TRUE_WORDS:
            return True
        if v in FALSE_WORDS:
            return False
    return None


def _is_human_feedback(assessment: Any) -> bool:
    source = getattr(assessment, "source", None)
    st = getattr(source, "source_type", None)
    if st is None:
        return False
    human = AssessmentSourceType.HUMAN
    is_human = st == human or str(getattr(st, "value", st)) == str(getattr(human, "value", human))
    if not is_human:
        return False
    # Only Feedback counts as a label; an Expectation is ground truth, not a judgement.
    try:
        from mlflow.entities import Expectation

        if isinstance(assessment, Expectation):
            return False
    except Exception:  # noqa: BLE001
        pass
    return hasattr(assessment, "value") and getattr(assessment, "valid", True) is not False


def _human_labels_on_trace(
    trace: Any, criteria: set[str]
) -> dict[str, tuple[bool, str | None, int]]:
    """criterion -> (value, rationale, update_time_ms); latest human feedback per name wins."""
    out: dict[str, tuple[bool, str | None, int]] = {}
    for a in getattr(trace.info, "assessments", None) or []:
        if not _is_human_feedback(a) or a.name not in criteria:
            continue
        val = coerce_bool(a.value)
        if val is None:
            continue
        ts = int(getattr(a, "last_update_time_ms", None) or getattr(a, "create_time_ms", None) or 0)
        if a.name not in out or ts >= out[a.name][2]:
            out[a.name] = (val, getattr(a, "rationale", None), ts)
    return out


def pull_labels(settings: Settings, run_id: str, log: Log = print) -> int:
    store = RunStore(settings, run_id)
    manifest = store.load_manifest()
    suite = load_suite(manifest.suite, settings)
    results = store.load_results()
    by_id = {t.id: t for t in suite.tasks}
    mlflow.set_tracking_uri(settings.effective_tracking_uri())
    client = MlflowClient()
    new: list[Label] = []
    missing = 0
    for r in results:
        tid = r.record.trace_id
        task = by_id.get(r.task_id)
        if not tid or task is None:
            continue
        try:
            trace = client.get_trace(tid, display=False)
        except Exception as e:  # noqa: BLE001
            missing += 1
            log(f"  trace {tid} unavailable: {type(e).__name__}")
            continue
        crits = {c.name for c in suite.judge_criteria_for(task)}
        for name, (val, why, ts) in _human_labels_on_trace(trace, crits).items():
            when = datetime.fromtimestamp(ts / 1000, tz=UTC) if ts else datetime.now(UTC)
            new.append(
                Label(
                    run_id=run_id,
                    trace_id=tid,
                    task_id=r.task_id,
                    rep=r.rep,
                    criterion=name,
                    human=val,
                    rationale=why,
                    labeled_at=when,
                )
            )
    if missing:
        log(f"  {missing} traces could not be fetched")
    labels = load_labels(settings, suite.name)
    count = labels.upsert(new)
    save_labels(settings, labels)
    return count


# Terminal labeling loop ---------------------------------------------------------------------
def _clip(text: str, limit: int) -> str:
    text = text or ""
    return (
        text if len(text) <= limit else text[:limit] + f"\n…[{len(text) - limit} more characters]"
    )


def label_tui(settings: Settings, run_id: str, input_fn: InputFn = input, out: Log = print) -> int:
    store = RunStore(settings, run_id)
    manifest = store.load_manifest()
    suite = load_suite(manifest.suite, settings)
    results = store.load_results()
    by_id = {t.id: t for t in suite.tasks}
    by_case = {(r.task_id, r.rep): r for r in results}

    sp = sample_path(settings, suite.name, run_id)
    if sp.is_file():
        sheet = json.loads(sp.read_text())
        cases = [by_case[k] for e in sheet["entries"] if (k := (e["task_id"], e["rep"])) in by_case]
        out(f"labeling {len(cases)} sampled cases from {sp.name}")
    else:
        cases = sorted(results, key=lambda r: (r.task_id, r.rep))
        out(f"no sample file; labeling all {len(cases)} cases")

    labels = load_labels(settings, suite.name)
    new: list[Label] = []
    quit_ = False
    for i, r in enumerate(cases, 1):
        task = by_id.get(r.task_id)
        if task is None:
            continue
        out("")
        out(f"=== [{i}/{len(cases)}] {r.task_id} rep{r.rep}  tags={','.join(task.tags)} ===")
        out("--- prompt ---")
        out(task.prompt.rstrip())
        if task.reference:
            out("--- reference (judge-only) ---")
            out(task.reference.rstrip())
        out("--- final output ---")
        out(_clip(r.record.final_text, TUI_OUTPUT_CHARS) or "(empty)")
        if task.kind == "workspace":
            out("--- diff ---")
            out(_clip(r.record.diff or "(no files changed)", TUI_DIFF_CHARS))
        for crit in suite.judge_criteria_for(task):
            while True:
                ans = (
                    input_fn(f"{crit.name} ({crit.description[:70]}) pass? [y/n/s(kip)/q(uit)] ")
                    .strip()
                    .lower()
                )
                if ans in {"y", "n", "s", "q"}:
                    break
                out("  answer y, n, s or q")
            if ans == "q":
                quit_ = True
                break
            if ans == "s":
                continue
            new.append(
                Label(
                    run_id=run_id,
                    trace_id=r.record.trace_id,
                    task_id=r.task_id,
                    rep=r.rep,
                    criterion=crit.name,
                    human=(ans == "y"),
                )
            )
        if quit_:
            break
    count = labels.upsert(new)
    save_labels(settings, labels)
    out(f"{count} labels written to {labels_path(settings, suite.name)}")
    return count


# Agreement ----------------------------------------------------------------------------------
class CriterionAgreement(BaseModel):
    n: int
    raw_agreement: float
    kappa: float | None


class CalibrationResult(BaseModel):
    path: Path
    suite: str
    judge_version: str
    judge_model: str | None
    judge_backend: str
    per_criterion: dict[str, CriterionAgreement]
    overall: CriterionAgreement
    flags: list[str]
    mlflow_run_id: str | None = None


def cohen_kappa(human: list[bool], judge: list[bool]) -> float | None:
    """Cohen's kappa for two binary raters.

    Returns 0.0 when only one rater is constant (a rubber-stamp judge scores zero, which is the
    signal kappa exists to give) and None only when both are constant (undefined, pe == 1).
    """
    if len(human) != len(judge) or not human:
        return None
    h = np.asarray(human, dtype=int)
    j = np.asarray(judge, dtype=int)
    table = np.zeros((2, 2), dtype=float)
    np.add.at(table, (h, j), 1)
    n = table.sum()
    po = float(np.trace(table) / n)
    pe = float((table.sum(axis=1) * table.sum(axis=0)).sum() / (n * n))
    if pe >= 1.0:
        return None
    return (po - pe) / (1.0 - pe)


def agreement(human: list[bool], judge: list[bool]) -> CriterionAgreement:
    n = len(human)
    raw = float(np.mean(np.asarray(human) == np.asarray(judge))) if n else 0.0
    return CriterionAgreement(n=n, raw_agreement=raw, kappa=cohen_kappa(human, judge))


def _stored_verdicts(results: list[CaseResult]) -> dict[tuple[str, int, str], bool | None]:
    out: dict[tuple[str, int, str], bool | None] = {}
    for r in results:
        for v in r.verdicts:
            if not v.determinism_rerun:
                out[(r.task_id, r.rep, v.criterion)] = v.passed
    return out


def _rejudge_labeled(
    settings: Settings,
    suite: Suite,
    results: list[CaseResult],
    labeled: list[Label],
    version: str,
    log: Log,
) -> tuple[dict[tuple[str, int, str], bool | None], str, str | None]:
    """Judge only the labeled (case, criterion) pairs with the current judge. Outputs stay frozen."""
    judge = get_judge(settings)
    by_id = {t.id: t for t in suite.tasks}
    by_case = {(r.task_id, r.rep): r for r in results}
    out: dict[tuple[str, int, str], bool | None] = {}
    log(
        f"  re-judging {len(labeled)} labeled verdicts with {judge.name} ({judge.model}) at version {version}"
    )
    for lab in labeled:
        res = by_case.get((lab.task_id, lab.rep))
        task = by_id.get(lab.task_id)
        if res is None or task is None:
            continue
        crit = next((c for c in suite.judge_criteria_for(task) if c.name == lab.criterion), None)
        if crit is None:
            continue
        try:
            out[(lab.task_id, lab.rep, lab.criterion)] = judge.judge(
                crit, task, res.record, version
            ).passed
        except Exception as e:  # noqa: BLE001
            log(
                f"  ! judge failed on {lab.task_id} rep{lab.rep} {lab.criterion}: {type(e).__name__}: {e}"
            )
            out[(lab.task_id, lab.rep, lab.criterion)] = None
    return out, judge.name, judge.model


def _flags(
    settings: Settings, per: dict[str, CriterionAgreement], overall: CriterionAgreement
) -> list[str]:
    flags: list[str] = []
    for name, row in [*per.items(), ("overall", overall)]:
        if row.raw_agreement < settings.agreement_min_raw:
            flags.append(
                f"{name}: raw agreement {row.raw_agreement:.2f} below {settings.agreement_min_raw}"
            )
        if row.kappa is not None and row.kappa < settings.agreement_min_kappa:
            flags.append(f"{name}: kappa {row.kappa:.2f} below {settings.agreement_min_kappa}")
    for name, row in per.items():
        if row.n < MIN_LABELS_PER_CRITERION:
            flags.append(f"n < {MIN_LABELS_PER_CRITERION} labels for {name} (n={row.n})")
    return flags


def _ensure_experiment(settings: Settings, name: str) -> str:
    exp = mlflow.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    kwargs: dict[str, Any] = {}
    if not settings.effective_tracking_uri().startswith("http"):
        settings.mlflow_artifacts.mkdir(parents=True, exist_ok=True)
        kwargs["artifact_location"] = (settings.mlflow_artifacts / "judge-calibration").as_uri()
    return mlflow.create_experiment(name, **kwargs)


def _log_calibration(
    settings: Settings, payload: dict[str, Any], path: Path, run_id: str, log: Log
) -> str | None:
    try:
        mlflow.set_tracking_uri(settings.effective_tracking_uri())
        exp_id = _ensure_experiment(settings, CALIBRATION_EXPERIMENT)
        with mlflow.start_run(
            experiment_id=exp_id, run_name=f"{payload['suite']}@{payload['judge_version']}"
        ) as run:
            mlflow.log_params(
                {
                    "suite": payload["suite"],
                    "judge_version": payload["judge_version"],
                    "judge_model": str(payload["judge_model"]),
                    "judge_backend": payload["judge_backend"],
                    "labeled_run_id": run_id,
                }
            )
            metrics: dict[str, float] = {"n_labels": float(payload["n_labels"])}
            for name, row in payload["per_criterion"].items():
                metrics[f"agreement.{name}"] = row["raw_agreement"]
                metrics[f"n.{name}"] = float(row["n"])
                if row["kappa"] is not None:
                    metrics[f"kappa.{name}"] = row["kappa"]
            metrics["agreement.overall"] = payload["overall"]["raw_agreement"]
            if payload["overall"]["kappa"] is not None:
                metrics["kappa.overall"] = payload["overall"]["kappa"]
            mlflow.log_metrics(metrics)
            mlflow.set_tags(
                {
                    "aeb.kind": "calibration",
                    "aeb.suite": payload["suite"],
                    "aeb.judge_version": payload["judge_version"],
                    "aeb.labeled_run_id": run_id,
                    "aeb.flagged": str(bool(payload["flags"])).lower(),
                }
            )
            mlflow.log_artifact(str(path))
            return run.info.run_id
    except Exception as e:  # noqa: BLE001
        log(f"  could not log calibration to MLflow: {type(e).__name__}: {e}")
        return None


def calibrate(
    settings: Settings, run_id: str, rejudge: bool = True, log: Log = print
) -> CalibrationResult:
    store = RunStore(settings, run_id)
    manifest: RunManifest = store.load_manifest()
    suite = load_suite(manifest.suite, settings)
    results = store.load_results()
    labeled = load_labels(settings, suite.name).for_run(run_id)
    if not labeled:
        raise ValueError(
            f"no labels for run {run_id[:8]} in {labels_path(settings, suite.name)}; "
            f"run `aeb labels sample {run_id[:8]}`, label, then `aeb labels pull {run_id[:8]}`"
        )

    current = judge_version(suite.criteria, get_judge(settings).model)
    if manifest.judge_version != current and rejudge:
        verdicts, backend, model = _rejudge_labeled(settings, suite, results, labeled, current, log)
        version = current
    else:
        if manifest.judge_version != current:
            log(
                f"  judge version changed ({manifest.judge_version} -> {current}); comparing against stored verdicts"
            )
        verdicts, backend, model = (
            _stored_verdicts(results),
            manifest.judge_backend,
            manifest.judge_model,
        )
        version = manifest.judge_version

    pairs: dict[str, tuple[list[bool], list[bool]]] = {}
    skipped = 0
    for lab in labeled:
        j = verdicts.get((lab.task_id, lab.rep, lab.criterion))
        if j is None:
            skipped += 1
            continue
        h, jj = pairs.setdefault(lab.criterion, ([], []))
        h.append(lab.human)
        jj.append(j)
    if skipped:
        log(f"  {skipped} labels had no judge verdict and were skipped")
    if not pairs:
        raise ValueError(
            f"none of the {len(labeled)} labels for run {run_id[:8]} matched a judge verdict"
        )

    per = {name: agreement(h, j) for name, (h, j) in sorted(pairs.items())}
    all_h = [x for h, _ in pairs.values() for x in h]
    all_j = [x for _, j in pairs.values() for x in j]
    overall = agreement(all_h, all_j)
    flags = _flags(settings, per, overall)

    path = settings.calibration_dir / suite.name / f"{version}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "suite": suite.name,
        "judge_version": version,
        "judge_model": model,
        "judge_backend": backend,
        "n_labels": overall.n,
        "labeled_run_ids": [run_id],
        "per_criterion": {k: v.model_dump() for k, v in per.items()},
        "overall": overall.model_dump(),
        "flags": flags,
        "computed_at": datetime.now(UTC).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2))
    mlrun = _log_calibration(settings, payload, path, run_id, log)
    return CalibrationResult(
        path=path,
        suite=suite.name,
        judge_version=version,
        judge_model=model,
        judge_backend=backend,
        per_criterion=per,
        overall=overall,
        flags=flags,
        mlflow_run_id=mlrun,
    )
