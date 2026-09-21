"""The run loop: agent runs -> programmatic checks -> traces -> judge -> cost -> MLflow metrics.

Compare, drift, and report are invoked at the end when their modules are available and a
baseline exists; each is also runnable on its own from the CLI (portability contract).
"""

from __future__ import annotations

import contextlib
import statistics
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType

from . import __version__
from .adapters.base import get_adapter
from .baseline import get_baseline
from .config import Settings
from .cost import PriceTable
from .judge.base import JudgeBackend, get_judge
from .judge.prompts import judge_version
from .models import CaseResult, Profile, RunManifest, RunRecord, Suite, Task, Verdict
from .storage import RunStore
from .tracing import flush, log_record_trace
from .workspace import prepare_workspace, qa_expected_match, run_checks, snapshot, unified_diff

Log = Callable[[str], None]


@dataclass
class RunOptions:
    reps: int | None = None
    concurrency: int | None = None
    judge: bool = True
    judge_backend: str | None = None
    judge_model: str | None = None
    compare: bool = True
    drift: bool = True
    report: bool = True
    promote: bool = False
    limit: int | None = None
    task_ids: list[str] | None = None
    tags: list[str] | None = None
    run_name: str | None = None
    extra_tags: dict[str, str] = field(default_factory=dict)


def experiment_name(suite: Suite) -> str:
    return f"aeb/{suite.name}"


def setup_mlflow(settings: Settings, suite: Suite) -> str:
    uri = settings.effective_tracking_uri()
    mlflow.set_tracking_uri(uri)
    name = experiment_name(suite)
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        kwargs = {}
        if not uri.startswith("http"):
            # Direct SQLite/file store: pin artifacts under AEB_HOME instead of ./mlruns.
            settings.mlflow_artifacts.mkdir(parents=True, exist_ok=True)
            kwargs["artifact_location"] = (settings.mlflow_artifacts / suite.name).as_uri()
        exp_id = mlflow.create_experiment(name, **kwargs)
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment_id=exp_id)
    return exp_id


def select_tasks(suite: Suite, opts: RunOptions) -> list[Task]:
    tasks = list(suite.tasks)
    if opts.task_ids:
        wanted = set(opts.task_ids)
        tasks = [t for t in tasks if t.id in wanted]
    if opts.tags:
        wanted_tags = set(opts.tags)
        tasks = [t for t in tasks if wanted_tags & set(t.tags)]
    if opts.limit:
        tasks = tasks[: opts.limit]
    if not tasks:
        raise ValueError("no tasks selected")
    return tasks


def register_dataset(suite: Suite, tasks: list[Task], exp_id: str, log: Log) -> str | None:
    """Best effort: MLflow evaluation datasets need a SQL backend; a file store raises."""
    try:
        from mlflow.genai import datasets as ds

        name = f"aeb-{suite.name}-v{suite.version}"
        try:
            dataset = ds.get_dataset(name=name)
        except Exception:
            dataset = ds.create_dataset(
                name=name,
                experiment_id=exp_id,
                tags={"aeb.suite": suite.name, "aeb.version": suite.version},
            )
        records = []
        for t in tasks:
            records.append(
                {
                    "inputs": {"task_id": t.id, "kind": t.kind, "prompt": t.prompt},
                    "expectations": {
                        k: v
                        for k, v in {"reference": t.reference, "expected": t.expected}.items()
                        if v
                    },
                    "tags": {"aeb.tags": ",".join(t.tags)},
                }
            )
        dataset.merge_records(records)
        # Also attach the dataset to the run as an input so the run page lists it.
        with contextlib.suppress(Exception):
            import mlflow.data

            mlflow.log_input(
                mlflow.data.from_pandas(dataset.to_df(), name=name), context="evaluation"
            )
        return getattr(dataset, "dataset_id", None) or name
    except Exception as e:  # noqa: BLE001
        log(f"  dataset registration skipped: {type(e).__name__}: {str(e)[:120]}")
        return None


def run_suite(
    suite: Suite,
    profile: Profile,
    settings: Settings,
    opts: RunOptions | None = None,
    log: Log = print,
) -> RunManifest:
    opts = opts or RunOptions()
    settings.ensure_dirs()
    reps = opts.reps or settings.reps
    concurrency = opts.concurrency or settings.concurrency
    tasks = select_tasks(suite, opts)
    prices = PriceTable.load(settings.price_table)
    judge: JudgeBackend | None = (
        get_judge(settings, opts.judge_backend, opts.judge_model) if opts.judge else None
    )
    version = judge_version(suite.criteria, judge.model if judge else None)
    exp_id = setup_mlflow(settings, suite)
    adapter = get_adapter(profile)
    started = datetime.now(UTC)

    run_name = opts.run_name or f"{profile.name}@{suite.name}"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        store = RunStore(settings, run_id).ensure()
        log(
            f"run {run_id}  suite={suite.name} profile={profile.name} tasks={len(tasks)} reps={reps}"
        )
        params = profile.flat_params()
        params.update(
            {
                "suite": suite.name,
                "suite_version": suite.version,
                "reps": str(reps),
                "n_tasks": str(len(tasks)),
                "judge_backend": judge.name if judge else "none",
                "judge_model": str(judge.model) if judge else "none",
                "judge_version": version,
                "kit_version": __version__,
                "price_table_version": prices.version,
            }
        )
        mlflow.log_params(params)
        mlflow.set_tags(
            {
                "aeb.kind": "eval",
                "aeb.suite": suite.name,
                "aeb.profile": profile.name,
                "aeb.agent": profile.adapter,
                "aeb.judge_version": version,
                **opts.extra_tags,
            }
        )
        dataset_id = register_dataset(suite, tasks, exp_id, log)
        if dataset_id:
            mlflow.set_tag("aeb.dataset", str(dataset_id))

        manifest = RunManifest(
            run_id=run_id,
            experiment=experiment_name(suite),
            experiment_id=exp_id,
            suite=suite.name,
            suite_version=suite.version,
            profile=profile,
            judge_backend=judge.name if judge else "none",
            judge_model=judge.model if judge else None,
            judge_version=version,
            reps=reps,
            n_tasks=len(tasks),
            started_at=started,
            kit_version=__version__,
            tracking_uri=settings.effective_tracking_uri(),
            baseline_run_id=get_baseline(settings, suite.name, profile.name),
        )
        store.save_manifest(manifest)

        # Phase 1: agent runs -------------------------------------------------------------
        jobs = [(t, r) for t in tasks for r in range(reps)]
        results: list[CaseResult] = []
        log(f"  phase 1: {len(jobs)} agent runs, concurrency {concurrency}")
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = {
                pool.submit(
                    _run_one, suite, task, rep, profile, adapter, settings, store, prices, run_id
                ): (task, rep)
                for task, rep in jobs
            }
            for fut in as_completed(futs):
                task, rep = futs[fut]
                try:
                    results.append(fut.result())
                except Exception as e:  # noqa: BLE001
                    log(f"  ! {task.id} rep{rep}: {type(e).__name__}: {e}")
                    rec = RunRecord(
                        task_id=task.id,
                        rep=rep,
                        agent=profile.adapter,
                        profile=profile.name,
                        model=profile.model,
                        exit_status=1,
                        error=f"{type(e).__name__}: {e}",
                    )
                    results.append(CaseResult(record=rec, agent_cost_usd=0.0))
        results.sort(key=lambda r: (r.task_id, r.rep))
        store.save_results(results)
        flush()  # traces export asynchronously; the judge phase logs feedback onto them

        # Phase 2: judge -------------------------------------------------------------------
        if judge is not None:
            judge_results(results, suite, judge, settings, version, prices, log)
            store.save_results(results)

        # Phase 3: metrics -----------------------------------------------------------------
        metrics = summarize_metrics(results, suite)
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, int | float)})
        manifest.finished_at = datetime.now(UTC)
        manifest.notes["metrics"] = metrics
        store.save_manifest(manifest)
        mlflow.log_artifact(str(store.results_path))
        mlflow.log_artifact(str(store.manifest_path))
        flush()
        log(
            f"  pass_rate={metrics.get('pass_rate'):.3f} cases={len(results)} agent_cost=${metrics.get('agent_cost_usd_total', 0):.4f} judge_cost=${metrics.get('judge_cost_usd_total', 0):.4f}"
        )

        # Phase 4: compare / drift / report -----------------------------------------------
        if opts.promote:
            from .baseline import promote

            promote(settings, run_id)
            manifest.baseline_run_id = run_id
            store.save_manifest(manifest)
            log("  promoted to baseline")
        _post_run(suite, manifest, results, settings, store, opts, log)
    return manifest


def judge_results(
    results: list[CaseResult],
    suite: Suite,
    judge: JudgeBackend,
    settings: Settings,
    version: str,
    prices: PriceTable,
    log: Log = print,
) -> None:
    """Phase 2: one verdict per (case, judge criterion), plus a determinism re-judgement sample.
    Existing verdicts are replaced so this is safe to call again when the judge version changes."""
    by_id = {t.id: t for t in suite.tasks}
    for res in results:
        res.verdicts = []
    jjobs = [
        (res, crit) for res in results for crit in suite.judge_criteria_for(by_id[res.task_id])
    ]
    n_det = (
        int(len(results) * settings.determinism_sample + 0.999)
        if settings.determinism_sample > 0
        else 0
    )
    det_sample = results[:: max(1, len(results) // n_det)][:n_det] if n_det else []
    djobs = [
        (res, crit) for res in det_sample for crit in suite.judge_criteria_for(by_id[res.task_id])
    ]
    log(
        f"  phase 2: {len(jjobs)} judge verdicts + {len(djobs)} determinism re-judgements, backend {judge.name} ({judge.model})"
    )
    with ThreadPoolExecutor(max_workers=settings.judge_concurrency) as pool:
        futs = {
            pool.submit(judge.judge, crit, by_id[res.task_id], res.record, version): (
                res,
                crit,
                False,
            )
            for res, crit in jjobs
        }
        futs.update(
            {
                pool.submit(judge.judge, crit, by_id[res.task_id], res.record, version): (
                    res,
                    crit,
                    True,
                )
                for res, crit in djobs
            }
        )
        for fut in as_completed(futs):
            res, crit, rerun = futs[fut]
            try:
                v = fut.result()
            except Exception as e:  # noqa: BLE001
                v = Verdict(
                    task_id=res.task_id,
                    rep=res.rep,
                    criterion=crit.name,
                    passed=None,
                    error=f"{type(e).__name__}: {e}",
                    judge_backend=judge.name,
                    judge_model=judge.model,
                    judge_version=version,
                )
            v.determinism_rerun = rerun
            v.cost_usd = prices.verdict_cost(v)
            res.verdicts.append(v)
    for res in results:
        res.verdicts.sort(key=lambda v: (v.determinism_rerun, v.criterion))
        res.judge_cost_usd = sum(v.cost_usd or 0.0 for v in res.verdicts if not v.determinism_rerun)
        _log_verdict_feedback(res, judge)


def rejudge_run(
    settings: Settings,
    run_id: str,
    backend: str | None = None,
    model: str | None = None,
    log: Log = print,
) -> RunManifest:
    """Re-judge the frozen outputs of an existing run with the current judge and criteria."""
    from .suites import load_suite

    store = RunStore(settings, run_id)
    manifest = store.load_manifest()
    suite = load_suite(manifest.suite, settings)
    results = store.load_results()
    prices = PriceTable.load(settings.price_table)
    judge = get_judge(settings, backend, model)
    version = judge_version(suite.criteria, judge.model)
    mlflow.set_tracking_uri(settings.effective_tracking_uri())
    judge_results(results, suite, judge, settings, version, prices, log)
    store.save_results(results)
    manifest.judge_backend, manifest.judge_model, manifest.judge_version = (
        judge.name,
        judge.model,
        version,
    )
    metrics = summarize_metrics(results, suite)
    manifest.notes["metrics"] = metrics
    store.save_manifest(manifest)
    try:
        with mlflow.start_run(run_id=run_id):
            mlflow.set_tags({"aeb.judge_version": version, "aeb.rejudged": "true"})
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, int | float)})
            mlflow.log_artifact(str(store.results_path))
    except Exception as e:  # noqa: BLE001
        log(f"  could not update MLflow run: {e}")
    return manifest


def _run_one(
    suite: Suite,
    task: Task,
    rep: int,
    profile: Profile,
    adapter,
    settings: Settings,
    store: RunStore,
    prices: PriceTable,
    run_id: str,
) -> CaseResult:
    workspace: Path | None = None
    before: dict[str, str] = {}
    if task.kind == "workspace":
        workspace = prepare_workspace(suite, task, store.work_dir(task.id, rep) / "ws")
        before = snapshot(workspace)
    try:
        record = adapter.run(task, rep, workspace, profile, settings.agent_timeout_s)
    except Exception as e:  # noqa: BLE001
        record = RunRecord(
            task_id=task.id,
            rep=rep,
            agent=profile.adapter,
            profile=profile.name,
            model=profile.model,
            exit_status=1,
            error=f"{type(e).__name__}: {e}",
            raw={"traceback": traceback.format_exc()[-2000:]},
        )
    programmatic: dict[str, bool] = {}
    wanted = [c.name for c in suite.programmatic_criteria_for(task)]
    if task.kind == "workspace" and workspace is not None:
        record.diff = unified_diff(suite, task, workspace, before)
        programmatic = run_checks(
            suite, task, workspace, before, wanted, log_dir=store.work_dir(task.id, rep)
        )
    elif "expected_match" in wanted:
        m = qa_expected_match(task, record.final_text)
        if m is not None:
            programmatic["expected_match"] = m
    cost, known = prices.record_cost(record)
    try:
        record.trace_id = log_record_trace(
            record, task, suite.name, run_id, prices.record_cost_split(record)
        )
    except Exception as e:  # noqa: BLE001
        record.raw["trace_error"] = f"{type(e).__name__}: {e}"
    return CaseResult(
        record=record, programmatic=programmatic, agent_cost_usd=cost, agent_cost_known=known
    )


def _log_verdict_feedback(res: CaseResult, judge: JudgeBackend) -> None:
    """Mirror judge verdicts onto the trace as LLM_JUDGE feedback so they show in the trace UI
    and so MLflow's judge alignment can pair them with HUMAN feedback of the same name."""
    if not res.record.trace_id:
        return
    for v in res.verdicts:
        if v.determinism_rerun or v.passed is None:
            continue
        try:
            mlflow.log_feedback(
                trace_id=res.record.trace_id,
                name=v.criterion,
                value=bool(v.passed),
                source=AssessmentSource(
                    source_type=AssessmentSourceType.LLM_JUDGE,
                    source_id=f"{judge.name}:{v.judge_model}",
                ),
                rationale=v.rationale[:2000],
                metadata={"judge_version": v.judge_version},
            )
        except Exception:
            return


def summarize_metrics(results: list[CaseResult], suite: Suite) -> dict[str, float]:
    m: dict[str, float] = {}
    if not results:
        return m
    criteria = [c.name for c in suite.criteria]
    all_vals: list[float] = []
    for c in criteria:
        vals = [res.passed(c) for res in results]
        vals = [float(v) for v in vals if v is not None]
        if vals:
            m[f"pass_rate.{c}"] = sum(vals) / len(vals)
            all_vals += vals
    m["pass_rate"] = sum(all_vals) / len(all_vals) if all_vals else 0.0
    m["n_cases"] = float(len(results))
    m["error_rate"] = sum(1 for r in results if r.record.error) / len(results)
    m["refusal_rate"] = sum(1 for r in results if r.record.refused) / len(results)
    costs = [r.agent_cost_usd or 0.0 for r in results]
    m["agent_cost_usd_total"] = sum(costs)
    m["agent_cost_usd_mean"] = sum(costs) / len(costs)
    m["agent_cost_known_share"] = sum(1 for r in results if r.agent_cost_known) / len(results)
    jc = [r.judge_cost_usd or 0.0 for r in results]
    m["judge_cost_usd_total"] = sum(jc)
    walls = [r.record.wall_ms for r in results]
    m["wall_ms_mean"] = statistics.fmean(walls)
    m["wall_ms_p50"] = statistics.median(walls)
    m["wall_ms_p95"] = sorted(walls)[int(0.95 * (len(walls) - 1))]
    m["input_tokens_total"] = float(sum(r.record.input_tokens for r in results))
    m["output_tokens_total"] = float(sum(r.record.output_tokens for r in results))
    m["cache_read_tokens_total"] = float(sum(r.record.cache_read_tokens for r in results))
    m["cache_write_tokens_total"] = float(sum(r.record.cache_write_tokens for r in results))
    m["tool_calls_per_case_mean"] = statistics.fmean(len(r.record.tool_calls) for r in results)
    m["output_chars_mean"] = statistics.fmean(len(r.record.final_text or "") for r in results)
    # Judge determinism: share of re-judged verdicts that agree with the original.
    agree = total = 0
    for r in results:
        orig = {v.criterion: v.passed for v in r.verdicts if not v.determinism_rerun}
        for v in r.verdicts:
            if v.determinism_rerun and v.passed is not None and orig.get(v.criterion) is not None:
                total += 1
                agree += int(v.passed == orig[v.criterion])
    if total:
        m["judge_determinism"] = agree / total
    m["judge_error_rate"] = sum(1 for r in results for v in r.verdicts if v.error) / max(
        1, sum(len(r.verdicts) for r in results)
    )
    return m


def _post_run(
    suite: Suite,
    manifest: RunManifest,
    results: list[CaseResult],
    settings: Settings,
    store: RunStore,
    opts: RunOptions,
    log: Log,
) -> None:
    baseline_id = manifest.baseline_run_id
    if baseline_id and baseline_id != manifest.run_id and (opts.compare or opts.drift):
        try:
            base_results = RunStore(settings, baseline_id).load_results()
        except FileNotFoundError:
            log(f"  baseline {baseline_id} has no local results; skipping compare/drift")
            base_results = None
        if base_results is not None:
            if opts.compare:
                try:
                    from .compare import compare_runs

                    cmp = compare_runs(
                        results,
                        base_results,
                        suite,
                        settings,
                        run_id=manifest.run_id,
                        baseline_run_id=baseline_id,
                    )
                    store.save_json("compare.json", cmp)
                    mlflow.log_metrics({f"delta.{k}": v for k, v in cmp.metric_deltas().items()})
                    mlflow.set_tag("aeb.quality_verdict", cmp.overall_verdict)
                    log(f"  compare vs {baseline_id[:8]}: {cmp.overall_verdict}")
                except ImportError:
                    log("  compare module not available yet")
            if opts.drift:
                try:
                    from .drift import detect_drift

                    d = detect_drift(
                        results,
                        base_results,
                        manifest.profile.adapter,
                        settings,
                        run_id=manifest.run_id,
                        baseline_run_id=baseline_id,
                    )
                    store.save_json("drift.json", d)
                    mlflow.set_tag("aeb.behavior_verdict", d.overall_verdict)
                    log(f"  drift vs {baseline_id[:8]}: {d.overall_verdict}")
                except ImportError:
                    log("  drift module not available yet")
    if opts.report:
        try:
            from .report.build import build_report

            path = build_report(manifest.run_id, settings)
            with contextlib.suppress(Exception):
                mlflow.log_artifact(str(path), artifact_path="report")
            log(f"  report: {path}")
        except ImportError:
            log("  report module not available yet")
        except Exception as e:  # noqa: BLE001
            log(f"  report failed: {type(e).__name__}: {e}")
