"""`aeb` command line. Every scheduler-facing action is a non-interactive subcommand."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from . import __version__
from .config import Settings, load_settings

app = typer.Typer(
    help="Agent Eval Bench: MLflow-based evaluation kit for AI agents.", no_args_is_help=True
)
mlflow_app = typer.Typer(help="Local MLflow tracking server lifecycle.", no_args_is_help=True)
suite_app = typer.Typer(help="Suites: list, validate, draft.", no_args_is_help=True)
profile_app = typer.Typer(help="Agent profiles.", no_args_is_help=True)
runs_app = typer.Typer(help="Recorded runs.", no_args_is_help=True)
baseline_app = typer.Typer(help="Explicit baseline promotion.", no_args_is_help=True)
report_app = typer.Typer(help="Static HTML/JSON/Markdown reports.", no_args_is_help=True)
labels_app = typer.Typer(help="Human calibration labels.", no_args_is_help=True)
app.add_typer(mlflow_app, name="mlflow")
app.add_typer(suite_app, name="suite")
app.add_typer(profile_app, name="profile")
app.add_typer(runs_app, name="runs")
app.add_typer(baseline_app, name="baseline")
app.add_typer(report_app, name="report")
app.add_typer(labels_app, name="labels")


def _settings(config: Path | None = None) -> Settings:
    s = load_settings(config)
    s.ensure_dirs()
    return s


def _echo(msg: str) -> None:
    typer.echo(msg)


@app.callback()
def _main(version: bool = typer.Option(False, "--version", help="Print version and exit.")):
    if version:
        typer.echo(f"aeb {__version__}")
        raise typer.Exit()


# run --------------------------------------------------------------------------------------
@app.command()
def run(
    suite: str = typer.Option(..., "--suite", "-s", help="Suite name or path."),
    profile: str = typer.Option(..., "--profile", "-p", help="Profile name or path."),
    reps: int | None = typer.Option(None, help="Repetitions per task (default from config)."),
    concurrency: int | None = typer.Option(None, help="Parallel agent runs."),
    judge: bool = typer.Option(True, "--judge/--no-judge"),
    judge_backend: str | None = typer.Option(None, help="claude_code | mlflow | fake"),
    judge_model: str | None = typer.Option(None),
    compare: bool = typer.Option(True, "--compare/--no-compare"),
    drift: bool = typer.Option(True, "--drift/--no-drift"),
    report: bool = typer.Option(True, "--report/--no-report"),
    promote: bool = typer.Option(
        False, "--promote", help="Promote this run to baseline when done."
    ),
    limit: int | None = typer.Option(None, help="Only the first N tasks."),
    task: list[str] = typer.Option([], "--task", help="Run only these task ids (repeatable)."),
    tag: list[str] = typer.Option([], "--tag", help="Run only tasks with any of these tags."),
    name: str | None = typer.Option(None, "--name", help="MLflow run name."),
    config: Path | None = typer.Option(None, "--config", help="Path to an aeb.toml."),
):
    """Run a suite against a profile: agent runs, checks, judge, cost, compare, drift, report."""
    from .profiles import load_profile
    from .runner import RunOptions, run_suite
    from .suites import load_suite

    s = _settings(config)
    st = load_suite(suite, s)
    pr = load_profile(profile, s)
    opts = RunOptions(
        reps=reps,
        concurrency=concurrency,
        judge=judge,
        judge_backend=judge_backend,
        judge_model=judge_model,
        compare=compare,
        drift=drift,
        report=report,
        promote=promote,
        limit=limit,
        task_ids=task or None,
        tags=tag or None,
        run_name=name,
    )
    manifest = run_suite(st, pr, s, opts, log=_echo)
    _echo(f"done. run_id={manifest.run_id}")
    _echo(
        f"mlflow: {s.effective_tracking_uri()}/#/experiments/{manifest.experiment_id}/runs/{manifest.run_id}"
    )
    rep = s.reports_dir / manifest.run_id / "report.html"
    if rep.exists():
        _echo(f"report: {rep}")


@app.command()
def judge(
    run_ref: str = typer.Argument("latest", help="Run id, unique prefix, or 'latest'."),
    backend: str | None = typer.Option(None, "--backend"),
    model: str | None = typer.Option(None, "--model"),
    config: Path | None = typer.Option(None, "--config"),
):
    """Re-judge a run's frozen outputs with the current judge prompt and criteria."""
    from .runner import rejudge_run
    from .storage import resolve_run_id

    s = _settings(config)
    rid = resolve_run_id(s, run_ref)
    m = rejudge_run(s, rid, backend, model, log=_echo)
    _echo(f"re-judged {rid} with {m.judge_backend} ({m.judge_model}) version {m.judge_version}")


@app.command()
def compare(
    run_ref: str = typer.Argument("latest"),
    baseline: str | None = typer.Option(
        None, "--baseline", help="Baseline run id (default: promoted baseline)."
    ),
    as_json: bool = typer.Option(False, "--json"),
    config: Path | None = typer.Option(None, "--config"),
):
    """Signed comparison of a run against its baseline (paired bootstrap CIs)."""
    from .baseline import get_baseline
    from .compare import compare_runs
    from .storage import RunStore, resolve_run_id
    from .suites import load_suite

    s = _settings(config)
    rid = resolve_run_id(s, run_ref)
    store = RunStore(s, rid)
    m = store.load_manifest()
    bid = resolve_run_id(s, baseline) if baseline else get_baseline(s, m.suite, m.profile.name)
    if not bid:
        raise typer.BadParameter("no baseline set for this suite/profile; promote one first")
    result = compare_runs(
        store.load_results(),
        RunStore(s, bid).load_results(),
        load_suite(m.suite, s),
        s,
        run_id=rid,
        baseline_run_id=bid,
    )
    store.save_json("compare.json", result)
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
    else:
        _echo(f"{rid[:8]} vs baseline {bid[:8]}: {result.overall_verdict}")
        for mv in result.metrics:
            _echo(
                f"  {mv.metric:<28} {mv.baseline_mean:9.4f} -> {mv.current_mean:9.4f}  delta {mv.delta:+.4f}  CI[{mv.ci_low:+.4f}, {mv.ci_high:+.4f}]  {mv.verdict}"
            )


@app.command()
def drift(
    run_ref: str = typer.Argument("latest"),
    baseline: str | None = typer.Option(None, "--baseline"),
    as_json: bool = typer.Option(False, "--json"),
    config: Path | None = typer.Option(None, "--config"),
):
    """Unsigned behavioral drift of a run against its baseline (tool mix, tokens, latency)."""
    from .baseline import get_baseline
    from .drift import detect_drift
    from .storage import RunStore, resolve_run_id

    s = _settings(config)
    rid = resolve_run_id(s, run_ref)
    store = RunStore(s, rid)
    m = store.load_manifest()
    bid = resolve_run_id(s, baseline) if baseline else get_baseline(s, m.suite, m.profile.name)
    if not bid:
        raise typer.BadParameter("no baseline set for this suite/profile; promote one first")
    result = detect_drift(
        store.load_results(),
        RunStore(s, bid).load_results(),
        m.profile.adapter,
        s,
        run_id=rid,
        baseline_run_id=bid,
    )
    store.save_json("drift.json", result)
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
    else:
        _echo(f"{rid[:8]} vs baseline {bid[:8]}: behavior {result.overall_verdict}")
        for sh in result.shifts:
            js = f"JS={sh.js_divergence:.3f}" if sh.js_divergence is not None else ""
            pv = f"p={sh.p_value:.3g}" if sh.p_value is not None else ""
            _echo(f"  {sh.name:<28} {sh.verdict:<10} {js} {pv}")


@app.command()
def calibrate(
    run_ref: str = typer.Argument("latest"),
    no_rejudge: bool = typer.Option(
        False, "--no-rejudge", help="Do not re-judge when the judge version changed."
    ),
    config: Path | None = typer.Option(None, "--config"),
):
    """Compute human-vs-judge agreement (raw and Cohen's kappa) from labels for a run."""
    from .calibration import calibrate as _cal
    from .storage import resolve_run_id

    s = _settings(config)
    rid = resolve_run_id(s, run_ref)
    res = _cal(s, rid, rejudge=not no_rejudge, log=_echo)
    _echo(f"calibration written: {res.path}")
    for name, row in res.per_criterion.items():
        _echo(
            f"  {name:<22} n={row.n:<4} raw={row.raw_agreement:.3f} kappa={row.kappa if row.kappa is None else round(row.kappa, 3)}"
        )
    for f in res.flags:
        _echo(f"  FLAG {f}")


@app.command("judge-selftest")
def judge_selftest(
    backend: str | None = typer.Option(None, "--backend"),
    model: str | None = typer.Option(None, "--model"),
    config: Path | None = typer.Option(None, "--config"),
):
    """Feed the judge known negatives (empty, non-answer, wrong question, grader injection); all must fail."""
    from .judge.base import get_judge
    from .judge.selftest import run_selftest

    s = _settings(config)
    j = get_judge(s, backend, model)
    res = run_selftest(j, log=_echo)
    if res.ok:
        _echo(
            f"judge self-test OK: {j.name} ({j.model}) failed all {len(res.rows)} known negatives"
        )
    else:
        _echo(
            f"judge self-test FAILED: {len(res.failures)} of {len(res.rows)} known negatives were passed"
        )
        raise typer.Exit(code=1)


# mlflow -----------------------------------------------------------------------------------
@mlflow_app.command("up")
def mlflow_up(config: Path | None = typer.Option(None, "--config")):
    """Start the local MLflow server (SQLite backend) detached."""
    from .mlflow_server import up

    st = up(_settings(config))
    _echo(f"mlflow server healthy at {st['uri']} (pid {st['pid']}); db {st['db']}")


@mlflow_app.command("down")
def mlflow_down(config: Path | None = typer.Option(None, "--config")):
    """Stop the local MLflow server."""
    from .mlflow_server import down

    _echo("stopped" if down(_settings(config)) else "not running")


@mlflow_app.command("status")
def mlflow_status(config: Path | None = typer.Option(None, "--config")):
    from .mlflow_server import status

    _echo(json.dumps(status(_settings(config)), indent=2))


@mlflow_app.command("open")
def mlflow_open(config: Path | None = typer.Option(None, "--config")):
    """Open the MLflow UI in the browser."""
    from .mlflow_server import open_ui

    open_ui(_settings(config))


# suites / profiles / runs -----------------------------------------------------------------
@suite_app.command("list")
def suite_list(config: Path | None = typer.Option(None, "--config")):
    from .suites import list_suites

    for name, path in list_suites(_settings(config)):
        _echo(f"{name:<24} {path}")


@suite_app.command("validate")
def suite_validate(suite: str, config: Path | None = typer.Option(None, "--config")):
    from .suites import load_suite, validate_suite

    st = load_suite(suite, _settings(config))
    warnings = validate_suite(st)
    _echo(f"{st.name}: {len(st.tasks)} tasks, {len(st.criteria)} criteria, ok")
    for w in warnings:
        _echo(f"  warning: {w}")


@suite_app.command("draft")
def suite_draft(
    kind: str = typer.Option("qa", help="qa | workspace"),
    n: int = typer.Option(10),
    topic: str = typer.Option("general reasoning, grounded answers, instruction following"),
    out: Path = typer.Option(..., "--out", help="Directory to write suite.draft.yaml into."),
    model: str = typer.Option("claude-opus-5"),
    config: Path | None = typer.Option(None, "--config"),
):
    """Draft candidate golden cases with Claude Code headless for a human to curate."""
    from .suite_tools import draft_suite

    path = draft_suite(
        _settings(config), kind=kind, n=n, topic=topic, out_dir=out, model=model, log=_echo
    )
    _echo(f"draft written: {path}  (edit, then rename to suite.yaml and run `aeb suite validate`)")


@profile_app.command("list")
def profile_list(config: Path | None = typer.Option(None, "--config")):
    from .profiles import list_profiles

    for p in list_profiles(_settings(config)):
        _echo(f"{p.name:<28} adapter={p.adapter:<12} model={p.model}")


@runs_app.command("list")
def runs_list(limit: int = typer.Option(20), config: Path | None = typer.Option(None, "--config")):
    from .baseline import load_baselines
    from .storage import list_runs

    s = _settings(config)
    bl = set(load_baselines(s).values())
    for m in list_runs(s)[:limit]:
        pr = (m.notes.get("metrics") or {}).get("pass_rate")
        mark = "*" if m.run_id in bl else " "
        _echo(
            f"{mark} {m.run_id[:12]} {m.started_at:%Y-%m-%d %H:%M} {m.suite:<12} {m.profile.name:<26} pass={pr if pr is None else round(pr, 3)}"
        )


# baseline ---------------------------------------------------------------------------------
@baseline_app.command("promote")
def baseline_promote(
    run_ref: str = typer.Argument("latest"), config: Path | None = typer.Option(None, "--config")
):
    from .baseline import promote
    from .storage import resolve_run_id

    s = _settings(config)
    rid = resolve_run_id(s, run_ref)
    key, prev = promote(s, rid)
    _echo(f"baseline for {key} is now {rid}" + (f" (was {prev})" if prev else ""))


@baseline_app.command("list")
def baseline_list(config: Path | None = typer.Option(None, "--config")):
    from .baseline import load_baselines

    for k, v in sorted(load_baselines(_settings(config)).items()):
        _echo(f"{k:<40} {v}")


# report -----------------------------------------------------------------------------------
@report_app.command("build")
def report_build(
    run_ref: str = typer.Argument("latest"), config: Path | None = typer.Option(None, "--config")
):
    from .report.build import build_report
    from .storage import resolve_run_id

    s = _settings(config)
    path = build_report(resolve_run_id(s, run_ref), s)
    _echo(str(path))


@report_app.command("open")
def report_open(
    run_ref: str | None = typer.Argument(None, help="Run id or prefix; omit for the index."),
    config: Path | None = typer.Option(None, "--config"),
):
    from .report.build import open_report

    path = open_report(_settings(config), run_ref)
    _echo(str(path))


@report_app.command("index")
def report_index(config: Path | None = typer.Option(None, "--config")):
    from .report.build import build_index

    _echo(str(build_index(_settings(config))))


# labels -----------------------------------------------------------------------------------
@labels_app.command("sample")
def labels_sample(
    run_ref: str = typer.Argument("latest"),
    n: int = typer.Option(40),
    seed: int = typer.Option(0),
    config: Path | None = typer.Option(None, "--config"),
):
    """Pick a stratified sample of cases from a run to label in the MLflow UI."""
    from .calibration import sample_for_labeling
    from .storage import resolve_run_id

    s = _settings(config)
    sheet = sample_for_labeling(s, resolve_run_id(s, run_ref), n=n, seed=seed)
    _echo(f"sample of {len(sheet.entries)} cases written to {sheet.path}")
    _echo(sheet.instructions)


@labels_app.command("pull")
def labels_pull(
    run_ref: str = typer.Argument("latest"), config: Path | None = typer.Option(None, "--config")
):
    """Pull HUMAN feedback from MLflow traces into the private labels file."""
    from .calibration import pull_labels
    from .storage import resolve_run_id

    s = _settings(config)
    n = pull_labels(s, resolve_run_id(s, run_ref))
    _echo(f"{n} labels merged")


@labels_app.command("tui")
def labels_tui(
    run_ref: str = typer.Argument("latest"), config: Path | None = typer.Option(None, "--config")
):
    """Label the sampled cases in the terminal instead of the MLflow UI."""
    from .calibration import label_tui
    from .storage import resolve_run_id

    s = _settings(config)
    n = label_tui(s, resolve_run_id(s, run_ref))
    _echo(f"{n} labels recorded")


if __name__ == "__main__":
    app()
