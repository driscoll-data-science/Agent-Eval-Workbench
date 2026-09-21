"""End-to-end: example suite, fake agent, fake judge, direct SQLite MLflow store. No network."""

from __future__ import annotations

import mlflow
import pytest

from agent_eval_bench.runner import RunOptions, rejudge_run, run_suite
from agent_eval_bench.storage import RunStore


@pytest.mark.timeout(600)
def test_run_suite_end_to_end(settings, example_suite, fake_profile):
    m = run_suite(
        example_suite,
        fake_profile,
        settings,
        RunOptions(reps=1, concurrency=3, compare=False, drift=False, report=False),
        log=lambda _: None,
    )
    store = RunStore(settings, m.run_id)
    results = store.load_results()
    assert len(results) == 10
    metrics = m.notes["metrics"]
    assert metrics["pass_rate"] == pytest.approx(1.0)
    assert metrics["pass_rate.tests_pass"] == pytest.approx(1.0)
    assert "judge_determinism" in metrics
    # Every case has a trace linked to the MLflow run, with agent/LLM/TOOL spans.
    assert all(r.record.trace_id for r in results)
    traces = mlflow.search_traces(run_id=m.run_id, return_type="list")
    assert len(traces) == 10
    span_types = {sp.span_type for sp in traces[0].data.spans}
    assert {"AGENT", "LLM"} <= span_types
    # Judge verdicts are mirrored onto the trace as LLM_JUDGE feedback.
    ws = next(r for r in results if r.task_id == "ws-slugify-unicode")
    t = mlflow.MlflowClient().get_trace(ws.record.trace_id)
    names = {a.name for a in (t.info.assessments or [])}
    assert {"minimal_diff", "readable", "explanation_accurate"} <= names
    # Workspace checks ran against the hidden tests and the diff was captured.
    assert ws.programmatic == {"tests_pass": True, "only_allowed_paths": True, "not_noop": True}
    assert "unicodedata" in (ws.record.diff or "")
    # MLflow run params and metrics recorded.
    run = mlflow.get_run(m.run_id)
    assert run.data.params["profile"] == "fake" and run.data.params["judge_backend"] == "fake"
    assert run.data.metrics["pass_rate"] == pytest.approx(1.0)
    assert run.data.tags.get("aeb.dataset")


@pytest.mark.timeout(600)
def test_regressed_profile_scores_lower_and_rejudge_is_idempotent(
    settings, example_suite, regressed_profile
):
    m = run_suite(
        example_suite,
        regressed_profile,
        settings,
        RunOptions(reps=1, concurrency=3, compare=False, drift=False, report=False),
        log=lambda _: None,
    )
    metrics = m.notes["metrics"]
    assert metrics["pass_rate"] < 1.0
    assert metrics["tool_calls_per_case_mean"] > 1.1  # extra_tool adds one call per case
    before = RunStore(settings, m.run_id).load_results()
    m2 = rejudge_run(settings, m.run_id, backend="fake", log=lambda _: None)
    after = RunStore(settings, m.run_id).load_results()
    assert m2.judge_version == m.judge_version
    assert [
        (v.criterion, v.passed) for r in before for v in r.verdicts if not v.determinism_rerun
    ] == [(v.criterion, v.passed) for r in after for v in r.verdicts if not v.determinism_rerun]


def test_task_selection_and_no_judge(settings, example_suite, fake_profile):
    m = run_suite(
        example_suite,
        fake_profile,
        settings,
        RunOptions(
            reps=1,
            judge=False,
            compare=False,
            drift=False,
            report=False,
            task_ids=["qa-leap-year", "ws-inventory-negative"],
        ),
        log=lambda _: None,
    )
    results = RunStore(settings, m.run_id).load_results()
    assert sorted(r.task_id for r in results) == ["qa-leap-year", "ws-inventory-negative"]
    assert all(r.verdicts == [] for r in results)
    assert m.judge_backend == "none"


def test_cli_smoke(settings):
    from typer.testing import CliRunner

    from agent_eval_bench.cli import app

    runner = CliRunner()
    r = runner.invoke(app, ["suite", "validate", "example"])
    assert r.exit_code == 0 and "10 tasks" in r.output
    r = runner.invoke(app, ["profile", "list"])
    assert r.exit_code == 0 and "fake" in r.output
    r = runner.invoke(
        app,
        [
            "run",
            "--suite",
            "example",
            "--profile",
            "fake",
            "--reps",
            "1",
            "--limit",
            "2",
            "--no-compare",
            "--no-drift",
            "--no-report",
        ],
    )
    assert r.exit_code == 0, r.output
    assert "done. run_id=" in r.output
    r = runner.invoke(app, ["runs", "list"])
    assert r.exit_code == 0 and "example" in r.output
    r = runner.invoke(app, ["baseline", "promote", "latest"])
    assert r.exit_code == 0 and "baseline for example::fake" in r.output
