"""compare.py: paired bootstrap comparison, no MLflow, no network.

CaseResults are synthesized by driving FakeAgent and FakeJudge directly over the example
suite, which is exactly what the runner does minus tracing and programmatic checks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from agent_eval_bench.adapters.fake import FakeAgent
from agent_eval_bench.compare import (
    ComparisonResult,
    bootstrap_mean_ci,
    classify,
    compare_runs,
)
from agent_eval_bench.judge.fake import FakeJudge
from agent_eval_bench.models import CaseResult, Profile, Suite

JUDGE_VERSION = "test"


def synth_results(
    suite: Suite, profile: Profile, reps: int, workdir: Path, cost_per_token: float = 1e-6
) -> list[CaseResult]:
    agent, judge = FakeAgent(), FakeJudge()
    out: list[CaseResult] = []
    for task in suite.tasks:
        for rep in range(reps):
            ws = None
            if task.kind == "workspace":
                ws = workdir / profile.name / f"{task.id}_{rep}"
                ws.mkdir(parents=True, exist_ok=True)
            record = agent.run(task, rep, ws, profile, timeout_s=30)
            verdicts = [
                judge.judge(crit, task, record, JUDGE_VERSION)
                for crit in suite.judge_criteria_for(task)
            ]
            out.append(
                CaseResult(
                    record=record,
                    verdicts=verdicts,
                    agent_cost_usd=record.total_tokens * cost_per_token,
                )
            )
    out.sort(key=lambda r: (r.task_id, r.rep))
    return out


def by_metric(result: ComparisonResult) -> dict[str, str]:
    return {m.metric: m.verdict for m in result.metrics}


@pytest.fixture()
def fake_results(example_suite, fake_profile, tmp_path):
    return synth_results(example_suite, fake_profile, reps=2, workdir=tmp_path / "ws")


@pytest.fixture()
def regressed_results(example_suite, regressed_profile, tmp_path):
    return synth_results(example_suite, regressed_profile, reps=2, workdir=tmp_path / "ws")


def test_identical_runs_are_no_change(settings, example_suite, fake_results):
    result = compare_runs(fake_results, fake_results, example_suite, settings, "cur", "base")

    assert result.pairing == "task_rep"
    assert result.n_pairs_total == len(fake_results) == 2 * len(example_suite.tasks)
    assert result.run_id == "cur" and result.baseline_run_id == "base"
    assert result.overall_verdict == "no_change"
    for m in result.metrics:
        if m.n_pairs == 0:
            # Programmatic criteria are never populated by direct synthesis.
            assert m.verdict == "insufficient", m.metric
            continue
        assert m.verdict == "no_change", m.metric
        assert m.delta == 0.0
        assert (m.ci_low, m.ci_high) == (0.0, 0.0)
        assert m.baseline_mean == m.current_mean


def test_regressed_profile_flags_quality_and_tokens(
    settings, example_suite, fake_results, regressed_results
):
    result = compare_runs(regressed_results, fake_results, example_suite, settings)
    verdicts = by_metric(result)

    assert result.pairing == "task_rep"
    assert verdicts["pass_rate"] == "regression"
    assert verdicts["pass_rate.correct"] == "regression"
    assert verdicts["total_tokens"] == "regression"  # more tokens, lower is better
    assert result.overall_verdict == "regression"

    pr = next(m for m in result.metrics if m.metric == "pass_rate")
    assert pr.kind == "quality" and pr.higher_is_better
    assert pr.current_mean < pr.baseline_mean
    assert pr.ci_low <= pr.delta <= pr.ci_high < 0

    tok = next(m for m in result.metrics if m.metric == "total_tokens")
    assert tok.kind == "tokens" and not tok.higher_is_better
    assert 0 < tok.ci_low <= tok.delta <= tok.ci_high


def test_cost_never_drives_overall_verdict(settings, example_suite, fake_results):
    # Same quality, ten times the cost: cost regresses, overall stays no_change.
    pricier = [
        r.model_copy(update={"agent_cost_usd": (r.agent_cost_usd or 0) * 10 + 0.01})
        for r in fake_results
    ]
    result = compare_runs(pricier, fake_results, example_suite, settings)
    verdicts = by_metric(result)

    assert verdicts["agent_cost_usd"] == "regression"
    assert verdicts["pass_rate"] == "no_change"
    assert result.overall_verdict == "no_change"


def test_two_cases_is_insufficient(settings, example_suite, fake_results, regressed_results):
    cur = regressed_results[:2]
    base = fake_results[:2]
    result = compare_runs(cur, base, example_suite, settings)

    assert result.n_pairs_total == 2
    assert all(m.verdict == "insufficient" for m in result.metrics)
    assert result.overall_verdict == "insufficient"


def test_mismatched_reps_pairs_by_task_mean(settings, example_suite, fake_profile, tmp_path):
    two_reps = synth_results(example_suite, fake_profile, reps=2, workdir=tmp_path / "a")
    one_rep = synth_results(example_suite, fake_profile, reps=1, workdir=tmp_path / "b")
    result = compare_runs(two_reps, one_rep, example_suite, settings)

    assert result.pairing == "task_mean"
    assert result.n_pairs_total == len(example_suite.tasks)
    assert by_metric(result)["pass_rate"] == "no_change"
    # Only tasks present on both sides count.
    partial = [r for r in one_rep if r.task_id != example_suite.tasks[0].id]
    result = compare_runs(two_reps, partial, example_suite, settings)
    assert result.n_pairs_total == len(example_suite.tasks) - 1


def test_metric_deltas_keys_match_metrics(settings, example_suite, fake_results, regressed_results):
    result = compare_runs(regressed_results, fake_results, example_suite, settings)
    deltas = result.metric_deltas()

    assert list(deltas) == [m.metric for m in result.metrics]
    assert deltas == {m.metric: m.delta for m in result.metrics}
    assert {"pass_rate", "agent_cost_usd", "wall_ms", "total_tokens"} <= set(deltas)
    assert all(f"pass_rate.{c.name}" in deltas for c in example_suite.criteria)


def test_result_round_trips_through_json(settings, example_suite, fake_results, regressed_results):
    result = compare_runs(regressed_results, fake_results, example_suite, settings, "r", "b")
    again = ComparisonResult.model_validate_json(result.model_dump_json())
    assert again == result


def test_compare_is_deterministic(settings, example_suite, fake_results, regressed_results):
    a = compare_runs(regressed_results, fake_results, example_suite, settings)
    b = compare_runs(regressed_results, fake_results, example_suite, settings)
    assert a == b


def test_practical_floor_suppresses_small_shifts(
    settings, example_suite, fake_results, regressed_results
):
    strict = settings.model_copy(update={"practical_floor": 0.0})
    lenient = settings.model_copy(update={"practical_floor": 1.0})
    args = (regressed_results, fake_results, example_suite)
    assert compare_runs(*args, strict).overall_verdict == "regression"
    assert compare_runs(*args, lenient).overall_verdict == "no_change"


@pytest.mark.parametrize(
    ("lo", "hi", "higher_is_better", "expected"),
    [
        (-0.5, -0.1, True, "regression"),
        (0.1, 0.5, True, "improvement"),
        (-0.1, 0.1, True, "no_change"),
        (0.1, 0.5, False, "regression"),
        (-0.5, -0.1, False, "improvement"),
        (-0.1, 0.1, False, "no_change"),
        (0.0, 0.0, True, "no_change"),
    ],
)
def test_classify(lo, hi, higher_is_better, expected):
    assert classify(lo, hi, n_pairs=10, higher_is_better=higher_is_better, floor=0.0) == expected


def test_classify_insufficient_and_floor():
    assert classify(-0.5, -0.1, n_pairs=2, higher_is_better=True, floor=0.0) == "insufficient"
    assert classify(-0.5, -0.1, n_pairs=10, higher_is_better=True, floor=0.6) == "no_change"


def test_bootstrap_ci_properties():
    assert bootstrap_mean_ci(np.zeros(10), 500, 0.95) == (0.0, 0.0)
    assert bootstrap_mean_ci(np.array([]), 500, 0.95) == (0.0, 0.0)
    diffs = np.array([-1.0, -1.0, 0.0, -1.0, -0.5, -1.0, 0.0, -1.0])
    lo, hi = bootstrap_mean_ci(diffs, 2000, 0.95)
    assert lo <= diffs.mean() <= hi
    assert hi < 0
    assert bootstrap_mean_ci(diffs, 2000, 0.95) == (lo, hi)  # seeded, reproducible
    lo90, hi90 = bootstrap_mean_ci(diffs, 2000, 0.90)
    assert lo <= lo90 and hi90 <= hi


def test_overall_improvement_and_regression_precedence(
    settings, example_suite, fake_profile, regressed_profile
):
    from agent_eval_bench.compare import compare_runs
    from agent_eval_bench.runner import RunOptions, run_suite
    from agent_eval_bench.storage import RunStore

    opts = RunOptions(reps=2, compare=False, drift=False, report=False)
    good = RunStore(
        settings, run_suite(example_suite, fake_profile, settings, opts, log=lambda _: None).run_id
    ).load_results()
    bad = RunStore(
        settings,
        run_suite(example_suite, regressed_profile, settings, opts, log=lambda _: None).run_id,
    ).load_results()
    # good measured against a bad baseline is an improvement
    up = compare_runs(good, bad, example_suite, settings)
    assert up.overall_verdict == "improvement"
    assert any(m.verdict == "improvement" for m in up.metrics if m.kind == "quality")
    # the reverse is a regression, and regression wins even though tokens "improve"
    down = compare_runs(bad, good, example_suite, settings)
    assert down.overall_verdict == "regression"


def test_unknown_cost_is_insufficient_not_zero(settings, example_suite, fake_profile):
    from agent_eval_bench.compare import compare_runs
    from agent_eval_bench.runner import RunOptions, run_suite
    from agent_eval_bench.storage import RunStore

    opts = RunOptions(reps=1, compare=False, drift=False, report=False)
    base = RunStore(
        settings, run_suite(example_suite, fake_profile, settings, opts, log=lambda _: None).run_id
    ).load_results()
    cur = RunStore(
        settings, run_suite(example_suite, fake_profile, settings, opts, log=lambda _: None).run_id
    ).load_results()
    for r in base:
        r.agent_cost_usd, r.agent_cost_known = 0.02, True
    for r in cur:
        r.agent_cost_usd, r.agent_cost_known = 0.0, False  # model missing from the price table
    res = compare_runs(cur, base, example_suite, settings)
    cost = next(m for m in res.metrics if m.metric == "agent_cost_usd")
    assert cost.verdict == "insufficient" and cost.n_pairs == 0
