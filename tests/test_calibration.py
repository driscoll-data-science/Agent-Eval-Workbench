"""Calibration: sampling, pulling HUMAN feedback, terminal labeling, agreement and kappa.

Offline: fake agent, fake judge, sqlite MLflow store from conftest.
"""

from __future__ import annotations

import json

import mlflow
import pytest
import yaml
from mlflow.entities import AssessmentSource, AssessmentSourceType

from agent_eval_bench.calibration import (
    CALIBRATION_EXPERIMENT,
    Label,
    LabelsFile,
    calibrate,
    cohen_kappa,
    label_tui,
    labels_path,
    load_labels,
    pull_labels,
    sample_for_labeling,
    save_labels,
)
from agent_eval_bench.judge.prompts import judge_version
from agent_eval_bench.runner import RunOptions, run_suite
from agent_eval_bench.storage import RunStore

OPTS = RunOptions(reps=1, compare=False, drift=False, report=False)


def _run(settings, suite, profile):
    manifest = run_suite(suite, profile, settings, OPTS, log=lambda _m: None)
    mlflow.flush_trace_async_logging()
    return manifest, RunStore(settings, manifest.run_id).load_results()


@pytest.fixture()
def run(settings, example_suite, fake_profile):
    return _run(settings, example_suite, fake_profile)


@pytest.fixture()
def regressed_run(settings, example_suite, regressed_profile):
    return _run(settings, example_suite, regressed_profile)


# 1. sampling ---------------------------------------------------------------------------------
def test_sample_is_stratified_and_written(settings, example_suite, run):
    manifest, results = run
    sheet = sample_for_labeling(settings, manifest.run_id, n=6, seed=1)
    assert 0 < len(sheet.entries) <= 6
    first_tags = {e.tags[0] for e in sheet.entries}
    assert len(first_tags) >= 2, "round-robin over first tags should cover several tags"
    assert len({(e.task_id, e.rep) for e in sheet.entries}) == len(sheet.entries)
    assert all(e.rep == 0 for e in sheet.entries)
    assert all(e.trace_id and e.trace_id.startswith("tr-") for e in sheet.entries)
    assert all(e.criteria and set(e.judge_verdicts) == set(e.criteria) for e in sheet.entries)
    assert all(e.trace_url and "selectedTraceId=" in e.trace_url for e in sheet.entries)
    assert "aeb labels pull" in sheet.instructions and "Human" in sheet.instructions

    assert sheet.path.is_file()
    assert sheet.path.name == f"example.sample.{manifest.run_id[:8]}.json"
    data = json.loads(sheet.path.read_text())
    assert data["suite"] == "example" and data["run_id"] == manifest.run_id
    assert [e["task_id"] for e in data["entries"]] == [e.task_id for e in sheet.entries]

    # Seeded: same seed reproduces the same sample; n above the case count returns everything.
    again = sample_for_labeling(settings, manifest.run_id, n=6, seed=1)
    assert [e.task_id for e in again.entries] == [e.task_id for e in sheet.entries]
    everything = sample_for_labeling(settings, manifest.run_id, n=100, seed=0)
    assert len(everything.entries) == len(results)


# 2. pulling HUMAN feedback -----------------------------------------------------------------
def _human(**kw):
    return AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="test", **kw)


def test_pull_labels_merges_human_feedback(settings, run):
    manifest, results = run
    by_task = {r.task_id: r for r in results}
    a = by_task["qa-capital-australia"].record.trace_id
    b = by_task["qa-leap-year"].record.trace_id
    c = by_task["ws-slugify-unicode"].record.trace_id
    mlflow.log_feedback(
        trace_id=a, name="correct", value=True, source=_human(), rationale="yes it is"
    )
    mlflow.log_feedback(trace_id=b, name="correct", value=False, source=_human())
    mlflow.log_feedback(trace_id=b, name="grounded", value="yes", source=_human())  # string coerced
    mlflow.log_feedback(trace_id=c, name="minimal_diff", value=1, source=_human())  # int coerced
    mlflow.log_feedback(trace_id=a, name="vibes", value=True, source=_human())  # not a criterion
    mlflow.log_feedback(
        trace_id=c, name="correct", value=True, source=_human()
    )  # not this task's criterion

    n = pull_labels(settings, manifest.run_id, log=lambda _m: None)
    assert n == 4

    path = labels_path(settings, "example")
    assert path.is_file()
    doc = yaml.safe_load(path.read_text())
    assert doc["suite"] == "example"
    got = {(e["task_id"], e["criterion"]): e for e in doc["entries"]}
    assert set(got) == {
        ("qa-capital-australia", "correct"),
        ("qa-leap-year", "correct"),
        ("qa-leap-year", "grounded"),
        ("ws-slugify-unicode", "minimal_diff"),
    }
    assert got[("qa-capital-australia", "correct")]["human"] is True
    assert got[("qa-capital-australia", "correct")]["rationale"] == "yes it is"
    assert got[("qa-leap-year", "correct")]["human"] is False
    assert got[("qa-leap-year", "grounded")]["human"] is True
    assert got[("ws-slugify-unicode", "minimal_diff")]["human"] is True
    for e in doc["entries"]:
        assert (
            e["run_id"] == manifest.run_id and e["trace_id"].startswith("tr-") and e["labeled_at"]
        )

    # Pulling again upserts rather than duplicating.
    assert pull_labels(settings, manifest.run_id, log=lambda _m: None) == 4
    assert len(load_labels(settings, "example").entries) == 4


def test_pull_labels_survives_missing_trace(settings, run):
    manifest, results = run
    store = RunStore(settings, manifest.run_id)
    results[0].record.trace_id = "tr-doesnotexist"
    store.save_results(results)
    assert pull_labels(settings, manifest.run_id, log=lambda _m: None) == 0


# 3. agreement and kappa ------------------------------------------------------------------------
def test_cohen_kappa_hand_computed():
    human = [True, True, False, False, True, False]
    judge = [True, False, False, False, True, True]
    # po = 4/6; marginals 3/6 each way -> pe = 0.25 + 0.25 = 0.5; kappa = (2/3 - 1/2) / (1/2) = 1/3
    assert cohen_kappa(human, judge) == pytest.approx(1 / 3)
    assert cohen_kappa([True, False, True], [True, False, True]) == pytest.approx(1.0)
    assert cohen_kappa([True, False], [False, True]) == pytest.approx(-1.0)
    assert cohen_kappa([True, True, True], [True, False, True]) == pytest.approx(
        0.0
    )  # human constant
    assert cohen_kappa([True, False, True], [True, True, True]) == pytest.approx(
        0.0
    )  # judge constant
    assert cohen_kappa([True, True], [True, True]) is None  # both constant: undefined
    assert cohen_kappa([], []) is None


def test_calibrate_matches_constructed_agreement(settings, example_suite, regressed_run):
    manifest, results = regressed_run
    qa = [r for r in results if r.task_id.startswith("qa-")]
    assert len(qa) == 8
    verdict = {(r.task_id, c): r.passed(c) for r in qa for c in ("correct", "concise")}
    assert all(v is not None for v in verdict.values())

    labels = LabelsFile(suite="example")
    new = []
    for i, r in enumerate(qa):
        j = verdict[(r.task_id, "correct")]
        human = (not j) if i < 2 else j  # two deliberate disagreements
        new.append(
            Label(
                run_id=manifest.run_id,
                trace_id=r.record.trace_id,
                task_id=r.task_id,
                rep=0,
                criterion="correct",
                human=human,
            )
        )
        new.append(
            Label(
                run_id=manifest.run_id,
                trace_id=r.record.trace_id,
                task_id=r.task_id,
                rep=0,
                criterion="concise",
                human=verdict[(r.task_id, "concise")],
            )
        )
    new.append(
        Label(run_id="other-run", task_id="qa-leap-year", rep=0, criterion="correct", human=False)
    )  # ignored
    labels.upsert(new)
    save_labels(settings, labels)

    res = calibrate(settings, manifest.run_id, rejudge=False, log=lambda _m: None)

    assert set(res.per_criterion) == {"correct", "concise"}
    correct = res.per_criterion["correct"]
    assert correct.n == 8
    assert correct.raw_agreement == pytest.approx(6 / 8)
    h = [lab.human for lab in new if lab.criterion == "correct" and lab.run_id == manifest.run_id]
    j = [
        verdict[(lab.task_id, "correct")]
        for lab in new
        if lab.criterion == "correct" and lab.run_id == manifest.run_id
    ]
    assert correct.kappa == cohen_kappa(h, j)
    assert res.per_criterion["concise"].raw_agreement == pytest.approx(1.0)
    assert res.overall.n == 16 and res.overall.raw_agreement == pytest.approx(14 / 16)

    assert f"correct: raw agreement 0.75 below {settings.agreement_min_raw}" in res.flags
    assert not any(f.startswith("concise: raw") for f in res.flags)
    assert "n < 10 labels for correct (n=8)" in res.flags
    assert "n < 10 labels for concise (n=8)" in res.flags

    assert (
        res.judge_version
        == manifest.judge_version
        == judge_version(example_suite.criteria, "fake-judge")
    )
    assert res.path == settings.calibration_dir / "example" / f"{res.judge_version}.json"
    assert res.path.is_file()
    doc = json.loads(res.path.read_text())
    assert doc["suite"] == "example" and doc["judge_version"] == res.judge_version
    assert doc["judge_backend"] == "fake" and doc["judge_model"] == "fake-judge"
    assert doc["n_labels"] == 16 and doc["labeled_run_ids"] == [manifest.run_id]
    assert doc["per_criterion"]["correct"] == {
        "n": 8,
        "raw_agreement": 0.75,
        "kappa": correct.kappa,
    }
    assert doc["overall"]["n"] == 16 and doc["flags"] == res.flags and doc["computed_at"]

    # Logged to the dedicated MLflow experiment.
    assert res.mlflow_run_id
    exp = mlflow.get_experiment_by_name(CALIBRATION_EXPERIMENT)
    assert exp is not None
    mlrun = mlflow.get_run(res.mlflow_run_id)
    assert mlrun.info.experiment_id == exp.experiment_id
    assert mlrun.data.tags["aeb.kind"] == "calibration"
    assert mlrun.data.params["suite"] == "example"
    assert mlrun.data.params["judge_version"] == res.judge_version
    assert mlrun.data.params["labeled_run_id"] == manifest.run_id
    assert mlrun.data.metrics["agreement.correct"] == pytest.approx(0.75)
    assert mlrun.data.metrics["agreement.overall"] == pytest.approx(14 / 16)
    assert mlrun.data.metrics["n_labels"] == 16
    if correct.kappa is not None:
        assert mlrun.data.metrics["kappa.correct"] == pytest.approx(correct.kappa)


def test_calibrate_rejudges_when_judge_version_changed(settings, example_suite, run, monkeypatch):
    manifest, results = run
    r = results[0]
    labels = LabelsFile(suite="example")
    labels.upsert(
        [
            Label(run_id=manifest.run_id, task_id=r.task_id, rep=0, criterion=c, human=True)
            for c in r.all_criteria()
            if c in example_suite.judge_criteria_names()
        ]
    )
    save_labels(settings, labels)

    store = RunStore(settings, manifest.run_id)
    manifest.judge_version = "stale0000000"
    store.save_manifest(manifest)
    current = judge_version(example_suite.criteria, "fake-judge")

    kept = calibrate(settings, manifest.run_id, rejudge=False, log=lambda _m: None)
    assert kept.judge_version == "stale0000000" and kept.path.name == "stale0000000.json"

    # The re-judge path must use the CURRENT judge, not the stored verdicts. Swap in a judge
    # that fails everything: agreement with the all-True human labels must drop to zero.
    from agent_eval_bench.models import Verdict

    class AlwaysFalseJudge:
        name = "fake"
        model = "fake-judge"

        def judge(self, criterion, task, record, version):
            return Verdict(
                task_id=task.id,
                rep=record.rep,
                criterion=criterion.name,
                passed=False,
                judge_backend=self.name,
                judge_model=self.model,
                judge_version=version,
            )

    import agent_eval_bench.calibration as cal_mod

    monkeypatch.setattr(cal_mod, "get_judge", lambda *a, **k: AlwaysFalseJudge())
    fresh = calibrate(settings, manifest.run_id, rejudge=True, log=lambda _m: None)
    assert fresh.judge_version == current and fresh.path.name == f"{current}.json"
    assert fresh.judge_backend == "fake"
    assert fresh.overall.raw_agreement == pytest.approx(0.0)
    assert kept.overall.raw_agreement == pytest.approx(1.0)


def test_calibrate_without_labels_raises(settings, run):
    manifest, _ = run
    with pytest.raises(ValueError, match="no labels"):
        calibrate(settings, manifest.run_id, rejudge=False, log=lambda _m: None)


# 4. terminal labeling ---------------------------------------------------------------------------
def test_label_tui_records_labels(settings, run):
    manifest, results = run
    sheet = sample_for_labeling(settings, manifest.run_id, n=2, seed=3)
    criteria = [(e.task_id, c) for e in sheet.entries for c in e.criteria]
    cycle = ["y", "n", "s"]
    answers = [cycle[i % 3] for i in range(len(criteria))]
    expected = {key: ans == "y" for key, ans in zip(criteria, answers, strict=True) if ans != "s"}

    script = iter(["maybe", *answers])  # first answer is invalid and must be re-asked
    printed: list[str] = []
    n = label_tui(
        settings, manifest.run_id, input_fn=lambda _prompt: next(script), out=printed.append
    )

    assert n == len(expected)
    doc = yaml.safe_load(labels_path(settings, "example").read_text())
    got = {(e["task_id"], e["criterion"]): e["human"] for e in doc["entries"]}
    assert got == expected
    assert all(e["run_id"] == manifest.run_id and e["rep"] == 0 for e in doc["entries"])
    joined = "\n".join(printed)
    assert "--- prompt ---" in joined and "--- final output ---" in joined
    assert "answer y, n, s or q" in joined
    if any(t.startswith("ws-") for t, _ in criteria):
        assert "--- diff ---" in joined


def test_label_tui_quit_saves_partial(settings, run):
    manifest, _ = run
    script = iter(["y", "q"])
    n = label_tui(settings, manifest.run_id, input_fn=lambda _p: next(script), out=lambda _m: None)
    assert n == 1
    entries = load_labels(settings, "example").entries
    assert len(entries) == 1 and entries[0].human is True
