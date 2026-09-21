"""Unit tests for the spine: models, cost, suites, workspace checks, storage, baselines, config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_eval_bench.config import DEFAULT_PRICES, load_settings
from agent_eval_bench.cost import PriceTable
from agent_eval_bench.models import (
    CaseResult,
    Criterion,
    ModelCall,
    RunRecord,
    Suite,
    Task,
    Verdict,
    WorkspaceChecks,
)
from agent_eval_bench.suites import load_suite, validate_suite
from agent_eval_bench.workspace import (
    changed_paths,
    prepare_workspace,
    qa_expected_match,
    run_checks,
    snapshot,
    unified_diff,
)


def test_programmatic_criterion_name_validated():
    with pytest.raises(ValueError):
        Criterion(name="made_up", description="x", kind="programmatic")
    Criterion(name="tests_pass", description="x", kind="programmatic")


def test_workspace_task_requires_fixture():
    with pytest.raises(ValueError):
        Task(id="t", kind="workspace", prompt="p")
    t = Task(id="t", kind="workspace", prompt="p", fixture="fixtures/x")
    assert isinstance(t.checks, WorkspaceChecks)


def test_suite_rejects_duplicates_and_unknown_criteria():
    c = [Criterion(name="correct", description="x")]
    with pytest.raises(ValueError):
        Suite(name="s", criteria=c, tasks=[Task(id="a", prompt="p"), Task(id="a", prompt="q")])
    s = Suite(name="s", criteria=c, tasks=[Task(id="a", prompt="p", criteria=["nope"])])
    with pytest.raises(ValueError):
        s.criteria_for(s.tasks[0])


def test_case_result_passed_prefers_programmatic_and_ignores_reruns():
    rec = RunRecord(task_id="a", rep=0, agent="fake", profile="p")
    res = CaseResult(
        record=rec,
        programmatic={"tests_pass": False},
        verdicts=[
            Verdict(task_id="a", rep=0, criterion="correct", passed=True),
            Verdict(task_id="a", rep=0, criterion="correct", passed=False, determinism_rerun=True),
        ],
    )
    assert res.passed("tests_pass") is False
    assert res.passed("correct") is True
    assert res.passed("missing") is None
    assert res.all_criteria() == ["tests_pass", "correct"]


def test_price_table_resolution_and_cost():
    t = PriceTable.load(DEFAULT_PRICES)
    assert t.resolve("claude-haiku-4-5-20251001") == "claude-haiku-4-5"
    assert t.resolve("opus") == "claude-opus-5"
    assert t.resolve("gpt-99-unknown") is None
    call = ModelCall(index=0, model="claude-opus-5", input_tokens=1_000_000, output_tokens=0)
    assert t.call_cost(call) == pytest.approx(5.0)
    rec = RunRecord(
        task_id="a",
        rep=0,
        agent="x",
        profile="p",
        model="claude-opus-5",
        model_calls=[call, ModelCall(index=1, model="mystery-model", input_tokens=10)],
    )
    cost, known = t.record_cost(rec)
    assert cost == pytest.approx(5.0) and known is False
    empty = RunRecord(task_id="a", rep=0, agent="x", profile="p")
    assert t.record_cost(empty) == (0.0, True)


def test_run_record_token_sums():
    rec = RunRecord(
        task_id="a",
        rep=0,
        agent="x",
        profile="p",
        model_calls=[
            ModelCall(
                index=0,
                input_tokens=10,
                output_tokens=5,
                cache_read_tokens=100,
                cache_write_tokens=7,
            )
        ],
    )
    assert (
        rec.input_tokens,
        rec.output_tokens,
        rec.cache_read_tokens,
        rec.cache_write_tokens,
        rec.total_tokens,
    ) == (10, 5, 100, 7, 122)


def test_example_suite_loads_and_validates(settings):
    suite = load_suite("example", settings)
    assert suite.name == "example" and len(suite.tasks) == 10
    assert validate_suite(suite) == []
    ws = [t for t in suite.tasks if t.kind == "workspace"]
    assert len(ws) == 2 and all(t.reference_patch for t in ws)


def test_workspace_checks_end_to_end(settings, tmp_path):
    suite = load_suite("example", settings)
    task = next(t for t in suite.tasks if t.id == "ws-inventory-negative")
    ws = prepare_workspace(suite, task, tmp_path / "ws")
    before = snapshot(ws)
    wanted = ["tests_pass", "only_allowed_paths", "not_noop", "must_exist"]
    # Untouched fixture: hidden tests fail, no-op detected.
    r0 = run_checks(suite, task, ws, before, wanted)
    assert (
        r0["tests_pass"] is False and r0["not_noop"] is False and r0["only_allowed_paths"] is True
    )
    # Apply the reference patch: everything passes.
    for rel, content in task.reference_patch.items():
        (ws / rel).write_text(content)
    r1 = run_checks(suite, task, ws, before, wanted)
    assert r1 == {
        "tests_pass": True,
        "only_allowed_paths": True,
        "not_noop": True,
        "must_exist": True,
    }
    diff = unified_diff(suite, task, ws, before)
    assert "inventory.py" in diff and "if qty > current" in diff
    # Touching a protected file fails only_allowed_paths.
    (ws / "tests/test_inventory.py").write_text("# tampered\n")
    r2 = run_checks(suite, task, ws, before, ["only_allowed_paths"])
    assert r2["only_allowed_paths"] is False


def test_changed_paths_and_expected_match():
    assert changed_paths({"a": "1", "b": "2"}, {"a": "1", "b": "3", "c": "4"}) == ["b", "c"]
    assert changed_paths({"a": "1"}, {}) == ["a"]
    t = Task(id="q", prompt="p", expected="Canberra")
    assert qa_expected_match(t, "It is canberra.") is True
    assert qa_expected_match(Task(id="q", prompt="p"), "x") is None


def test_settings_env_and_toml(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "aeb.toml").write_text(
        '[judge]\nbackend = "fake"\nmodel = "claude-sonnet-5"\n[run]\nreps = 3\n[mlflow]\nport = 6000\n'
    )
    monkeypatch.setenv("AEB_HOME", str(home))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("AEB_JUDGE_BACKEND", raising=False)
    monkeypatch.chdir(tmp_path)
    s = load_settings()
    assert s.aeb_home == home and s.judge_backend == "fake" and s.judge_model == "claude-sonnet-5"
    assert (
        s.reps == 3
        and s.mlflow_port == 6000
        and s.effective_tracking_uri() == "http://127.0.0.1:6000"
    )
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://remote:5000")
    assert load_settings().effective_tracking_uri() == "http://remote:5000"


def test_storage_roundtrip_and_baseline(settings):
    from datetime import UTC, datetime

    from agent_eval_bench.baseline import get_baseline, promote
    from agent_eval_bench.models import Profile, RunManifest
    from agent_eval_bench.storage import RunStore, list_runs, resolve_run_id

    prof = Profile(name="fake", adapter="fake")
    for rid in ("aaaa1111", "bbbb2222"):
        st = RunStore(settings, rid).ensure()
        st.save_manifest(
            RunManifest(
                run_id=rid,
                experiment="aeb/example",
                suite="example",
                suite_version="1",
                profile=prof,
                judge_backend="fake",
                judge_model="fake-judge",
                judge_version="v",
                reps=1,
                n_tasks=1,
                started_at=datetime.now(UTC),
            )
        )
        st.save_results(
            [CaseResult(record=RunRecord(task_id="a", rep=0, agent="fake", profile="fake"))]
        )
    assert len(list_runs(settings)) == 2
    assert resolve_run_id(settings, "aaaa") == "aaaa1111"
    with pytest.raises(FileNotFoundError):
        resolve_run_id(settings, "zzzz")
    assert get_baseline(settings, "example", "fake") is None
    key, prev = promote(settings, "aaaa1111")
    assert key == "example::fake" and prev is None
    key, prev = promote(settings, "bbbb2222")
    assert prev == "aaaa1111" and get_baseline(settings, "example", "fake") == "bbbb2222"
    assert json.loads(settings.baselines_file.read_text()) == {"example::fake": "bbbb2222"}
    assert RunStore(settings, "bbbb2222").load_results()[0].record.task_id == "a"


def test_judge_version_changes_with_criteria():
    from agent_eval_bench.judge.prompts import build_user_message, judge_version

    a = [Criterion(name="correct", description="x")]
    b = [Criterion(name="correct", description="y")]
    assert judge_version(a) != judge_version(b)
    task = Task(id="t", prompt="Q?", reference="R", kind="qa")
    msg = build_user_message(
        a[0], task, RunRecord(task_id="t", rep=0, agent="x", profile="p", final_text="A")
    )
    assert "Reference material" in msg and "<output>\nA\n</output>" in msg


def test_fixture_dirs_exist():
    root = Path(__file__).resolve().parent.parent / "suites" / "example"
    assert (root / "fixtures/slugify/slug.py").is_file()
    assert (root / "hidden_tests/inventory/test_hidden_inventory.py").is_file()
