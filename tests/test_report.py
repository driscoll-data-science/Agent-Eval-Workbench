"""Report builder: two offline runs of the example suite, synthetic compare/drift inputs on
the second, then assertions on the HTML, JSON, Markdown and index outputs."""

from __future__ import annotations

import json
import re

import pytest

from agent_eval_bench.report.build import (
    _trace_url,
    build_index,
    build_report,
    open_report,
    report_data,
    same_model_family,
)
from agent_eval_bench.runner import RunOptions, run_suite
from agent_eval_bench.storage import RunStore

OPTS = RunOptions(reps=1, compare=False, drift=False, report=False)
SECTION_HEADINGS = (
    "Verdict grid",
    "Per-criterion pass rates",
    "Cost and latency",
    "Drift",
    "Calibration status",
    "Profile params and run metadata",
    "Failures",
)


def _compare_json(run_id: str, baseline_id: str) -> dict:
    return {
        "run_id": run_id,
        "baseline_run_id": baseline_id,
        "ci_level": 0.95,
        "bootstrap_n": 200,
        "practical_floor": 0.0,
        "pairing": "task_rep",
        "n_pairs_total": 10,
        "metrics": [
            {
                "metric": "pass_rate.correct",
                "kind": "pass_rate",
                "higher_is_better": True,
                "n_pairs": 8,
                "baseline_mean": 1.0,
                "current_mean": 0.625,
                "delta": -0.375,
                "ci_low": -0.625,
                "ci_high": -0.125,
                "verdict": "regression",
            },
            {
                "metric": "pass_rate.concise",
                "kind": "pass_rate",
                "higher_is_better": True,
                "n_pairs": 8,
                "baseline_mean": 1.0,
                "current_mean": 1.0,
                "delta": 0.0,
                "ci_low": 0.0,
                "ci_high": 0.0,
                "verdict": "no_change",
            },
            {
                "metric": "agent_cost_usd_mean",
                "kind": "cost",
                "higher_is_better": False,
                "n_pairs": 10,
                "baseline_mean": 0.001,
                "current_mean": 0.0015,
                "delta": 0.0005,
                "ci_low": 0.0002,
                "ci_high": 0.0008,
                "verdict": "regression",
            },
            {
                "metric": "wall_ms_mean",
                "kind": "latency",
                "higher_is_better": False,
                "n_pairs": 10,
                "baseline_mean": 120.0,
                "current_mean": 110.0,
                "delta": -10.0,
                "ci_low": -40.0,
                "ci_high": 15.0,
                "verdict": "no_change",
            },
        ],
        "overall_verdict": "regression",
    }


def _drift_json(run_id: str, baseline_id: str) -> dict:
    return {
        "run_id": run_id,
        "baseline_run_id": baseline_id,
        "adapter": "fake",
        "js_threshold": 0.1,
        "alpha": 0.05,
        "tool_histogram_baseline": {"read": 6, "write": 4, "exec": 2},
        "tool_histogram_current": {"read": 5, "write": 4, "exec": 9, "web": 1},
        "raw_tool_histogram_baseline": {"Read": 6, "Edit": 4, "Bash": 2},
        "raw_tool_histogram_current": {"Read": 5, "Edit": 4, "Bash": 9, "WebFetch": 1},
        "shifts": [
            {
                "name": "tool_mix",
                "kind": "categorical",
                "baseline_summary": {"n": 12},
                "current_summary": {"n": 19},
                "js_divergence": 0.21,
                "statistic": 7.4,
                "p_value": 0.024,
                "verdict": "changed",
                "note": "exec share rose",
            },
            {
                "name": "output_chars",
                "kind": "numeric",
                "baseline_summary": {"mean": 80.2, "p50": 75.0},
                "current_summary": {"mean": 140.5, "p50": 130.0},
                "js_divergence": 0.05,
                "statistic": 1.9,
                "p_value": 0.41,
                "verdict": "stable",
                "note": "",
            },
        ],
        "overall_verdict": "changed",
    }


@pytest.fixture()
def two_runs(settings, example_suite, fake_profile, regressed_profile):
    quiet = lambda _msg: None  # noqa: E731
    base = run_suite(example_suite, fake_profile, settings, OPTS, log=quiet)
    cur = run_suite(example_suite, regressed_profile, settings, OPTS, log=quiet)
    store = RunStore(settings, cur.run_id)
    store.save_json("compare.json", _compare_json(cur.run_id, base.run_id))
    store.save_json("drift.json", _drift_json(cur.run_id, base.run_id))
    settings.baselines_file.write_text(json.dumps({"example::fake": base.run_id}))
    return base, cur


def _write_calibration(settings, judge_version: str, labeled_run_id: str) -> None:
    # The judge version is a hash of template + criteria, so it is the same for every run of
    # a suite; a calibration file applies to all of them. Only tests that want the
    # "calibrated" path call this.
    cal_dir = settings.calibration_dir / "example"
    cal_dir.mkdir(parents=True, exist_ok=True)
    (cal_dir / f"{judge_version}.json").write_text(
        json.dumps(
            {
                "suite": "example",
                "judge_version": judge_version,
                "judge_model": "fake-judge",
                "n_labels": 40,
                "labeled_run_ids": [labeled_run_id],
                "per_criterion": {
                    "correct": {"n": 40, "raw_agreement": 0.95, "kappa": 0.82},
                    "concise": {"n": 40, "raw_agreement": 0.70, "kappa": 0.31},
                },
                "flags": ["concise: raw agreement 0.70 below 0.85"],
                "computed_at": "2026-09-17T10:00:00Z",
            }
        )
    )


def test_build_report_with_baseline(settings, two_runs):
    base, cur = two_runs
    _write_calibration(settings, cur.judge_version, base.run_id)
    html_path = build_report(cur.run_id, settings)
    out_dir = settings.reports_dir / cur.run_id
    assert html_path == out_dir / "report.html"
    for name in ("report.html", "report.json", "report.md"):
        assert (out_dir / name).is_file(), name

    page = html_path.read_text()
    assert len(page.encode()) < 1_500_000
    for heading in SECTION_HEADINGS:
        assert heading in page, heading
    # Verdict words from the synthetic inputs, plus the cell reading.
    assert "regression" in page
    assert "changed" in page
    assert "behavior changed, quality regressed" in page
    # Self-contained: no scripts, no external stylesheets/fonts/images.
    assert "<script" not in page
    # No external assets: the only <link> allowed is the inline data: favicon.
    assert 'rel="stylesheet"' not in page and 'href="http' not in page
    assert page.count("<link") == page.count('<link rel="icon" href="data:,">')
    assert not re.search(r"""(src|href)=["']https?://""", page)
    assert "@import" not in page
    # Charts are inline SVG with a baseline tick and CI bracket.
    assert page.count("<svg") == 2
    assert 'class="baseline-tick"' in page
    assert 'class="ci"' in page
    assert 'class="bar-base"' in page
    # Sections carry their data.
    assert "CI of the delta" in page
    assert "exec share rose" in page
    assert "below threshold" in page
    assert "concise: raw agreement 0.70 below 0.85" in page
    assert "fail_rate" in page  # profile param of fake_regressed
    assert "never a regression flag by itself" in page
    assert "prefers-color-scheme: dark" in page

    data = json.loads((out_dir / "report.json").read_text())
    assert data["run_id"] == cur.run_id
    assert data["verdict"]["quality"] == "regression"
    assert data["verdict"]["behavior"] == "changed"
    assert data["verdict"]["active_cell"] == ["regression", "changed"]
    assert data["verdict"]["baseline_run_id"] == base.run_id
    by_name = {r["name"]: r for r in data["criteria"]["rows"]}
    assert by_name["correct"]["verdict"] == "regression"
    assert by_name["correct"]["baseline_mean"] == 1.0
    assert by_name["correct"]["n"] > 0
    assert {m["metric"] for m in data["cost"]["relative"]} == {
        "agent_cost_usd_mean",
        "wall_ms_mean",
    }
    absolute = data["cost"]["absolute"]
    assert absolute["n_cases"] == 10
    # fake-model-1 has no list price, so agent cost is unknown for every case and the page
    # must say so rather than silently reporting $0.
    assert absolute["agent_cost_unknown_cases"] == 10
    assert absolute["agent_cost_unknown_share"] == 1.0
    assert absolute["agent_cost_unknown_models"] == ["fake-model-1"]
    assert "Agent cost is incomplete" in page
    cats = {c["category"]: c for c in data["drift"]["categories"]}
    assert cats["exec"]["current_count"] == 9
    assert abs(sum(c["current_share"] for c in cats.values()) - 1.0) < 1e-9
    assert data["calibration"]["calibrated"] is True
    assert data["calibration"]["same_family_warning"] is False
    flagged = {c["name"]: c["below_threshold"] for c in data["calibration"]["per_criterion"]}
    assert flagged == {"correct": False, "concise": True}
    assert data["run"]["profile"]["params"]["fail_rate"] == 0.4
    assert data["run"]["mlflow_run_url"] is None  # sqlite tracking URI in tests
    # The regressed profile fails some cases; each failure carries rationale and output.
    assert data["failures"], "fake_regressed should produce failures"
    f = data["failures"][0]
    assert f["failed_criteria"] or f["error"]
    assert len(f["final_output"]) <= 600
    assert all(len(r["rationale"]) <= 500 for r in f["rationales"])

    md = (out_dir / "report.md").read_text()
    assert "# aeb report" in md
    assert "| quality \\ behavior | stable | changed |" in md
    assert "**(this run)**" in md
    assert "## 2. Per-criterion pass rates" in md
    assert "## 7. Failures" in md
    assert "exec share rose" in md


def test_no_baseline_report(settings, two_runs):
    base, _cur = two_runs
    html_path = build_report(base.run_id, settings)
    page = html_path.read_text()
    assert "no baseline set" in page
    assert "judge not yet calibrated for version" in page
    assert page.count("<svg") == 1  # pass-rate chart only; no drift histogram
    assert "current baseline" in page  # baselines.json marks this run
    for heading in SECTION_HEADINGS:
        assert heading in page, heading
    data = report_data(base.run_id, settings)
    assert data["verdict"]["quality"] == "no baseline set"
    assert data["verdict"]["behavior"] == "no baseline set"
    assert data["verdict"]["active_cell"] is None
    assert data["criteria"]["has_baseline"] is False
    assert data["cost"]["relative"] == []
    assert data["drift"]["has_baseline"] is False
    assert data["calibration"]["calibrated"] is False
    assert data["run"]["is_baseline"] is True
    md = (out_dir := settings.reports_dir / base.run_id) and (out_dir / "report.md").read_text()
    assert "no baseline set" in md
    assert "judge not yet calibrated" in md


def test_index_lists_both_runs(settings, two_runs):
    base, cur = two_runs
    build_report(cur.run_id, settings)
    index = build_index(settings)
    assert index == settings.reports_dir / "index.html"
    page = index.read_text()
    assert f'href="{base.run_id}/report.html"' in page
    assert f'href="{cur.run_id}/report.html"' in page
    assert page.index(cur.run_id[:8]) < page.index(base.run_id[:8])  # newest first
    assert "baseline" in page
    assert "fake_regressed" in page
    assert "regression" in page and "changed" in page  # verdict badges
    assert "no baseline set" in page  # the base run has no baseline of its own
    assert "<script" not in page
    assert not re.search(r"""(src|href)=["']https?://""", page)


def test_open_report_builds_when_missing(settings, two_runs, monkeypatch):
    _base, cur = two_runs
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url) or True)
    path = open_report(settings, cur.run_id[:8])
    assert path.is_file() and path.name == "report.html"
    assert opened == [path.as_uri()]
    index = open_report(settings, None)
    assert index == settings.reports_dir / "index.html"
    assert opened[-1] == index.as_uri()


def test_same_model_family_and_trace_url():
    assert same_model_family("claude-opus-5", "claude-opus-5")
    assert same_model_family("claude-opus-5", "claude-opus-4-1")
    assert not same_model_family("claude-opus-5", "claude-sonnet-5")
    assert not same_model_family("gpt-5", "claude-sonnet-5")
    assert same_model_family("gpt-5", "gpt-5")
    assert not same_model_family("fake-judge", "fake-model-1")
    assert not same_model_family(None, "claude-opus-5")
    assert (
        _trace_url("http://localhost:5050", "7", "tr-abc")
        == "http://localhost:5050/#/experiments/7/traces/tr-abc"
    )
    assert _trace_url("sqlite:///x.db", "7", "tr-abc") is None
    assert _trace_url("http://localhost:5050", "7", None) is None
