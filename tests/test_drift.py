"""Drift detection: unsigned distribution shifts between two result sets.

Cases come from FakeAgent, either directly (fast, no MLflow) or through ``run_suite`` on the
example suite with the SQLite store from conftest (offline, about a second per run).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_eval_bench.adapters.fake import FakeAgent
from agent_eval_bench.config import DEFAULT_TOOL_MAP, Settings
from agent_eval_bench.drift import (
    DriftResult,
    categorical_shift,
    continuous_shift,
    detect_drift,
    load_tool_map,
    normalize_tool,
    overall_verdict,
    proportion_shift,
)
from agent_eval_bench.models import CaseResult, Profile, RunRecord, Suite, ToolCall

# Helpers ------------------------------------------------------------------------------------


def fake_cases(suite: Suite, profile: Profile, reps: int, root: Path) -> list[CaseResult]:
    """Run FakeAgent directly on every task; workspace tasks get a scratch directory."""
    agent = FakeAgent()
    out: list[CaseResult] = []
    for task in suite.tasks:
        for rep in range(reps):
            ws = None
            if task.kind == "workspace":
                ws = root / profile.name / f"{task.id}_{rep}"
                ws.mkdir(parents=True, exist_ok=True)
            out.append(CaseResult(record=agent.run(task, rep, ws, profile, timeout_s=10)))
    return out


def record(
    tools: list[str],
    task_id: str = "t",
    rep: int = 0,
    text: str = "ok",
    refused: bool = False,
    error: str | None = None,
) -> CaseResult:
    calls = [ToolCall(index=i, name=n) for i, n in enumerate(tools)]
    rec = RunRecord(
        task_id=task_id,
        rep=rep,
        agent="claude_code",
        profile="p",
        final_text=text,
        tool_calls=calls,
        refused=refused,
        error=error,
        exit_status=1 if error else 0,
    )
    return CaseResult(record=rec)


def shift_by_name(result: DriftResult, name: str):
    matches = [s for s in result.shifts if s.name == name]
    assert len(matches) == 1, f"expected exactly one shift named {name!r}"
    return matches[0]


# 1. same profile twice -----------------------------------------------------------------------


def test_same_profile_twice_is_stable(settings, example_suite, fake_profile):
    from agent_eval_bench.runner import RunOptions, run_suite
    from agent_eval_bench.storage import RunStore

    opts = RunOptions(reps=1, compare=False, drift=False, report=False)
    first = run_suite(example_suite, fake_profile, settings, opts, log=lambda _m: None)
    second = run_suite(example_suite, fake_profile, settings, opts, log=lambda _m: None)
    base = RunStore(settings, first.run_id).load_results()
    cur = RunStore(settings, second.run_id).load_results()
    assert len(base) == len(cur) == len(example_suite.tasks)

    result = detect_drift(
        cur,
        base,
        fake_profile.adapter,
        settings,
        run_id=second.run_id,
        baseline_run_id=first.run_id,
    )

    cats = shift_by_name(result, "tool_categories")
    assert cats.verdict in ("stable", "insufficient"), cats.note
    if cats.js_divergence is not None:
        assert cats.js_divergence == pytest.approx(0.0, abs=1e-9), cats.note
    assert cats.baseline_summary == cats.current_summary
    assert result.tool_histogram_baseline == result.tool_histogram_current

    # One rep gives 10 cases per side, which clears the 5-case floor for the per-case
    # continuous and proportion shifts, and identical samples give p = 1 there. So at
    # least one shift is stable and the overall verdict is "stable", not "insufficient".
    assert result.overall_verdict == "stable", [(s.name, s.verdict, s.note) for s in result.shifts]
    assert not any(s.verdict == "changed" for s in result.shifts), [
        (s.name, s.note) for s in result.shifts if s.verdict == "changed"
    ]
    assert result.run_id == second.run_id and result.baseline_run_id == first.run_id


# 2. fake vs fake_regressed ------------------------------------------------------------------


def test_regressed_profile_changes_tool_mix_and_output_length(
    settings, example_suite, fake_profile, regressed_profile, tmp_path
):
    base = fake_cases(example_suite, fake_profile, reps=3, root=tmp_path)
    cur = fake_cases(example_suite, regressed_profile, reps=3, root=tmp_path)

    result = detect_drift(cur, base, "fake", settings)

    cats = shift_by_name(result, "tool_categories")
    names = shift_by_name(result, "tool_names")
    assert "changed" in (cats.verdict, names.verdict), (cats.note, names.note)
    # extra_tool=Bash adds one exec call per case; the histogram must show it.
    assert result.tool_histogram_current["exec"] > result.tool_histogram_baseline["exec"]
    assert result.raw_tool_histogram_current["Bash"] > result.raw_tool_histogram_baseline.get(
        "Bash", 0
    )
    assert cats.js_divergence is not None and 0.0 <= cats.js_divergence <= 1.0
    assert cats.p_value is not None and cats.p_value < settings.drift_alpha

    out = shift_by_name(result, "output_chars")
    assert out.verdict == "changed", out.note
    assert out.current_summary["median"] > out.baseline_summary["median"]

    assert result.overall_verdict == "changed"
    # Drift is unsigned: no directional judgment anywhere in the output.
    blob = result.model_dump_json().lower()
    assert "regression" not in blob and "improvement" not in blob


# 3. tool normalization ----------------------------------------------------------------------


def test_normalize_tool_maps_known_prefix_and_unknown_names():
    tool_map = load_tool_map(DEFAULT_TOOL_MAP)
    assert normalize_tool("claude_code", "Bash", tool_map) == "exec"
    assert normalize_tool("claude_code", "Read", tool_map) == "read"
    assert normalize_tool("codex", "shell", tool_map) == "exec"
    assert normalize_tool("codex", "mcp:foo.bar", tool_map) == "other"
    assert normalize_tool("claude_code", "mcp__server__tool", tool_map) == "other"
    assert normalize_tool("claude_code", "NoSuchTool", tool_map) == "other"
    assert normalize_tool("no_such_agent", "Bash", tool_map) == "other"


def test_load_tool_map_shape():
    tool_map = load_tool_map(DEFAULT_TOOL_MAP)
    assert set(tool_map) >= {"claude_code", "claude_sdk", "codex", "fake"}
    allowed = {"read", "write", "exec", "search", "web", "other"}
    for agent, mapping in tool_map.items():
        assert set(mapping.values()) <= allowed, agent


# 4. empty baseline --------------------------------------------------------------------------


def test_empty_baseline_is_insufficient_everywhere(settings, example_suite, fake_profile, tmp_path):
    cur = fake_cases(example_suite, fake_profile, reps=1, root=tmp_path)

    result = detect_drift(cur, [], "fake", settings)

    assert result.overall_verdict == "insufficient"
    assert {s.verdict for s in result.shifts} == {"insufficient"}
    assert result.tool_histogram_baseline == {}
    assert result.raw_tool_histogram_baseline == {}
    assert sum(result.tool_histogram_current.values()) > 0
    for s in result.shifts:
        assert s.js_divergence is None and s.statistic is None and s.p_value is None, s.name
    # Serializes without NaN or None-in-float trouble and reads back.
    again = DriftResult.model_validate_json(result.model_dump_json())
    assert again == result

    both_empty = detect_drift([], [], "fake", settings)
    assert both_empty.overall_verdict == "insufficient"


# Unit-level checks on the individual shift computations -----------------------------------


def test_bigrams_and_histograms_from_handcrafted_records(settings):
    base = [record(["Read", "Edit", "Bash"]) for _ in range(10)]
    cur = [record(["Read", "Edit", "Bash", "Bash"]) for _ in range(10)]

    result = detect_drift(cur, base, "claude_code", settings)

    assert result.tool_histogram_baseline == {"exec": 10, "read": 10, "write": 10}
    assert result.tool_histogram_current == {"exec": 20, "read": 10, "write": 10}
    assert result.raw_tool_histogram_current == {"Bash": 20, "Edit": 10, "Read": 10}
    bigrams = shift_by_name(result, "tool_sequence_bigrams")
    assert set(bigrams.baseline_summary) == {"read->write", "write->exec"}
    assert set(bigrams.current_summary) == {"read->write", "write->exec", "exec->exec"}
    assert bigrams.baseline_summary["read->write"] == pytest.approx(0.5)
    assert bigrams.current_summary["exec->exec"] == pytest.approx(1 / 3)
    assert bigrams.js_divergence is not None and bigrams.js_divergence > 0


def test_categorical_shift_thresholds():
    from collections import Counter

    s = Settings(drift_js_threshold=0.10, drift_alpha=0.05)
    same = categorical_shift("x", Counter(a=10, b=10), Counter(a=10, b=10), s)
    assert same.verdict == "stable" and same.js_divergence == pytest.approx(0.0)
    assert same.p_value is not None and same.p_value == pytest.approx(1.0)

    flipped = categorical_shift("x", Counter(a=40, b=5), Counter(a=5, b=40), s)
    assert flipped.verdict == "changed", flipped.note
    assert flipped.js_divergence is not None and flipped.js_divergence > 0.10

    few = categorical_shift("x", Counter(a=4), Counter(a=1, b=30), s)
    assert few.verdict == "insufficient", few.note

    single = categorical_shift("x", Counter(a=20), Counter(a=20), s)
    assert single.p_value is None and single.statistic is None
    assert single.verdict == "stable"

    # Verdict is threshold-driven: a stricter JS threshold flips a mild shift to changed.
    mild = categorical_shift("x", Counter(a=60, b=40), Counter(a=40, b=60), s)
    assert mild.verdict == "stable", mild.note
    strict = categorical_shift(
        "x", Counter(a=60, b=40), Counter(a=40, b=60), Settings(drift_js_threshold=0.01)
    )
    assert strict.verdict == "changed", strict.note


def test_continuous_shift_median_floor_and_zero_baseline():
    s = Settings()
    zeros = [0.0] * 20
    ones = [1.0] * 20
    up = continuous_shift("x", zeros, ones, s)
    assert up.verdict == "changed", up.note  # divide-by-zero path: relative to current
    assert up.baseline_summary["median"] == 0.0 and up.current_summary["median"] == 1.0

    tiny = continuous_shift("x", [100.0 + i for i in range(20)], [101.0 + i for i in range(20)], s)
    assert tiny.verdict == "stable", tiny.note  # median moves 1 percent, below the floor

    short = continuous_shift("x", [1.0, 2.0, 3.0], ones, s)
    assert short.verdict == "insufficient"
    assert short.statistic is not None  # KS still reported when both sides are non-empty

    empty = continuous_shift("x", [], ones, s)
    assert empty.verdict == "insufficient" and empty.statistic is None
    assert empty.baseline_summary == {"n": 0.0}


def test_proportion_shift_z_test():
    s = Settings()
    none_refused = [False] * 20
    changed = proportion_shift("refusal_rate", none_refused, [True] * 12 + [False] * 8, s)
    assert changed.verdict == "changed", changed.note
    assert changed.current_summary["rate"] == pytest.approx(0.6)
    assert changed.p_value is not None and changed.p_value < s.drift_alpha

    degenerate = proportion_shift("refusal_rate", none_refused, none_refused, s)
    assert degenerate.verdict == "stable" and degenerate.p_value == 1.0

    small = proportion_shift("refusal_rate", [False] * 3, [True] * 3, s)
    assert small.verdict == "insufficient"


def test_refusal_and_error_rates_from_records(settings):
    base = [record(["Read"], task_id=f"t{i}") for i in range(20)]
    cur = [record(["Read"], task_id=f"t{i}", refused=i < 12) for i in range(20)]
    result = detect_drift(cur, base, "claude_code", settings)
    assert shift_by_name(result, "refusal_rate").verdict == "changed"
    assert shift_by_name(result, "error_rate").verdict == "stable"

    errs = [record([], task_id=f"t{i}", error="boom" if i < 10 else None) for i in range(20)]
    result = detect_drift(errs, base, "claude_code", settings)
    assert shift_by_name(result, "error_rate").verdict == "changed"
    assert result.overall_verdict == "changed"


def test_overall_verdict_rules():
    from agent_eval_bench.drift import DistributionShift

    def mk(v):
        return DistributionShift(
            name="x",
            kind="proportion",
            baseline_summary={},
            current_summary={},
            js_divergence=None,
            statistic=None,
            p_value=None,
            verdict=v,
        )

    assert overall_verdict([]) == "insufficient"
    assert overall_verdict([mk("insufficient"), mk("insufficient")]) == "insufficient"
    assert overall_verdict([mk("insufficient"), mk("stable")]) == "stable"
    assert overall_verdict([mk("stable"), mk("changed"), mk("insufficient")]) == "changed"
