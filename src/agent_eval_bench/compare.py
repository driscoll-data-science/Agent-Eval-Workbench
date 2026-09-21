"""Signed comparison of a run against its baseline: paired bootstrap CIs per metric.

Method
- Cases are paired by (task_id, rep). When the two runs disagree on which reps exist (or a
  run holds duplicate keys), pairing falls back to task_id with each task's mean over its
  reps (``pairing="task_mean"``). Only tasks present in both runs are compared.
- For every metric the statistic is the mean paired difference (current - baseline). Its
  two-sided confidence interval comes from a percentile bootstrap over the paired
  differences (``settings.bootstrap_n`` resamples, ``numpy.random.default_rng(0)``, so
  results are reproducible).
- Verdicts read the interval against ``settings.practical_floor``. For a higher-is-better
  metric: regression when the whole interval lies below -floor, improvement when it lies
  above +floor; mirrored for lower-is-better. Fewer than three pairs is "insufficient".
- ``overall_verdict`` is driven only by quality metrics. Cost, latency, and token verdicts
  are reported alongside but never trigger a regression flag on their own (see SPEC,
  "Baseline and comparison" and "Drift").

This module is pure: no MLflow, no filesystem. The runner and CLI persist the result.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from statistics import fmean
from typing import Literal

import numpy as np
from pydantic import BaseModel

from .config import Settings
from .models import CaseResult, Suite

MetricKind = Literal["quality", "cost", "latency", "tokens"]
VerdictLabel = Literal["regression", "improvement", "no_change", "insufficient"]
Pairing = Literal["task_rep", "task_mean"]

MIN_PAIRS = 3
BOOTSTRAP_SEED = 0

# A per-case extractor returns None when the metric is undefined for that case
# (for example a criterion that was never judged); such pairs are skipped.
CaseValue = Callable[[CaseResult], float | None]

_SAFE_METRIC_CHARS = re.compile(r"[^A-Za-z0-9_.\-/ ]")


class MetricVerdict(BaseModel):
    metric: str
    kind: MetricKind
    higher_is_better: bool
    n_pairs: int
    baseline_mean: float
    current_mean: float
    delta: float
    ci_low: float
    ci_high: float
    verdict: VerdictLabel


class ComparisonResult(BaseModel):
    run_id: str | None
    baseline_run_id: str | None
    ci_level: float
    bootstrap_n: int
    practical_floor: float
    pairing: Pairing
    n_pairs_total: int
    metrics: list[MetricVerdict]
    overall_verdict: VerdictLabel

    def metric_deltas(self) -> dict[str, float]:
        """{metric: delta} with names restricted to characters MLflow accepts."""
        return {_SAFE_METRIC_CHARS.sub("_", m.metric): m.delta for m in self.metrics}

    def quality_metrics(self) -> list[MetricVerdict]:
        return [m for m in self.metrics if m.kind == "quality"]


# Pairing ---------------------------------------------------------------------------------


def _pairing_mode(current: list[CaseResult], baseline: list[CaseResult]) -> Pairing:
    cur_keys = [(r.task_id, r.rep) for r in current]
    base_keys = [(r.task_id, r.rep) for r in baseline]
    if len(set(cur_keys)) != len(cur_keys) or len(set(base_keys)) != len(base_keys):
        return "task_mean"
    if {rep for _, rep in cur_keys} != {rep for _, rep in base_keys}:
        return "task_mean"
    return "task_rep"


def _paired_units(
    current: list[CaseResult], baseline: list[CaseResult], pairing: Pairing
) -> list[tuple[list[CaseResult], list[CaseResult]]]:
    """Units to compare: (current cases, baseline cases) per pair key, in stable key order.

    Under task_rep each side holds exactly one case. Under task_mean each side holds every
    rep of that task, and the metric value is the mean over them.
    """
    if pairing == "task_rep":
        cur = {(r.task_id, r.rep): r for r in current}
        base = {(r.task_id, r.rep): r for r in baseline}
        keys = sorted(set(cur) & set(base))
        return [([cur[k]], [base[k]]) for k in keys]
    cur_by_task: dict[str, list[CaseResult]] = {}
    base_by_task: dict[str, list[CaseResult]] = {}
    for r in current:
        cur_by_task.setdefault(r.task_id, []).append(r)
    for r in baseline:
        base_by_task.setdefault(r.task_id, []).append(r)
    tasks = sorted(set(cur_by_task) & set(base_by_task))
    return [(cur_by_task[t], base_by_task[t]) for t in tasks]


def _unit_value(cases: list[CaseResult], value: CaseValue) -> float | None:
    vals = [v for v in (value(c) for c in cases) if v is not None]
    return fmean(vals) if vals else None


def _paired_values(
    units: list[tuple[list[CaseResult], list[CaseResult]]], value: CaseValue
) -> tuple[np.ndarray, np.ndarray]:
    cur_vals: list[float] = []
    base_vals: list[float] = []
    for cur_cases, base_cases in units:
        c = _unit_value(cur_cases, value)
        b = _unit_value(base_cases, value)
        if c is None or b is None:
            continue
        cur_vals.append(c)
        base_vals.append(b)
    return np.asarray(cur_vals, dtype=float), np.asarray(base_vals, dtype=float)


# Per-case metric extractors --------------------------------------------------------------


def _criterion_value(name: str) -> CaseValue:
    def value(case: CaseResult) -> float | None:
        p = case.passed(name)
        return None if p is None else float(p)

    return value


def _pass_rate_value(criteria: list[str]) -> CaseValue:
    def value(case: CaseResult) -> float | None:
        vals = [case.passed(c) for c in criteria]
        known = [float(v) for v in vals if v is not None]
        return fmean(known) if known else None

    return value


def _cost_value(case: CaseResult) -> float | None:
    # Unknown cost (model missing from the price table) must not masquerade as $0.
    if case.agent_cost_usd is None or not case.agent_cost_known:
        return None
    return float(case.agent_cost_usd)


def _wall_ms_value(case: CaseResult) -> float | None:
    return float(case.record.wall_ms)


def _total_tokens_value(case: CaseResult) -> float | None:
    return float(case.record.total_tokens)


# Statistics ------------------------------------------------------------------------------


def bootstrap_mean_ci(
    diffs: np.ndarray, n_boot: int, ci_level: float, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float]:
    """Two-sided percentile bootstrap CI for the mean of ``diffs``.

    Deterministic for a given seed. An empty or all-zero sample yields [0, 0].
    """
    n = int(diffs.size)
    if n == 0 or not np.any(diffs):
        return 0.0, 0.0
    if n_boot < 1:
        m = float(diffs.mean())
        return m, m
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diffs[idx].mean(axis=1)
    alpha = 1.0 - ci_level
    lo, hi = np.percentile(means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(lo), float(hi)


def classify(
    ci_low: float, ci_high: float, n_pairs: int, higher_is_better: bool, floor: float
) -> VerdictLabel:
    if n_pairs < MIN_PAIRS:
        return "insufficient"
    floor = abs(floor)
    below = ci_high < -floor  # the whole interval is below -floor
    above = ci_low > floor  # the whole interval is above +floor
    if higher_is_better:
        if below:
            return "regression"
        if above:
            return "improvement"
        return "no_change"
    if above:
        return "regression"
    if below:
        return "improvement"
    return "no_change"


def _metric_verdict(
    metric: str,
    kind: MetricKind,
    higher_is_better: bool,
    units: list[tuple[list[CaseResult], list[CaseResult]]],
    value: CaseValue,
    settings: Settings,
) -> MetricVerdict:
    cur, base = _paired_values(units, value)
    n = int(cur.size)
    diffs = cur - base
    baseline_mean = float(base.mean()) if n else 0.0
    current_mean = float(cur.mean()) if n else 0.0
    delta = float(diffs.mean()) if n else 0.0
    ci_low, ci_high = bootstrap_mean_ci(diffs, settings.bootstrap_n, settings.ci_level)
    return MetricVerdict(
        metric=metric,
        kind=kind,
        higher_is_better=higher_is_better,
        n_pairs=n,
        baseline_mean=baseline_mean,
        current_mean=current_mean,
        delta=delta,
        ci_low=ci_low,
        ci_high=ci_high,
        verdict=classify(ci_low, ci_high, n, higher_is_better, settings.practical_floor),
    )


def _overall(quality: list[MetricVerdict]) -> VerdictLabel:
    verdicts = {m.verdict for m in quality}
    if "regression" in verdicts:
        return "regression"
    if "improvement" in verdicts:
        return "improvement"
    if quality and verdicts == {"insufficient"}:
        return "insufficient"
    if not quality:
        return "insufficient"
    return "no_change"


# Entry point -----------------------------------------------------------------------------


def compare_runs(
    current: list[CaseResult],
    baseline: list[CaseResult],
    suite: Suite,
    settings: Settings,
    run_id: str | None = None,
    baseline_run_id: str | None = None,
) -> ComparisonResult:
    """Compare ``current`` against ``baseline`` case by case. Pure and deterministic."""
    pairing = _pairing_mode(current, baseline)
    units = _paired_units(current, baseline, pairing)
    criteria = [c.name for c in suite.criteria]

    metrics: list[MetricVerdict] = [
        _metric_verdict("pass_rate", "quality", True, units, _pass_rate_value(criteria), settings)
    ]
    for name in criteria:
        metrics.append(
            _metric_verdict(
                f"pass_rate.{name}", "quality", True, units, _criterion_value(name), settings
            )
        )
    metrics.append(_metric_verdict("agent_cost_usd", "cost", False, units, _cost_value, settings))
    metrics.append(_metric_verdict("wall_ms", "latency", False, units, _wall_ms_value, settings))
    metrics.append(
        _metric_verdict("total_tokens", "tokens", False, units, _total_tokens_value, settings)
    )

    return ComparisonResult(
        run_id=run_id,
        baseline_run_id=baseline_run_id,
        ci_level=settings.ci_level,
        bootstrap_n=settings.bootstrap_n,
        practical_floor=settings.practical_floor,
        pairing=pairing,
        n_pairs_total=len(units),
        metrics=metrics,
        overall_verdict=_overall([m for m in metrics if m.kind == "quality"]),
    )
