"""Unsigned behavioral drift between a run and its baseline.

Drift answers "does the agent behave differently?" and never "is it better or worse?".
Every shift is a distribution comparison whose verdict is ``changed``, ``stable`` or
``insufficient``. Direction words (regression, improvement) belong to ``compare``, which
grades quality; drift only describes behavior.

Shifts computed, in order
- categorical: ``tool_categories`` (raw tool names normalized through tool_map.yaml),
  ``tool_names`` (raw), ``tool_sequence_bigrams`` (consecutive normalized categories inside
  one case, e.g. ``read->write``). Jensen-Shannon divergence in base 2 (range 0..1) on the
  two share vectors, plus a chi-square test of independence on the 2xK count table.
- continuous, one value per case: ``tool_calls_per_case``, ``total_tokens``, ``wall_ms``,
  ``output_chars``. Two-sample Kolmogorov-Smirnov test plus a practical floor of ten
  percent on the relative change in median.
- proportion, one flag per case: ``refusal_rate``, ``error_rate``. Two-proportion z-test.

Everything here is a pure, deterministic function of the two result lists and Settings.
No MLflow calls; the runner persists the result and sets the tag.
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel
from scipy import stats
from scipy.spatial.distance import jensenshannon

from .config import Settings
from .models import CaseResult

ShiftVerdict = Literal["changed", "stable", "insufficient"]
ShiftKind = Literal["categorical", "continuous", "proportion"]

# Minimum evidence before a verdict other than "insufficient" is allowed.
MIN_CATEGORICAL_EVENTS = 10  # tool events per side
MIN_CASES = 5  # cases per side for continuous and proportion shifts
# Practical floor: a continuous shift only counts as changed when the median moved this much.
MEDIAN_REL_CHANGE = 0.10

OTHER = "other"
# MCP tools arrive with a server prefix ("mcp:server.tool" from Codex, "mcp__server__tool"
# from Claude Code). They map through the agent's "mcp_tool_call" entry when it has one.
MCP_PREFIXES = ("mcp:", "mcp__")
MCP_MAP_KEY = "mcp_tool_call"


class DistributionShift(BaseModel):
    name: str
    kind: ShiftKind
    # categorical: normalized shares per category; continuous: n, mean, median, p95;
    # proportion: n, rate. An empty side carries only {"n": 0} (continuous/proportion) or {}.
    baseline_summary: dict[str, float]
    current_summary: dict[str, float]
    # categorical only: Jensen-Shannon divergence, base 2, 0..1 (square of scipy's distance).
    js_divergence: float | None
    # chi-square statistic, KS statistic, or two-proportion z statistic.
    statistic: float | None
    p_value: float | None
    verdict: ShiftVerdict
    note: str = ""


class DriftResult(BaseModel):
    run_id: str | None
    baseline_run_id: str | None
    adapter: str
    js_threshold: float
    alpha: float
    tool_histogram_baseline: dict[str, int]  # normalized category -> count
    tool_histogram_current: dict[str, int]
    raw_tool_histogram_baseline: dict[str, int]  # raw tool name -> count
    raw_tool_histogram_current: dict[str, int]
    shifts: list[DistributionShift]
    overall_verdict: ShiftVerdict


# Tool normalization ------------------------------------------------------------------------


def load_tool_map(path: Path) -> dict[str, dict[str, str]]:
    """Read tool_map.yaml into ``{agent: {raw_tool: category}}``.

    Only the ``agents`` section is used; the ``categories`` list is documentation.
    """
    data = yaml.safe_load(Path(path).read_text()) or {}
    agents = data.get("agents") or {}
    out: dict[str, dict[str, str]] = {}
    for agent, mapping in agents.items():
        out[str(agent)] = {str(raw): str(cat) for raw, cat in (mapping or {}).items()}
    return out


def normalize_tool(adapter_name: str, raw: str, tool_map: dict[str, dict[str, str]]) -> str:
    """Map a raw tool name to a shared category. Unknown names fall back to ``other``."""
    mapping = tool_map.get(adapter_name, {})
    if raw in mapping:
        return mapping[raw]
    if raw.startswith(MCP_PREFIXES):
        return mapping.get(MCP_MAP_KEY, OTHER)
    return OTHER


def _tool_counters(
    results: list[CaseResult], adapter_name: str, tool_map: dict[str, dict[str, str]]
) -> tuple[Counter[str], Counter[str], Counter[str]]:
    """Raw-name, normalized-category, and within-case category-bigram counts."""
    raw: Counter[str] = Counter()
    cats: Counter[str] = Counter()
    bigrams: Counter[str] = Counter()
    for r in results:
        seq: list[str] = []
        for call in r.record.tool_calls:
            raw[call.name] += 1
            cat = normalize_tool(adapter_name, call.name, tool_map)
            cats[cat] += 1
            seq.append(cat)
        for a, b in pairwise(seq):
            bigrams[f"{a}->{b}"] += 1
    return raw, cats, bigrams


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return {k: int(v) for k, v in sorted(counter.items())}


# Shift computations -----------------------------------------------------------------------


def _shares(counts: Counter[str], total: int) -> dict[str, float]:
    if total <= 0:
        return {}
    return {k: float(v / total) for k, v in sorted(counts.items())}


def categorical_shift(
    name: str, baseline: Counter[str], current: Counter[str], settings: Settings
) -> DistributionShift:
    n_b, n_c = sum(baseline.values()), sum(current.values())
    keys = sorted(set(baseline) | set(current))
    b = np.array([baseline.get(k, 0) for k in keys], dtype=float)
    c = np.array([current.get(k, 0) for k in keys], dtype=float)

    js: float | None = None
    statistic: float | None = None
    p_value: float | None = None
    if n_b > 0 and n_c > 0 and keys:
        dist = float(jensenshannon(b / n_b, c / n_c, base=2))
        js = 0.0 if math.isnan(dist) else min(1.0, max(0.0, dist * dist))
        table = np.vstack([b, c])
        table = table[:, table.sum(axis=0) > 0]  # chi2 rejects all-zero columns
        if table.shape[1] >= 2:
            res = stats.chi2_contingency(table)
            statistic = float(res.statistic)
            p_value = float(res.pvalue)

    thr, alpha = settings.drift_js_threshold, settings.drift_alpha
    if n_b < MIN_CATEGORICAL_EVENTS or n_c < MIN_CATEGORICAL_EVENTS:
        verdict: ShiftVerdict = "insufficient"
        note = (
            f"needs at least {MIN_CATEGORICAL_EVENTS} tool events per side; "
            f"baseline has {n_b}, current has {n_c}"
        )
    elif js is not None and js >= thr and (p_value is None or p_value < alpha):
        verdict = "changed"
        if p_value is None:
            chi = "no chi-square test (fewer than two categories)"
        else:
            chi = f"chi-square p={p_value:.3g} < {alpha}"
        note = f"JS divergence {js:.3f} >= {thr} and {chi}"
    else:
        verdict = "stable"
        if js is not None and js < thr:
            note = f"JS divergence {js:.3f} below threshold {thr}"
        else:
            note = f"JS divergence {js:.3f} >= {thr} but chi-square p={p_value:.3g} >= {alpha}"

    return DistributionShift(
        name=name,
        kind="categorical",
        baseline_summary=_shares(baseline, n_b),
        current_summary=_shares(current, n_c),
        js_divergence=js,
        statistic=statistic,
        p_value=p_value,
        verdict=verdict,
        note=note,
    )


def _continuous_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0.0}
    arr = np.asarray(values, dtype=float)
    return {
        "n": float(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


def _relative_change(baseline_median: float, current_median: float) -> float:
    """Relative change in median. Relative to the baseline, or to the current value when the
    baseline median is zero, so a move away from zero always registers."""
    if baseline_median == current_median:
        return 0.0
    denom = abs(baseline_median) or abs(current_median)
    return abs(current_median - baseline_median) / denom


def continuous_shift(
    name: str, baseline: list[float], current: list[float], settings: Settings
) -> DistributionShift:
    n_b, n_c = len(baseline), len(current)
    base_summary = _continuous_summary(baseline)
    cur_summary = _continuous_summary(current)

    statistic: float | None = None
    p_value: float | None = None
    if n_b > 0 and n_c > 0:
        res = stats.ks_2samp(baseline, current)
        statistic = float(res.statistic)
        p_value = float(res.pvalue)

    alpha = settings.drift_alpha
    if n_b < MIN_CASES or n_c < MIN_CASES:
        verdict: ShiftVerdict = "insufficient"
        note = f"needs at least {MIN_CASES} cases per side; baseline has {n_b}, current has {n_c}"
    else:
        rel = _relative_change(base_summary["median"], cur_summary["median"])
        medians = f"median {base_summary['median']:.3g} -> {cur_summary['median']:.3g}"
        if p_value is not None and p_value < alpha and rel >= MEDIAN_REL_CHANGE:
            verdict = "changed"
            note = f"KS p={p_value:.3g} < {alpha}; {medians} ({rel:.0%} relative)"
        elif p_value is not None and p_value < alpha:
            verdict = "stable"
            note = (
                f"KS p={p_value:.3g} < {alpha} but {medians} moves less than "
                f"{MEDIAN_REL_CHANGE:.0%}"
            )
        else:
            verdict = "stable"
            note = f"KS p={p_value:.3g} >= {alpha}; {medians}"

    return DistributionShift(
        name=name,
        kind="continuous",
        baseline_summary=base_summary,
        current_summary=cur_summary,
        js_divergence=None,
        statistic=statistic,
        p_value=p_value,
        verdict=verdict,
        note=note,
    )


def _proportion_summary(flags: list[bool]) -> dict[str, float]:
    if not flags:
        return {"n": 0.0}
    return {"n": float(len(flags)), "rate": float(sum(flags) / len(flags))}


def proportion_shift(
    name: str, baseline: list[bool], current: list[bool], settings: Settings
) -> DistributionShift:
    n_b, n_c = len(baseline), len(current)
    k_b, k_c = sum(baseline), sum(current)
    base_summary = _proportion_summary(baseline)
    cur_summary = _proportion_summary(current)

    statistic: float | None = None
    p_value: float | None = None
    if n_b > 0 and n_c > 0:
        pooled = (k_b + k_c) / (n_b + n_c)
        se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n_b + 1.0 / n_c))
        if se == 0.0:
            # Both sides all-true or all-false: identical proportions, nothing to test.
            statistic, p_value = 0.0, 1.0
        else:
            z = (k_c / n_c - k_b / n_b) / se
            statistic = float(z)
            p_value = float(2.0 * stats.norm.sf(abs(z)))

    alpha = settings.drift_alpha
    if n_b < MIN_CASES or n_c < MIN_CASES:
        verdict: ShiftVerdict = "insufficient"
        note = f"needs at least {MIN_CASES} cases per side; baseline has {n_b}, current has {n_c}"
    else:
        rates = f"rate {base_summary['rate']:.3g} -> {cur_summary['rate']:.3g}"
        floor = getattr(settings, "drift_min_rate_delta", 0.0)
        moved = abs(cur_summary["rate"] - base_summary["rate"]) >= floor
        if p_value is not None and p_value < alpha and moved:
            verdict = "changed"
            note = f"two-proportion z p={p_value:.3g} < {alpha}; {rates}"
        elif p_value is not None and p_value < alpha and not moved:
            verdict = "stable"
            note = f"p={p_value:.3g} < {alpha} but rate moved less than {floor:.2f}; {rates}"
        else:
            verdict = "stable"
            note = f"two-proportion z p={p_value:.3g} >= {alpha}; {rates}"

    return DistributionShift(
        name=name,
        kind="proportion",
        baseline_summary=base_summary,
        current_summary=cur_summary,
        js_divergence=None,
        statistic=statistic,
        p_value=p_value,
        verdict=verdict,
        note=note,
    )


# Entry point --------------------------------------------------------------------------------


def overall_verdict(shifts: list[DistributionShift]) -> ShiftVerdict:
    verdicts = [s.verdict for s in shifts]
    if not verdicts or all(v == "insufficient" for v in verdicts):
        return "insufficient"
    if any(v == "changed" for v in verdicts):
        return "changed"
    return "stable"


def detect_drift(
    current: list[CaseResult],
    baseline: list[CaseResult],
    adapter_name: str,
    settings: Settings,
    run_id: str | None = None,
    baseline_run_id: str | None = None,
) -> DriftResult:
    """Compare the behavior distributions of ``current`` against ``baseline``.

    Both lists hold one CaseResult per (task, rep). Cases are not paired; each shift compares
    the two samples as a whole. Either list may be empty, which yields ``insufficient``.
    """
    tool_map = load_tool_map(settings.tool_map)
    raw_b, cats_b, bigrams_b = _tool_counters(baseline, adapter_name, tool_map)
    raw_c, cats_c, bigrams_c = _tool_counters(current, adapter_name, tool_map)

    shifts: list[DistributionShift] = [
        categorical_shift("tool_categories", cats_b, cats_c, settings),
        categorical_shift("tool_names", raw_b, raw_c, settings),
        categorical_shift("tool_sequence_bigrams", bigrams_b, bigrams_c, settings),
    ]

    def per_case(results: list[CaseResult], fn) -> list[float]:
        return [float(fn(r.record)) for r in results]

    continuous = (
        ("tool_calls_per_case", lambda rec: len(rec.tool_calls)),
        ("total_tokens", lambda rec: rec.total_tokens),
        ("wall_ms", lambda rec: rec.wall_ms),
        ("output_chars", lambda rec: len(rec.final_text)),
    )
    for name, fn in continuous:
        shifts.append(
            continuous_shift(name, per_case(baseline, fn), per_case(current, fn), settings)
        )

    proportions = (
        ("refusal_rate", lambda rec: bool(rec.refused)),
        ("error_rate", lambda rec: rec.error is not None),
    )
    for name, fn in proportions:
        shifts.append(
            proportion_shift(
                name,
                [bool(fn(r.record)) for r in baseline],
                [bool(fn(r.record)) for r in current],
                settings,
            )
        )

    return DriftResult(
        run_id=run_id,
        baseline_run_id=baseline_run_id,
        adapter=adapter_name,
        js_threshold=settings.drift_js_threshold,
        alpha=settings.drift_alpha,
        tool_histogram_baseline=_sorted_counts(cats_b),
        tool_histogram_current=_sorted_counts(cats_c),
        raw_tool_histogram_baseline=_sorted_counts(raw_b),
        raw_tool_histogram_current=_sorted_counts(raw_c),
        shifts=shifts,
        overall_verdict=overall_verdict(shifts),
    )
