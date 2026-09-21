"""Static per-run reports: report.html, report.json and report.md under
<reports_dir>/<run_id>/, plus <reports_dir>/index.html listing every recorded run.

Inputs are the run's own files (manifest.json, results.jsonl) plus three optional JSON
inputs written by sibling modules: compare.json, drift.json and the suite's calibration
file under <calibration_dir>/<suite>/<judge_version>.json. Every optional input degrades
to an explicit reading ("no baseline set", "judge not yet calibrated for version ..."),
never to a blank panel.

Charts are inline SVG built here in Python; the HTML is a single self-contained file
with no scripts, external stylesheets, fonts or images. Chart colors follow the dataviz
reference palette in its "emphasis" form: one series hue (blue) for the current run plus
a de-emphasis gray for baseline context. The two-slot palette (#2a78d6 + #898781 light,
#3987e5 + #898781 dark) was run through the dataviz palette validator on 2026-09-17:
lightness band, CVD separation (worst adjacent dE 15.9), normal-vision floor (17.8 light,
17.0 dark) and 3:1 contrast against both surfaces all PASS. The only FAIL is the chroma
floor on the gray, which is the intended de-emphasis slot rather than a categorical
series, so it is accepted by design.
"""

from __future__ import annotations

import html
import json
import statistics
import webbrowser
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup

from ..baseline import baseline_key, load_baselines
from ..config import Settings
from ..models import CaseResult, RunManifest
from ..storage import RunStore, list_runs, resolve_run_id

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

OUTPUT_TRUNCATE = 600
RATIONALE_TRUNCATE = 500
NO_BASELINE = "no baseline set"

QUALITY_VERDICTS = ("improvement", "no_change", "regression")
BEHAVIOR_VERDICTS = ("stable", "changed")

# Reading for each (quality, behavior) cell of the verdict grid.
READINGS: dict[tuple[str, str], str] = {
    (
        "improvement",
        "stable",
    ): "quality improved, behavior stable: a clean gain; consider promoting this run",
    ("improvement", "changed"): "behavior changed, quality improved: consider promoting this run",
    ("no_change", "stable"): "no significant change on either axis: nothing to act on",
    ("no_change", "changed"): (
        "behavior changed, quality held: not a regression; decide whether the new behavior is acceptable"
    ),
    ("regression", "stable"): (
        "quality regressed, behavior stable: inspect the failed cases and judge rationales, not tool usage"
    ),
    (
        "regression",
        "changed",
    ): "behavior changed, quality regressed: the behavior shift is the first place to look",
}

HOW_TO_READ = (
    "Quality verdicts are signed: regression means the paired bootstrap confidence interval on the "
    "delta sits entirely below zero, improvement entirely above, no change otherwise. Drift is "
    "unsigned: it reports that behavioral distributions (tool mix, call counts, tokens, latency, "
    "output length) differ from the baseline, not whether that is good or bad. A drift verdict of "
    "changed is never a regression flag by itself; read it together with the quality verdict in the "
    "grid. Pass rates count cases with a definite verdict; n is that count. Costs are list price "
    "from the versioned price table; judge cost is recorded separately and never added to agent cost."
)

# Status colors are reserved for state and always ship with a glyph and a label, never alone.
VERDICT_STYLE: dict[str, tuple[str, str, str]] = {
    "regression": ("critical", "▼", "regression"),
    "improvement": ("good", "▲", "improvement"),
    "no_change": ("neutral", "●", "no change"),
    "insufficient": ("warning", "○", "insufficient"),
    "changed": ("neutral", "◆", "changed"),
    "stable": ("neutral", "●", "stable"),
}

_MODEL_TIERS = ("opus", "sonnet", "haiku")


# Public API -------------------------------------------------------------------------------
def build_report(run_id: str, settings: Settings) -> Path:
    """Write report.html, report.json and report.md for ``run_id``, rebuild the index and
    return the HTML path."""
    data = report_data(run_id, settings)
    out_dir = settings.reports_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(data, indent=2, default=str) + "\n")
    (out_dir / "report.md").write_text(render_markdown(data))
    html_path = out_dir / "report.html"
    html_path.write_text(render_html(data))
    build_index(settings)
    return html_path


def build_index(settings: Settings) -> Path:
    """<reports_dir>/index.html: every recorded run, newest first."""
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    baselines = load_baselines(settings)
    rows: list[dict[str, Any]] = []
    for m in list_runs(settings):
        store = RunStore(settings, m.run_id)
        metrics = m.notes.get("metrics") or {}
        is_baseline = baselines.get(baseline_key(m.suite, m.profile.name)) == m.run_id
        compare = _safe_json(store, "compare.json")
        drift = _safe_json(store, "drift.json")
        rows.append(
            {
                "run_id": m.run_id,
                "short_id": m.run_id[:8],
                "suite": m.suite,
                "suite_version": m.suite_version,
                "profile": m.profile.name,
                "adapter": m.profile.adapter,
                "model": m.profile.model,
                "started_at": _iso(m.started_at),
                "finished": m.finished_at is not None,
                "n_cases": metrics.get("n_cases"),
                "pass_rate": metrics.get("pass_rate"),
                "quality_verdict": (compare or {}).get("overall_verdict")
                or _missing_quality(m.baseline_run_id, m.run_id),
                "behavior_verdict": (drift or {}).get("overall_verdict")
                or _missing_behavior(m.baseline_run_id, m.run_id),
                "is_baseline": is_baseline,
                "agent_cost_usd": metrics.get("agent_cost_usd_total"),
                "judge_cost_usd": metrics.get("judge_cost_usd_total"),
                "report_href": f"{m.run_id}/report.html",
                "report_built": (settings.reports_dir / m.run_id / "report.html").is_file(),
            }
        )
    page = (
        _env()
        .get_template("index.html.j2")
        .render(
            rows=rows,
            n_runs=len(rows),
            n_baselines=len(baselines),
            reports_dir=str(settings.reports_dir),
            generated_at=_now_iso(),
        )
    )
    path = settings.reports_dir / "index.html"
    path.write_text(page)
    return path


def open_report(settings: Settings, run_ref: str | None) -> Path:
    """Open a run's report (or the index when ``run_ref`` is None) in the default browser,
    building it first when it is missing."""
    if run_ref is None:
        path = settings.reports_dir / "index.html"
        if not path.is_file():
            path = build_index(settings)
    else:
        run_id = resolve_run_id(settings, run_ref)
        path = settings.reports_dir / run_id / "report.html"
        if not path.is_file():
            path = build_report(run_id, settings)
    webbrowser.open(path.as_uri())
    return path


def report_data(run_id: str, settings: Settings) -> dict[str, Any]:
    """Everything the HTML shows, as plain JSON-serializable data (this is report.json)."""
    store = RunStore(settings, run_id)
    manifest = store.load_manifest()
    results = store.load_results() if store.results_path.is_file() else []
    compare = _safe_json(store, "compare.json")
    drift = _safe_json(store, "drift.json")
    calibration = _load_calibration(settings, manifest.suite, manifest.judge_version)
    baselines = load_baselines(settings)
    is_baseline = baselines.get(baseline_key(manifest.suite, manifest.profile.name)) == run_id
    baseline_run_id = (
        (compare or {}).get("baseline_run_id")
        or (drift or {}).get("baseline_run_id")
        or manifest.baseline_run_id
    )

    verdict = _verdict_section(compare, drift, baseline_run_id, run_id, is_baseline)
    criteria = _criteria_section(results, compare, _criteria_order(manifest, settings, results))
    failures = _failures_section(results, manifest)
    overall = _overall_pass_rate(criteria["rows"])
    return {
        "run_id": run_id,
        "generated_at": _now_iso(),
        "summary": {
            "suite": manifest.suite,
            "profile": manifest.profile.name,
            "pass_rate": overall,
            "n_cases": len(results),
            "n_failed_cases": len(failures),
            "quality_verdict": verdict["quality"],
            "behavior_verdict": verdict["behavior"],
            "reading": verdict["reading"],
        },
        "verdict": verdict,
        "criteria": criteria,
        "cost": _cost_section(results, compare),
        "drift": _drift_section(drift, baseline_run_id),
        "calibration": _calibration_section(manifest, calibration, settings),
        "run": _run_section(manifest, results, settings, is_baseline, baseline_run_id),
        "failures": failures,
        "how_to_read": HOW_TO_READ,
    }


# Rendering --------------------------------------------------------------------------------
def render_html(data: dict[str, Any]) -> str:
    crit = data["criteria"]
    drift = data["drift"]
    charts = {
        "pass_rates": Markup(_svg_pass_rates(crit["rows"], crit["has_baseline"])),
        "drift": Markup(_svg_histograms(drift["categories"])) if drift["has_baseline"] else None,
    }
    return (
        _env()
        .get_template("report.html.j2")
        .render(
            d=data,
            charts=charts,
            quality_levels=QUALITY_VERDICTS,
            behavior_levels=BEHAVIOR_VERDICTS,
            readings={f"{q}|{b}": r for (q, b), r in READINGS.items()},
        )
    )


def render_markdown(d: dict[str, Any]) -> str:
    run, v, cost, drift, cal = d["run"], d["verdict"], d["cost"], d["drift"], d["calibration"]
    lines: list[str] = []
    add = lines.append
    add(f"# aeb report: {run['suite']} / {run['profile']['name']} / {run['run_id'][:8]}")
    add("")
    add(
        f"Run `{run['run_id']}` · suite {run['suite']} v{run['suite_version']} · profile "
        f"{run['profile']['name']} ({run['profile']['adapter']}, {run['profile']['model']}) · "
        f"{run['n_cases']} cases ({run['n_tasks']} tasks x {run['reps']} reps) · started {run['started_at']}"
    )
    if run["is_baseline"]:
        add("")
        add("This run is the current baseline for its suite and profile.")
    add("")
    add("## 1. Verdict grid")
    add("")
    add(f"Quality: **{_label(v['quality'])}** · Behavior: **{_label(v['behavior'])}**")
    add("")
    add(f"Reading: {v['reading']}")
    add("")
    add("| quality \\ behavior | stable | changed |")
    add("|---|---|---|")
    for q in QUALITY_VERDICTS:
        cells = []
        for b in BEHAVIOR_VERDICTS:
            mark = " **(this run)**" if (v["quality"], v["behavior"]) == (q, b) else ""
            cells.append(READINGS[(q, b)] + mark)
        add(f"| {_label(q)} | {cells[0]} | {cells[1]} |")
    if v["baseline_run_id"]:
        add("")
        add(f"Baseline run: `{v['baseline_run_id']}`")
    add("")
    add("## 2. Per-criterion pass rates")
    add("")
    if d["criteria"]["has_baseline"]:
        add("| criterion | kind | pass rate | n | baseline | delta | CI low | CI high | verdict |")
        add("|---|---|---|---|---|---|---|---|---|")
        for r in d["criteria"]["rows"]:
            add(
                f"| {r['name']} | {r['kind']} | {fmt_pct(r['pass_rate'])} | {r['n']} | "
                f"{fmt_pct(r['baseline_mean'])} | {fmt_pp(r['delta'])} | {fmt_pp(r['ci_low'])} | "
                f"{fmt_pp(r['ci_high'])} | {_label(r['verdict']) if r['verdict'] else 'n/a'} |"
            )
    else:
        add("| criterion | kind | pass rate | passed | n |")
        add("|---|---|---|---|---|")
        for r in d["criteria"]["rows"]:
            add(
                f"| {r['name']} | {r['kind']} | {fmt_pct(r['pass_rate'])} | {r['n_pass']} | {r['n']} |"
            )
        add("")
        add(f"Baseline: {NO_BASELINE}.")
    add("")
    add("## 3. Cost and latency")
    add("")
    a = cost["absolute"]
    add("| metric | value |")
    add("|---|---|")
    add(f"| agent cost total | {fmt_usd(a['agent_cost_usd_total'])} |")
    add(f"| agent cost per case | {fmt_usd(a['agent_cost_usd_per_case'])} |")
    add(f"| judge cost total | {fmt_usd(a['judge_cost_usd_total'])} |")
    add(f"| judge cost per case | {fmt_usd(a['judge_cost_usd_per_case'])} |")
    add(
        f"| cases with unknown agent cost | {a['agent_cost_unknown_cases']} of {a['n_cases']} "
        f"({fmt_pct(a['agent_cost_unknown_share'])}) |"
    )
    add(
        f"| wall time mean / p50 / p95 | {fmt_ms(a['wall_ms_mean'])} / {fmt_ms(a['wall_ms_p50'])} / {fmt_ms(a['wall_ms_p95'])} |"
    )
    add(f"| input / output tokens | {fmt_num(a['input_tokens'])} / {fmt_num(a['output_tokens'])} |")
    add(
        f"| cache read / write tokens | {fmt_num(a['cache_read_tokens'])} / {fmt_num(a['cache_write_tokens'])} |"
    )
    add(f"| total tokens | {fmt_num(a['total_tokens'])} |")
    add(f"| tool calls per case | {fmt_num(a['tool_calls_per_case_mean'])} |")
    if cost["relative"]:
        add("")
        add("Relative to baseline:")
        add("")
        add("| metric | baseline | current | delta | CI low | CI high | verdict |")
        add("|---|---|---|---|---|---|---|")
        for m in cost["relative"]:
            add(
                f"| {m['metric']} | {fmt_metric(m['baseline_mean'], m['metric'])} | "
                f"{fmt_metric(m['current_mean'], m['metric'])} | {fmt_metric(m['delta'], m['metric'], True)} | "
                f"{fmt_metric(m['ci_low'], m['metric'], True)} | {fmt_metric(m['ci_high'], m['metric'], True)} | "
                f"{_label(m['verdict']) if m['verdict'] else 'n/a'} |"
            )
    add("")
    add("## 4. Drift")
    add("")
    if drift["has_baseline"]:
        add(
            f"Behavior verdict: **{_label(drift['overall_verdict'])}** (unsigned; never a regression flag by itself)"
        )
        add("")
        add("| tool category | baseline count | baseline share | current count | current share |")
        add("|---|---|---|---|---|")
        for c in drift["categories"]:
            add(
                f"| {c['category']} | {c['baseline_count']} | {fmt_pct(c['baseline_share'])} | "
                f"{c['current_count']} | {fmt_pct(c['current_share'])} |"
            )
        add("")
        add("| distribution | kind | JS divergence | statistic | p-value | verdict | note |")
        add("|---|---|---|---|---|---|---|")
        for s in drift["shifts"]:
            add(
                f"| {s['name']} | {s['kind']} | {fmt_float(s['js_divergence'])} | {fmt_float(s['statistic'])} | "
                f"{fmt_float(s['p_value'])} | {_label(s['verdict']) if s['verdict'] else 'n/a'} | {s['note'] or ''} |"
            )
    else:
        add(NO_BASELINE + ".")
    add("")
    add("## 5. Calibration status")
    add("")
    add(f"Judge: {cal['judge_backend']} / {cal['judge_model']} / version `{cal['judge_version']}`")
    add("")
    add(cal["status"] + ".")
    if cal["same_family_warning"]:
        add("")
        add(f"WARNING: {cal['same_family_message']}")
    if cal["per_criterion"]:
        add("")
        add("| criterion | n | raw agreement | kappa | flag |")
        add("|---|---|---|---|---|")
        for c in cal["per_criterion"]:
            add(
                f"| {c['name']} | {c['n']} | {fmt_pct(c['raw_agreement'])} | {fmt_float(c['kappa'])} | "
                f"{'below threshold' if c['below_threshold'] else ''} |"
            )
    if cal["flags"]:
        add("")
        add("Flags:")
        add("")
        lines.extend(f"- {f}" for f in cal["flags"])
    add("")
    add("## 6. Profile params and run metadata")
    add("")
    add("| field | value |")
    add("|---|---|")
    for k in (
        "run_id",
        "experiment",
        "experiment_id",
        "suite",
        "suite_version",
        "reps",
        "n_tasks",
        "n_cases",
        "started_at",
        "finished_at",
        "kit_version",
        "tracking_uri",
        "mlflow_run_url",
        "baseline_run_id",
    ):
        add(f"| {k} | {run[k] if run[k] is not None else 'n/a'} |")
    for k, val in run["profile"].items():
        if k != "params":
            add(f"| profile.{k} | {val} |")
    for k, val in sorted(run["profile"]["params"].items()):
        add(f"| param.{k} | {val} |")
    add("")
    add(f"## 7. Failures ({len(d['failures'])})")
    add("")
    if not d["failures"]:
        add("No case failed a criterion or errored.")
    for f in d["failures"]:
        head = f"- **{f['task_id']}** rep {f['rep']}: failed {', '.join(f['failed_criteria']) or 'none'}"
        if f["error"]:
            head += f" · error: {f['error']}"
        if f["trace_url"]:
            head += f" · [trace]({f['trace_url']})"
        elif f["trace_id"]:
            head += f" · trace {f['trace_id']}"
        add(head)
        lines.extend(
            f"  - {r['criterion']}: {r['rationale'] or r['error']}" for r in f["rationales"]
        )
        if f["final_output"]:
            add("  - output: " + f["final_output"].replace("\n", " ")[:300])
    add("")
    add("## How to read this")
    add("")
    add(HOW_TO_READ)
    add("")
    return "\n".join(lines)


# Sections ---------------------------------------------------------------------------------
def _verdict_section(compare, drift, baseline_run_id, run_id, is_baseline) -> dict[str, Any]:
    quality = (compare or {}).get("overall_verdict") or _missing_quality(baseline_run_id, run_id)
    behavior = (drift or {}).get("overall_verdict") or _missing_behavior(baseline_run_id, run_id)
    reading = READINGS.get((quality, behavior))
    if reading is None:
        if quality == NO_BASELINE or behavior == NO_BASELINE:
            reading = (
                "no baseline set: promote a run with `aeb baseline promote <run>` (or run with "
                "--promote) to enable signed comparison and drift detection"
            )
        elif is_baseline:
            reading = "this run is the baseline its suite and profile are compared against"
        elif "insufficient" in (quality, behavior):
            axes = [
                a
                for a, val in (("quality", quality), ("behavior", behavior))
                if val == "insufficient"
            ]
            reading = (
                f"too few paired cases to call {' and '.join(axes)}; add reps or tasks and rerun"
            )
        else:
            reading = f"quality {_label(quality)}, behavior {_label(behavior)}"
    active = (quality, behavior) if (quality, behavior) in READINGS else None
    return {
        "quality": quality,
        "behavior": behavior,
        "reading": reading,
        "active_cell": list(active) if active else None,
        "baseline_run_id": baseline_run_id,
        "is_baseline": is_baseline,
        "ci_level": (compare or {}).get("ci_level"),
        "n_pairs_total": (compare or {}).get("n_pairs_total"),
        "pairing": (compare or {}).get("pairing"),
        "js_threshold": (drift or {}).get("js_threshold"),
        "alpha": (drift or {}).get("alpha"),
    }


def _criteria_section(
    results: list[CaseResult], compare, order: list[tuple[str, str]]
) -> dict[str, Any]:
    by_criterion = _compare_metrics_by_criterion(compare)
    rows: list[dict[str, Any]] = []
    for name, kind in order:
        vals = [v for v in (r.passed(name) for r in results) if v is not None]
        n, n_pass = len(vals), sum(1 for v in vals if v)
        row: dict[str, Any] = {
            "name": name,
            "kind": kind,
            "n": n,
            "n_pass": n_pass,
            "pass_rate": n_pass / n if n else None,
            "baseline_mean": None,
            "delta": None,
            "ci_low": None,
            "ci_high": None,
            "verdict": None,
            "n_pairs": None,
        }
        m = by_criterion.get(name)
        if m:
            row.update(
                baseline_mean=m.get("baseline_mean"),
                delta=m.get("delta"),
                ci_low=m.get("ci_low"),
                ci_high=m.get("ci_high"),
                verdict=m.get("verdict"),
                n_pairs=m.get("n_pairs"),
            )
        rows.append(row)
    return {
        "rows": rows,
        "has_baseline": compare is not None,
        "ci_level": (compare or {}).get("ci_level"),
        "practical_floor": (compare or {}).get("practical_floor"),
    }


def _cost_section(results: list[CaseResult], compare) -> dict[str, Any]:
    n = len(results)
    agent_costs = [r.agent_cost_usd or 0.0 for r in results]
    judge_costs = [r.judge_cost_usd or 0.0 for r in results]
    unknown = [r for r in results if not r.agent_cost_known]
    walls = sorted(r.record.wall_ms for r in results)
    absolute = {
        "n_cases": n,
        "agent_cost_usd_total": sum(agent_costs),
        "agent_cost_usd_per_case": sum(agent_costs) / n if n else None,
        "judge_cost_usd_total": sum(judge_costs),
        "judge_cost_usd_per_case": sum(judge_costs) / n if n else None,
        "agent_cost_unknown_cases": len(unknown),
        "agent_cost_unknown_share": len(unknown) / n if n else 0.0,
        "agent_cost_unknown_models": sorted({r.record.model or "unknown" for r in unknown}),
        "wall_ms_mean": statistics.fmean(walls) if walls else None,
        "wall_ms_p50": statistics.median(walls) if walls else None,
        "wall_ms_p95": walls[int(0.95 * (len(walls) - 1))] if walls else None,
        "input_tokens": sum(r.record.input_tokens for r in results),
        "output_tokens": sum(r.record.output_tokens for r in results),
        "cache_read_tokens": sum(r.record.cache_read_tokens for r in results),
        "cache_write_tokens": sum(r.record.cache_write_tokens for r in results),
        "total_tokens": sum(r.record.total_tokens for r in results),
        "tool_calls_per_case_mean": statistics.fmean(len(r.record.tool_calls) for r in results)
        if n
        else None,
    }
    criterion_metrics = set(_compare_metrics_by_criterion(compare))
    relative: list[dict[str, Any]] = []
    for m in (compare or {}).get("metrics", []):
        name = str(m.get("metric", ""))
        if name.startswith("pass_rate.") and name.split(".", 1)[1] in criterion_metrics:
            continue
        base = m.get("baseline_mean")
        delta = m.get("delta")
        relative.append(
            {
                "metric": name,
                "kind": m.get("kind"),
                "higher_is_better": m.get("higher_is_better"),
                "n_pairs": m.get("n_pairs"),
                "baseline_mean": base,
                "current_mean": m.get("current_mean"),
                "delta": delta,
                "relative_delta": (delta / base) if (delta is not None and base) else None,
                "ci_low": m.get("ci_low"),
                "ci_high": m.get("ci_high"),
                "verdict": m.get("verdict"),
            }
        )
    return {"absolute": absolute, "relative": relative, "has_baseline": compare is not None}


def _drift_section(drift, baseline_run_id: str | None = None) -> dict[str, Any]:
    if not drift:
        return {
            "has_baseline": False,
            "overall_verdict": NO_BASELINE,
            "categories": [],
            "raw_tools": [],
            "shifts": [],
            "totals": {"baseline": 0, "current": 0},
        }
    base = drift.get("tool_histogram_baseline") or {}
    cur = drift.get("tool_histogram_current") or {}
    categories = _histogram_rows(base, cur, "category")
    raw_tools = _histogram_rows(
        drift.get("raw_tool_histogram_baseline") or {},
        drift.get("raw_tool_histogram_current") or {},
        "name",
    )
    shifts = [
        {
            "name": s.get("name"),
            "kind": s.get("kind"),
            "baseline_summary": s.get("baseline_summary") or {},
            "current_summary": s.get("current_summary") or {},
            "js_divergence": s.get("js_divergence"),
            "statistic": s.get("statistic"),
            "p_value": s.get("p_value"),
            "verdict": s.get("verdict"),
            "note": s.get("note"),
        }
        for s in drift.get("shifts", [])
    ]
    return {
        "has_baseline": True,
        "overall_verdict": drift.get("overall_verdict"),
        # The runner may write drift.json with a null baseline_run_id; fall back to the
        # id the manifest or compare.json knows about.
        "baseline_run_id": drift.get("baseline_run_id") or baseline_run_id,
        "adapter": drift.get("adapter"),
        "js_threshold": drift.get("js_threshold"),
        "alpha": drift.get("alpha"),
        "categories": categories,
        "raw_tools": raw_tools,
        "shifts": shifts,
        "totals": {"baseline": sum(base.values()), "current": sum(cur.values())},
    }


def _histogram_rows(base: dict[str, int], cur: dict[str, int], key: str) -> list[dict[str, Any]]:
    bt, ct = sum(base.values()), sum(cur.values())
    names = sorted(
        set(base) | set(cur),
        key=lambda c: (-(cur.get(c, 0) / ct if ct else 0), -(base.get(c, 0) / bt if bt else 0), c),
    )
    return [
        {
            key: c,
            "baseline_count": base.get(c, 0),
            "current_count": cur.get(c, 0),
            "baseline_share": base.get(c, 0) / bt if bt else 0.0,
            "current_share": cur.get(c, 0) / ct if ct else 0.0,
        }
        for c in names
    ]


def _calibration_section(manifest: RunManifest, cal, settings: Settings) -> dict[str, Any]:
    agent_model = manifest.profile.model
    same = same_model_family(manifest.judge_model, agent_model)
    per_criterion: list[dict[str, Any]] = []
    for name, row in sorted(((cal or {}).get("per_criterion") or {}).items()):
        raw, kappa = row.get("raw_agreement"), row.get("kappa")
        below = (raw is not None and raw < settings.agreement_min_raw) or (
            kappa is not None and kappa < settings.agreement_min_kappa
        )
        per_criterion.append(
            {
                "name": name,
                "n": row.get("n"),
                "raw_agreement": raw,
                "kappa": kappa,
                "below_threshold": bool(below),
            }
        )
    if cal is None:
        status = f"judge not yet calibrated for version {manifest.judge_version}"
    else:
        status = (
            f"calibrated on {cal.get('n_labels', 0)} labels from {len(cal.get('labeled_run_ids') or [])} run(s), "
            f"computed {cal.get('computed_at') or 'at an unknown time'}"
        )
    return {
        "judge_backend": manifest.judge_backend,
        "judge_model": manifest.judge_model,
        "judge_version": manifest.judge_version,
        "agent_model": agent_model,
        "same_family_warning": same,
        "same_family_message": (
            f"judge model {manifest.judge_model} and agent model {agent_model} are the same model family; "
            "judge verdicts may favor this agent's own style. Prefer a judge from a different family "
            "or a different tier, and weigh the agreement numbers below accordingly."
        )
        if same
        else None,
        "thresholds": {
            "raw_agreement": settings.agreement_min_raw,
            "kappa": settings.agreement_min_kappa,
        },
        "calibrated": cal is not None,
        "status": status,
        "n_labels": (cal or {}).get("n_labels", 0),
        "labeled_run_ids": (cal or {}).get("labeled_run_ids") or [],
        "computed_at": (cal or {}).get("computed_at"),
        "calibrated_judge_model": (cal or {}).get("judge_model"),
        "per_criterion": per_criterion,
        "flags": list((cal or {}).get("flags") or []),
    }


def _run_section(
    manifest: RunManifest,
    results: list[CaseResult],
    settings: Settings,
    is_baseline: bool,
    baseline_run_id: str | None,
) -> dict[str, Any]:
    uri = manifest.tracking_uri or settings.effective_tracking_uri()
    duration = (
        (manifest.finished_at - manifest.started_at).total_seconds()
        if manifest.finished_at
        else None
    )
    n = len(results)
    return {
        "run_id": manifest.run_id,
        "experiment": manifest.experiment,
        "experiment_id": manifest.experiment_id,
        "suite": manifest.suite,
        "suite_version": manifest.suite_version,
        "reps": manifest.reps,
        "n_tasks": manifest.n_tasks,
        "n_cases": n,
        "task_ids": sorted({r.task_id for r in results}),
        "started_at": _iso(manifest.started_at),
        "finished_at": _iso(manifest.finished_at) if manifest.finished_at else None,
        "duration_s": duration,
        "tracking_uri": uri,
        "mlflow_run_url": _mlflow_run_url(uri, manifest.experiment_id, manifest.run_id),
        "kit_version": manifest.kit_version,
        "is_baseline": is_baseline,
        "baseline_run_id": baseline_run_id,
        "profile": {
            "name": manifest.profile.name,
            "adapter": manifest.profile.adapter,
            "model": manifest.profile.model,
            "params": dict(manifest.profile.params),
        },
        "judge": {
            "backend": manifest.judge_backend,
            "model": manifest.judge_model,
            "version": manifest.judge_version,
        },
        "error_rate": sum(1 for r in results if r.record.error) / n if n else None,
        "refusal_rate": sum(1 for r in results if r.record.refused) / n if n else None,
    }


def _failures_section(results: list[CaseResult], manifest: RunManifest) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in sorted(results, key=lambda r: (r.task_id, r.rep)):
        failed = [c for c in r.all_criteria() if r.passed(c) is False]
        judge_errors = [v for v in r.verdicts if v.error and not v.determinism_rerun]
        if not failed and not r.record.error and not judge_errors:
            continue
        rationales = [
            {
                "criterion": v.criterion,
                "passed": v.passed,
                "rationale": _truncate(v.rationale, RATIONALE_TRUNCATE),
                "error": v.error,
            }
            for v in r.verdicts
            if not v.determinism_rerun and (v.passed is False or v.error)
        ]
        text = r.record.final_text or ""
        out.append(
            {
                "task_id": r.task_id,
                "rep": r.rep,
                "failed_criteria": failed,
                "error": r.record.error,
                "refused": r.record.refused,
                "rationales": rationales,
                "final_output": _truncate(text, OUTPUT_TRUNCATE),
                "final_output_chars": len(text),
                "trace_id": r.record.trace_id,
                "trace_url": _trace_url(
                    manifest.tracking_uri, manifest.experiment_id, r.record.trace_id
                ),
                "wall_ms": r.record.wall_ms,
                "agent_cost_usd": r.agent_cost_usd,
                "tool_calls": len(r.record.tool_calls),
            }
        )
    return out


# Helpers ----------------------------------------------------------------------------------
def same_model_family(judge_model: str | None, agent_model: str | None) -> bool:
    """True when judge and agent are the same model id, or both Claude models of the same tier."""
    if not judge_model or not agent_model:
        return False
    j, a = judge_model.lower(), agent_model.lower()
    if j == a:
        return True
    if "claude" in j and "claude" in a:
        return bool({t for t in _MODEL_TIERS if t in j} & {t for t in _MODEL_TIERS if t in a})
    return False


def _trace_url(
    tracking_uri: str | None, experiment_id: str | None, trace_id: str | None
) -> str | None:
    # Route confirmed against the installed MLflow 3.16.1 UI bundle
    # (mlflow/server/js/build/static/js/*.js): the route table defines
    # experimentPageTabTraceDetail = "/experiments/:experimentId/traces/:traceId" next to
    # experimentPageTabTraces = "/experiments/:experimentId/traces". The UI uses a hash
    # router, so the deep link is <tracking_uri>/#/experiments/<exp>/traces/<trace_id>.
    if not (tracking_uri and tracking_uri.startswith("http") and experiment_id and trace_id):
        return None
    return f"{tracking_uri.rstrip('/')}/#/experiments/{experiment_id}/traces/{trace_id}"


def _mlflow_run_url(tracking_uri: str | None, experiment_id: str | None, run_id: str) -> str | None:
    if not (tracking_uri and tracking_uri.startswith("http") and experiment_id):
        return None
    return f"{tracking_uri.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}"


def _missing_quality(baseline_run_id: str | None, run_id: str) -> str:
    if not baseline_run_id:
        return NO_BASELINE
    if baseline_run_id == run_id:
        return "this run is the baseline"
    return "not compared"


def _missing_behavior(baseline_run_id: str | None, run_id: str) -> str:
    if not baseline_run_id:
        return NO_BASELINE
    if baseline_run_id == run_id:
        return "this run is the baseline"
    return "not measured"


def _compare_metrics_by_criterion(compare) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for m in (compare or {}).get("metrics", []):
        name = str(m.get("metric", ""))
        kind = str(m.get("kind", ""))
        if name.startswith("pass_rate."):
            out[name.split(".", 1)[1]] = m
        elif kind in ("criterion", "pass_rate") and name != "pass_rate":
            out[name] = m
    return out


def _criteria_order(
    manifest: RunManifest, settings: Settings, results: list[CaseResult]
) -> list[tuple[str, str]]:
    """Suite order when the suite is loadable, then any criteria seen only in the results."""
    order: dict[str, str] = {}
    try:
        from ..suites import load_suite

        for c in load_suite(manifest.suite, settings).criteria:
            order[c.name] = c.kind
    except Exception:  # noqa: BLE001 - the suite may live elsewhere; results are enough
        pass
    for r in results:
        for name in r.programmatic:
            order.setdefault(name, "programmatic")
        for v in r.verdicts:
            if not v.determinism_rerun:
                order.setdefault(v.criterion, "judge")
    return list(order.items())


def _overall_pass_rate(rows: list[dict[str, Any]]) -> float | None:
    n = sum(r["n"] for r in rows)
    return sum(r["n_pass"] for r in rows) / n if n else None


def _load_calibration(settings: Settings, suite: str, judge_version: str) -> dict[str, Any] | None:
    p = settings.calibration_dir / suite / f"{judge_version}.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _safe_json(store: RunStore, name: str) -> dict[str, Any] | None:
    if not store.has(name):
        return None
    try:
        obj = store.load_json(name)
    except (json.JSONDecodeError, OSError):
        return None
    return obj if isinstance(obj, dict) else None


def _truncate(text: str | None, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _iso(dt: datetime) -> str:
    return (
        dt.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        if dt.tzinfo
        else dt.strftime("%Y-%m-%d %H:%M:%S")
    )


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def _label(v: str | None) -> str:
    if v is None:
        return "n/a"
    return VERDICT_STYLE.get(v, ("neutral", "", v.replace("_", " ")))[2]


# Formatting (also exposed as Jinja filters) -------------------------------------------------
def fmt_pct(v: float | None, digits: int = 1) -> str:
    return "n/a" if v is None else f"{v * 100:.{digits}f}%"


def fmt_pp(v: float | None) -> str:
    """Signed percentage-point delta for pass-rate differences."""
    return "n/a" if v is None else f"{v * 100:+.1f} pp"


def fmt_usd(v: float | None, signed: bool = False) -> str:
    if v is None:
        return "n/a"
    sign = "+" if signed and v > 0 else ""
    return f"{sign}${v:,.4f}" if abs(v) < 1 else f"{sign}${v:,.2f}"


def fmt_num(v: float | int | None, signed: bool = False) -> str:
    if v is None:
        return "n/a"
    sign = "+" if signed and v > 0 else ""
    if isinstance(v, int) or float(v).is_integer():
        return f"{sign}{int(v):,}"
    return f"{sign}{v:,.2f}"


def fmt_ms(v: float | None, signed: bool = False) -> str:
    if v is None:
        return "n/a"
    sign = "+" if signed and v > 0 else ""
    return f"{sign}{v:,.0f} ms" if abs(v) < 1000 else f"{sign}{v / 1000:,.2f} s"


def fmt_float(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:.3g}" if abs(v) < 0.01 else f"{v:.3f}"


def fmt_metric(v: float | None, metric: str, signed: bool = False) -> str:
    """Pick a unit from the metric name: cost -> USD, *_ms -> time, *rate -> %, else count."""
    name = metric.lower()
    if "cost" in name or "usd" in name:
        return fmt_usd(v, signed)
    if "_ms" in name or "latency" in name:
        return fmt_ms(v, signed)
    if "rate" in name or "share" in name or "determinism" in name:
        return fmt_pp(v) if signed else fmt_pct(v)
    return fmt_num(v, signed)


def fmt_kv(d: dict[str, Any] | None) -> str:
    if not d:
        return ""
    parts = []
    for k, v in d.items():
        parts.append(f"{k}={fmt_float(v) if isinstance(v, float) else v}")
    return ", ".join(parts)


def _verdict_class(v: str | None) -> str:
    return VERDICT_STYLE.get(v or "", ("neutral", "", ""))[0]


def _verdict_glyph(v: str | None) -> str:
    return VERDICT_STYLE.get(v or "", ("neutral", "", ""))[1]


_ENV: Environment | None = None


def _env() -> Environment:
    global _ENV
    if _ENV is None:
        env = Environment(
            loader=FileSystemLoader(str(TEMPLATES_DIR)),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        env.filters.update(
            pct=fmt_pct,
            pp=fmt_pp,
            usd=fmt_usd,
            num=fmt_num,
            ms=fmt_ms,
            flt=fmt_float,
            metric=fmt_metric,
            kv=fmt_kv,
            label=_label,
            vclass=_verdict_class,
            vglyph=_verdict_glyph,
            short=lambda s: (s or "")[:8],
        )
        _ENV = env
    return _ENV


# SVG charts -------------------------------------------------------------------------------
def _bar_path(x: float, y: float, w: float, h: float, r: float = 4.0) -> str:
    """A horizontal bar rounded on its data end (right) and square at the baseline (left)."""
    if w <= 0:
        return ""
    r = min(r, w / 2, h / 2)
    return (
        f"M{x:.1f},{y:.1f} H{x + w - r:.1f} A{r:.1f},{r:.1f} 0 0 1 {x + w:.1f},{y + r:.1f} "
        f"V{y + h - r:.1f} A{r:.1f},{r:.1f} 0 0 1 {x + w - r:.1f},{y + h:.1f} H{x:.1f} Z"
    )


def _svg_pass_rates(rows: list[dict[str, Any]], has_baseline: bool) -> str:
    """Horizontal bar per criterion (current pass rate, series-1). With a baseline, a gray
    tick marks the baseline pass rate and a bracket shows the delta CI anchored at it."""
    if not rows:
        return ""
    width, label_w, right_pad, top, row_h, bar_h = 720, 200, 64, 8, 30, 16
    plot_x, plot_w = label_w, width - label_w - right_pad
    plot_h = row_h * len(rows)
    height = top + plot_h + 30
    e = html.escape
    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Pass rate per criterion{", with baseline and confidence interval" if has_baseline else ""}">'
    ]
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = plot_x + t * plot_w
        parts.append(
            f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}"/>'
        )
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{height - 10}" text-anchor="middle">{int(t * 100)}%</text>'
        )
    for i, row in enumerate(rows):
        y = top + i * row_h + (row_h - bar_h) / 2
        cy = y + bar_h / 2
        rate = row["pass_rate"]
        tip = f"{row['name']}: {fmt_pct(rate)} ({row['n_pass']} of {row['n']} cases)"
        if has_baseline and row["baseline_mean"] is not None:
            tip += f"; baseline {fmt_pct(row['baseline_mean'])}, delta {fmt_pp(row['delta'])}"
            if row["ci_low"] is not None and row["ci_high"] is not None:
                tip += f", CI [{fmt_pp(row['ci_low'])}, {fmt_pp(row['ci_high'])}]"
            if row["verdict"]:
                tip += f", verdict {_label(row['verdict'])}"
        parts.append(f"<g><title>{e(tip)}</title>")
        parts.append(
            f'<text class="label" x="{label_w - 10}" y="{cy + 4:.1f}" text-anchor="end">{e(row["name"])}</text>'
        )
        label_x = plot_x + 6
        if rate is None:
            parts.append(
                f'<text class="value muted" x="{label_x:.1f}" y="{cy + 4:.1f}">no verdicts</text>'
            )
        else:
            w = rate * plot_w
            parts.append(f'<path class="bar" d="{_bar_path(plot_x, y, w, bar_h)}"/>')
            label_x = plot_x + w + 6
        if has_baseline and row["baseline_mean"] is not None:
            bx = plot_x + min(max(row["baseline_mean"], 0.0), 1.0) * plot_w
            parts.append(
                f'<rect class="baseline-tick" x="{bx - 1:.1f}" y="{y - 4}" width="2" height="{bar_h + 8}"/>'
            )
            if row["ci_low"] is not None and row["ci_high"] is not None:
                lo = min(max(row["baseline_mean"] + row["ci_low"], 0.0), 1.0)
                hi = min(max(row["baseline_mean"] + row["ci_high"], 0.0), 1.0)
                x1, x2 = plot_x + lo * plot_w, plot_x + hi * plot_w
                parts.append(
                    f'<line class="ci-ring" x1="{x1:.1f}" y1="{cy:.1f}" x2="{x2:.1f}" y2="{cy:.1f}"/>'
                )
                parts.append(
                    f'<line class="ci" x1="{x1:.1f}" y1="{cy:.1f}" x2="{x2:.1f}" y2="{cy:.1f}"/>'
                )
                for cap in (x1, x2):
                    parts.append(
                        f'<line class="ci" x1="{cap:.1f}" y1="{cy - 5:.1f}" x2="{cap:.1f}" y2="{cy + 5:.1f}"/>'
                    )
                label_x = max(label_x, x2 + 8)
        if rate is not None:
            parts.append(
                f'<text class="value" x="{label_x:.1f}" y="{cy + 4:.1f}">{fmt_pct(rate, 0)}</text>'
            )
        parts.append("</g>")
    parts.append("</svg>")
    return "".join(parts)


def _svg_histograms(categories: list[dict[str, Any]]) -> str:
    """Side-by-side tool-category shares: baseline (gray, top) vs current (series-1, bottom)."""
    if not categories:
        return ""
    width, label_w, right_pad, top, row_h, bar_h, gap = 720, 130, 64, 8, 34, 11, 2
    plot_x, plot_w = label_w, width - label_w - right_pad
    plot_h = row_h * len(categories)
    height = top + plot_h + 30
    vmax = max(
        [c["baseline_share"] for c in categories]
        + [c["current_share"] for c in categories]
        + [0.05]
    )
    axis_max = min(1.0, (int(vmax * 10) + 1) / 10)
    e = html.escape
    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Share of tool calls per category, baseline versus current run">'
    ]
    for k in range(5):
        t = axis_max * k / 4
        x = plot_x + (t / axis_max) * plot_w
        parts.append(
            f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}"/>'
        )
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{height - 10}" text-anchor="middle">{t * 100:.0f}%</text>'
        )
    for i, c in enumerate(categories):
        y0 = top + i * row_h + (row_h - (2 * bar_h + gap)) / 2
        cy = y0 + bar_h + gap / 2
        tip = (
            f"{c['category']}: baseline {fmt_pct(c['baseline_share'])} ({c['baseline_count']} calls), "
            f"current {fmt_pct(c['current_share'])} ({c['current_count']} calls)"
        )
        parts.append(f"<g><title>{e(tip)}</title>")
        parts.append(
            f'<text class="label" x="{label_w - 10}" y="{cy + 4:.1f}" text-anchor="end">{e(str(c["category"]))}</text>'
        )
        wb = (c["baseline_share"] / axis_max) * plot_w
        wc = (c["current_share"] / axis_max) * plot_w
        parts.append(f'<path class="bar-base" d="{_bar_path(plot_x, y0, wb, bar_h)}"/>')
        parts.append(f'<path class="bar" d="{_bar_path(plot_x, y0 + bar_h + gap, wc, bar_h)}"/>')
        parts.append(
            f'<text class="value" x="{plot_x + max(wb, wc) + 6:.1f}" y="{cy + 4:.1f}">{fmt_pct(c["current_share"], 0)}</text>'
        )
        parts.append("</g>")
    parts.append("</svg>")
    return "".join(parts)
