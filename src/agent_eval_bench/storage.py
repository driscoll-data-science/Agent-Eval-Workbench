"""On-disk layout for a run under $AEB_HOME/runs/<run_id>/.

manifest.json    RunManifest
results.jsonl    one CaseResult per line (records, programmatic checks, verdicts, costs)
compare.json     ComparisonResult (when a baseline exists)
drift.json       DriftResult (when a baseline exists)
work/<task>_<rep>/ws   workspace copies for workspace tasks (kept for inspection)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Settings
from .models import CaseResult, RunManifest


class RunStore:
    def __init__(self, settings: Settings, run_id: str):
        self.settings = settings
        self.run_id = run_id
        self.dir = settings.runs_dir / run_id

    def ensure(self) -> RunStore:
        self.dir.mkdir(parents=True, exist_ok=True)
        return self

    def work_dir(self, task_id: str, rep: int) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in task_id)
        d = self.dir / "work" / f"{safe}_{rep}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # manifest ---------------------------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    def save_manifest(self, m: RunManifest) -> None:
        self.manifest_path.write_text(m.model_dump_json(indent=2))

    def load_manifest(self) -> RunManifest:
        return RunManifest.model_validate_json(self.manifest_path.read_text())

    # results ----------------------------------------------------------------------------
    @property
    def results_path(self) -> Path:
        return self.dir / "results.jsonl"

    def save_results(self, results: list[CaseResult]) -> None:
        with self.results_path.open("w") as fh:
            for r in results:
                fh.write(r.model_dump_json() + "\n")

    def load_results(self) -> list[CaseResult]:
        out: list[CaseResult] = []
        with self.results_path.open() as fh:
            for line in fh:
                if line.strip():
                    out.append(CaseResult.model_validate_json(line))
        return out

    # generic json -----------------------------------------------------------------------
    def save_json(self, name: str, obj: Any) -> Path:
        p = self.dir / name
        if hasattr(obj, "model_dump"):
            p.write_text(obj.model_dump_json(indent=2))
        else:
            p.write_text(json.dumps(obj, indent=2, default=str))
        return p

    def load_json(self, name: str) -> Any:
        return json.loads((self.dir / name).read_text())

    def has(self, name: str) -> bool:
        return (self.dir / name).is_file()


def list_runs(settings: Settings) -> list[RunManifest]:
    out: list[RunManifest] = []
    if not settings.runs_dir.is_dir():
        return out
    for d in settings.runs_dir.iterdir():
        m = d / "manifest.json"
        if m.is_file():
            try:
                out.append(RunManifest.model_validate_json(m.read_text()))
            except Exception:
                continue
    out.sort(key=lambda m: m.started_at, reverse=True)
    return out


def resolve_run_id(settings: Settings, ref: str | None) -> str:
    """Accept a full run id, a unique prefix, or 'latest'."""
    runs = list_runs(settings)
    if ref in (None, "", "latest"):
        if not runs:
            raise FileNotFoundError("no runs recorded yet")
        return runs[0].run_id
    exact = [m for m in runs if m.run_id == ref]
    if exact:
        return exact[0].run_id
    pref = [m for m in runs if m.run_id.startswith(ref)]
    if len(pref) == 1:
        return pref[0].run_id
    if len(pref) > 1:
        raise ValueError(f"run prefix {ref!r} is ambiguous ({len(pref)} matches)")
    raise FileNotFoundError(f"run {ref!r} not found under {settings.runs_dir}")
