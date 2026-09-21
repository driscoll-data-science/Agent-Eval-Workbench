"""Explicit baseline promotion. A baseline is one run per (suite, profile) and changes only
when someone promotes another run. Stored in $AEB_HOME/baselines.json and mirrored as the
MLflow run tag ``aeb.baseline``."""

from __future__ import annotations

import json

import mlflow

from .config import Settings
from .storage import RunStore


def baseline_key(suite: str, profile: str) -> str:
    return f"{suite}::{profile}"


def load_baselines(settings: Settings) -> dict[str, str]:
    p = settings.baselines_file
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


def get_baseline(settings: Settings, suite: str, profile: str) -> str | None:
    return load_baselines(settings).get(baseline_key(suite, profile))


def promote(
    settings: Settings, run_id: str, tracking_uri: str | None = None
) -> tuple[str, str | None]:
    """Promote ``run_id`` to baseline for its suite and profile. Returns (key, previous_run_id)."""
    manifest = RunStore(settings, run_id).load_manifest()
    key = baseline_key(manifest.suite, manifest.profile.name)
    baselines = load_baselines(settings)
    previous = baselines.get(key)
    baselines[key] = run_id
    settings.baselines_file.parent.mkdir(parents=True, exist_ok=True)
    settings.baselines_file.write_text(json.dumps(baselines, indent=2, sort_keys=True))
    try:
        mlflow.set_tracking_uri(tracking_uri or settings.effective_tracking_uri())
        client = mlflow.MlflowClient()
        if previous and previous != run_id:
            client.set_tag(previous, "aeb.baseline", "false")
        client.set_tag(run_id, "aeb.baseline", "true")
    except Exception:
        # MLflow tagging is a mirror, not the source of truth.
        pass
    return key, previous
