"""Settings: one config file (aeb.toml) plus environment overrides.

Resolution order, later wins: packaged defaults -> $AEB_HOME/aeb.toml -> ./aeb.toml -> env.
Credentials are never read from config; adapters and judges take them from the environment
or the OS keychain that Claude Code and Codex already use.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_PRICES = PACKAGE_DIR / "prices.yaml"
DEFAULT_TOOL_MAP = PACKAGE_DIR / "tool_map.yaml"

ENV_HOME = "AEB_HOME"
ENV_TRACKING = "MLFLOW_TRACKING_URI"


class Settings(BaseModel):
    aeb_home: Path = Field(default_factory=lambda: Path.home() / ".aeb")
    tracking_uri: str | None = None
    mlflow_host: str = "127.0.0.1"
    mlflow_port: int = 5050

    reps: int = 2
    concurrency: int = 4
    agent_timeout_s: int = 900

    judge_backend: Literal["claude_code", "mlflow", "fake"] = "claude_code"
    judge_model: str = "claude-opus-5"
    judge_concurrency: int = 4
    judge_timeout_s: int = 180
    judge_effort: str | None = "high"
    determinism_sample: float = 0.10

    agreement_min_raw: float = 0.85
    agreement_min_kappa: float = 0.60

    ci_level: float = 0.95
    bootstrap_n: int = 2000
    practical_floor: float = 0.0

    drift_js_threshold: float = 0.10
    drift_alpha: float = 0.05
    drift_min_rate_delta: float = 0.10

    price_table: Path = DEFAULT_PRICES
    tool_map: Path = DEFAULT_TOOL_MAP
    extra_suite_dirs: list[Path] = Field(default_factory=list)
    extra_profile_dirs: list[Path] = Field(default_factory=list)

    # Derived locations -------------------------------------------------------------------
    @property
    def runs_dir(self) -> Path:
        return self.aeb_home / "runs"

    @property
    def reports_dir(self) -> Path:
        return self.aeb_home / "reports"

    @property
    def labels_dir(self) -> Path:
        return self.aeb_home / "labels"

    @property
    def calibration_dir(self) -> Path:
        return self.aeb_home / "calibration"

    @property
    def baselines_file(self) -> Path:
        return self.aeb_home / "baselines.json"

    @property
    def mlflow_db(self) -> Path:
        return self.aeb_home / "mlflow.db"

    @property
    def mlflow_artifacts(self) -> Path:
        return self.aeb_home / "mlartifacts"

    @property
    def mlflow_pidfile(self) -> Path:
        return self.aeb_home / "mlflow.pid"

    @property
    def server_uri(self) -> str:
        return f"http://{self.mlflow_host}:{self.mlflow_port}"

    def effective_tracking_uri(self) -> str:
        """Server URI by default; MLFLOW_TRACKING_URI or [mlflow].tracking_uri overrides."""
        return self.tracking_uri or self.server_uri

    def suite_dirs(self) -> list[Path]:
        dirs = [self.aeb_home / "suites", Path.cwd() / "suites", _repo_root() / "suites"]
        dirs += self.extra_suite_dirs
        return _dedupe_existing(dirs)

    def profile_dirs(self) -> list[Path]:
        dirs = [self.aeb_home / "profiles", Path.cwd() / "profiles", _repo_root() / "profiles"]
        dirs += self.extra_profile_dirs
        return _dedupe_existing(dirs)

    def ensure_dirs(self) -> None:
        for d in (
            self.aeb_home,
            self.runs_dir,
            self.reports_dir,
            self.labels_dir,
            self.calibration_dir,
            self.aeb_home / "suites",
            self.aeb_home / "profiles",
            self.mlflow_artifacts,
        ):
            d.mkdir(parents=True, exist_ok=True)


def _repo_root() -> Path:
    # src/agent_eval_bench/config.py -> repo root is three levels up when running from source.
    return PACKAGE_DIR.parent.parent


def _dedupe_existing(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        p = p.expanduser().resolve()
        if p.is_dir() and p not in out:
            out.append(p)
    return out


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _flatten(cfg: dict[str, Any]) -> dict[str, Any]:
    """aeb.toml uses sections; Settings is flat. Map known sections onto field names."""
    flat: dict[str, Any] = {}
    for section, body in cfg.items():
        if not isinstance(body, dict):
            flat[section] = body
            continue
        for k, v in body.items():
            key = {
                ("mlflow", "host"): "mlflow_host",
                ("mlflow", "port"): "mlflow_port",
                ("mlflow", "tracking_uri"): "tracking_uri",
                ("run", "reps"): "reps",
                ("run", "concurrency"): "concurrency",
                ("run", "agent_timeout_s"): "agent_timeout_s",
                ("judge", "backend"): "judge_backend",
                ("judge", "model"): "judge_model",
                ("judge", "concurrency"): "judge_concurrency",
                ("judge", "timeout_s"): "judge_timeout_s",
                ("judge", "effort"): "judge_effort",
                ("judge", "determinism_sample"): "determinism_sample",
                ("calibration", "min_raw_agreement"): "agreement_min_raw",
                ("calibration", "min_kappa"): "agreement_min_kappa",
                ("compare", "ci_level"): "ci_level",
                ("compare", "bootstrap_n"): "bootstrap_n",
                ("compare", "practical_floor"): "practical_floor",
                ("drift", "js_threshold"): "drift_js_threshold",
                ("drift", "alpha"): "drift_alpha",
                ("drift", "min_rate_delta"): "drift_min_rate_delta",
                ("paths", "home"): "aeb_home",
                ("paths", "price_table"): "price_table",
                ("paths", "tool_map"): "tool_map",
                ("paths", "extra_suite_dirs"): "extra_suite_dirs",
                ("paths", "extra_profile_dirs"): "extra_profile_dirs",
            }.get((section, k), f"{section}_{k}")
            flat[key] = v
    return flat


def load_settings(config_path: Path | None = None) -> Settings:
    env_home = os.environ.get(ENV_HOME)
    home = Path(env_home).expanduser() if env_home else Path.home() / ".aeb"

    merged: dict[str, Any] = {}
    merged.update(_flatten(_read_toml(home / "aeb.toml")))
    merged.update(_flatten(_read_toml(Path.cwd() / "aeb.toml")))
    if config_path is not None:
        merged.update(_flatten(_read_toml(config_path)))

    # Environment overrides. AEB_HOME always wins over any file so a scheduler can relocate data.
    if env_home:
        merged["aeb_home"] = home
    if os.environ.get(ENV_TRACKING):
        merged["tracking_uri"] = os.environ[ENV_TRACKING]
    for env, field in (
        ("AEB_JUDGE_BACKEND", "judge_backend"),
        ("AEB_JUDGE_MODEL", "judge_model"),
        ("AEB_CONCURRENCY", "concurrency"),
        ("AEB_REPS", "reps"),
    ):
        if os.environ.get(env):
            merged[field] = os.environ[env]

    settings = Settings(**{k: v for k, v in merged.items() if k in Settings.model_fields})
    settings.aeb_home = settings.aeb_home.expanduser()
    return settings
