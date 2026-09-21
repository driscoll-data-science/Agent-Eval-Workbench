"""Shared fixtures: an isolated AEB_HOME and a direct SQLite MLflow store, no server, no keys."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture()
def aeb_env(tmp_path, monkeypatch):
    home = tmp_path / "aeb-home"
    home.mkdir()
    monkeypatch.setenv("AEB_HOME", str(home))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("AEB_JUDGE_BACKEND", "fake")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.chdir(REPO)
    return home


@pytest.fixture()
def settings(aeb_env):
    from agent_eval_bench.config import load_settings

    s = load_settings()
    s.ensure_dirs()
    return s


@pytest.fixture()
def example_suite(settings):
    from agent_eval_bench.suites import load_suite

    return load_suite("example", settings)


@pytest.fixture()
def fake_profile(settings):
    from agent_eval_bench.profiles import load_profile

    return load_profile("fake", settings)


@pytest.fixture()
def regressed_profile(settings):
    from agent_eval_bench.profiles import load_profile

    return load_profile("fake_regressed", settings)


def pytest_configure(config):
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
