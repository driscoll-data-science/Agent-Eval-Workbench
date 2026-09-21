"""Lifecycle for the local MLflow tracking server: SQLite backend, local artifact root, no
always-on service. ``aeb mlflow up`` starts it detached and records a pid file; ``down`` stops it."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from .config import Settings


def health(uri: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{uri.rstrip('/')}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _pid(settings: Settings) -> int | None:
    p = settings.mlflow_pidfile
    if not p.is_file():
        return None
    try:
        pid = int(p.read_text().strip())
    except ValueError:
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        return None
    return pid


def status(settings: Settings) -> dict:
    return {
        "uri": settings.server_uri,
        "healthy": health(settings.server_uri),
        "pid": _pid(settings),
        "db": str(settings.mlflow_db),
        "artifacts": str(settings.mlflow_artifacts),
    }


def up(settings: Settings, wait_s: float = 45.0) -> dict:
    settings.ensure_dirs()
    if health(settings.server_uri):
        return status(settings)
    log = settings.aeb_home / "mlflow-server.log"
    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        f"sqlite:///{settings.mlflow_db}",
        "--default-artifact-root",
        str(settings.mlflow_artifacts),
        "--host",
        settings.mlflow_host,
        "--port",
        str(settings.mlflow_port),
    ]
    with log.open("a") as fh:
        proc = subprocess.Popen(
            cmd, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True, cwd=settings.aeb_home
        )
    settings.mlflow_pidfile.write_text(str(proc.pid))
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if health(settings.server_uri):
            return status(settings)
        if proc.poll() is not None:
            raise RuntimeError(f"mlflow server exited early (code {proc.returncode}); see {log}")
        time.sleep(0.5)
    raise TimeoutError(f"mlflow server did not become healthy within {wait_s}s; see {log}")


def down(settings: Settings, wait_s: float = 15.0) -> bool:
    pid = _pid(settings)
    if pid is None:
        if settings.mlflow_pidfile.exists():
            settings.mlflow_pidfile.unlink()
        return False
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except OSError:
        os.kill(pid, signal.SIGTERM)
    deadline = time.time() + wait_s
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            break
        time.sleep(0.25)
    else:
        with contextlib.suppress(OSError):
            os.killpg(os.getpgid(pid), signal.SIGKILL)
    settings.mlflow_pidfile.unlink(missing_ok=True)
    return True


def open_ui(settings: Settings) -> None:
    import webbrowser

    webbrowser.open(settings.server_uri)


def ui_path(p: Path) -> str:
    return p.as_uri()
