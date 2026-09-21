"""Workspace preparation and end-state checks for ``workspace`` tasks.

The environment is the answer: after the agent runs, we look at what it left behind rather
than at what it said. Hidden tests are copied in only at grading time so the agent cannot
read the answer key.
"""

from __future__ import annotations

import contextlib
import fnmatch
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .models import Suite, Task

IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "node_modules"}


def snapshot(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in root.rglob("*"):
        if any(part in IGNORED_DIRS for part in p.relative_to(root).parts):
            continue
        if p.is_file():
            out[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def prepare_workspace(suite: Suite, task: Task, dest: Path) -> Path:
    assert suite.root is not None and task.fixture
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(suite.root / task.fixture, dest)
    return dest


def changed_paths(before: dict[str, str], after: dict[str, str]) -> list[str]:
    changed = [p for p, h in after.items() if before.get(p) != h]
    changed += [p for p in before if p not in after]
    return sorted(set(changed))


def _matches_any(path: str, patterns: list[str]) -> bool:
    return any(
        fnmatch.fnmatch(path, pat) or path == pat or path.startswith(pat.rstrip("/") + "/")
        for pat in patterns
    )


def run_checks(
    suite: Suite,
    task: Task,
    workspace: Path,
    before: dict[str, str],
    wanted: list[str],
    log_dir: Path | None = None,
) -> dict[str, bool]:
    """Compute the programmatic criteria named in ``wanted`` for a finished workspace."""
    checks = task.checks
    assert checks is not None
    after = snapshot(workspace)
    changed = changed_paths(before, after)
    results: dict[str, bool] = {}

    if "not_noop" in wanted:
        results["not_noop"] = bool(changed) if checks.no_op_fail else True

    if "must_exist" in wanted:
        results["must_exist"] = all((workspace / p).exists() for p in checks.must_exist)

    if "only_allowed_paths" in wanted:
        ok = True
        if checks.must_not_change and any(_matches_any(p, checks.must_not_change) for p in changed):
            ok = False
        if checks.allowed_paths is not None and any(
            not _matches_any(p, checks.allowed_paths) for p in changed
        ):
            ok = False
        results["only_allowed_paths"] = ok

    if "tests_pass" in wanted:
        results["tests_pass"] = _run_tests(suite, task, workspace, log_dir)

    return results


def _run_tests(suite: Suite, task: Task, workspace: Path, log_dir: Path | None) -> bool:
    assert suite.root is not None and task.checks is not None
    copied: list[Path] = []
    if task.hidden_tests:
        src = suite.root / task.hidden_tests
        for p in src.rglob("*"):
            if p.is_file():
                target = workspace / p.relative_to(src)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)
                copied.append(target)
    command = task.checks.command
    if not command:
        _remove(copied)
        return False
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env.pop("PYTHONPATH", None)
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=task.checks.timeout_s,
            env=env,
        )
        rc, out = proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired as e:
        rc, out = 124, f"TIMEOUT after {task.checks.timeout_s}s\n{e}"
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "tests.log").write_text(f"$ {command}\n(exit {rc})\n\n{out}")
    _remove(copied)  # leave only the agent's end state on disk
    return rc == 0


def _remove(paths: list[Path]) -> None:
    for p in paths:
        with contextlib.suppress(OSError):
            p.unlink()


def qa_expected_match(task: Task, answer: str) -> bool | None:
    if task.expected is None:
        return None
    return task.expected.strip().lower() in (answer or "").lower()


def unified_diff(
    suite: Suite, task: Task, workspace: Path, before: dict[str, str], limit_chars: int = 200_000
) -> str:
    """Unified diff of every changed text file between the fixture and the workspace end state."""
    import difflib

    assert suite.root is not None and task.fixture
    fixture = suite.root / task.fixture
    after = snapshot(workspace)
    chunks: list[str] = []
    for rel in changed_paths(before, after):
        old_p, new_p = fixture / rel, workspace / rel
        old = _read_text(old_p) if old_p.exists() else ""
        new = _read_text(new_p) if new_p.exists() else ""
        if old is None or new is None:
            chunks.append(f"Binary file {rel} changed\n")
            continue
        diff = difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{rel}" if old_p.exists() else "/dev/null",
            tofile=f"b/{rel}" if new_p.exists() else "/dev/null",
        )
        chunks.append("".join(diff))
    text = "".join(chunks)
    if len(text) > limit_chars:
        # Deliberate truncation for judge prompts; the full workspace is kept on disk.
        text = text[:limit_chars] + f"\n…[diff truncated at {limit_chars} characters]…\n"
    return text


def _read_text(p: Path) -> str | None:
    try:
        return p.read_text()
    except UnicodeDecodeError:
        return None
