"""Suite loading and validation. A suite is a directory with ``suite.yaml`` (criteria + tasks)
and optional ``fixtures/`` and ``hidden_tests/`` subdirectories for workspace tasks."""

from __future__ import annotations

from pathlib import Path

import yaml

from .config import Settings
from .models import Suite


class SuiteNotFound(FileNotFoundError):
    pass


def find_suite_dir(name_or_path: str, settings: Settings) -> Path:
    p = Path(name_or_path).expanduser()
    if p.is_dir() and (p / "suite.yaml").is_file():
        return p.resolve()
    if p.is_file() and p.name == "suite.yaml":
        return p.parent.resolve()
    for base in settings.suite_dirs():
        cand = base / name_or_path
        if (cand / "suite.yaml").is_file():
            return cand.resolve()
    searched = ", ".join(str(d) for d in settings.suite_dirs()) or "(no suite dirs exist)"
    raise SuiteNotFound(f"suite {name_or_path!r} not found; searched {searched}")


def load_suite(name_or_path: str, settings: Settings) -> Suite:
    root = find_suite_dir(name_or_path, settings)
    data = yaml.safe_load((root / "suite.yaml").read_text()) or {}
    suite = Suite(**data)
    suite.root = root
    validate_suite(suite)
    return suite


def validate_suite(suite: Suite) -> list[str]:
    """Raise on hard errors; return soft warnings."""
    warnings: list[str] = []
    assert suite.root is not None
    for task in suite.tasks:
        suite.criteria_for(task)  # raises on unknown criterion names
        if task.kind == "workspace":
            fixture = suite.root / task.fixture  # type: ignore[operator]
            if not fixture.is_dir():
                raise FileNotFoundError(f"task {task.id!r}: fixture dir missing: {fixture}")
            if task.hidden_tests and not (suite.root / task.hidden_tests).is_dir():
                raise FileNotFoundError(f"task {task.id!r}: hidden_tests dir missing")
            if task.checks and task.checks.command is None and not task.hidden_tests:
                warnings.append(
                    f"task {task.id!r}: no command and no hidden_tests; end state is only checked for changes"
                )
        else:
            if not task.reference and not task.expected:
                warnings.append(
                    f"task {task.id!r}: qa task has neither reference nor expected; judge has no anchor"
                )
    if not suite.tasks:
        raise ValueError("suite has no tasks")
    return warnings


def list_suites(settings: Settings) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for base in settings.suite_dirs():
        for child in sorted(base.iterdir()):
            if (child / "suite.yaml").is_file():
                found.append((child.name, child))
    return found
