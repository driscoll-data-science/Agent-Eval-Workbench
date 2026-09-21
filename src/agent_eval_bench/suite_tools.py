"""Suite drafting: Claude Code headless drafts candidate golden cases; a human curates them.

For workspace cases the draft includes fixture files, hidden tests, and a reference patch. Each
candidate is validated mechanically before it is written: the fixture must FAIL the hidden
tests and the reference patch must PASS them, otherwise the case is dropped. That keeps the
drafts honest without a model call.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import yaml

from .config import Settings
from .models import Suite, Task

Log = Callable[[str], None]

DRAFT_SYSTEM = """You design evaluation cases for AI agents. Produce diverse, unambiguous golden cases with a single correct answer or end state. Each case must be gradeable by an independent expert with no context beyond what you provide. Avoid trivia that changes over time, avoid opinion, and avoid tasks whose answer is in the prompt. Vary difficulty: include clearly easy cases and known-hard negatives (traps that a careless agent fails). Return ONLY the JSON object requested."""

QA_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "kebab-case, prefixed qa-"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "prompt": {"type": "string"},
                    "reference": {
                        "type": "string",
                        "description": "authoritative answer, judge-only",
                    },
                    "expected": {
                        "type": "string",
                        "description": "short substring the answer must contain, or empty",
                    },
                    "rubric_notes": {"type": "string", "description": "grading notes, or empty"},
                },
                "required": ["id", "tags", "prompt", "reference", "expected", "rubric_notes"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["tasks"],
    "additionalProperties": False,
}

WS_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "kebab-case, prefixed ws-"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "prompt": {
                        "type": "string",
                        "description": "what the agent must do; mention running the tests",
                    },
                    "fixture_files": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "path -> content of the starting workspace, including its own passing tests",
                    },
                    "hidden_tests": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "path -> pytest file content that the fixture FAILS and the reference PASSES",
                    },
                    "reference_patch": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "path -> full new content of files a correct solution changes",
                    },
                    "allowed_paths": {"type": "array", "items": {"type": "string"}},
                    "must_not_change": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "id",
                    "tags",
                    "prompt",
                    "fixture_files",
                    "hidden_tests",
                    "reference_patch",
                    "allowed_paths",
                    "must_not_change",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["tasks"],
    "additionalProperties": False,
}


def _claude_json(
    system: str, user: str, schema: dict, model: str, timeout_s: int, work_dir: Path
) -> dict:
    cmd = [
        shutil.which("claude") or "claude",
        "-p",
        "--model",
        model,
        "--strict-mcp-config",
        "--setting-sources",
        "",
        "--system-prompt",
        system,
        "--tools",
        "",
        "--max-turns",
        "2",
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(schema),
    ]
    proc = subprocess.run(
        cmd, input=user, capture_output=True, text=True, timeout=timeout_s, cwd=work_dir
    )
    data = json.loads(proc.stdout)
    if data.get("is_error"):
        raise RuntimeError(f"draft call failed: {str(data.get('result'))[:300]}")
    so = data.get("structured_output")
    if not isinstance(so, dict):
        so = json.loads(data.get("result") or "{}")
    so["_usage"] = data.get("usage")
    return so


def _write_files(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        rel = rel.lstrip("/")
        if ".." in Path(rel).parts:
            raise ValueError(f"unsafe path in draft: {rel}")
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content if content.endswith("\n") else content + "\n")


def _pytest_passes(root: Path, timeout_s: int = 120) -> bool:
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env.pop("PYTHONPATH", None)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return False
    return proc.returncode == 0


def validate_workspace_candidate(cand: dict, log: Log) -> bool:
    """Fixture must fail the hidden tests; reference patch must pass them (and the fixture's own tests)."""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        fixture = base / "fixture"
        _write_files(fixture, cand["fixture_files"])
        if not _pytest_passes(fixture):
            log(f"    drop {cand['id']}: fixture's own tests do not pass")
            return False
        broken = base / "broken"
        shutil.copytree(fixture, broken)
        _write_files(broken, cand["hidden_tests"])
        if _pytest_passes(broken):
            log(f"    drop {cand['id']}: fixture already passes the hidden tests (task is a no-op)")
            return False
        solved = base / "solved"
        shutil.copytree(fixture, solved)
        _write_files(solved, cand["reference_patch"])
        _write_files(solved, cand["hidden_tests"])
        if not _pytest_passes(solved):
            log(f"    drop {cand['id']}: reference patch does not pass the hidden tests")
            return False
    return True


def draft_suite(
    settings: Settings,
    kind: str,
    n: int,
    topic: str,
    out_dir: Path,
    model: str = "claude-opus-5",
    timeout_s: int = 900,
    log: Log = print,
) -> Path:
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = settings.aeb_home / "judge_cwd"
    work_dir.mkdir(parents=True, exist_ok=True)
    if kind == "qa":
        user = f"Draft {n} question-answer evaluation cases about: {topic}.\nMix factual, reasoning, extraction, grounding (passage-only questions where the passage may not contain the answer), and instruction-following cases. Every prompt must include an explicit format or length instruction. References must be short and unambiguous."
        data = _claude_json(DRAFT_SYSTEM, user, QA_SCHEMA, model, timeout_s, work_dir)
        tasks = []
        for c in data["tasks"]:
            tasks.append(
                {
                    "id": c["id"],
                    "kind": "qa",
                    "tags": c["tags"] or ["general"],
                    "prompt": c["prompt"],
                    "reference": c["reference"],
                    **({"expected": c["expected"]} if c.get("expected") else {}),
                    **({"rubric_notes": c["rubric_notes"]} if c.get("rubric_notes") else {}),
                    "criteria": ["correct", "grounded", "follows_format", "concise"],
                }
            )
        log(
            f"  drafted {len(tasks)} qa cases (usage: {data.get('_usage', {}).get('input_tokens')} in / {data.get('_usage', {}).get('output_tokens')} out)"
        )
    elif kind == "workspace":
        user = f"Draft {n} small Python coding tasks about: {topic}.\nEach task: a fixture of 1-3 short modules plus a tests/ directory with 2-4 passing pytest tests; a realistic bug or missing feature; hidden tests (2-4) that the fixture fails and a correct solution passes; a reference patch giving the FULL new content of each changed file; allowed_paths listing only the files a solution should touch; must_not_change listing the fixture's test files. Prompts must describe the observable problem, forbid new dependencies, and tell the agent to run the tests. Modules must import without third-party packages."
        data = _claude_json(DRAFT_SYSTEM, user, WS_SCHEMA, model, timeout_s, work_dir)
        tasks = []
        kept = 0
        for c in data["tasks"]:
            if not validate_workspace_candidate(c, log):
                continue
            kept += 1
            fx = f"fixtures/{c['id']}"
            ht = f"hidden_tests/{c['id']}"
            _write_files(out_dir / fx, c["fixture_files"])
            _write_files(out_dir / ht, c["hidden_tests"])
            tasks.append(
                {
                    "id": c["id"],
                    "kind": "workspace",
                    "tags": c["tags"] or ["coding"],
                    "prompt": c["prompt"],
                    "fixture": fx,
                    "hidden_tests": ht,
                    "checks": {
                        "command": "python -m pytest -q",
                        "timeout_s": 120,
                        "allowed_paths": c["allowed_paths"],
                        "must_not_change": c["must_not_change"],
                    },
                    "criteria": [
                        "tests_pass",
                        "only_allowed_paths",
                        "not_noop",
                        "minimal_diff",
                        "readable",
                        "explanation_accurate",
                    ],
                    "reference_patch": c["reference_patch"],
                }
            )
        log(
            f"  drafted {len(data['tasks'])} workspace candidates, kept {kept} after mechanical validation"
        )
    else:
        raise ValueError("kind must be qa or workspace")

    example = yaml.safe_load(
        (
            Path(__file__).resolve().parent.parent.parent / "suites" / "example" / "suite.yaml"
        ).read_text()
    )
    draft = {
        "name": out_dir.name,
        "description": f"DRAFT ({kind}) about {topic}. Curate before use.",
        "version": "1",
        "criteria": example["criteria"],
        "tasks": tasks,
    }
    path = out_dir / "suite.draft.yaml"
    path.write_text(yaml.safe_dump(draft, sort_keys=False, allow_unicode=True, width=100))
    return path


def merge_drafts(paths: list[Path], name: str, out_dir: Path) -> Path:
    """Combine several drafts (e.g. a qa draft and workspace drafts) into one suite.yaml.

    Colliding task ids get a numeric suffix, and their fixture and hidden-test directories
    are copied under the new id so nothing is silently dropped or overwritten.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    criteria = None
    tasks: list[dict] = []
    seen: set[str] = set()
    for p in paths:
        d = yaml.safe_load(Path(p).read_text())
        criteria = criteria or d["criteria"]
        src_root = Path(p).parent
        for t in d["tasks"]:
            original = t["id"]
            new_id, n = original, 2
            while new_id in seen:
                new_id, n = f"{original}-{n}", n + 1
            seen.add(new_id)
            t["id"] = new_id
            if t["kind"] == "workspace":
                for key in ("fixture", "hidden_tests"):
                    if not t.get(key):
                        continue
                    src = src_root / t[key]
                    dest_rel = f"{Path(t[key]).parent.as_posix()}/{new_id}"
                    dest = out_dir / dest_rel
                    if src.is_dir() and not dest.exists():
                        shutil.copytree(src, dest)
                    t[key] = dest_rel
            tasks.append(t)
    suite = {
        "name": name,
        "description": "Curated suite.",
        "version": "1",
        "criteria": criteria,
        "tasks": tasks,
    }
    validated = Suite(**suite)  # raises before anything is written
    validated.root = out_dir
    path = out_dir / "suite.yaml"
    path.write_text(yaml.safe_dump(suite, sort_keys=False, allow_unicode=True, width=100))
    return path


__all__ = ["draft_suite", "merge_drafts", "validate_workspace_candidate", "Task"]
