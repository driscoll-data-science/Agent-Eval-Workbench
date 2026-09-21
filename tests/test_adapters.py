"""Adapter tests. No network, no credentials, no live CLI.

Fixtures:
- ``fixtures/claude_code_stream.jsonl`` is a REAL ``claude -p --verbose --output-format
  stream-json`` capture (Claude Code 2.1.275, claude-haiku-4-5, 2026-09-17) with the session id,
  socket path, rate-limit numbers and absolute paths redacted.
- ``fixtures/codex_exec.jsonl`` is SYNTHETIC: hand-written from the documented
  ``codex exec --json`` event schema for 0.154.0. No authenticated Codex run has been recorded
  yet; replace it with a real capture once ``codex login`` has been done.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import textwrap
from pathlib import Path

import pytest

from agent_eval_bench.adapters.base import get_adapter
from agent_eval_bench.adapters.claude_code import (
    EXIT_NOT_FOUND,
    EXIT_TIMEOUT,
    STRIP_ENV,
    ClaudeCodeCLIAdapter,
    clean_env,
    parse_stream_json,
    refusal_heuristic,
)
from agent_eval_bench.adapters.claude_sdk import (
    DEFAULT_ENTRYPOINT,
    ClaudeSDKAdapter,
    collect,
    resolve_entrypoint,
)
from agent_eval_bench.adapters.codex import DEFAULT_MODEL, CodexCLIAdapter, parse_jsonl
from agent_eval_bench.models import Profile, Task

FIXTURES = Path(__file__).parent / "fixtures"
CLAUDE_FIXTURE = FIXTURES / "claude_code_stream.jsonl"
CODEX_FIXTURE = FIXTURES / "codex_exec.jsonl"
REDACTED_SESSION = "00000000-0000-4000-8000-000000000000"

QA_TASK = Task(id="qa-1", kind="qa", prompt="Create hello.txt containing hello, then say done.")


def _fake_cli(tmp_path: Path, name: str, body: str) -> Path:
    """Write an executable python script standing in for a CLI binary."""
    script = tmp_path / name
    script.write_text(f"#!{sys.executable}\n" + textwrap.dedent(body))
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


# ---------------------------------------------------------------------------------------------
# Claude Code CLI
# ---------------------------------------------------------------------------------------------


def test_claude_code_parse_real_fixture():
    parsed = parse_stream_json(CLAUDE_FIXTURE.read_text().splitlines())
    assert parsed.final_text == "Done."
    assert parsed.model == "claude-haiku-4-5-20251001"
    assert parsed.session_id == REDACTED_SESSION
    assert parsed.is_error is False and parsed.error is None
    assert parsed.num_turns == 2
    assert parsed.reported_cost_usd == pytest.approx(0.0210858)

    # Four assistant events, but only two distinct message ids -> two model calls.
    assert [c.index for c in parsed.model_calls] == [0, 1]
    first, last = parsed.model_calls
    assert first.model == "claude-haiku-4-5-20251001"
    assert (first.input_tokens, first.cache_write_tokens, first.cache_read_tokens) == (
        9,
        7220,
        13774,
    )
    assert first.stop_reason == "tool_use"
    assert last.stop_reason == "end_turn"
    # Totals reconcile exactly with result.usage; output shortfall lands on the last call.
    totals = (
        sum(c.input_tokens for c in parsed.model_calls),
        sum(c.output_tokens for c in parsed.model_calls),
        sum(c.cache_read_tokens for c in parsed.model_calls),
        sum(c.cache_write_tokens for c in parsed.model_calls),
    )
    assert totals == (17, 285, 34768, 7597)
    assert parsed.extras["usage_reconciliation"] == {"output_tokens": 280}

    assert len(parsed.tool_calls) == 1
    tc = parsed.tool_calls[0]
    assert tc.name == "Write"
    assert tc.input_full == {"file_path": "/tmp/aeb-fixture-ws/hello.txt", "content": "hello"}
    assert isinstance(tc.output_full, str) and tc.output_full.startswith(
        "File created successfully"
    )
    assert tc.is_error is False
    assert tc.output_summary and "File created" in tc.output_summary

    assert parsed.extras["init"]["claude_code_version"] == "2.1.275"
    assert parsed.extras["init"]["mcp_servers"] == []
    assert parsed.extras["thinking_blocks"] == 2
    assert parsed.extras["other_event:rate_limit_event"] == 1
    assert "claude-haiku-4-5" in parsed.extras["model_usage"]
    # modelUsage lists an auxiliary 908-token call that result.usage does not include.
    assert parsed.extras["auxiliary_tokens"] == {"input_tokens": 908, "output_tokens": 13}


def test_claude_code_parse_error_result_and_missing_result():
    lines = [
        json.dumps(
            {"type": "system", "subtype": "init", "session_id": "s1", "model": "claude-sonnet-5"}
        ),
        json.dumps(
            {
                "type": "result",
                "subtype": "error_max_turns",
                "is_error": True,
                "num_turns": 3,
                "result": "Reached max turns",
                "usage": {"input_tokens": 5, "output_tokens": 7},
                "session_id": "s1",
            }
        ),
    ]
    parsed = parse_stream_json(lines)
    assert parsed.is_error and parsed.error == "claude_code error_max_turns: Reached max turns"
    assert parsed.model == "claude-sonnet-5"
    assert len(parsed.model_calls) == 1 and parsed.model_calls[0].output_tokens == 7

    truncated = parse_stream_json(CLAUDE_FIXTURE.read_text().splitlines()[:-1])
    assert truncated.is_error and "without a result" in (truncated.error or "")
    assert truncated.final_text == "Done."  # last assistant text survives
    assert len(truncated.model_calls) == 2

    junk = parse_stream_json(["not json", "", "[1,2]"])
    assert junk.extras["bad_lines"] == 2 and junk.is_error


def test_claude_code_build_command_honors_every_param(tmp_path):
    sp = tmp_path / "extra.md"
    sp.write_text("be terse")
    profile = Profile(
        name="p",
        adapter="claude_code",
        model="claude-sonnet-5",
        params={
            "clean_slate": True,
            "permission_mode": "acceptEdits",
            "allowed_tools": "Read,Edit,Write,Glob,Grep,Bash",
            "tools": ["Read", "Bash"],
            "system_prompt_file": str(sp),
            "max_turns": 40,
            "max_budget_usd": 2.0,
            "effort": "high",
            "extra_args": ["--bare"],
        },
    )
    cmd = ClaudeCodeCLIAdapter(cli="claude").build_command(QA_TASK, None, profile)
    assert cmd == [
        "claude", "-p", "--verbose", "--output-format", "stream-json", "--no-session-persistence",
        "--model", "claude-sonnet-5",
        "--strict-mcp-config", "--setting-sources", "",
        "--permission-mode", "acceptEdits",
        "--allowedTools", "Read,Edit,Write,Glob,Grep,Bash",
        "--tools", "Read,Bash",
        "--append-system-prompt-file", str(sp),
        "--max-turns", "40",
        "--max-budget-usd", "2.0",
        "--effort", "high",
        "--bare",
    ]  # fmt: skip


def test_claude_code_build_command_full_config_and_no_tools():
    full = Profile(
        name="p", adapter="claude_code", model=None, params={"clean_slate": False, "tools": ""}
    )
    cmd = ClaudeCodeCLIAdapter(cli="claude").build_command(QA_TASK, None, full)
    assert "--strict-mcp-config" not in cmd and "--setting-sources" not in cmd
    assert "--model" not in cmd
    assert cmd[-2:] == ["--tools", ""]
    bare = ClaudeCodeCLIAdapter(cli="claude").build_command(
        QA_TASK,
        None,
        Profile(name="p", adapter="claude_code", params={"no_session_persistence": False}),
    )
    assert bare == [
        "claude",
        "-p",
        "--verbose",
        "--output-format",
        "stream-json",
        "--strict-mcp-config",
        "--setting-sources",
        "",
    ]  # noqa: E501


def test_clean_env_strips_nested_session_vars():
    base = {"CLAUDECODE": "1", "CLAUDE_EFFORT": "max", "CLAUDE_CODE_OAUTH_TOKEN": "t", "HOME": "/h"}
    out = clean_env(base)
    assert out == {"CLAUDE_CODE_OAUTH_TOKEN": "t", "HOME": "/h"}
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in STRIP_ENV


@pytest.fixture()
def fake_claude(tmp_path, monkeypatch):
    argv_file, stdin_file = tmp_path / "argv.json", tmp_path / "stdin.txt"
    monkeypatch.setenv("AEB_FAKE_ARGV", str(argv_file))
    monkeypatch.setenv("AEB_FAKE_STDIN", str(stdin_file))
    monkeypatch.setenv("AEB_FAKE_FIXTURE", str(CLAUDE_FIXTURE))
    monkeypatch.setenv("CLAUDE_EFFORT", "max")  # must not reach the child
    script = _fake_cli(
        tmp_path,
        "fake-claude",
        """
        import json, os, sys, time
        from pathlib import Path
        Path(os.environ["AEB_FAKE_ARGV"]).write_text(json.dumps({
            "argv": sys.argv[1:], "cwd": os.getcwd(),
            "effort_env": os.environ.get("CLAUDE_EFFORT"),
        }))
        Path(os.environ["AEB_FAKE_STDIN"]).write_text(sys.stdin.read())
        if os.environ.get("AEB_FAKE_SLEEP"):
            time.sleep(float(os.environ["AEB_FAKE_SLEEP"]))
        sys.stdout.write(Path(os.environ["AEB_FAKE_FIXTURE"]).read_text())
        sys.stderr.write("fake stderr\\n")
        sys.exit(int(os.environ.get("AEB_FAKE_EXIT", "0")))
        """,
    )
    return script, argv_file, stdin_file


def test_claude_code_run_qa_with_fake_cli(fake_claude):
    script, argv_file, stdin_file = fake_claude
    profile = Profile(
        name="clean", adapter="claude_code", model="claude-haiku-4-5", params={"max_turns": 3}
    )
    rec = ClaudeCodeCLIAdapter(cli=str(script)).run(QA_TASK, 0, None, profile, timeout_s=30)
    assert rec.ok, rec.error
    assert rec.agent == "claude_code" and rec.profile == "clean"
    assert rec.model == "claude-haiku-4-5-20251001"  # what the API served, not the alias
    assert rec.final_text == "Done."
    assert rec.tool_names() == ["Write"]
    assert rec.total_tokens == 17 + 285 + 34768 + 7597
    assert rec.reported_cost_usd == pytest.approx(0.0210858)
    assert rec.refused is False and rec.wall_ms >= 0 and rec.workspace_path is None
    assert rec.raw["session_id"] == REDACTED_SESSION and rec.raw["num_turns"] == 2
    assert rec.raw["stderr_tail"] == "fake stderr"
    assert rec.raw["command"][0] == str(script)
    seen = json.loads(argv_file.read_text())
    assert seen["argv"][:5] == [
        "-p",
        "--verbose",
        "--output-format",
        "stream-json",
        "--no-session-persistence",
    ]
    assert seen["effort_env"] is None
    assert not Path(seen["cwd"]).exists() or seen["cwd"] != os.getcwd()  # neutral temp dir, removed
    assert stdin_file.read_text() == QA_TASK.prompt


def test_claude_code_run_workspace_cwd_and_failure_modes(fake_claude, tmp_path, monkeypatch):
    script, argv_file, _ = fake_claude
    ws = tmp_path / "ws"
    ws.mkdir()
    task = Task(id="ws-1", kind="workspace", prompt="fix it", fixture="x")
    profile = Profile(name="p", adapter="claude_code", model="claude-haiku-4-5")
    adapter = ClaudeCodeCLIAdapter(cli=str(script))

    rec = adapter.run(task, 1, ws, profile, timeout_s=30)
    assert rec.ok and rec.workspace_path == str(ws)
    assert Path(json.loads(argv_file.read_text())["cwd"]).resolve() == ws.resolve()

    monkeypatch.setenv("AEB_FAKE_EXIT", "1")
    rec = adapter.run(task, 1, ws, profile, timeout_s=30)
    assert rec.exit_status == 1 and not rec.ok
    assert rec.error is not None and "exited 1" in rec.error and "fake stderr" in rec.error
    assert rec.final_text == "Done."  # partial output is still parsed
    monkeypatch.delenv("AEB_FAKE_EXIT")

    monkeypatch.setenv("AEB_FAKE_SLEEP", "5")
    rec = adapter.run(task, 1, ws, profile, timeout_s=1)
    assert rec.exit_status == EXIT_TIMEOUT and rec.error == "timeout after 1s"
    monkeypatch.delenv("AEB_FAKE_SLEEP")

    rec = ClaudeCodeCLIAdapter(cli=str(tmp_path / "missing-claude")).run(task, 0, ws, profile, 5)
    assert rec.exit_status == EXIT_NOT_FOUND and "not found" in (rec.error or "")


def test_claude_code_run_marks_refusal(fake_claude, tmp_path, monkeypatch):
    script, _, _ = fake_claude
    refusal = tmp_path / "refusal.jsonl"
    refusal.write_text(
        "\n".join(
            json.dumps(ev)
            for ev in [
                {
                    "type": "system",
                    "subtype": "init",
                    "session_id": "s",
                    "model": "claude-sonnet-5",
                },
                {
                    "type": "assistant",
                    "message": {
                        "id": "m1",
                        "model": "claude-sonnet-5",
                        "content": [{"type": "text", "text": "I can't help with that request."}],
                        "usage": {"input_tokens": 10, "output_tokens": 9},
                        "stop_reason": "end_turn",
                    },
                },
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "num_turns": 1,
                    "result": "I can't help with that request.",
                    "usage": {"input_tokens": 10, "output_tokens": 9},
                    "total_cost_usd": 0.001,
                    "session_id": "s",
                },
            ]
        )
    )
    monkeypatch.setenv("AEB_FAKE_FIXTURE", str(refusal))
    rec = ClaudeCodeCLIAdapter(cli=str(script)).run(
        QA_TASK, 0, None, Profile(name="p", adapter="claude_code", model="claude-sonnet-5"), 30
    )
    assert rec.ok and rec.refused is True
    assert rec.model_calls[0].stop_reason == "end_turn" and rec.output_tokens == 9


@pytest.mark.parametrize(
    "text",
    [
        "I can't help with that request.",
        "I'm sorry, but I cannot assist with creating malware.",
        "I am unable to provide that information.",
        "I must decline this request.",
        "That request goes against my guidelines, so I won't proceed.",
        "Sorry, I don't feel comfortable writing that.",
        "  I won't be able to help with this.",
    ],
)
def test_refusal_heuristic_positive(text):
    assert refusal_heuristic(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "",
        None,
        "Done.",
        "The answer is 42.",
        "I can't find the file, so I created it and then ran the tests.",
        "It won't compile until you add the import; here is the fix.",
        "Here is the summary you asked for. I can't guarantee the dates are exact.",
    ],
)
def test_refusal_heuristic_negative(text):
    assert refusal_heuristic(text) is False


# ---------------------------------------------------------------------------------------------
# Codex CLI
# ---------------------------------------------------------------------------------------------


def test_codex_parse_synthetic_fixture():
    parsed = parse_jsonl(CODEX_FIXTURE.read_text().splitlines())
    assert parsed.session_id == "thr_synthetic_0001"
    assert parsed.final_text == "I fixed the off-by-one in calc.py.\n\nAll tests pass now."
    assert parsed.model == DEFAULT_MODEL and parsed.num_turns == 1
    assert parsed.is_error is False and parsed.reported_cost_usd is None

    names = [t.name for t in parsed.tool_calls]
    assert names == ["shell", "shell", "apply_patch", "mcp:docs.lookup", "web_search"]
    ls, pytest_call, patch, mcp, web = parsed.tool_calls
    assert ls.input_full == {"command": "ls -la"} and ls.is_error is False
    assert ls.output_full.startswith("total 8") and "README.md" in ls.output_summary
    assert pytest_call.is_error is True and pytest_call.output_full == "F\n1 failed in 0.02s\n"
    assert patch.input_full == {
        "changes": [
            {"path": "calc.py", "kind": "update"},
            {"path": "tests/test_calc.py", "kind": "add"},
        ]
    }
    assert patch.output_full == "completed" and patch.is_error is False
    assert mcp.input_full == {"q": "python sum"} and mcp.is_error is False
    assert mcp.output_full == {"content": [{"type": "text", "text": "sum(iterable, /, start=0)"}]}
    assert web.input_full == {"query": "python builtin sum"} and web.output_full is None

    # Usage: cached is a subset of input, reasoning a subset of output. No double counting.
    assert len(parsed.model_calls) == 1
    call = parsed.model_calls[0]
    assert (call.input_tokens, call.cache_read_tokens, call.output_tokens, call.cache_write_tokens) == (
        200, 1000, 300, 0,
    )  # fmt: skip
    assert parsed.extras["reasoning_output_tokens"] == 100
    assert parsed.extras["reasoning_items"] == 1 and parsed.extras["todo_items"] == 1

    with_model = parse_jsonl(CODEX_FIXTURE.read_text().splitlines(), model="gpt-5.3-codex")
    assert (
        with_model.model == "gpt-5.3-codex" and with_model.model_calls[0].model == "gpt-5.3-codex"
    )


def test_codex_parse_failure_events():
    lines = [
        json.dumps({"type": "thread.started", "thread_id": "t"}),
        json.dumps({"type": "turn.started"}),
        json.dumps(
            {
                "type": "item.started",
                "item": {
                    "id": "c1",
                    "type": "command_execution",
                    "command": "sleep 99",
                    "status": "in_progress",
                },
            }
        ),
        json.dumps({"type": "turn.failed", "error": {"message": "rate limited"}}),
        json.dumps({"type": "error", "message": "stream closed"}),
        "garbage",
    ]
    parsed = parse_jsonl(lines)
    assert parsed.is_error and parsed.error == "codex: rate limited; stream closed"
    assert parsed.model_calls == [] and parsed.final_text == ""
    assert [t.name for t in parsed.tool_calls] == ["shell"]
    assert parsed.tool_calls[0].output_full is None and parsed.tool_calls[0].is_error is False
    assert parsed.extras["bad_lines"] == 1


def test_codex_build_command_honors_every_param(tmp_path):
    profile = Profile(
        name="codex_default",
        adapter="codex",
        model="gpt-5.3-codex",
        params={
            "sandbox": "workspace-write",
            "isolate_home": True,
            "disable_agents_md": True,
            "ignore_user_config": True,
            "reasoning_effort": "high",
            "config_overrides": {"mcp_servers": {}, "features.web_search": False},
        },
    )
    out = tmp_path / "last.txt"
    cmd = CodexCLIAdapter(cli="codex").build_command(
        QA_TASK, tmp_path / "ws", profile, output_file=out
    )
    assert cmd == [
        "codex", "exec", "--json", "--skip-git-repo-check", "--ephemeral",
        "-C", str(tmp_path / "ws"),
        "-m", "gpt-5.3-codex",
        "-s", "workspace-write",
        "--ignore-user-config",
        "-c", "project_doc_max_bytes=0",
        "-c", "model_reasoning_effort=high",
        "-c", "mcp_servers={}",
        "-c", "features.web_search=false",
        "-o", str(out),
        "-",
    ]  # fmt: skip
    assert "--full-auto" not in cmd

    minimal = Profile(
        name="m", adapter="codex", model=None,
        params={"disable_agents_md": False, "ignore_user_config": False, "reasoning_effort": None},
    )  # fmt: skip
    cmd = CodexCLIAdapter(cli="codex").build_command(QA_TASK, None, minimal)
    assert cmd == ["codex", "exec", "--json", "--skip-git-repo-check", "--ephemeral", "-"]


def test_codex_prepare_home_copies_only_auth(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".codex").mkdir(parents=True)
    (home / ".codex" / "auth.json").write_text('{"token": "fake"}')
    (home / ".codex" / "config.toml").write_text("model = 'x'\n")
    monkeypatch.setenv("HOME", str(home))

    codex_home = CodexCLIAdapter.prepare_home(tmp_path / "run1")
    assert codex_home.is_dir() and codex_home == tmp_path / "run1" / "codex_home"
    assert (codex_home / "auth.json").read_text() == '{"token": "fake"}'
    assert stat.S_IMODE((codex_home / "auth.json").stat().st_mode) == 0o600
    assert sorted(p.name for p in codex_home.iterdir()) == ["auth.json"]

    monkeypatch.setenv("HOME", str(tmp_path / "nohome"))
    empty = CodexCLIAdapter.prepare_home(tmp_path / "run2")
    assert empty.is_dir() and list(empty.iterdir()) == []


def test_codex_run_with_fake_cli(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".codex").mkdir(parents=True)
    (home / ".codex" / "auth.json").write_text("{}")
    monkeypatch.setenv("HOME", str(home))
    seen_file = tmp_path / "seen.json"
    monkeypatch.setenv("AEB_FAKE_ARGV", str(seen_file))
    monkeypatch.setenv("AEB_FAKE_FIXTURE", str(CODEX_FIXTURE))
    script = _fake_cli(
        tmp_path,
        "fake-codex",
        """
        import json, os, sys
        from pathlib import Path
        argv = sys.argv[1:]
        codex_home = os.environ.get("CODEX_HOME")
        Path(os.environ["AEB_FAKE_ARGV"]).write_text(json.dumps({
            "argv": argv, "cwd": os.getcwd(), "codex_home": codex_home,
            "auth_present": bool(codex_home) and os.path.exists(os.path.join(codex_home, "auth.json")),
            "prompt": sys.stdin.read(),
        }))
        Path(argv[argv.index("-o") + 1]).write_text("last message from -o file")
        sys.stdout.write(Path(os.environ["AEB_FAKE_FIXTURE"]).read_text())
        sys.exit(int(os.environ.get("AEB_FAKE_EXIT", "0")))
        """,
    )
    ws = tmp_path / "ws"
    ws.mkdir()
    task = Task(id="ws-1", kind="workspace", prompt="fix the bug", fixture="x")
    profile = Profile(
        name="codex_default", adapter="codex", model=None, params={"sandbox": "workspace-write"}
    )
    rec = CodexCLIAdapter(cli=str(script)).run(task, 0, ws, profile, timeout_s=30)
    assert rec.ok, rec.error
    assert rec.agent == "codex" and rec.model == DEFAULT_MODEL
    assert rec.final_text.startswith("I fixed the off-by-one")  # agent_message wins over -o
    assert rec.tool_names() == ["shell", "shell", "apply_patch", "mcp:docs.lookup", "web_search"]
    assert (rec.input_tokens, rec.cache_read_tokens, rec.output_tokens) == (200, 1000, 300)
    assert rec.reported_cost_usd is None and rec.refused is False
    assert rec.raw["session_id"] == "thr_synthetic_0001" and rec.raw["codex_home_isolated"] is True
    assert rec.raw["last_message_file"] is True and rec.raw["reasoning_output_tokens"] == 100
    seen = json.loads(seen_file.read_text())
    assert seen["prompt"] == "fix the bug" and Path(seen["cwd"]).resolve() == ws.resolve()
    assert seen["codex_home"] and seen["auth_present"] is True
    assert not Path(seen["codex_home"]).exists()  # throwaway home removed after the run
    assert seen["argv"][:5] == ["exec", "--json", "--skip-git-repo-check", "--ephemeral", "-C"]

    # No agent_message in the stream: the -o file is the fallback for final_text.
    bare = tmp_path / "bare.jsonl"
    bare.write_text(json.dumps({"type": "thread.started", "thread_id": "t2"}) + "\n")
    monkeypatch.setenv("AEB_FAKE_FIXTURE", str(bare))
    rec = CodexCLIAdapter(cli=str(script)).run(task, 0, ws, profile, timeout_s=30)
    assert rec.final_text == "last message from -o file" and rec.model_calls == []

    monkeypatch.setenv("AEB_FAKE_EXIT", "2")
    rec = CodexCLIAdapter(cli=str(script)).run(task, 0, ws, profile, timeout_s=30)
    assert rec.exit_status == 2 and "exited 2" in (rec.error or "")


# ---------------------------------------------------------------------------------------------
# Claude Agent SDK
# ---------------------------------------------------------------------------------------------


def test_sdk_build_options_honors_every_param(tmp_path):
    sp = tmp_path / "extra.md"
    sp.write_text("be terse")
    profile = Profile(
        name="claude_sdk_reference",
        adapter="claude_sdk",
        model="claude-sonnet-5",
        params={
            "entrypoint": DEFAULT_ENTRYPOINT,
            "permission_mode": "acceptEdits",
            "allowed_tools": ["Read", "Edit", "Write", "Glob", "Grep", "Bash"],
            "tools": "Read,Bash",
            "max_turns": 40,
            "max_budget_usd": 2.0,
            "effort": "high",
            "system_prompt_file": str(sp),
            "extra_args": {"bare": None},
        },
    )
    opts = ClaudeSDKAdapter().build_options(QA_TASK, tmp_path / "ws", profile)
    assert opts.model == "claude-sonnet-5"
    assert opts.cwd == str(tmp_path / "ws")
    assert opts.setting_sources == [] and opts.strict_mcp_config is True and opts.mcp_servers == {}
    assert opts.permission_mode == "acceptEdits"
    assert opts.allowed_tools == ["Read", "Edit", "Write", "Glob", "Grep", "Bash"]
    assert opts.tools == ["Read", "Bash"]
    assert opts.max_turns == 40 and opts.max_budget_usd == 2.0 and opts.effort == "high"
    assert opts.system_prompt == {"type": "preset", "preset": "claude_code", "append": "be terse"}
    assert opts.extra_args == {"bare": None}

    full = Profile(
        name="full", adapter="claude_sdk", params={"clean_slate": False, "system_prompt": "x"}
    )
    opts = ClaudeSDKAdapter().build_options(QA_TASK, None, full)
    assert opts.setting_sources == ["user", "project", "local"] and opts.strict_mcp_config is False
    assert opts.cwd is None and opts.tools is None and opts.allowed_tools == []
    assert opts.system_prompt == "x" and opts.max_turns is None


def _sdk_messages(prompt: str = "hi"):
    from claude_agent_sdk import (
        AssistantMessage,
        ResultMessage,
        SystemMessage,
        TextBlock,
        ThinkingBlock,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
    )

    usage1 = {
        "input_tokens": 12,
        "output_tokens": 30,
        "cache_read_input_tokens": 100,
        "cache_creation_input_tokens": 50,
    }
    usage2 = {
        "input_tokens": 8,
        "output_tokens": 20,
        "cache_read_input_tokens": 160,
        "cache_creation_input_tokens": 0,
    }
    return [
        SystemMessage(subtype="init", data={"session_id": "sdk-sess", "model": "claude-sonnet-5"}),
        AssistantMessage(
            content=[ThinkingBlock(thinking="", signature="sig")],
            model="claude-sonnet-5-20260601",
            usage=usage1,
            message_id="m1",
        ),
        AssistantMessage(
            content=[ToolUseBlock(id="t1", name="Read", input={"file_path": "README.md"})],
            model="claude-sonnet-5-20260601",
            usage=usage1,
            message_id="m1",
        ),
        UserMessage(
            content=[ToolResultBlock(tool_use_id="t1", content="# readme", is_error=False)]
        ),
        AssistantMessage(
            content=[TextBlock(text=f"echo: {prompt}")],
            model="claude-sonnet-5-20260601",
            usage=usage2,
            stop_reason="end_turn",
            message_id="m2",
        ),
        ResultMessage(
            subtype="success",
            duration_ms=1200,
            duration_api_ms=900,
            is_error=False,
            num_turns=2,
            session_id="sdk-sess",
            total_cost_usd=0.0123,
            usage={
                "input_tokens": 20,
                "output_tokens": 50,
                "cache_read_input_tokens": 260,
                "cache_creation_input_tokens": 50,
            },
            result=f"echo: {prompt}",
            model_usage={
                "claude-sonnet-5": {"inputTokens": 20, "outputTokens": 50, "costUSD": 0.0123}
            },
        ),
    ]


def test_sdk_collect_hand_built_messages():
    parsed = collect(_sdk_messages("hello"))
    assert parsed.final_text == "echo: hello"
    assert parsed.model == "claude-sonnet-5-20260601" and parsed.session_id == "sdk-sess"
    assert parsed.num_turns == 2 and parsed.reported_cost_usd == pytest.approx(0.0123)
    assert [c.index for c in parsed.model_calls] == [0, 1]  # m1 deduplicated across two events
    c0, c1 = parsed.model_calls
    assert (c0.input_tokens, c0.output_tokens, c0.cache_read_tokens, c0.cache_write_tokens) == (
        12,
        30,
        100,
        50,
    )
    assert (c1.input_tokens, c1.output_tokens, c1.cache_read_tokens, c1.cache_write_tokens) == (
        8,
        20,
        160,
        0,
    )
    assert c0.stop_reason == "tool_use" and c1.stop_reason == "end_turn"
    assert "usage_reconciliation" not in parsed.extras  # totals already match the result
    assert len(parsed.tool_calls) == 1
    tc = parsed.tool_calls[0]
    assert tc.name == "Read" and tc.input_full == {"file_path": "README.md"}
    assert tc.output_full == "# readme" and tc.is_error is False and tc.output_summary == "# readme"
    assert parsed.extras["thinking_blocks"] == 1
    assert parsed.extras["model_usage"]["claude-sonnet-5"]["costUSD"] == 0.0123
    assert parsed.extras["duration_ms"] == 1200


def test_sdk_collect_error_result():
    from claude_agent_sdk import ResultMessage

    parsed = collect(
        [
            ResultMessage(
                subtype="error_during_execution",
                duration_ms=5,
                duration_api_ms=0,
                is_error=True,
                num_turns=0,
                session_id="s",
                result="boom",
                errors=["authentication_failed"],
            )
        ]
    )
    assert (
        parsed.is_error
        and parsed.error == "claude_code error_during_execution: boom: authentication_failed"
    )
    assert parsed.model_calls == [] and parsed.final_text == "boom"


@pytest.fixture()
def fake_entrypoint(tmp_path, monkeypatch):
    mod = tmp_path / "aeb_fake_sdk_agent.py"
    mod.write_text(
        textwrap.dedent(
            """
            import asyncio, os
            from pathlib import Path
            from tests.test_adapters import _sdk_messages

            SEEN = {}

            async def run(prompt, options):
                SEEN["prompt"] = prompt
                SEEN["cwd"] = options.cwd
                SEEN["model"] = options.model
                if os.environ.get("AEB_FAKE_SLEEP"):
                    yield _sdk_messages(prompt)[0]
                    await asyncio.sleep(float(os.environ["AEB_FAKE_SLEEP"]))
                for m in _sdk_messages(prompt):
                    yield m

            async def boom(prompt, options):
                raise RuntimeError("agent exploded")
                yield  # pragma: no cover - makes this an async generator

            def not_async(prompt, options):
                return None
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parent.parent))
    sys.modules.pop("aeb_fake_sdk_agent", None)
    yield "aeb_fake_sdk_agent"
    sys.modules.pop("aeb_fake_sdk_agent", None)


def test_sdk_run_with_fake_entrypoint(fake_entrypoint, tmp_path, monkeypatch):
    import importlib

    profile = Profile(
        name="sdk", adapter="claude_sdk", model="claude-sonnet-5",
        params={"entrypoint": f"{fake_entrypoint}:run", "allowed_tools": ["Read"]},
    )  # fmt: skip
    monkeypatch.setenv("CLAUDECODE", "1")
    rec = ClaudeSDKAdapter().run(QA_TASK, 0, None, profile, timeout_s=30)
    assert rec.ok, rec.error
    assert "CLAUDECODE" not in os.environ  # nested-session vars scrubbed before the SDK spawns
    assert rec.agent == "claude_sdk" and rec.model == "claude-sonnet-5-20260601"
    assert rec.final_text == f"echo: {QA_TASK.prompt}"
    assert rec.tool_names() == ["Read"] and len(rec.model_calls) == 2
    assert rec.total_tokens == 20 + 50 + 260 + 50 and rec.reported_cost_usd == pytest.approx(0.0123)
    assert rec.raw["entrypoint"] == f"{fake_entrypoint}:run" and rec.raw["n_messages"] == 6
    seen = importlib.import_module(fake_entrypoint).SEEN
    assert seen["prompt"] == QA_TASK.prompt and seen["model"] == "claude-sonnet-5"
    assert seen["cwd"] and not Path(seen["cwd"]).exists()  # neutral temp dir, removed afterwards

    ws = tmp_path / "ws"
    ws.mkdir()
    task = Task(id="ws-1", kind="workspace", prompt="fix", fixture="x")
    rec = ClaudeSDKAdapter().run(task, 0, ws, profile, timeout_s=30)
    assert rec.ok and rec.workspace_path == str(ws) and seen["cwd"] == str(ws)

    monkeypatch.setenv("AEB_FAKE_SLEEP", "5")
    rec = ClaudeSDKAdapter().run(task, 0, ws, profile, timeout_s=1)
    assert rec.exit_status == EXIT_TIMEOUT and rec.error == "timeout after 1s"
    assert rec.raw["n_messages"] == 1  # messages yielded before the timeout survive
    monkeypatch.delenv("AEB_FAKE_SLEEP")

    boom = Profile(name="b", adapter="claude_sdk", params={"entrypoint": f"{fake_entrypoint}:boom"})
    rec = ClaudeSDKAdapter().run(task, 0, ws, boom, timeout_s=5)
    assert rec.exit_status == 1 and rec.error == "RuntimeError: agent exploded"

    bad = Profile(name="b", adapter="claude_sdk", params={"entrypoint": "no.such.module:run"})
    rec = ClaudeSDKAdapter().run(task, 0, ws, bad, timeout_s=5)
    assert rec.exit_status == 1 and rec.error is not None and "ModuleNotFoundError" in rec.error


def test_resolve_entrypoint_validation():
    assert callable(resolve_entrypoint(DEFAULT_ENTRYPOINT))
    with pytest.raises(ValueError):
        resolve_entrypoint("agent_eval_bench.agents.reference_sdk_agent")
    with pytest.raises(ValueError):
        resolve_entrypoint("agent_eval_bench.agents.reference_sdk_agent:__doc__")


# ---------------------------------------------------------------------------------------------
# Registry and shipped profiles
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("profile_file", "cls", "name"),
    [
        ("claude_code_clean_sonnet.yaml", ClaudeCodeCLIAdapter, "claude_code"),
        ("claude_code_full_sonnet.yaml", ClaudeCodeCLIAdapter, "claude_code"),
        ("codex_default.yaml", CodexCLIAdapter, "codex"),
        ("claude_sdk_reference.yaml", ClaudeSDKAdapter, "claude_sdk"),
    ],
)
def test_get_adapter_and_shipped_profiles_build(profile_file, cls, name, tmp_path):
    import yaml

    repo = Path(__file__).resolve().parent.parent
    profile = Profile(**yaml.safe_load((repo / "profiles" / profile_file).read_text()))
    adapter = get_adapter(profile)
    assert isinstance(adapter, cls) and adapter.name == name
    if isinstance(adapter, ClaudeSDKAdapter):
        opts = adapter.build_options(QA_TASK, tmp_path, profile)
        assert opts.model == profile.model and opts.cwd == str(tmp_path)
    else:
        cmd = adapter.build_command(QA_TASK, tmp_path, profile)
        assert cmd[0] == adapter.cli and all(isinstance(a, str) for a in cmd)
        if name == "claude_code":
            clean = profile.params["clean_slate"]
            assert ("--strict-mcp-config" in cmd) is clean
            assert "--verbose" in cmd and "--model" in cmd
