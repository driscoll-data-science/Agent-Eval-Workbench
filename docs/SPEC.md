# Agent Eval Bench — Design Spec (v1)

Status: CONFIRMED by J on 2026-09-17 ("looks good on paper, let's build"). Authoritative for the
build. Deviations forced by reality are listed at the end under "Deviations".

## Purpose

A generic, reusable evaluation kit built on MLflow for scoring AI agents
against golden task sets. Five components: golden-task runner, LLM-as-judge
with a human-labeled calibration sample and tracked agreement rate,
token-cost accounting, baseline comparison, and a drift detector with a
tool-call-distribution monitor.

Primary subjects: agents J builds next (Claude Agent SDK, Python). Reference
subjects: Claude Code CLI and OpenAI Codex CLI. Results are private. The
project is a public portfolio repo with a passing-tests badge.

## Decisions already made (do not revisit)

### Scope and audience
- Consumers: J, and later agents tasked with improving an agent under test.
- Results private. Project public under MIT. CI badge must be green. CI never
  touches the network or credentials.
- v1 runs locally on J's Mac, triggered manually. Portable by design (see
  Runtime, hosting, portability).

### Tasks and suites
- One schema, two task kinds: `qa` (prompt in, answer out, judge-graded) and
  `workspace` (fixture directory copied to a temp dir, end-state graded, judge
  for taste dimensions only).
- Golden entries are YAML files, one directory per suite. Git is the source of
  truth (private suites under AEB_HOME, the example suite in the repo). Each
  run registers the suite into an MLflow evaluation dataset for lineage.
- Authoring: agent-drafted, J-curated. Target 50 cases per suite, tagged, with
  a mix of clear positives and known-hard negatives. 2 reps per run.
- First suites are general-domain because J's future agents do not exist yet:
  `qa` (about 30 reasoning, grounded-answer, instruction-following cases) and
  `workspace` (about 20 small Python repo fixes with hidden tests). The public
  example suite is a 10-case subset runnable end to end by the fake agent.
- Workspace checks declared per case: a command that must exit 0 (tests),
  paths that must exist, paths that must not change, and a no-op detector
  (workspace byte-identical after a claimed success fails the case). Hidden
  tests live outside the fixture and are copied in only at grading time.
- Isolation: temp directory plus the agent's own sandbox (Claude Code
  permission mode, Codex sandbox mode). No containers in v1.

### Agents and adapters
- Python adapter protocol: run(task, workspace) returns a normalized run
  record: final text, ordered model calls with usage, ordered tool calls with
  name, input, and output summary, wall time, exit status, model. Every
  adapter also writes an MLflow trace with spans so the trace UI and MLflow's
  tool-call judges work identically for all agents.
- v1 adapters: Claude Agent SDK (in-process), Claude Code CLI (subprocess,
  `--output-format stream-json`), Codex CLI (subprocess, `--json` JSONL), and
  FakeAgent (deterministic, for CI). A generic subprocess contract for
  non-Python agents is deferred.
- An agent configuration is a named YAML profile recorded as MLflow run
  params. Claude Code knobs: model, clean vs full config, appended instruction
  file, tool allowlist, permission mode, effort, max turns, budget cap. Codex
  knobs: model, sandbox mode, AGENTS.md on or off, MCP on or off, reasoning
  effort.
- Clean slate is the default for evaluated runs. Locally for Claude Code:
  `--strict-mcp-config --setting-sources ""`, which keeps the keychain login
  and was verified to drop CLAUDE.md, plugins, and MCP tools. Remote or CI:
  `--bare` plus `CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token` (not
  generated until remote mode is built). Codex: throwaway `CODEX_HOME`,
  `--ephemeral`, `-c project_doc_max_bytes=0`, MCP disabled via override.
- First comparison matrix: model by clean-vs-full config.

### Judge
- Per-criterion binary pass/fail with a rationale, one structured-output
  judge call per criterion. Aggregated to per-criterion pass rates and an
  overall rate.
- Judge model `claude-opus-5`, pinned and recorded per run. The report warns
  when judge family equals agent family. No jury in v1.
- Two backends behind one interface, selected per run. Default: Claude Code
  headless on J's subscription using the stripped recipe, `--system-prompt`
  carrying the judge prompt, `--tools ""`, `--max-turns 1`, `--json-schema`.
  Measured: about 1.2K prompt tokens and about 3.7 s of process overhead per
  verdict; the verdict arrives in `structured_output`. Optional: MLflow
  `make_judge` with `anthropic:/claude-opus-5`, which needs an API key and
  unlocks `align()`.
- Judge prompt rules: criteria as concrete checkable claims; candidate text is
  untrusted data; explicit instruction against rewarding length; known
  negatives (empty string, "I don't know", confident wrong answer) must fail.
- Default criteria. qa: correct, grounded, follows_format, concise. workspace
  programmatic: tests_pass, only_allowed_paths, not_noop. workspace judge:
  minimal_diff, readable, explanation_accurate. Per-suite overrides allowed.

### Calibration and agreement
- 40 cases per suite, stratified random by tag, labeled per criterion by J.
  Re-labeled whenever the rubric or judge-prompt version changes.
- Labels: primary via the MLflow UI assessments panel and review queues on the
  trace, exported by `aeb labels pull` into private YAML (source of truth).
  Secondary: a terminal labeling loop writing the same YAML.
- Agreement is per-criterion raw agreement plus Cohen's kappa, computed by the
  kit, logged per judge-prompt version in a dedicated MLflow experiment.
  Flags: raw below 0.85 or kappa below 0.6. When the judge version changes,
  calibration re-judges the same frozen outputs so agreement compares like
  with like.

### Cost
- Tokens come from each agent's API-reported usage per call, including cache
  read and write, surfaced by the adapter. Never estimated from text.
- USD is computed by the kit from a versioned price table (YAML with effective
  dates, Claude and OpenAI models). List price is canonical for every agent
  regardless of billing, subscriptions included. Judge cost is recorded
  separately. MLflow's LiteLLM-based cost is a cross-check only.

### Baseline and comparison (signed)
- A baseline is an explicitly promoted run per agent profile and suite, via
  `aeb baseline promote <run>` or `--promote` on a run. Never automatic. The
  report says "no baseline set" until one is promoted.
- Comparison is paired per case with a bootstrap 95% confidence interval on
  the delta for each criterion pass rate, cost, and latency. Verdict per
  metric: regression (interval below zero), improvement (above zero),
  otherwise no significant change. Practical-significance floor configurable,
  default zero.
- Notification: dashboard flag only. No push, email, or chat.

### Drift (unsigned)
- Behavioral distributions versus the baseline run: tool-call mix, calls per
  task, sequence shape, tokens, latency, output length, refusal rate. Verdict
  is changed or stable. Drift never triggers a regression flag on its own.
- Tool naming: raw names per agent plus an optional per-agent mapping to
  normalized categories (read, write, exec, search, web, other).
- The report shows the two-by-two of quality verdict against behavior verdict.

### Reports and dashboard
- Static `report.html` and `report.json` per run plus an `index.html`, written
  under AEB_HOME/reports and attached to the MLflow run as artifacts.
  `aeb report open` opens the index. The MLflow UI remains for trace-level
  drilling.
- Report sections, in order: verdict grid; per-criterion pass rates with CIs
  against baseline; cost and latency, absolute first and relative second;
  drift panel with tool-call histograms and divergence; calibration status
  (agreement, kappa, judge version, same-family warning); profile params;
  failures with judge rationale and MLflow trace links.
- Improvement agents consume `report.json`, `report.md`, and
  `aeb compare --json`. An MCP server is deferred.

### Runtime, hosting, portability
- `aeb mlflow up` and `aeb mlflow down` manage a local `mlflow server` with
  SQLite and an artifact directory on port 5050. No always-on service in v1.
- `AEB_HOME` (default `~/.aeb`) holds private suites, labels, reports,
  mlflow.db, and artifacts.
- Portability contract: every scheduler-facing action is a non-interactive
  CLI subcommand (run, judge, compare, drift, report, baseline, labels,
  mlflow); one config file `aeb.toml` with environment overrides (AEB_HOME,
  MLFLOW_TRACKING_URI); credentials only from the environment or keychain,
  never from config; no macOS-specific paths; a documented remote mode
  (remote tracking URI, any scheduler, `--bare` plus setup token).
- Scheduling: manual only in v1.

### Repo, CI, license
- Public: kit code, the 10-case example suite, example calibration labels,
  price table, CI. Private and never committed: real suites, labels, results,
  MLflow data, fixtures beyond the example.
- CI on GitHub Actions: ruff, pytest, and an end-to-end run with FakeAgent, a
  fake judge, and an ephemeral SQLite MLflow. No network, no keys. Badge in
  README.
- License: MIT.

## Decisions delegated to Claude (chosen and logged)
- Python 3.13 via uv. Resolution verified 2026-09-17: mlflow 3.16.1,
  claude-agent-sdk 0.2.154, anthropic 1.6.0, typer 0.27.2. ruff and pytest.
  LiteLLM added explicitly for MLflow's cost table.
- Package `agent_eval_bench`, CLI `aeb` (typer). Layout:
  `src/agent_eval_bench/{adapters,judge,runner,compare,drift,report,cli}`,
  `suites/example`, `tests`, `docs`.
- MLflow port 5050 locally (5000 is held by macOS Control Center). One MLflow
  experiment per suite; runs tagged with profile, agent, and judge version.
- First-matrix models: `claude-opus-5` and `claude-sonnet-5`. Codex model:
  account default, recorded from run output.
- Criteria defined per suite with per-case overrides. Judge-prompt version is
  a content hash of template plus criteria.
- Distribution shift measured by Jensen-Shannon divergence plus a chi-square
  p-value, thresholds configurable.
- Defaults: 2 reps, concurrency 4, judge determinism measured by re-judging a
  10 percent sample twice per run.
- Example suite: 8 qa plus 2 workspace cases. A minimal reference Agent SDK
  agent ships so the SDK adapter path is exercised.
- Suite drafting: `aeb suite draft` writes candidates for J to edit;
  `aeb suite validate` checks the schema.
- README quickstart runs end to end with FakeAgent and no credentials, with a
  Mermaid architecture diagram.
- Commits: local commits at milestones; push only when J says so.

## Out of scope (v1)
- Scheduled runs, remote hosting on cortex, setup-token generation,
  notifications of any kind.
- Container isolation for workspace tasks.
- Generic subprocess adapter for non-Python agents; MCP server for results.
- Jury judges, pairwise judging, MLflow online monitoring.
- Synthetic bulk generation of golden cases; J's domain-specific private
  suite (deferred until the future agents exist).

## Acceptance (what J will look at)
1. `aeb mlflow up`, then `aeb run --suite example --profile fake` completes
   with no credentials. The MLflow UI at localhost:5050 shows the run, traces
   with tool spans, and the registered dataset.
2. `aeb report open` shows a report with every section, "no baseline set",
   and per-criterion pass rates.
3. A real run of Claude Code (clean slate, `claude-sonnet-5`) on the qa suite
   via subscription, judged via subscription, with the cost column populated
   at list price and the process visible in the MLflow trace.
4. Promote that run, run again: the report shows signed verdicts with CIs and
   an unsigned drift panel with tool-call histograms.
5. Label 40 cases in the MLflow UI, `aeb labels pull`: the report shows
   agreement and kappa per criterion and flags anything below threshold.
6. The Codex adapter passes fixture tests; a live Codex run works after
   `codex login`.
7. GitHub Actions is green, the badge is in the README, and `git status`
   shows no private data tracked.

## J's actions at implementation time
- Install Codex CLI (`npm i -g @openai/codex`) and run `codex login` on the
  ChatGPT subscription when ready for live Codex runs.
- Review and curate the agent-drafted suites and criteria.
- Label the 40 calibration cases per suite.
- Say when to push to GitHub.

## Facts verified this session (2026-09-17)
- Mac: Python 3.14.6 system, uv 0.11.2, Docker 29, Claude Code 2.1.274
  logged in via claude.ai subscription, gh authenticated, no API keys in env,
  no MLflow, no Codex CLI, port 5000 occupied by Control Center.
- cortex: reachable, Ubuntu 26.04, 2 vCPU, 3.7 GiB, 33 GB free, uv with
  Python 3.12 and 3.13, no Docker, no MLflow, MLflow ports free, twelve
  cortex systemd timers. Not a v1 target.
- mac-studio: offline, last seen 14 days ago.
- MLflow 3.16.1: make_judge, judge.align (needs provider API key), evaluation
  datasets (SQL backend required), anthropic and openai autolog with token and
  cost capture, no built-in drift detection, no built-in agreement metric.
- Claude Code headless: `--bare` and an empty `CLAUDE_CONFIG_DIR` both lose
  the keychain login. `--strict-mcp-config --setting-sources ""` keeps it and
  drops CLAUDE.md, plugins, and MCP (about 6.5K prompt tokens; about 450 with
  a custom system prompt versus 62K for the default config). Structured output
  lands in `structured_output` and costs one extra internal turn. `--effort`
  exists with five levels.
- Codex CLI 0.154.0: `codex exec --json` emits JSONL with usage but no USD;
  isolation via CODEX_HOME, `--ephemeral`, `project_doc_max_bytes=0`; exit 1
  on failure; `--full-auto` rejected.

## Facts established during the build (2026-09-17)
- Claude Code `--output-format stream-json` in `-p` mode requires `--verbose`; without it the
  CLI exits 1 with an error on stderr. The adapter always adds it.
- A clean-slate Claude Code run with tools enabled costs about 29K prompt tokens per case
  (system prompt plus tool schemas), most of it cache reads shared across concurrent cases.
  Measured: example suite, 20 cases, Sonnet 5: $0.75 agent + $0.06 judge (Opus 5) list price,
  83 s wall at concurrency 4.
- The Opus 5 judge fails all known negatives (empty, non-answer, wrong question, grader
  injection) and costs about $0.002 to $0.004 per verdict.
- MLflow drops a whole trace if any span attribute value is None where a float is expected;
  the exporter logs it only at DEBUG level. Cost attributes must be numeric.
- Setting MLFLOW_ENABLE_ASYNC_TRACE_LOGGING=false left traces "not fully exported" and
  unreadable in 3.16.1; async export (default) plus an explicit flush between phases works.
- The claude.ai subscription's 5-hour window was exhausted once by five parallel Opus
  workers plus drafting; jobs failed with "You've hit your session limit" and recovered after
  the reset. Parallel Opus work on this account must be paced.
- Codex `--json` never reports the model id, so cost needs the model pinned in the profile.

## Deviations from the confirmed design
- Codex profile pins `model: gpt-5.6` instead of "account default, recorded from run output"
  (not recoverable from the JSONL).
- LiteLLM is an optional extra (`aeb[crosscheck]`) rather than a hard dependency; MLflow's
  own cost table is only a cross-check.
- ruff E501 is ignored; the formatter enforces line length on code, long prompt strings
  are left readable.
