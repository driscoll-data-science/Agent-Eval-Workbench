# STATE — agent_eval_bench
Updated: 2026-09-20 (README dress-up; first push to GitHub)

## Works (verified by running it, not by reading docs)
- `aeb mlflow up` starts a local MLflow 3.16.1 server on http://127.0.0.1:5050 (SQLite at
  ~/.aeb/mlflow.db, artifacts at ~/.aeb/mlartifacts). Verified healthy; pid file at ~/.aeb/mlflow.pid.
- `aeb run --suite example --profile fake --judge-backend fake`: 10 cases, 42 verdicts, 10 traces
  with AGENT/LLM/TOOL spans, judge verdicts attached to traces as LLM_JUDGE feedback, dataset
  registered and linked to the run. Seen in the MLflow UI (run page and traces tab).
- Real runs on the subscription, no API key:
  - claude_code_clean_sonnet on the example suite, 2 reps: 20 cases, pass 1.000, agent cost
    $0.75 list, judge (Opus 5) $0.06, 83 s wall. Promoted as baseline (run 54ed1e53).
  - Rerun of the same profile: pass 0.920, compare = no_change (paired bootstrap CIs include
    zero at n=20), drift = stable, 100 s wall. Report shows verdict grid, per-criterion bars
    with baseline markers and CIs, cost/latency absolute then relative, tool-category
    histogram, three failures with judge rationale and MLflow trace links.
  - Real finding surfaced by the bench: Sonnet 5 answers the leap-year case with
    "Yes ... No, actually not" on 3 of 3 attempts; judge fails concise and follows_format.
- `aeb judge-selftest --backend claude_code --model claude-opus-5`: all 8 known negatives
  (empty, non-answer, wrong question, grader injection) correctly FAIL.
- `aeb compare` / `aeb drift` against an explicit baseline: fake vs fake_regressed shows
  regression on correct/grounded/follows_format and drift on output length; tool mix stable.
- `aeb labels sample|tui|pull` and `aeb calibrate`: exercised end to end against the live
  server with throwaway labels (then deleted). Flags fire below 0.85 raw / 0.60 kappa.
- `aeb suite draft`: drafted 30 qa cases (one Opus call, 13.8K output tokens) and 20
  workspace cases in four batches, each validated mechanically (fixture fails hidden tests,
  reference patch passes). Merged into the private suite ~/.aeb/suites/general (50 tasks,
  validates). UNCURATED: J has not reviewed it yet.
- Uncurated private suite `general` (50 tasks) with claude_code_clean_sonnet, 1 rep: pass 0.954,
  agent cost $2.30 list, judge $0.42, 4.2M tokens (mostly cache reads), p95 wall 21 s.
  Workspace end-state checks 20/20; misses are on qa correct/grounded (27/30) and
  explanation_accurate (17/20). Report: ~/.aeb/reports/6106dfe3.../report.html (experiment aeb/general).
- Adversarial review of compare/drift/calibration (1 high, 2 medium, 5 low) applied: unknown
  agent cost now yields "insufficient" instead of $0; Cohen's kappa is 0 (not None) when only
  one rater is constant so a rubber-stamp judge is flagged; the re-judge test can now fail;
  Expectations are not counted as human labels; proportion drift needs a 0.10 rate move;
  the judge version hash now includes the judge model, so swapping models forces re-calibration.
- Tests: `uv run pytest -q` -> 100 passed (offline: fake agent, fake judge, SQLite store).
  `uv run ruff check src tests` -> clean. CI workflow at .github/workflows/ci.yml mirrors this.

## Broken / degraded
- Remote is https://github.com/driscoll-data-science/Agent-Eval-Workbench (public). First push
  of the kit happened 2026-09-20; CI status and badge are checked after that push (see below).
- `claude -p --bare` and an empty CLAUDE_CONFIG_DIR both lose the keychain login. Local runs
  use `--strict-mcp-config --setting-sources ""` instead; remote mode needs a setup token.
- Subscription capacity is real: the 5-hour window was exhausted once during the build
  (five parallel Opus workers plus drafting). Three drafting batches failed with
  "You've hit your session limit" and were re-run after the reset. Expect this if many
  Opus jobs run in parallel; the kit itself is modest (a 20-case run is ~$0.80 list price).
- Codex adapter is tested against a synthetic JSONL fixture only. Codex CLI is not installed
  and no `codex login` has been done. Codex never reports its model in JSONL, so the profile
  pins `gpt-5.6` (spec deviation: "recorded from run output" is not possible).
- Report index table is wider than a 1200px viewport (horizontal scroll); acceptable.
- Reports are static files; the Playwright check served them via a throwaway
  `python -m http.server 8765` on ~/.aeb/reports (not part of the kit; `aeb report open`
  uses the browser's file:// handling).

## Where things live
- README gallery images: docs/images/*.png, generated from the fake demo (never real-agent
  results). Demo data home for regenerating them: <repo>/.aeb (gitignored) with its own MLflow
  instance on :5051 (`AEB_HOME=$PWD/.aeb uv run aeb mlflow up`); runs d2d731a7 (baseline,
  fake) and dff5dad4 (fake_regressed) plus 38 synthetic demo labels.
- Repo: /Users/jonathandriscoll/agent_eval_bench (public portfolio, MIT)
  - src/agent_eval_bench/  kit; cli.py is `aeb`
  - suites/example/        public 10-case suite; profiles/ agent configurations
  - docs/SPEC.md           authoritative design (confirmed by J 2026-09-17)
  - aeb.example.toml       config template; README.md quickstart + architecture
- Private data (never committed): ~/.aeb
  - suites/general/        50-task drafted suite (30 qa + 20 workspace), uncurated
  - suites/drafts/         raw drafts (qa_general, ws_part1..4)
  - runs/<run_id>/         results.jsonl, manifest.json, compare.json, drift.json, work/
  - reports/               index.html + <run_id>/report.{html,json,md}
  - labels/, calibration/  empty (rehearsal artifacts removed)
  - mlflow.db, mlartifacts/, mlflow.pid, mlflow-server.log
- MLflow: experiments aeb/example (5 runs), aeb/general (1 run); UI on :5050 while the server is up.

## Decisions made this session
- README rewritten as the portfolio front page: badges, feature list, gallery of the fake demo
  (report top, drift panel, failures, index, MLflow trace), CLI excerpts for compare/drift/
  calibrate, a suite-format example, architecture diagram, layout, config, testing.
- Report index no longer prints the absolute reports path (privacy in screenshots).
- Fake judge failure rationales now say why they failed instead of "Heuristic pass."
- All design decisions: docs/SPEC.md. Additional build-time decisions (inconsequential):
  - ruff E501 ignored; the formatter owns line length (long prompt strings stay readable).
  - Async trace export kept (MLflow default); the earlier lost trace was a None value in
    the span cost attribute, now fixed. Synchronous export made traces unreadable.
  - Hidden tests are copied into the workspace for grading and removed afterwards so the
    kept workspace reflects only the agent's changes.
  - `python` in check commands resolves to the kit's interpreter (PATH prepend) so hidden
    tests find pytest on any host.
  - Codex token semantics: cached_input_tokens is a subset of input_tokens and reasoning
    tokens a subset of output (OpenAI semantics); the adapter does not double-count.
  - Claude Code stream-json requires `--verbose` in -p mode; the adapter always adds it.
  - Judge effort defaults to "high"; judge runs with cwd ~/.aeb/judge_cwd so no project
    CLAUDE.md leaks in.
  - LiteLLM is an optional extra (`crosscheck`), not a hard dependency.
  - `aeb compare/drift --baseline` accept run-id prefixes and "latest".
  - Draft merging renames colliding task ids with a numeric suffix and copies fixtures under
    the new id; validation happens before anything is written.

## Open questions for J
1. Curate ~/.aeb/suites/general/suite.yaml (50 drafted tasks). The four workspace batches
   used the same topic prompt and overlap in theme (four slugify, four add-months, four
   config-parser, four money-rounding, four CLI-parser variants). Options: keep, prune to
   one per theme, or re-draft three batches with distinct topics (~3 Opus calls).
2. Label 40 calibration cases (`aeb labels sample <run>` then MLflow UI or `aeb labels tui`)
   so the judge gets an agreement rate. Until then reports say "judge not yet calibrated".
3. (done) Repo created and pushed; confirm the CI badge turns green on GitHub.
4. Codex: install (`npm i -g @openai/codex`) and `codex login` when you want live Codex runs.
5. The one `general` run is on the uncurated suite; treat its numbers as a preview until curated.
6. Known design caveat from review: drift calls behavior "changed" if any of nine shifts changes at alpha 0.05 with no multiplicity correction. Options: leave (sensitive, unsigned), Bonferroni, or raise the JS floor.
