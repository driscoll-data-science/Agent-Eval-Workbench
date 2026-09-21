from agent_eval_bench.judge.fake import FakeJudge
from agent_eval_bench.judge.selftest import run_selftest


def test_fake_judge_fails_all_known_negatives():
    res = run_selftest(FakeJudge(), log=lambda _: None)
    assert res.ok, [(r.negative, r.criterion) for r in res.failures]
    assert len(res.rows) == 8


def test_cli_selftest(settings):
    from typer.testing import CliRunner

    from agent_eval_bench.cli import app

    r = CliRunner().invoke(app, ["judge-selftest", "--backend", "fake"])
    assert r.exit_code == 0 and "self-test OK" in r.output
