"""Research assistant end to end with the offline FakeProvider on the diffusion system (no network, ~1 min)."""
import json
import re

import pytest

from kolmo.assistant import agent, tools
from kolmo.assistant.provider import FakeProvider, get_provider, parse_question

Q = "How well does a CNN surrogate of heat diffusion with kappa = 0.2 on a grid of 16 hold up? Use 40 training steps."


def test_parse_and_offline_by_default():
    s, p, o = parse_question(Q)
    assert s == "diffusion" and p == {"kappa": 0.2, "n": 16} and o["train_steps"] == 40 and o["kind"] == "cnn"
    assert get_provider({"KOLMO_LLM_LIVE": "1"}).name == "offline · rule-based"  # no key -> offline
    assert get_provider({"KOLMO_LLM_LIVE": "0", "GROQ_API_KEY": "x"}).name == "offline · rule-based"
    assert get_provider({"KOLMO_LLM_LIVE": "1", "GROQ_API_KEY": "x", "KOLMO_LLM_MODEL": "m"}).name == "live · m"


def test_tools_validate_limits_and_confirmation(tmp_path):
    cfg = FakeProvider.config(Q, "diffusion", {"n": 16}, {"kind": "cnn", "compare": False, "train_steps": 40})
    assert tools.validate_config(cfg, str(tmp_path))["ok"]
    big = json.loads(json.dumps(cfg)); big["data"]["params"]["n"] = 512
    assert "max_grid" in tools.validate_config(big)["error"]
    bad = json.loads(json.dumps(cfg)); bad["models"]["one_step"]["bogus"] = 1
    assert not tools.validate_config(bad)["ok"]
    costly = json.loads(json.dumps(cfg)); costly["models"]["one_step"]["steps"] = 4000; costly["data"]["params"]["n"] = 64
    e = tools.estimate_cost(costly)
    assert e["needs_confirmation"] and e["seconds"] > tools.LIMITS["confirm_seconds"]
    r = tools.call("run_experiment", {"config": costly, "confirmed": True}, str(tmp_path))  # model can't self-confirm
    assert r["needs_confirmation"] and not list(tmp_path.iterdir())


def test_clarification_when_system_missing(tmp_path):
    r = agent.Agent(FakeProvider(), str(tmp_path)).ask("How fast does the error grow?", log=lambda *_: None)
    assert r["status"] == "clarify" and "diffusion" in r["text"] and r["run_dir"] is None


@pytest.fixture(scope="module")
def runs(tmp_path_factory):  # shared: the second run reuses the data cache and checkpoints
    return str(tmp_path_factory.mktemp("runs"))


def test_end_to_end_diffusion(runs):
    r = agent.Agent(FakeProvider(), runs).ask(Q, log=lambda *_: None)
    assert r["status"] == "done", r["text"]
    assert [t["tool"] for t in r["trace"]] == ["list_systems", "validate_config", "estimate_cost", "run_experiment",
                                               "read_results", "diagnose"]
    for s in agent.SECTIONS:
        assert f"## {s}" in r["text"]
    assert r["removed"] == [], r["removed"]  # the fake only quotes tool numbers
    text = open(r["report"], encoding="utf-8").read()
    assert "test" in text and re.search(r"relL2@\d+ \d", text)


def test_audit_strips_invented_numbers(runs):
    class Liar(FakeProvider):
        def chat(self, messages, tools_):
            r = super().chat(messages, tools_)
            return {**r, "content": r["content"] + "\nThe surrogate is 123.456x faster."} if not r["tool_calls"] else r
    r = agent.Agent(Liar(), runs).ask(Q, log=lambda *_: None)
    assert "123.456" not in r["text"] and "123.456" in r["removed"] and "[unverified]" in r["text"]
