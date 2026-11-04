"""Autonomous research assistant with the offline FakeProvider (no network): diffusion and a composed Cahn-Hilliard PDE
end to end on tiny grids, events per docs/contracts.md, recovery, budgets, number audit."""
import os

import pytest

from kolmo.assistant import agent, tools
from kolmo.assistant.provider import FakeProvider, get_provider, parse_question

FAST = " Use 0.1 min per model and 12 steps."


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    return str(tmp_path_factory.mktemp("runs"))


def go(q, runs, provider=None, **kw):
    ev = []
    r = agent.run(q, on_event=ev.append, provider=provider or FakeProvider(), out_dir=runs, **kw)
    return r, ev


def test_parse_and_offline_by_default():
    a = parse_question("Train nca and FNO on Cahn-Hilliard phase separation, grid of 16, 0.5 min")
    assert a["system"] == "cahn_hilliard" and a["pde"] and a["kinds"] == ["nca_hybrid", "fno"] and a["n"] == 16
    assert a["minutes"] == 0.5
    assert get_provider({"KOLMO_LLM_LIVE": "1"}).name == "offline · rule-based"  # no key -> offline
    assert get_provider({"KOLMO_LLM_LIVE": "1", "GROQ_API_KEY": "x", "KOLMO_LLM_MODEL": "m"}).name == "live · m"


def test_clarify_when_no_system(runs):
    r, ev = go("How fast does the error grow?", runs)
    assert "cahn_hilliard" in r["text"] and ev[-1] == {"type": "done"}
    assert not any(e["type"] == "tool" and e["name"] == "train" for e in ev)


def test_diffusion_end_to_end_events_report(runs):
    r, ev = go("How reliable is a CNN surrogate of heat diffusion on a grid of 16?" + FAST, runs)
    assert r["status"] == "done", r["text"]
    types = [e["type"] for e in ev]
    assert types[-1] == "done" and "plan" in types and "progress" in types and "result" in types
    assert [e["on"] for e in ev if e["type"] == "zen"] == [True, False]
    assert all(e["ok"] for e in ev if e["type"] == "tool"), [e for e in ev if e["type"] == "tool" and not e["ok"]]
    for s in agent.SECTIONS:
        assert f"## {s}" in r["text"]
    assert r["removed"] == [], r["removed"]
    res = next(e for e in ev if e["type"] == "result")
    assert "cnn" in res["metrics"] and res["figures"] and os.path.exists(res["report"])
    assert os.path.exists(os.path.join(r["dir"], "transcript.json")) and os.path.exists(os.path.join(r["dir"], "results.json"))


def test_cahn_hilliard_nca_vs_fno_and_registry(runs):
    r, ev = go("Lightweight NCA surrogate for Cahn-Hilliard phase separation vs FNO, grid of 16, compare long-horizon "
               "reliability." + FAST, runs)
    assert r["status"] == "done", r["text"]
    assert {"nca_hybrid", "fno"} <= set(next(e for e in ev if e["type"] == "result")["metrics"])
    t = tools.Task(r["dir"], runs, lambda e: None, 60)  # registry ops on that task's checkpoints
    assert t.call("registry_op", {"op": "register", "model": "fno"})["ok"]
    assert any(e["name"] == "fno" for e in t.call("registry_op", {"op": "list"})["entries"])


def test_compose_pde_errors_come_back(runs):
    t = tools.Task(os.path.join(runs, "pde"), runs, lambda e: None, 60)
    bad = t.call("compose_pde", {"spec": {"name": "x", "fields": ["u"], "rhs": {"u": "__import__('os')"}, "n": 16}})
    assert not bad["ok"] and bad["error"]
    assert t.call("compose_pde", {"spec": "allen_cahn"})["ok"]


def test_budget_and_audit(runs):
    class Liar(FakeProvider):
        def chat(self, messages, tools_):
            r = super().chat(messages, tools_)
            return {**r, "content": r["content"] + "\nIt is 123.456x faster."} if not r["tool_calls"] else r
    r, _ = go("Heat diffusion CNN on a grid of 16." + FAST, runs, provider=Liar())
    assert "123.456" in r["removed"] and "123.456" not in r["text"]
    r, ev = go("Heat diffusion CNN on a grid of 16." + FAST, runs, max_calls=3)
    assert r["status"] == "budget" and len([e for e in ev if e["type"] == "tool" and e["ok"]]) == 3


def test_recovery_halves_dt(runs, monkeypatch):
    from kolmo.surrogates import data
    real, dts = data.build, []

    def flaky(cfg, *a, **k):
        dts.append(cfg.params.get("dt"))
        if len(dts) == 1:
            raise ValueError("unstable: linear growth * dt too large")
        return real(cfg, *a, **k)
    monkeypatch.setattr(data, "build", flaky)
    t = tools.Task(os.path.join(runs, "rec"), runs, lambda e: None, 120)
    r = t.call("generate_data", {"system": "diffusion", "params": {"n": 16}, "steps": 6, "n_train": 2, "n_val": 1, "n_test": 1})
    assert r["ok"] and r["recovered"] and dts[1] == pytest.approx(0.025), (r, dts)
