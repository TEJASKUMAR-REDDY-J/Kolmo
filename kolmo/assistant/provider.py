"""Chat providers for the research assistant. Both speak one duck-typed interface:

    provider.name -> str (shown in the UI: "offline · rule-based" or "live · <model>")
    provider.chat(messages, tools) -> {"content": str, "tool_calls": [{"id", "name", "arguments": dict}]}

messages / tools use the OpenAI chat-completions format. A live provider is used ONLY when F:/Kolmo/.env (or the
environment) sets KOLMO_LLM_LIVE=1 and GROQ_API_KEY; everything else (tests, development) runs the deterministic
FakeProvider, which plans from fixed rules and explains only numbers it read from tool results."""
import json
import os
import re
import urllib.request

ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")


def load_env(path=ENV_PATH):
    """KEY=VALUE lines; '#' comments and blanks skipped; surrounding quotes stripped. Process env wins."""
    out = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                k, sep, v = line.strip().partition("=")
                if sep and not k.startswith("#"):
                    out[k.strip()] = v.strip().strip("'\"")
    return {**out, **{k: v for k, v in os.environ.items() if k in out or k.startswith(("KOLMO_", "GROQ_"))}}


class OpenAIChatProvider:
    def __init__(self, base_url, model, key, timeout=120):
        self.url, self.model, self.key, self.timeout = base_url.rstrip("/") + "/chat/completions", model, key, timeout
        self.name = f"live · {model}"

    def chat(self, messages, tools):
        body = {"model": self.model, "messages": messages, "tools": tools, "tool_choice": "auto", "temperature": 0}
        req = urllib.request.Request(self.url, json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            msg = json.load(r)["choices"][0]["message"]
        calls = [{"id": c["id"], "name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"] or "{}")}
                 for c in msg.get("tool_calls") or []]
        return {"content": msg.get("content") or "", "tool_calls": calls}


def get_provider(env=None):
    e = load_env() if env is None else env
    if e.get("KOLMO_LLM_LIVE") == "1" and e.get("GROQ_API_KEY"):
        return OpenAIChatProvider(e.get("KOLMO_LLM_BASE_URL", "https://api.groq.com/openai/v1"),
                                  e.get("KOLMO_LLM_MODEL", "meta-models/muse-spark-open"), e["GROQ_API_KEY"])
    return FakeProvider()


# ----------------------------------------------------------------------------------------------- offline provider
SYSTEM_WORDS = {"diffusion": ("diffusion", "heat", "temperature"),
                "fluid2d": ("fluid", "kolmogorov", "vortic", "turbul", "navier", "flow")}
NUM = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"


def parse_question(q):
    """Rule-based reading of a question -> (system or None, params, options)."""
    t = q.lower()
    system = next((s for s, ws in SYSTEM_WORDS.items() if any(w in t for w in ws)), None)
    params = {k: float(v) for k, v in re.findall(r"\b(kappa|nu|re|mu|dt|a|kf)\s*(?:=|of|is)\s*" + NUM, t)}
    params = {("Re" if k == "re" else "A" if k == "a" else k): v for k, v in params.items()}
    m = re.search(r"\b(?:n|grid)\s*(?:=|of|is)?\s*(\d+)", t)
    if m:
        params["n"] = int(m.group(1))
    opts = {"kind": "fno" if "fno" in t or "fourier" in t else "cnn",
            "compare": any(w in t for w in ("rollout", "unroll", "compare", "versus", " vs")),
            "train_steps": int(m.group(1)) if (m := re.search(r"(\d+)\s*(?:training|train|optimi[sz]er)\s*steps", t)) else 200}
    return system, params, opts


class FakeProvider:
    """Deterministic scripted 'model': list_systems -> validate_config -> estimate_cost -> run_experiment ->
    read_results -> diagnose -> explanation. Asks for clarification when the question names no known system."""
    name = "offline · rule-based"

    def chat(self, messages, tools):
        q = next(m["content"] for m in messages if m["role"] == "user")
        done = {}  # tool name -> last parsed result
        names = {}
        for m in messages:
            for c in m.get("tool_calls") or []:
                names[c["id"]] = c["function"]["name"]
            if m["role"] == "tool":
                done[names[m["tool_call_id"]]] = json.loads(m["content"])
        system, params, o = parse_question(q)
        call = lambda name, **a: {"content": "", "tool_calls": [{"id": f"c{len(names)}", "name": name, "arguments": a}]}
        if "list_systems" not in done:
            return call("list_systems")
        if system is None:
            return {"content": "Which system should I simulate? Known systems: " + ", ".join(done["list_systems"]) +
                    ". Please name one (and any parameter values you care about).", "tool_calls": []}
        cfg = self.config(q, system, params, o)
        v = done.get("validate_config")
        if v is None:
            return call("validate_config", config=cfg)
        if not v["ok"]:
            return {"content": "The planned experiment is not valid: " + v["error"] +
                    "\nPlease adjust the question (e.g. a smaller grid or fewer steps).", "tool_calls": []}
        if "estimate_cost" not in done:
            return call("estimate_cost", config=cfg)
        r = done.get("run_experiment")
        if r is None:
            return call("run_experiment", config=cfg)
        if "run_dir" not in r:
            return {"content": r.get("error", "The run did not happen."), "tool_calls": []}
        if "read_results" not in done:
            return call("read_results", run_dir=r["run_dir"])
        if "diagnose" not in done:
            return call("diagnose", run_dir=r["run_dir"])
        return {"content": self.explain(q, cfg, done), "tool_calls": []}

    @staticmethod
    def config(q, system, params, o):
        n = params.get("n", 32)
        p = {**params, "n": n}
        model = {"kind": o["kind"], "width": 8, "depth": 2, **({"modes": min(8, n // 4)} if o["kind"] == "fno" else {})}
        tc = {"model": model, "steps": o["train_steps"], "batch": 4, "eval_every": max(10, o["train_steps"] // 4),
              "val_horizon": 10}
        models = {"one_step": tc, **({"rollout": {**tc, "unroll": 3}} if o["compare"] else {})}
        slug = re.sub(r"[^a-z0-9]+", "-", q.lower()).strip("-")[:32] or "question"
        return {"name": f"assistant-{slug}", "data": {"system": system, "params": p, "steps": 20, "spacing": 2,
                "train_seeds": [0, 1, 2, 3], "val_seeds": [100], "test_seeds": [200, 201]},
                "models": models, "eval_steps": 20, "threshold": 0.5}

    @staticmethod
    def explain(q, cfg, done):
        est, run, res, diag = done["estimate_cost"], done["run_experiment"], done["read_results"], done["diagnose"]
        models = ", ".join(f"{k} ({v['model']['kind']}, unroll {v.get('unroll', 1)}, {v['steps']} training steps)"
                           for k, v in cfg["models"].items())
        d = cfg["data"]
        rows = [f"- {r['model']} / {r['split']}: " + ", ".join(f"{k} {v}" for k, v in r.items() if k not in ("model", "split"))
                for r in res["table"]]
        rows += [f"- {m}: reference {t['reference_s_per_step']} s/step, surrogate {t['surrogate_s_per_step']} s/step, "
                 f"speedup {t['speedup']}x" for m, t in res["runtime"].items()]
        interp = [f"- [{f['level']}] {f['interpretation']}" for f in diag["findings"]] or ["- No rule fired."]
        nxt = list(dict.fromkeys(f["next"] for f in diag["findings"] if f.get("next")))
        return "\n".join([
            "## Hypothesis", f"A learned surrogate of {d['system']} can reproduce the reference solver on held-out seeds "
            f"for the horizon asked about (question: \"{q}\").", "",
            "## Planned", f"- system {d['system']}, params {json.dumps(d['params'])}, {d['steps']} recorded steps "
            f"(spacing {d['spacing']}), test seeds {d['test_seeds']}", f"- models: {models}",
            f"- estimated cost {est['seconds']} s", "",
            "## Ran", f"- run directory {run['run_dir']}, wall time {run['seconds']} s", "",
            "## Measured", *rows, *[f"- {f['fact']}" for f in diag["findings"]], "",
            "## Interpretation", *interp, "",
            "## Open questions", *([f"- {x}" for x in nxt] or ["- None suggested by the rules."])])
