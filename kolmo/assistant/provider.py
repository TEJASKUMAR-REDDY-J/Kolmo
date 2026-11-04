"""Chat providers for the research assistant. Both speak one duck-typed interface:

    provider.name -> str (shown in the UI: "offline · rule-based" or "live")
    provider.chat(messages, tools) -> {"content": str, "tool_calls": [{"id", "name", "arguments": dict}]}

messages / tools use the OpenAI chat-completions format. A live provider is used ONLY when F:/Kolmo/.env (or the
environment) sets KOLMO_LLM_LIVE=1 and GROQ_API_KEY; everything else (tests, development) runs the deterministic
FakeProvider, which plans from fixed rules and explains only numbers it read from tool results."""
import json
import os
import re
import time
import urllib.error
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


class RateLimited(RuntimeError):
    def __init__(self, wait_s, msg):
        super().__init__(msg)
        self.wait_s = wait_s


class OpenAIChatProvider:
    """OpenAI-compatible chat endpoint. Several API keys (GROQ_API_KEY, GROQ_API_KEY_2, ...) are used round-robin, so
    the per-key tokens-per-minute limit is shared out; a key that returns HTTP 429 cools down for the time the provider
    states and is skipped meanwhile. RateLimited is raised only when every key is cooling (wait = the shortest one).
    The model id is never shown in the UI (name is just "live")."""
    GUARD_MODEL = "meta-llama/llama-prompt-guard-2-86m"  # Meta Llama Prompt Guard 2: prompt-injection / jailbreak score

    def __init__(self, base_url, model, keys, timeout=120):
        keys = [keys] if isinstance(keys, str) else [k for k in keys if k]
        if not keys:
            raise ValueError("no API key")
        self.base, self.model, self.keys, self.timeout = base_url.rstrip("/"), model, keys, timeout
        self.url = self.base + "/chat/completions"
        self.cool = [0.0] * len(keys)  # monotonic time until which each key is rate limited
        self.next = 0
        self.name = "live"

    def _key(self):
        now = time.monotonic()
        for i in range(len(self.keys)):
            j = (self.next + i) % len(self.keys)
            if self.cool[j] <= now:
                self.next = j + 1
                return j
        raise RateLimited(min(self.cool) - now, f"all {len(self.keys)} API key(s) are rate limited")

    def _post(self, body, j):
        req = urllib.request.Request(self.url, json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {self.keys[j]}", "Content-Type": "application/json",
                                              "User-Agent": "kolmo/0.1"})  # the API's firewall rejects urllib's default agent (HTTP 403)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)["choices"][0]["message"]

    def _request(self, body):
        while True:  # rotate through keys until one answers or all are cooling (RateLimited from _key)
            j = self._key()
            try:
                return self._post(body, j)
            except urllib.error.HTTPError as e:  # surface the provider's reason (e.g. terms not accepted, rate limit)
                text = e.read().decode(errors="replace")
                if e.code != 429:
                    raise RuntimeError(f"LLM provider HTTP {e.code}: {text[:400]}") from None
                m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", text)
                wait = (60 * int(m.group(1) or 0) + float(m.group(2))) if m else float(e.headers.get("retry-after") or 30)
                self.cool[j] = time.monotonic() + wait + 0.5

    def chat(self, messages, tools):
        body = {"model": self.model, "messages": messages, "temperature": 0}
        if tools:  # "auto" + one call per turn: Groq gpt-oss is unreliable with "required" and parallel calls
            body.update(tools=tools, tool_choice="auto", parallel_tool_calls=False)
        if "gpt-oss" in self.model:
            body["reasoning_effort"] = "medium"
        msg = self._request(body)
        calls = [{"id": c["id"], "name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"] or "{}")}
                 for c in msg.get("tool_calls") or []]
        return {"content": msg.get("content") or "", "tool_calls": calls}

    def guard(self, text):
        """Llama Prompt Guard 2 probability that `text` is a prompt attack (0..1), or None if the check is unavailable
        (fail open: the shield never blocks a task because of a network error)."""
        try:
            return float(self._request({"model": self.GUARD_MODEL, "messages": [{"role": "user", "content": text[:2000]}]})
                         ["content"])
        except (RuntimeError, RateLimited, OSError, ValueError, KeyError):
            return None


def get_provider(env=None):
    e = load_env() if env is None else env
    keys = [e.get("GROQ_API_KEY")] + [e[k] for k in sorted(e) if re.fullmatch(r"GROQ_API_KEY_\d+", k)]
    keys = [k for k in keys if k]
    if e.get("KOLMO_LLM_LIVE") == "1" and keys:
        return OpenAIChatProvider(e.get("KOLMO_LLM_BASE_URL", "https://api.groq.com/openai/v1"),
                                  e.get("KOLMO_LLM_MODEL", "openai/gpt-oss-120b"), keys)
    return FakeProvider()


# ----------------------------------------------------------------------------------------------- offline provider
SYSTEM_WORDS = {"diffusion": ("diffusion", "heat"), "fluid2d": ("fluid", "kolmogorov", "vortic", "turbul", "navier"),
                "boussinesq": ("boussinesq", "convection", "rayleigh")}
PDE_WORDS = {"cahn_hilliard": ("cahn", "phase separation"), "allen_cahn": ("allen",), "gray_scott": ("gray", "scott"),
             "kuramoto_sivashinsky": ("kuramoto", "sivashinsky"), "swift_hohenberg": ("swift", "hohenberg"),
             "burgers": ("burgers",)}
KIND_WORDS = (("nca_hybrid", ("nca",)), ("fno", ("fno", "fourier")), ("latent", ("latent", "jepa")), ("cnn", ("cnn",)))


def parse_question(q):
    """Rule-based reading of a task -> {"system", "pde" (bool), "kinds", "n", "minutes", "steps"}."""
    t = q.lower()
    pde = next((s for s, ws in PDE_WORDS.items() if any(w in t for w in ws)), None)
    system = pde or next((s for s, ws in SYSTEM_WORDS.items() if any(w in t for w in ws)), None)
    num = lambda pat, d, f=float: f(m.group(1)) if (m := re.search(pat, t)) else d
    return {"system": system, "pde": bool(pde), "kinds": [k for k, ws in KIND_WORDS if any(w in t for w in ws)] or ["cnn"],
            "n": num(r"\b(?:n\s*=|grid(?: of)?)\s*(\d+)", None, int), "minutes": num(r"(\d+\.?\d*)\s*min", 1.0),
            "steps": num(r"(\d+)\s*(?:time|rollout|trajectory)?\s*steps", 30, int)}


class FakeProvider:
    """Deterministic scripted planner (no model, no network): plan -> list_systems -> compose_pde | describe_system ->
    generate_data -> set_zen on -> train each kind -> set_zen off -> evaluate each -> compare -> make_figure ->
    write_report -> answer. Stops and explains on the first failed tool; asks when no system is recognisable."""
    name = "offline · rule-based"
    full_context = True  # it re-reads every tool result; no request-size limit offline

    def chat(self, messages, tools):
        q = next(m["content"] for m in messages if m["role"] == "user")
        names, done = {}, []  # [(tool, result)] in order
        for m in messages:
            for c in m.get("tool_calls") or []:
                names[c["id"]] = c["function"]["name"]
            if m["role"] == "tool":
                done.append((names[m["tool_call_id"]], json.loads(m["content"])))
        if done and not done[-1][1].get("ok"):
            return {"content": f"Stopped: {done[-1][0]} failed: {done[-1][1].get('error')}", "tool_calls": []}
        got = {k: r for k, r in done}
        a = parse_question(q)
        sysname, kinds = a["system"], a["kinds"]
        script = [("plan", {"steps": ["identify the system", "generate reference data", f"train {', '.join(kinds)}",
                                      "evaluate long-horizon rollouts", "compare and diagnose", "figure and report"]}),
                  ("list_systems", {})]
        if sysname:
            params = {"n": a["n"]} if a["n"] else {}
            script += [("compose_pde", {"spec": sysname}) if a["pde"] else ("describe_system", {"name": sysname}),
                       ("generate_data", {"system": sysname, "params": params, "steps": a["steps"], "n_train": 4,
                                          "n_val": 1, "n_test": 2}),
                       ("set_zen", {"on": True})]
            ds = got.get("generate_data", {}).get("dataset", "d1")
            script += [("train", {"dataset": ds, "kind": k, "budget_minutes": a["minutes"]}) for k in kinds]
            script += [("set_zen", {"on": False})] + [("evaluate", {"model": k}) for k in kinds]
            script += [("compare", {}), ("make_figure", {})]
        if len(done) < len(script):
            name, args = script[len(done)]
            return {"content": "", "tool_calls": [{"id": f"c{len(done)}", "name": name, "arguments": args}]}
        if not sysname:
            return {"content": "Which system or PDE should I study? Registered: " + ", ".join(got["list_systems"]["systems"]) +
                    "; PDE library: " + ", ".join(got["list_systems"].get("pde_library", {})) + ".", "tool_calls": []}
        text = self.explain(q, kinds, done)
        if "write_report" not in got:
            return {"content": "", "tool_calls": [{"id": f"c{len(done)}", "name": "write_report",
                                                   "arguments": {"markdown": text, "title": q[:80]}}]}
        return {"content": text, "tool_calls": []}

    @staticmethod
    def explain(q, kinds, done):
        got = dict(done)
        d, cmp_ = got["generate_data"], got["compare"]
        trains = [r for k, r in done if k == "train"]
        evals = {r["model"]: r for k, r in done if k == "evaluate"}
        rec = [x for r in [d, *trains] for x in r.get("recovered", [])]
        lines = ["## Hypothesis", f"Question: \"{q}\". Hypothesis: the models differ in how long their autoregressive "
                 "rollouts stay within the failure threshold on held-out seeds.", "",
                 "## Planned", f"- reference data {d['dataset']}: {d['system']}, train shape {d['shape_train']}, "
                 f"{d['steps']} steps, dt {d['dt']}", f"- models: {', '.join(kinds)}", "",
                 "## Ran", *[f"- {r['model']}: {r['optimizer_steps']} optimiser steps in {r['minutes']} min, "
                             f"{r['weights']} weights, best val rollout relL2 {r['best_val_rollout_rel_l2']}" for r in trains],
                 *[f"- recovery: {x}" for x in rec], "",
                 "## Measured", *[f"- {m}: test relL2 by horizon {e['test']['rel_l2_at']}, median steps to failure "
                                  f"{e['test']['steps_to_failure_median']} (failed {e['test']['failed']}), speedup "
                                  f"{e['runtime']['speedup']}x" for m, e in evals.items()],
                 *[f"- {f['model']}: {f['fact']}" for f in cmp_["findings"]], "",
                 "## Interpretation", *([f"- [{f['level']}] {f['model']}: {f['interpretation']}" for f in cmp_["findings"]]
                                        or ["- No diagnosis rule fired."]), "",
                 "## Open questions", *([f"- {x}" for x in dict.fromkeys(f["next"] for f in cmp_["findings"] if f.get("next"))]
                                        or ["- None suggested by the rules."])]
        return "\n".join(lines)