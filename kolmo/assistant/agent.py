"""Research-assistant loop: question -> config -> validate -> (clarify) -> run -> read -> explain.

The provider plans through tool calls (kolmo.assistant.tools); the agent executes them, enforces confirmation of
expensive runs, and audits the final explanation: every number in it must appear in a tool result or in the user's
question, otherwise it is replaced by "[unverified]". The audited explanation is saved as <run dir>/assistant_report.md."""
import json
import os
import re

from kolmo.assistant import tools
from kolmo.assistant.provider import get_provider

SECTIONS = ("Hypothesis", "Planned", "Ran", "Measured", "Interpretation", "Open questions")
SYSTEM = (
    "You are Kolmo's research assistant. Turn the user's question into ONE small, valid Kolmo experiment and explain "
    "its real results. Steps: list_systems; build an ExperimentConfig; validate_config (fix errors and retry); "
    "estimate_cost; run_experiment; read_results; diagnose. If the system or the quantity of interest is unclear, "
    "ask ONE clarifying question instead of calling tools. Keep runs small (grid <= 64, < 1000 training steps) unless "
    "asked. Final answer: markdown with sections " + ", ".join(f"## {s}" for s in SECTIONS) + ". Use ONLY numbers that "
    "appear in tool results; label interpretations as hypotheses; never invent measurements.")
NUMBER = re.compile(r"(?<![\w.^@/\\-])[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")  # "12x", "5s" count; "relL2", "fluid2d", hashes do not


def _nums(text):
    out = set()
    for m in NUMBER.finditer(text):
        try:
            out.add(abs(float(m.group().rstrip("."))))
        except ValueError:
            pass
    return out


def _supported(tok, allowed):
    s = tok.rstrip(".").lstrip("+-")
    x = float(s)
    if x in allowed:
        return True
    digits = len(re.sub(r"[eE].*|^0*\.?0*|\.", "", s)) or 1  # significant digits shown
    return any(v != 0 and float(f"{v:.{digits}g}") == x for v in allowed)


def audit(text, sources):
    """Replace numbers not backed by `sources` (tool-result texts + question). Returns (text, removed tokens).
    Numbered-list markers ("1. ") and heading levels are not numbers."""
    allowed = set().union(*map(_nums, sources)) if sources else set()
    removed = []

    def sub(m):
        tok = m.group()
        line_start = m.string.rfind("\n", 0, m.start()) + 1
        rest = m.string[m.end():m.end() + 2]
        if not m.string[line_start:m.start()].strip() and (tok.endswith(".") and rest[:1] in (" ", "") or rest in (". ", ") ")):
            return tok  # list marker
        if _supported(tok, allowed):
            return tok
        removed.append(tok)
        return "[unverified]" + ("." if tok.endswith(".") else "")
    return NUMBER.sub(sub, text), removed


class Agent:
    def __init__(self, provider=None, out_dir="runs", max_turns=16):
        self.provider = provider or get_provider()
        self.out_dir, self.max_turns = out_dir, max_turns

    @property
    def status(self):
        return self.provider.name

    def ask(self, question, confirmed=False, log=print):
        """-> {"status": "done" | "clarify" | "confirm" | "error", "text", "run_dir", "report", "removed", "trace"}.
        status "confirm": the run needs the user's OK; call ask(question, confirmed=True) to proceed."""
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
        sources, trace, run_dir = [question], [], None
        for _ in range(self.max_turns):
            try:
                reply = self.provider.chat(msgs, tools.SCHEMAS)
            except Exception as e:  # network / provider failure: report, never fabricate
                return self._out("error", f"Provider error: {type(e).__name__}: {e}", run_dir, trace)
            calls = reply.get("tool_calls") or []
            if not calls:
                return self._finish(reply.get("content") or "", sources, run_dir, trace)
            msgs.append({"role": "assistant", "content": reply.get("content") or "", "tool_calls": [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                for c in calls]})
            for c in calls:
                log(f"tool {c['name']}")
                res = tools.call(c["name"], c["arguments"], self.out_dir, confirmed)
                text = json.dumps(res, default=str)
                sources.append(text)
                trace.append({"tool": c["name"], "arguments": c["arguments"], "result": res})
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": text})
                if res.get("needs_confirmation") and c["name"] == "run_experiment":
                    return self._out("confirm", f"This run is estimated at {res['estimated_seconds']} s (above "
                                     f"{tools.LIMITS['confirm_seconds']} s). Run it?", run_dir, trace)
                run_dir = res.get("run_dir", run_dir)
        return self._out("error", f"Stopped after {self.max_turns} turns without a final answer.", run_dir, trace)

    def _finish(self, text, sources, run_dir, trace):
        if run_dir is None:  # no experiment ran: a clarification or a refusal, nothing to audit against results
            text, removed = audit(text, sources)
            return self._out("clarify", text, None, trace, removed)
        missing = [s for s in SECTIONS if not re.search(rf"^#+\s*{s}\b", text, re.M | re.I)]
        text += "".join(f"\n\n## {s}\n(not provided by the assistant)" for s in missing)
        text, removed = audit(text, sources)
        if removed:
            text += f"\n\n> Audit: {len(removed)} number(s) not found in tool results were replaced by [unverified]."
        report = os.path.join(run_dir, "assistant_report.md")
        with open(report, "w", encoding="utf-8") as f:
            f.write(f"# Assistant report\n\nProvider: {self.status}\n\n{text}\n")
        return self._out("done", text, run_dir, trace, removed, report)

    @staticmethod
    def _out(status, text, run_dir, trace, removed=(), report=None):
        return {"status": status, "text": text, "run_dir": run_dir, "report": report, "removed": list(removed), "trace": trace}


if __name__ == "__main__":
    t, r = audit("1. error 0.0123 grew to 42 at step 10; R^2 0.98", ['{"e": 0.012345, "s": 10, "r2": 0.9811}'])
    assert r == ["42"] and "0.0123" in t and "0.98" in t and t.startswith("1. "), (t, r)
    print("ok")
