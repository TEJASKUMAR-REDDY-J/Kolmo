"""Autonomous research loop: plan -> act (tool) -> observe -> adjust, until the model answers or a budget runs out.

    run(question, on_event=cb) -> {"status", "text", "dir", "report", "removed", "events"}

Events (docs/contracts.md): plan, status, tool, progress, result, zen, error, done. Budgets: tool calls and compute
seconds per task (tools time themselves). Recovery (unstable dt, NaN training) happens inside the tools and is reported
in their results; refusals (limits, invalid specs) come back as {"ok": false, "error"} so the model can adjust.
Every number in the final answer and in write_report must appear in a tool result or the question; others become
"[unverified]". The task directory runs/assistant/<time>-<slug>/ gets transcript.json and report.md."""
import json
import os
import re
import time

from kolmo.assistant import tools
from kolmo.assistant.provider import RateLimited, get_provider

SECTIONS = ("Hypothesis", "Planned", "Ran", "Measured", "Interpretation", "Open questions")
SYSTEM = (
    "You are Kolmo's autonomous research assistant for physics world models. Work in a loop: first call plan(steps), "
    "then act with ONE tool at a time, read the result, and adjust (call plan again if the plan changes). Typical task: "
    "list_systems/describe_system or compose_pde -> generate_data -> train (cnn | fno | latent | nca_hybrid, with a "
    "budget in minutes) -> evaluate -> compare -> make_figure -> write_report. Tools fix unstable dt and NaN training "
    "themselves and tell you; if a tool returns ok=false, read the error and fix the arguments or explain why you stop. "
    "Use set_zen(true) before long training and set_zen(false) after. Older tool results are shortened to their summary (the task keeps datasets, models and results by id): note key numbers as you go and call compare right before write_report. Stay inside the budgets given in each tool "
    "result. Use ONLY numbers that appear in tool results; label interpretations as hypotheses. write_report and the "
    "final answer use markdown sections " + ", ".join(f"## {s}" for s in SECTIONS) + ". If the question is unclear, "
    "ask ONE clarifying question instead of calling tools.")
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
    Numbered-list markers ("1. ") are not numbers."""
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


MAX_RATE_WAIT_S = 20 * 60  # total time a task may wait on provider rate limits
KEEP_FULL = 3  # most recent tool results sent in full; older ones as their summary (full copies stay in the transcript)


def compact(msgs):
    """Bound the request size (Groq on-demand tier: 8000 tokens/min): old tool results shrink to ok/ids/summary.
    The task memory keeps every dataset/model/result, so the model refers to them by id."""
    idx = [i for i, m in enumerate(msgs) if m["role"] == "tool"]
    out = list(msgs)
    for i in idx[:-KEEP_FULL]:
        r = json.loads(msgs[i]["content"])
        keep = {k: r[k] for k in ("ok", "summary", "error", "dataset", "model", "system", "report", "figures") if k in r}
        out[i] = {**msgs[i], "content": json.dumps(keep, default=str)}
    return out


def sections(text):
    missing = [s for s in SECTIONS if not re.search(rf"^#+\s*{s}\b", text, re.M | re.I)]
    return text + "".join(f"\n\n## {s}\n(not provided by the assistant)" for s in missing)


def run(question, on_event=None, provider=None, out_dir="runs", max_calls=40, max_compute_s=25 * 60, should_stop=None):
    provider = provider or get_provider()
    events = []

    def emit(ev):
        events.append(ev)
        if on_event:
            on_event(ev)
    slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")[:40] or "task"
    task = tools.Task(os.path.join(out_dir, "assistant", f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}"), out_dir, emit,
                      max_compute_s, should_stop)
    task.sources.append(question)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    status, text, refused, malformed, waited = "error", "", 0, 0, 0.0
    emit({"type": "status", "text": f"{provider.name}: budget {max_calls} tool calls, {max_compute_s / 60:.0f} min compute"})
    while True:
        if should_stop and should_stop():
            text, status = "Cancelled by the user.", "cancelled"
            emit({"type": "status", "text": text})
            break
        try:
            reply = provider.chat(msgs if getattr(provider, "full_context", False) else compact(msgs), tools.SCHEMAS)
        except RateLimited as e:  # honour the provider's stated wait, bounded per task
            if e.wait_s > 120 or waited + e.wait_s > MAX_RATE_WAIT_S:
                text = f"Provider error: {e}"
                emit({"type": "error", "text": text})
                break
            waited += e.wait_s + 1
            emit({"type": "status", "text": f"rate limited by the provider: waiting {e.wait_s + 1:.0f} s"})
            time.sleep(e.wait_s + 1)
            continue
        except Exception as e:  # provider failure: report once, never retry-loop, never fabricate
            if "tool_use_failed" in str(e) and malformed < 2:  # Groq: malformed call -> show the model its error once
                malformed += 1
                msgs.append({"role": "user", "content": f"Your last tool call was rejected by the API ({str(e)[:300]}). "
                             "Call ONE tool with arguments that match its JSON schema exactly."})
                continue
            text = f"Provider error: {e}"
            emit({"type": "error", "text": text})
            break
        calls = reply.get("tool_calls") or []
        if not calls:
            text = reply.get("content") or ""
            status = "budget" if refused else "done" if task.calls else "clarify"
            break
        msgs.append({"role": "assistant", "content": reply.get("content") or "", "tool_calls": [
            {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
            for c in calls]})
        for c in calls:
            if task.calls >= max_calls:
                refused += 1
                res = {"ok": False, "error": f"tool-call budget of {max_calls} exhausted: answer now with what you have"}
                emit({"type": "tool", "name": c["name"], "args": c["arguments"], "ok": False, "summary": res["error"]})
            else:
                res = task.call(c["name"], c["arguments"])
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(res, default=str)})
        if refused > 2:  # model ignored the budget message
            text, status = "Stopped: tool-call budget exhausted.", "budget"
            emit({"type": "error", "text": text})
            break
    text, removed = audit(text, task.sources)
    if status in ("done", "budget") and task.datasets:
        text = sections(text)
        if removed:
            text += f"\n\n> Audit: {len(removed)} number(s) not found in tool results were replaced by [unverified]."
        if not task.report:
            task.call("write_report", {"markdown": text})
        emit({"type": "result", "metrics": task.metrics(), "figures": task.figures, "report": task.report})
    os.makedirs(task.dir, exist_ok=True)
    with open(os.path.join(task.dir, "transcript.json"), "w", encoding="utf-8") as f:
        json.dump({"question": question, "provider": provider.name, "status": status, "messages": msgs,
                   "events": events, "final": text}, f, indent=1, default=str)
    emit({"type": "done"})
    return {"status": status, "text": text, "dir": task.dir, "report": task.report, "removed": removed + task.removed,
            "events": events}


class Agent:
    """Window-facing wrapper: .status for the badge, .run(question, on_event, should_stop) -> run() result."""
    def __init__(self, provider=None, out_dir="runs", **budgets):
        self.provider, self.out_dir, self.budgets = provider or get_provider(), out_dir, budgets

    @property
    def status(self):
        return self.provider.name

    def run(self, question, on_event=None, should_stop=None):
        return run(question, on_event, self.provider, self.out_dir, should_stop=should_stop, **self.budgets)


if __name__ == "__main__":
    t, r = audit("1. error 0.0123 grew to 42 at step 10; R^2 0.98", ['{"e": 0.012345, "s": 10, "r2": 0.9811}'])
    assert r == ["42"] and "0.0123" in t and "0.98" in t and t.startswith("1. "), (t, r)
    ms = [{"role": "tool", "tool_call_id": str(i), "content": json.dumps({"ok": True, "summary": "s", "big": [i] * 50})}
          for i in range(5)]
    c = compact(ms)
    assert "big" not in c[0]["content"] and "big" in c[-1]["content"] and "big" in ms[0]["content"]
    print("ok")
