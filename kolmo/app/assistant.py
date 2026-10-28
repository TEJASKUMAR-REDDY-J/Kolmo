"""Assistant panel logic. Short commands (run, reset, explain, plan, train) are answered by fixed rules about the
interactive session; longer questions go to the research agent (kolmo.assistant.agent), which designs, validates,
runs and explains a real experiment. The window runs the agent off the UI thread.

context: {"summary": backend.summarize() result or None, "mode": str, "has_checkpoint": bool, "pending": question
awaiting confirmation or None}
action: None | "run" | "reset" | "research" (ask the agent) | "confirm" (re-ask the pending question, confirmed)."""
YES = ("yes", "y", "ok", "confirm", "go ahead", "run it", "sure")


class RuleBased:
    def __init__(self, agent=None):
        self.agent = agent

    @property
    def name(self):  # "offline · rule-based" or "live · <model>"
        return self.agent.status if self.agent else "offline · rule-based"

    HELP = ("Short commands are answered by fixed rules (no language model, no network):\n"
            "• run — run the current setup\n• explain — summarise the last run's measured facts and findings\n"
            "• plan — list the next experiments suggested by kolmo.diagnose\n• train — how to train a surrogate\n"
            "• or ask a research question (e.g. \"how stable is a CNN surrogate of diffusion with kappa = 0.1?\"): I plan, "
            "validate, run and explain a real experiment")

    def reply(self, text, context):
        t = (text or "").lower()
        s = context.get("summary")
        if context.get("pending"):
            return ("Running the confirmed experiment.", "confirm") if t.strip(" .!") in YES else ("Cancelled.", None)
        if self.agent and len(t.split()) > 3:
            return "Designing an experiment for that question (validate, run, read results)…", "research"
        if any(w in t for w in ("run", "start", "go")):
            return "Running the current setup.", "run"
        if "reset" in t or "clear" in t:
            return "Cleared.", "reset"
        if "train" in t or "model" in t:
            return ("Train and evaluate surrogates reproducibly with\n  python -m kolmo.experiments run <config.json>\n"
                    "Its checkpoints then appear in the Surrogate checkpoint picker, with their test-set findings."), None
        if any(w in t for w in ("explain", "why", "diagnos", "result", "what")):
            if not s or isinstance(s, str):
                return (s or "No finished run yet. Run something first, then ask me to explain it."), None
            lines = list(s["facts"]) + [f"[{f['level']}] {f['fact']}" for f in s["findings"]]
            return ("\n".join(lines) if lines else "Nothing to report for this run."), None
        if "plan" in t or "next" in t:
            steps = plan(s)
            return ("\n".join(f"{i}. {x}" for i, x in enumerate(steps, 1)) if steps else
                    "No plan yet: rule-based plans come from a finished surrogate or hybrid run."), None
        return self.HELP, None


def plan(summary):
    """Next experiments = the 'next' field of each kolmo.diagnose finding (deduplicated, in order)."""
    if not summary or isinstance(summary, str):
        return []
    return list(dict.fromkeys(f["next"] for f in summary.get("findings", []) if f.get("next")))


if __name__ == "__main__":
    a = RuleBased()
    assert a.reply("run it", {})[1] == "run"
    assert "no language model" in a.reply("hello", {})[0] and a.name == "offline · rule-based"
    b = RuleBased(agent=type("A", (), {"status": "live · m"})())
    assert b.name == "live · m" and b.reply("how stable is the diffusion surrogate", {})[1] == "research"
    assert b.reply("yes", {"pending": "q"})[1] == "confirm" and b.reply("no", {"pending": "q"})[1] is None
    s = {"facts": ["x"], "findings": [{"level": "info", "fact": "f", "next": "n1"}, {"level": "w", "fact": "g", "next": "n1"}]}
    assert plan(s) == ["n1"] and "f" in a.reply("explain", {"summary": s})[0]
    print("ok")
