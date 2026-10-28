# Research assistant

Turns a question ("how stable is a CNN surrogate of diffusion with kappa = 0.2 on a 16 grid?") into one small,
validated Kolmo experiment, runs it, and explains the measured results.

| File | Role |
|---|---|
| `kolmo/assistant/provider.py` | `FakeProvider` (offline, deterministic, scripted from rules) and `OpenAIChatProvider` (OpenAI-compatible `/chat/completions` with tools, stdlib `urllib`). `get_provider()` reads `.env`. |
| `kolmo/assistant/tools.py` | Tools over real code: `list_systems`, `validate_config`, `estimate_cost`, `run_experiment` (`kolmo.experiments.run`), `read_results`, `diagnose` (`kolmo.diagnose`). Limits and confirmation live here. |
| `kolmo/assistant/agent.py` | Loop: list_systems, config, validate, (clarify), estimate, run, read, diagnose, explain. Audits numbers and writes `<run dir>/assistant_report.md`. |
| `kolmo/app/assistant.py` | Panel logic: short commands (run, reset, explain, plan, train) use fixed rules; questions longer than 3 words go to the agent. |

## Provider selection (`.env`)

```
GROQ_API_KEY=...
KOLMO_LLM_BASE_URL=https://api.groq.com/openai/v1
KOLMO_LLM_MODEL=meta-models/muse-spark-open
KOLMO_LLM_LIVE=0
```

The provider is live only when `KOLMO_LLM_LIVE=1` and `GROQ_API_KEY` is set. Otherwise the offline `FakeProvider`
runs, so tests and development never use the network. The UI badge shows `offline · rule-based` or `live · <model>`.
`meta-models/muse-spark-open` appears on Groq's models page as a Preview model (Enterprise, 131,072-token context).
Groq's tool-use page lists it as supporting tool use. Preview models are meant for evaluation only and can be removed.

## Guarantees

- **Real numbers only.** Every number in the final explanation must appear in a tool result or in the question.
  Numbers are matched at the precision they are displayed with, so `0.0854` matches `0.08542`. Any other number
  is replaced by `[unverified]`, and the report says how many were replaced. List markers are not checked. Tokens
  glued to identifiers are not checked either (`relL2`, `fluid2d`, run-directory hashes).
- **Sections.** Hypothesis / Planned / Ran / Measured / Interpretation / Open questions. If a section is missing,
  it is added and marked "(not provided by the assistant)".
- **Clarification.** If the question names no known system, the agent asks which system to use and runs nothing.
- **Limits** (`tools.LIMITS`): grid n <= 128, data steps x spacing <= 400, <= 5000 training steps per model,
  <= 4 models. A run is refused when its estimate is above 1800 s. A run whose estimate is above 60 s waits for
  the user to confirm. In the UI, reply `yes`. In code, call `Agent.ask(q, confirmed=True)`. The model cannot
  confirm a run itself: the agent drops any `confirmed` or `out_dir` argument the model sends.
- **Off the UI thread.** The window runs `Agent.ask` in a `Task` (QThread) and answers one question at a time.

## Cost estimate

`seconds = 10 + 8 x models + trajectories x solver steps x n^2 x ref_rate + sum(steps x batch x unroll) x n^2 x 7e-6`

The rates were measured on this machine, a shared 2-core CPU:

- **Training:** `runs/ui-tiny-diffusion-v2`, 300 x 8 samples at 32^2 in 17.1 s, about 7e-6 s per cell-sample.
- **Diffusion reference:** 1.36e-4 s/step at 32^2.
- **fluid2d reference:** about 400 steps/s at 64^2 (`docs/physics.md`).

Treat the result as an order of magnitude. The calibration used a CNN; FNO costs were not measured. In the test
run (n = 16, 40 steps) the estimate was 18.3 s. The actual time was 6.0 s, measured on a second run that reused
the cached data and checkpoints.

## Tests

`python -m pytest tests/test_assistant.py -q` runs offline with `FakeProvider`, about 80 s on the shared CPU.
It covers parsing and provider selection, limits and confirmation, clarification, a full diffusion run with
every section present and no number stripped, and a provider that invents a number (the number is stripped).
