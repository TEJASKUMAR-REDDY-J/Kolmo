# Research assistant (autonomous)

Give the assistant a physics world-model research task, for example "Build a lightweight NCA surrogate for
Cahn-Hilliard phase separation, compare it with an FNO on long-horizon reliability, and write a report". It runs a
loop until the task is done:

1. **Plan.** The model calls `plan`.
2. **Act.** It calls one tool per turn.
3. **Observe.** The tool result comes back, with the budget used so far.
4. **Adjust.** It calls `plan` again when the plan changes.

The loop stops when the model answers or a budget runs out.

| File | Role |
|---|---|
| `kolmo/assistant/agent.py` | `run(question, on_event, provider, out_dir, max_calls=40, max_compute_s=1500, should_stop)`. Also the loop, number audit, transcript, and the `Agent` wrapper used by the UI (`.status`, `.run`). |
| `kolmo/assistant/tools.py` | `Task`: memory for one task (specs, datasets, models, results), budgets, events, argument checks. Also the tools. |
| `kolmo/assistant/provider.py` | `OpenAIChatProvider` for any OpenAI-compatible `/chat/completions` endpoint (stdlib urllib), and the offline scripted `FakeProvider`. |

## Tools

- **Systems:** `list_systems`, `describe_system`.
- **PDEs:** `compose_pde`. Pass a spec or a library name; it is validated by `kolmo.systems.make`, and errors come back so the model can fix the spec.
- **Data:** `generate_data`.
- **Training:** `train`. `kind` is `cnn`, `fno`, `latent` or `nca_hybrid`, and the time budget is in minutes.
- **Evaluation:** `evaluate`, which rolls a model out over a long horizon on held-out seeds.
- **Comparison:** `compare`, which builds a table and the `kolmo.diagnose` findings.
- **Output:** `make_figure` and `write_report`.
- **Model registry:** `registry_op`, with `list`, `register` or `remove`.
- **UI:** `set_zen`.

Each task writes to `runs/assistant/<time>-<slug>/`:

- `checkpoints/`
- `results.json`
- figures (`*.png`)
- `report.md`
- `transcript.json`, which holds the messages and events

The layout matches an experiment run, so `registry.register` works on it.

## Reliability rules

These are enforced in code, not left to the model.

- **Only measured numbers.** Every number in `write_report` and in the final answer must appear in a tool result or in
  the question. Any other number is replaced by `[unverified]`, and the replacements are counted.
- **Argument checks.** Before a tool runs, its arguments are checked against the tool's schema. Unknown or missing
  arguments come back as `ok: false`.
- **Errors go back to the model.** A tool error is returned as a result, never raised, so the model can correct
  itself or explain why it stops.
- **Automatic recovery:**
  - Unstable or non-finite reference data: dt is halved, up to 3 times.
  - Non-finite training: one retry with lr divided by 3 and unroll set to 1.
  - Malformed tool call (Groq returns `tool_use_failed`): the error is shown to the model, at most twice per task.
- **Budgets:**
  - 40 tool calls per task.
  - 25 minutes of compute.
  - At most 15 minutes of training per model.
  - Grid n ≤ 128.
  - At most 400 solver steps per trajectory.
  - At most 64 trajectories.

  Once a budget is used up, the tool results tell the model to wrap up.
- **Provider errors stop the task.** An HTTP error is reported once, with the provider's reason, and never retried in
  a loop.
- **Cancel.** `should_stop()` is checked between turns and inside training.

## Events (docs/contracts.md)

| Event | Contents |
|---|---|
| `plan` | `steps` |
| `status` | `text` |
| `tool` | `name`, `args`, `ok`, `summary` |
| `progress` | the training progress dict, plus `model` |
| `result` | `metrics`, `figures`, `report` |
| `zen` | `on` |
| `error` | `text` |
| `done` | |

## Provider (`.env`)

```
GROQ_API_KEY=...
KOLMO_LLM_BASE_URL=https://api.groq.com/openai/v1
KOLMO_LLM_MODEL=openai/gpt-oss-120b
KOLMO_LLM_LIVE=1
```

The provider is live only when `KOLMO_LLM_LIVE=1` and the key is set. Otherwise the deterministic `FakeProvider` is
used. It runs with no network and is what the tests use.

Calls are sent with:

- `tool_choice="auto"`
- `parallel_tool_calls=false`
- `temperature=0`
- `reasoning_effort="medium"` (gpt-oss models only)

Groq users report that gpt-oss tool calling fails more often with `tool_choice` set to `required`, with parallel
calls, or with loose schemas. For that reason the schemas set `additionalProperties: false` at the top level.

Any OpenAI-compatible endpoint works. `meta-models/muse-spark-open` is listed on Groq as a Preview, Enterprise-only
model with tool use. Once enterprise access exists, switch to it by changing `KOLMO_LLM_MODEL`.

## Tests

`python -m pytest tests/test_assistant.py -q` runs offline with tiny grids. It covers:

- question parsing and provider selection
- clarification when no system is named
- diffusion end to end (events, zen on/off, sections, report, transcript)
- Cahn-Hilliard: compose the PDE, train `nca_hybrid` and `fno`, compare, then register a model
- PDE spec errors returned to the model
- the tool-call budget and number audit
- dt-halving recovery
