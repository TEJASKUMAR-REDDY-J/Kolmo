# Kolmo: demo video (2:40)

**Project settings:** 1920×1080 (16:9), 60 fps timeline. Export H.264 (High profile) at 25–40 Mbps, AAC audio at
320 kbps.
- The simulation clips (`video/clips/`) are 1080p60.
- The dashboard shots (`video/tour/`) are 1080p30. Drop them straight onto the 60 fps timeline: each frame shows
  twice, and that is fine for UI footage.

**Voice:** paste the "Voiceover for text-to-speech" section into your TTS tool. It is about 330 words, which reads in
about 2:30 at a natural pace, leaving roughly 10 s for music-only moments. `<break time="…"/>` tags are ElevenLabs
pauses; delete them if your tool doesn't support them.

---

## Scene plan

| # | Time | Length | Picture | Voiceover |
|---|---|---|---|---|
| 1 | 0:00–0:08 | 8 s | `clips/car_reveal.mp4`: photo, SAM mask sweep, cut-out, glide, wind on | "One photo. One word. A wind tunnel." |
| 2 | 0:08–0:22 | 14 s | `clips/car_tunnel.mp4` (6 s), then `clips/heat_plume.mp4` (8 s, from 2 s in) | the problem |
| 3 | 0:22–0:50 | 28 s | `tour/01_photo_wind_tunnel.mp4`: type "car", Find it with SAM 3.1, glow mask, Run wind tunnel, Zen. Speed up the SAM wait 4×. | Meta SAM 3.1 |
| 4 | 0:50–1:14 | 24 s | `tour/08_assistant_autonomous.mp4`: typing the task, plan ticking, Zen fading in, training curve (already a time-lapse) | autonomous assistant + Llama Prompt Guard |
| 5 | 1:14–1:38 | 24 s | `tour/07_models_ai_vs_solver.mp4` (10 s) → `clips/ai_vs_truth.mp4` (6 s) → `tour/06_experiments.mp4` leaderboard (8 s) | how long can you trust the AI |
| 6 | 1:38–2:00 | 22 s | Montage, 3–4 s each, cut on the beat: `clips/rings3d.mp4` → `tour/02_heated_plumes.mp4` → `clips/heat_rt.mp4` → `tour/04_cyclones.mp4` → `clips/gray_scott.mp4` → `clips/cahn_hilliard.mp4` (from 7 s) | range of physics |
| 7 | 2:00–2:18 | 18 s | `clips/rings3d.mp4` (slow, 8 s) → `tour/03_rayleigh_taylor.mp4` in Zen (10 s) | engineering depth |
| 8 | 2:18–2:40 | 22 s | `clips/car_tunnel.mp4` with a slow push-in (1.0 → 1.15 scale), fade to the Kolmo logo at 2:34, then black | impact and close |

Music: build up through 1–2, settle under 3–5, peak into the 6 montage (hit the cuts), drop out for 2–3 s at the start
of 7, then resolve under 8.

On-screen text (optional, lower-left, one font, at most 4 words):
- Scene 3: "Meta SAM 3.1 · on-device"
- Scene 4: "Llama Prompt Guard 2"
- Scene 5: "145 steps · 0.3 % error"
- Scene 6: "3D Navier–Stokes · 96³"
- Scene 8: "Kolmo"

---

## Voiceover for text-to-speech

One photo. One word. A wind tunnel. <break time="1.2s" />

Simulating physics, like air over a car or heat rising off a chip, still takes experts and hours of compute. AI can
predict it far faster, but nobody tells you when that prediction quietly stops being true. <break time="0.6s" /> Kolmo
does. <break time="1.0s" />

Drop in any photo and say what it is. Meta's Segment Anything 3.1 runs right here on the laptop and finds the object
from a single word. Kolmo levels it, sets it on the road, and turns it into a solid body for a real Navier–Stokes
solver. <break time="0.4s" /> One click, and the air flows. <break time="1.2s" />

Or just ask. Every request is first screened by Meta's Llama Prompt Guard. Then the assistant plans the study, runs
the solver, trains the models, tests them on cases they have never seen, and writes the report itself. Any number it
can't trace back to a measurement gets flagged. <break time="1.0s" />

And this is the part most AI tools skip. Left: the exact physics. Middle: a tiny neural network. On these chemical
patterns it stays accurate for a hundred and forty-five steps, at three tenths of a percent error. On turbulence,
every model breaks within about twenty, and Kolmo shows you exactly where. <break time="1.0s" />

Full 3D vortex rings. <break time="0.3s" /> Rising heat. <break time="0.3s" /> Storms on a spinning planet.
<break time="0.3s" /> Patterns that copy themselves. <break time="0.3s" /> Type an equation, and Kolmo builds the
solver. <break time="1.2s" />

Under the hood: spectral solvers in JAX, each checked against exact answers, neural models that learn the physics,
and a hybrid mode where the real solver steps in the moment the AI drifts. The physics and the models run offline, on one laptop CPU.
<break time="1.2s" />

For engineers, students and researchers: simulation that's fast, plus an honest answer to the question that matters
most. <break time="0.5s" /> When does it stop being right? <break time="1.0s" />

Kolmo. <break time="0.4s" /> Built on Meta AI.

---

## Tips
- Cut on the music's downbeats in scene 6; use 0.3–0.5 s cross-dissolves everywhere else.
- In scene 3, speed-ramp the SAM wait (8 s → 2 s) and land back at real speed when the glowing mask appears.
- Grade lightly: a little contrast and a slight teal-in-shadows tint. The clips are already on black.
- Regenerate the footage: `python tools/render_clips.py`, `python tools/render_3d.py`, `python tools/record_tour.py`.
