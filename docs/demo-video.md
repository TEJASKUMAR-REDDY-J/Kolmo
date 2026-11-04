# Kolmo: demo video (2:40)

**Project settings:** 1920×1080 (16:9), 60 fps timeline. Export H.264 (High profile) at 25–40 Mbps, AAC audio at
320 kbps.
- The simulation clips (`video/clips/`) are 1080p60.
- The dashboard shots (`video/tour/`) are 1080p30. They play fine on the 60 fps timeline.

A ready-made picture cut that follows this plan exactly is `video/kolmo_final_2m40.mp4` (no audio). It includes a
placeholder demo card at 0:26; drop your own card over it.

**How to make the voice:** generate **one TTS clip per scene** (the "VO" blocks below) and place each one at its
scene's start time. If a clip comes out a little long or short, it only shifts within its own scene, never the whole
video.
- Voice: a calm, deep, confident narrator. In ElevenLabs, "Brian" or "Adam" works, with stability ≈ 45 % and style
  ≈ 20 %.
- `<break time="…"/>` tags are ElevenLabs pauses; delete them if your tool doesn't support them.
- Target pace is about 145 words a minute. Each scene lists how many seconds its line should take.

---

## Timeline

| # | Time | Length | Picture | VO length |
|---|---|---|---|---|
| 1 | 0:00–0:08 | 8 s | `clips/car_reveal.mp4`: photo → SAM mask sweep → cut-out → glide → wind on | ~4 s |
| 2 | 0:08–0:26 | 18 s | `clips/car_tunnel.mp4` (6 s) → `clips/heat_plume.mp4` (6 s) → `clips/cyclone.mp4` (6 s) | ~15 s |
| **CARD** | 0:26–0:31 | 5 s | **Your "live demo" card.** No voice: whoosh + bass hit (see ideas) | 0 |
| 3 | 0:31–0:55 | 24 s | `tour/01_photo_wind_tunnel.mp4`: type "car" → Find it with SAM 3.1 → glow mask → Run wind tunnel → Zen | ~20 s |
| 4 | 0:55–1:17 | 22 s | `tour/08_assistant_autonomous.mp4`: typing the task (6 s), then a zoom on the agent panel as the plan ticks off, a failed step is retried, and the report is written | ~19 s |
| 5 | 1:17–1:39 | 22 s | `tour/07_models_ai_vs_solver.mp4` (9 s) → `clips/ai_vs_truth.mp4` (6 s) → `tour/06_experiments.mp4` leaderboard (7 s) | ~20 s |
| 6 | 1:39–1:59 | 20 s | Montage cut on the beat: `rings3d` → `02_heated_plumes` → `heat_rt` → `04_cyclones` → `gray_scott` → `cahn_hilliard` | ~12 s |
| 7 | 1:59–2:17 | 18 s | `clips/rings3d.mp4` (8 s) → `tour/03_rayleigh_taylor.mp4` in Zen (10 s) | ~15 s |
| 8 | 2:17–2:40 | 23 s | `clips/car_tunnel.mp4` with a slow push-in → end card ("Kolmo · Know when to trust your AI simulator · Built on Meta AI") at 2:34 | ~16 s |

---

## Voiceover, one clip per scene

**Scene 1 (starts 0:00)**
> One photo. <break time="0.4s" /> One word. <break time="0.4s" /> A wind tunnel.

**Scene 2 (starts 0:08)**
> Every car, every chip, every turbine is shaped by simulations that take experts and hours of compute.
> <break time="0.4s" /> AI makes them fast. <break time="0.3s" /> But AI fails silently, and nobody tells you
> when. <break time="0.7s" /> Kolmo does.

**CARD (0:26–0:31): no voice.**

**Scene 3 (starts 0:31)**
> Any photo. Name it. <break time="0.3s" /> Meta's Segment Anything 3.1 runs right here on the laptop and finds
> the car from one word. <break time="0.3s" /> Kolmo levels it, puts it on the road, and turns it into a solid body
> inside a real Navier–Stokes solver. <break time="0.6s" /> One click. <break time="0.4s" /> And the wind hits.

**Scene 4 (starts 0:55)**
> Or don't click at all. Just ask. <break time="0.3s" /> Meta's Llama Prompt Guard screens the request first.
> Then the agent takes over: it plans the study, generates the data, trains the models, fixes its own failed steps,
> and writes the report. <break time="0.3s" /> And it can't bluff: any number it can't prove gets flagged.

**Scene 5 (starts 1:17)**
> Now the part every other AI tool hides. <break time="0.4s" /> Left: the exact physics. Middle: a tiny neural
> network. On these patterns, it holds for a hundred and forty-five steps, at three tenths of a percent error.
> <break time="0.4s" /> On turbulence, every model breaks within about twenty steps. <break time="0.3s" /> Kolmo
> doesn't hide that. It shows you exactly where.

**Scene 6 (starts 1:39), with the montage cuts on the beat**
> Full 3D vortex rings. <break time="0.7s" /> Rising heat. <break time="0.7s" /> Storms on a spinning planet.
> <break time="0.7s" /> Patterns that copy themselves. <break time="0.7s" /> Type an equation, and Kolmo builds the
> solver.

**Scene 7 (starts 1:59)**
> Under the hood: spectral solvers in JAX, each checked against exact answers. Neural models that learn the physics.
> And a hybrid mode where the real solver steps in the moment the AI drifts. <break time="0.3s" /> The physics and the
> AI run on one laptop CPU.

**Scene 8 (starts 2:17)**
> Fast simulation is easy to fake. Trust isn't. <break time="0.5s" /> Kolmo gives engineers, students and
> researchers both, and answers the one question that matters. <break time="0.7s" /> When does it stop being right?
> <break time="1.2s" /> Kolmo. <break time="0.4s" /> Built on Meta AI.

---

## Make it hit harder

**Sound design (biggest upgrade for the least work)**
- Scene 1: a soft "scan" tone while the SAM mask sweeps across the photo, then a deep **whoosh + sub-bass hit**
  exactly when the wind switches on (≈ 0:06).
- Under the car tunnel shots: a low wind / air rumble at −24 dB.
- **Demo card:** a riser for the last 2 s of scene 2, silence for 0.3 s, then a hard bass hit as the card appears,
  and a reverse-cymbal whoosh out of it into scene 3.
- Scene 3: subtle UI clicks on the cursor ripples (the recordings show every click), and a soft "ping" when the
  glowing mask appears.
- Scene 6: cut every clip on a downbeat, with a short impact on each cut.
- Scene 8: let the music drop out completely on "When does it stop being right?", then one final hit on "Kolmo".

**On-screen text (one clean font, bottom-left, 2–4 words, 1.5 s each)**
- Scene 3: `Meta SAM 3.1 · on-device` and later `photo → solver in 1 click`
- Scene 4: `Llama Prompt Guard 2` and `autonomous · self-correcting`
- Scene 5: a big counter animating up to **145 steps**, then `0.3 % error`. On turbulence: `breaks at ~20 steps`
  in red.
- Scene 6: `3D Navier–Stokes · 96³` on the rings
- Scene 7: `runs on a laptop CPU`

**Edit moves**
- Speed-ramp the SAM wait in scene 3 (already shortened in the cut), and the moment the mask glows, slow to 100 %
  with a tiny 1.03× punch-in.
- In scene 5, freeze-frame for 0.5 s where the AI starts to drift and circle the error map in red. That's the "trust"
  moment the judges will remember.
- Use the same push-in on the car tunnel in scene 8 as on the end card, so the video ends where it started (the car),
  which closes the loop.

**Content that would make it even stronger (if you have 10 minutes before you edit)**
- Do the wind tunnel live with **your own photo** taken on your phone (a mug, a bottle, your bike) to prove it's not
  a canned demo. Add `your photo, live` on screen.
- Add one sentence of real-world stakes to scene 2, for example "every new car, chip and turbine is designed this
  way."
- End with the GitHub link on the end card for 2 s.

## Tips
- Use 0.3–0.5 s cross-dissolves except in scene 6 (hard cuts).
- Grade lightly: a little contrast and a slight teal tint in the shadows. The clips are already on black.
- Regenerate the footage: `python tools/render_clips.py`, `python tools/render_3d.py`, `python tools/record_tour.py`;
  rebuild the cut with `python video/make_final.py video/final_cfg.json`.
