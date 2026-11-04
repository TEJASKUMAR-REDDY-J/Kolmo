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
| 5 | 1:17–1:39 | 22 s | `clips/ai_vs_truth.mp4` (6 s) → `tour/06_experiments.mp4` leaderboard showing 145.5 (7 s) → `tour/07_models_ai_vs_solver.mp4`, the turbulence error map (9 s) | ~20 s |
| 6 | 1:39–1:59 | 20 s | Montage cut on the beat: `rings3d` → `tour/02_heated_plumes` → `heat_rt` → `tour/05_gray_scott` → `cyclone` → `cahn_hilliard` | ~13 s |
| 7 | 1:59–2:17 | 18 s | `clips/rings3d.mp4` (8 s) → `tour/03_rayleigh_taylor.mp4` in Zen (10 s) | ~15 s |
| 8 | 2:17–2:40 | 23 s | `clips/car_tunnel.mp4` with a slow push-in → end card ("Kolmo · Know when to trust your AI simulator · Built on Meta AI") at 2:34 | ~16 s |

---

## Voiceover, one clip per scene

Write-up style: one continuous story, spoken by you in the first person. The problem leads to the idea, the demo
proves it, the twist is about trust, and the close calls back to the opening. Each scene's last line hands over to
the next one, so the clips sound like one take.

**Scene 1 (0:00, about 4 s)**
> What if you could drop a photo… <break time="0.3s" /> into a wind tunnel?

**Scene 2 (0:08, about 16 s)**
> Cars, chips, turbines: they're all designed with physics simulations that take hours to run. <break time="0.3s" />
> AI can do it faster. <break time="0.4s" /> The catch? <break time="0.3s" /> It drifts, and it never tells you when.
> <break time="0.5s" /> So we built Kolmo: a physics lab where the AI has to prove it's right.

**CARD (0:26–0:31): no voice.**

**Scene 3 (0:31, about 20 s)**
> Here's the car from that photo. I type one word: car. <break time="0.3s" /> Meta's SAM 3.1, running right on this
> laptop, cuts it out. <break time="0.2s" /> Kolmo straightens it, sets it on the road, and drops it into a real
> fluid solver. <break time="0.4s" /> One click, and you watch the wake peel off the roof.

**Scene 4 (0:55, about 19 s)**
> Want to go further? <break time="0.2s" /> Just ask. <break time="0.3s" /> Meta's Llama Prompt Guard checks the
> request, then the agent runs the whole study on its own: data, training, testing. <break time="0.3s" /> When a step
> fails, it fixes it. <break time="0.2s" /> And every number in its report has to come from a real measurement.

**Scene 5 (1:17, about 20 s)**
> Because that's the real question: <break time="0.2s" /> how long can you trust the AI? <break time="0.4s" /> Here's
> the solver and the neural network, side by side. On patterns like these, the AI holds up: the best one lasts a
> hundred and forty-five steps, at under half a percent error. <break time="0.3s" /> On turbulence, it breaks in about
> twenty, and Kolmo shows you exactly where.

**Scene 6 (1:39, about 13 s; each phrase lands on a montage cut)**
> And it's not just air. <break time="0.5s" /> Full 3D turbulence. <break time="0.5s" /> Rising heat.
> <break time="0.5s" /> Patterns that copy themselves. <break time="0.5s" /> Storms on a spinning planet.
> <break time="0.5s" /> Type in an equation, and Kolmo builds the solver.

**Scene 7 (1:59, about 13 s)**
> Under the hood, every solver is checked against exact answers, <break time="0.2s" /> and the moment the AI starts to
> drift, the real physics steps back in. <break time="0.3s" /> The physics and the AI run on a single laptop.

**Scene 8 (2:17, about 18 s)**
> Fast answers are easy. <break time="0.3s" /> Trustworthy ones aren't. <break time="0.5s" /> Kolmo gives you both,
> so the next time an AI tells you how the air will flow, <break time="0.2s" /> you'll know exactly how far to
> believe it. <break time="1.2s" /> Kolmo. <break time="0.4s" /> Built on Meta AI.

All in one go (if you'd rather generate a single file and cut it yourself): paste the eight blocks in order, without
the scene headers.

## 2:13 version (current edit)

| # | Starts | Length | Voice (about 150 words a minute) |
|---|---|---|---|
| 1 | 0:00 | 7 s | What if you could drop a photo… into a wind tunnel? |
| 2 | 0:07 | 19 s | Cars, chips, turbines: all designed with physics simulations that take hours. AI can do it faster. The catch? It drifts, and it never tells you when. So we built Kolmo: a physics lab where the AI has to prove it's right. |
| Card | 0:26 | 5 s | (no voice) |
| 3 | 0:31 | 20 s | Here's the car from that photo. One word: car. Meta's SAM 3.1 cuts it out, right on this laptop. Kolmo sets it on the road, drops it into a real fluid solver, and one click later, the wake peels off the roof. |
| 4 | 0:51 | 17 s | Want more? Just ask. Llama Prompt Guard checks the request, then the agent runs the whole study itself: data, training, testing. It fixes its own failures, and every number it reports is measured. |
| 5 | 1:08 | 19 s | Which brings us to the real question: how long can you trust the AI? On patterns like these, our best model holds for a hundred and forty-five steps. On turbulence, it breaks in about twenty, and Kolmo shows you exactly where. |
| 6 | 1:27 | 15 s | And it's not just air. 3D turbulence. Rising heat. Living patterns. Storms on a spinning planet. Type in an equation, and Kolmo builds the solver. |
| 7 | 1:42 | 13 s | Every solver is checked against exact answers, and when the AI drifts, the real physics steps back in. |
| 8 | 1:55 | 18 s | Fast answers are easy. Trustworthy ones aren't. Kolmo gives you both. Kolmo. Built on Meta AI. |

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
