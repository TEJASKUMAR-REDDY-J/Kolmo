# Kolmo: 3-minute demo video

Target: 3:00. About 400 words of voiceover, read at a calm pace (about 140 words a minute). Each block lists the
time, what is on screen, and the line to read. Clips marked `clip:` are in `video/clips/` (1920×1080, 60 fps,
no text, made by `python tools/render_clips.py`). Shots marked `screen:` are recordings of the app (see the
recording notes at the end).

Rubric mapping: impact (30 %) → 0:12–0:32 and 2:38–3:00; Meta AI tooling (25 %) → 0:00–0:12, 0:32–1:02 and the
Prompt Guard beat at 1:05; technical depth (20 %) → 1:30–2:38; demo (25 %) → everything on `screen:`.

---

### 0:00–0:12 · Hook
- clip: `car_reveal.mp4`. A photo, Meta SAM 3.1's mask sweeping over it, the background falling away, the car
  gliding into the tunnel, the wind switching on.
- **VO:** "This is a photo of a car. Ten seconds later, it's sitting in a wind tunnel, and nobody wrote any CAD."

### 0:12–0:32 · The problem
- clip: `car_tunnel.mp4` (wake shedding behind the car), then `heat_plume.mp4`.
- **VO:** "Simulating physics, like air over a car or heat rising off a chip, takes experts and hours of compute.
  AI models can predict it far faster, but nobody can tell you when that prediction quietly stops being
  true. Kolmo is a lab that does both. It runs the real physics, trains small AI models to imitate it, and measures
  exactly how long you can trust them."

### 0:32–1:02 · Meta SAM 3.1: photo to wind tunnel (live)
- screen: the Photo wind tunnel page. Drop `car_side.png` and type "car". Press **Find it with SAM 3.1** (cut the
  wait), and the glowing mask appears. Show the tunnel preview, toggle **Road under it**, press **Run wind tunnel**.
  The app switches to Simulate and the wake starts shedding behind the real photo of the car.
- **VO:** "Drop in any photo and say what it is. Meta's Segment Anything 3.1 runs locally on this laptop and cuts
  the object out from a single word. Kolmo levels it, scales it, sets it on the road, and turns it into a solid
  obstacle for a real Navier–Stokes solver. One click, and the air flows left to right around your object."

### 1:02–1:30 · The autonomous research assistant
- screen: the assistant panel. Type: *"Build a lightweight NCA surrogate for Gray-Scott spot patterns, compare it
  with an FNO over 200 steps, and tell me which one keeps the spots stable."* Show the "Llama Prompt Guard 2: injection
  risk 0.00" status line. The plan steps tick off, **Zen mode** fades in, and the live training-loss chart runs. Speed
  this up 8–16× in the edit.
- **VO:** "Or just ask. Every request is screened by Meta's Llama Prompt Guard first. Then the assistant plans the
  study, generates data with the solver, trains the models, tests them on runs it has never seen, and writes the
  report. It doesn't invent results: any number it can't trace back to a measurement is flagged."

### 1:30–2:00 · How long can you trust the AI?
- clip: `ai_vs_truth.mp4` (solver | neural surrogate | difference, side by side). Then screen: the Experiments page
  with the leaderboard boxes, and the Models page.
- **VO:** "Here's the honest part. On the left is the exact physics; in the middle, a 23-thousand-weight neural
  cellular automaton. On Gray-Scott patterns it stays accurate for about 145 steps, with errors near three tenths
  of a percent. On turbulence, every model breaks within about twenty steps, and Kolmo shows you that instead of
  hiding it."

### 2:00–2:22 · Range of physics (montage, 3–4 s per clip, cut on the beat)
- clip: `heat_rt.mp4` → `cyclone.mp4` → `gray_scott.mp4` → `cahn_hilliard.mp4`. End on screen: the **Try a
  scenario…** menu opening.
- **VO:** "Heat and buoyancy. Cyclones drifting on a rotating planet. Chemical patterns that copy themselves.
  Alloys separating. Type an equation and Kolmo builds the solver for it."

### 2:22–2:38 · Engineering depth
- screen: Simulate running in Zen mode (press Z), with the error-vs-horizon chart and hybrid mode re-anchoring.
- **VO:** "Under the hood there are spectral solvers in JAX, each checked against exact solutions; three neural
  architectures; and a hybrid mode where the solver steps in whenever the AI drifts. All of it runs offline on a
  laptop CPU."

### 2:38–3:00 · Impact and close
- clip: `car_tunnel.mp4` again (slow push-in in the edit). Fade to the Kolmo logo.
- **VO:** "Engineers, students and researchers get fast simulation, plus a measured answer to the question that
  matters: when does it stop being right? Kolmo. Built on Meta AI."

---

## Recording the app (`screen:` shots)
- Use OBS Studio: 1920×1080, 60 fps, recording format MKV (remux to MP4 afterwards), encoder x264 at CRF 16, or the
  hardware encoder at "Indistinguishable" quality. Capture the Kolmo window only.
- Before recording, maximise the window, turn on **Animated background** (⚙ menu), close other apps, and set Windows
  display scaling to 100 % so the text is crisp.
- Warm up first: run SAM 3.1 once (the model loads in about 1 minute), so the take shows the ~10 s re-prompt rather
  than the cold start. Cut the wait in the edit anyway.
- In Simulate, use **Speed 4×** for smooth, fast flow.
- Assistant takes are long. Record the whole run and speed it up in the edit. Keep the moment where the plan steps
  turn green at real speed.

## Edit notes
- Use 0.3–0.5 s cross-dissolves between clips and hard cuts on the music beats.
- For text overlays, use one sans-serif font (Inter or SF Pro), white at 85 %, lower-left, three words at most, for
  example "Meta SAM 3.1 · local" or "145 steps · 0.3 % error".
- Mix the music around −18 LUFS under the voice and duck it 6 dB while the VO is speaking.
