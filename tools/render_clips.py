"""Render 1920x1080, 60 fps H.264 clips of Kolmo simulations for the demo video (clean: no text unless --caption).

    python tools/render_clips.py                    # every clip
    python tools/render_clips.py car_reveal heat_rt # some clips
    python tools/render_clips.py --preview          # 2 s of each, fast check

Clips land in video/clips/ (gitignored). Simulations run at 256-512^2 and are upsampled bicubically on the scalar
field before colouring, so edges stay smooth. Colour scales are fixed per clip (no flicker)."""
import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import colormaps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kolmo.geometry.mask import tunnel_grid  # noqa: E402
from kolmo.systems import make  # noqa: E402

W, H, FPS = 1920, 1080, 60
OUT = ROOT / "video" / "clips"
PREVIEW = False


# ------------------------------------------------------------------ output
class Writer:
    def __init__(self, name):
        OUT.mkdir(parents=True, exist_ok=True)
        self.path, self.n, self.t0 = OUT / f"{name}.mp4", 0, time.perf_counter()
        self.p = subprocess.Popen(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)], stdin=subprocess.PIPE)

    def __call__(self, frame):
        self.p.stdin.write(np.ascontiguousarray(frame, np.uint8).tobytes())
        self.n += 1
        if self.n in (FPS, 6 * FPS):  # stills for checking
            cv2.imwrite(str(OUT / "stills" / f"{self.path.stem}_{self.n // FPS}s.png"), frame[..., ::-1])

    def close(self):
        self.p.stdin.close()
        self.p.wait()
        print(f"{self.path.name}: {self.n / FPS:.1f} s in {time.perf_counter() - self.t0:.0f} s", flush=True)


def frames(seconds):
    return int((min(seconds, 2.0) if PREVIEW else seconds) * FPS)


# ------------------------------------------------------------------ colour
def ramp(points):
    """[(position 0..1, (r, g, b))] -> (256, 3) uint8 lookup table."""
    x = np.linspace(0, 1, 256)
    pos = [p for p, _ in points]
    return np.stack([np.interp(x, pos, [c[i] for _, c in points]) for i in range(3)], 1).astype(np.uint8)


VORT = ramp([(0, (210, 245, 255)), (0.18, (40, 160, 255)), (0.36, (12, 40, 120)), (0.5, (3, 4, 9)),
             (0.64, (120, 26, 12)), (0.82, (255, 120, 30)), (1, (255, 240, 190))])
PHASE = ramp([(0, (8, 18, 40)), (0.35, (20, 70, 130)), (0.5, (230, 230, 235)), (0.65, (240, 140, 60)), (1, (70, 12, 20))])
INFERNO = (colormaps["inferno"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
MAKO = ramp([(0, (4, 6, 14)), (0.3, (30, 40, 90)), (0.55, (40, 120, 150)), (0.8, (110, 210, 180)), (1, (230, 255, 235))])


def colour(field, lo, hi, lut, size=(W, H), soft=False):
    """Bicubic upsample of the scalar field, then map [lo, hi] (soft: tanh compression, for signed fields)."""
    f = cv2.resize(np.asarray(field, np.float32), size, interpolation=cv2.INTER_CUBIC)
    if soft:
        x = 0.5 + 0.5 * np.tanh(f / hi)
    else:
        x = (f - lo) / (hi - lo)
    return lut[np.clip(x * 255, 0, 255).astype(np.uint8)]


def tile_x(field, centre=0.5):
    """Square periodic field (rows y-up) -> a 16:9 band of rows around `centre` (wrapped), row 0 = screen top."""
    n = field.shape[-1]
    rows = int(round(n * H / W))
    idx = (np.arange(rows) + int(centre * n) - rows // 2) % n
    return field[idx][::-1]


_ADV = {}


def advance(step, k):
    key = (id(step), k)
    if key not in _ADV:
        _ADV[key] = jax.jit(lambda s: jax.lax.fori_loop(0, k, lambda _, x: step(x), s))
    return _ADV[key]


def run_steps(step, s, count):
    """count solver steps using a few compiled chunk sizes (no recompiles for varying counts)."""
    for chunk in (256, 64, 16, 4, 1):
        while count >= chunk:
            s, count = advance(step, chunk)(s), count - chunk
    return s


# ------------------------------------------------------------------ car wind tunnel (Meta SAM 3.1 silhouette)
CAR = ROOT / "data" / "samples" / "car_side.png"
TUN_SCALE = 6.0  # vorticity at which the colour is ~76 % saturated (tanh)
TUN = {"n": 512, "chord": 0.36, "x": 0.13, "ground": 0.30, "rows": (0.27, 0.27 + 0.5625)}


def car_setup():
    photo = cv2.cvtColor(cv2.imread(str(CAR)), cv2.COLOR_BGR2RGB)
    mask_path = CAR.with_name("car_side_mask.npy")
    if mask_path.exists():
        mask = np.load(mask_path)
    else:  # Meta SAM 3.1, local (~2 min the first time)
        from kolmo.geometry.segment import get_segmenter
        mask = get_segmenter("auto").segment(photo, "car")
        np.save(mask_path, mask)
    n = TUN["n"]
    grid, meta = tunnel_grid(mask, n=n, chord=TUN["chord"], x=TUN["x"], ground=TUN["ground"], ground_cells=6)
    return photo, mask, grid, meta


def to_screen(meta):
    """Car box (y-up cells) -> screen rectangle (x0, y0, x1, y1) in the tunnel view."""
    n, (r_lo, r_hi) = meta["n"], TUN["rows"]
    c0, r0, w, h = meta["box"]
    sx, sy = W / n, H / ((r_hi - r_lo) * n)
    top = H - (r0 + h - r_lo * n) * sy
    return c0 * sx, top, (c0 + w) * sx, top + h * sy


def cutout(photo, mask):
    x0, y0, x1, y1 = tunnel_grid.__globals__["orient"](mask)[2]
    return photo[y0:y1, x0:x1], mask[y0:y1, x0:x1].astype(np.float32)


def paste(frame, img, alpha, rect, opacity=1.0):
    x0, y0, x1, y1 = [int(round(v)) for v in rect]
    w, h = x1 - x0, y1 - y0
    if w < 2 or h < 2:
        return frame
    im = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    a = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_AREA)[..., None] * opacity
    fx0, fy0, fx1, fy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if fx1 <= fx0 or fy1 <= fy0:
        return frame
    sl = (slice(fy0, fy1), slice(fx0, fx1))
    ss = (slice(fy0 - y0, fy1 - y0), slice(fx0 - x0, fx1 - x0))
    frame[sl] = (frame[sl] * (1 - a[ss]) + im[ss] * a[ss]).astype(np.uint8)
    return frame


def ground(frame, opacity=1.0):
    n, (r_lo, r_hi) = TUN["n"], TUN["rows"]
    top = int(round(H - (TUN["ground"] - r_lo) / (r_hi - r_lo) * H))
    band = frame[top:].astype(np.float32)
    g = np.linspace(46, 14, H - top)[:, None, None] * np.ones((1, W, 3)) * np.array([1, 1.04, 1.12])
    frame[top:] = (band * (1 - opacity) + g * opacity).astype(np.uint8)
    frame[top: top + 2] = (frame[top: top + 2] * (1 - opacity) + 150 * opacity).astype(np.uint8)
    return frame


def tunnel_sim(grid):
    f = make("fluid2d")
    p = f.validate({"n": TUN["n"], "U_inf": 1.0, "nu": 6e-4, "mask": grid, "sponge": 0.08, "ic": "zero"})
    return f, p, f.stepper(p), f.initial(p, 0)


def tunnel_view(w, scale):
    n, (r_lo, r_hi) = TUN["n"], TUN["rows"]
    rows = cv2.GaussianBlur(np.asarray(w[0]), (0, 0), 1.2)[int(r_lo * n): int(r_hi * n)][::-1]  # hides Gibbs ripples
    return colour(rows, -scale, scale, VORT, soft=True)


def car_tunnel(seconds=14):
    photo, mask, grid, meta = car_setup()
    f, p, step, s = tunnel_sim(grid)
    k = max(1, round(0.026 / p["dt"]))
    s = run_steps(step, s, int(18.0 / p["dt"])).block_until_ready()
    scale = TUN_SCALE
    img, alpha = cutout(photo, mask)
    go, out = advance(step, k), Writer("car_tunnel")
    rect = to_screen(meta)
    for _ in range(frames(seconds)):
        s = go(s)
        out(paste(ground(tunnel_view(s, scale)), img, alpha, rect))
    out.close()


def car_reveal(seconds=7.5):
    """Photo -> SAM 3.1 mask sweeps in with a glowing contour -> background falls away -> the car glides into the
    tunnel -> the wind starts."""
    photo, mask, grid, meta = car_setup()
    ph, pw = photo.shape[:2]
    s0 = min(0.62 * H / ph, 0.62 * W / pw)
    big = (int(pw * s0), int(ph * s0))
    px0, py0 = (W - big[0]) // 2, (H - big[1]) // 2
    P = cv2.resize(photo, big, interpolation=cv2.INTER_CUBIC).astype(np.float32)
    M = cv2.resize(mask.astype(np.float32), big, interpolation=cv2.INTER_AREA)
    edge = cv2.morphologyEx((M > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)).astype(np.float32)
    glow = cv2.GaussianBlur(edge, (0, 0), 9) * 3 + edge
    tint = np.array([90, 170, 255], np.float32)
    img, alpha = cutout(photo, mask)
    ys, xs = np.nonzero(mask)
    src = (px0 + xs.min() * s0, py0 + ys.min() * s0, px0 + (xs.max() + 1) * s0, py0 + (ys.max() + 1) * s0)
    dst = to_screen(meta)
    f, p, step, s = tunnel_sim(grid)
    k = max(1, round(0.026 / p["dt"]))
    go, out = advance(step, k), Writer("car_reveal")
    ease = lambda t: 0.5 - 0.5 * math.cos(math.pi * min(max(t, 0), 1))
    for i in range(frames(seconds)):
        t = i / FPS
        fr = np.zeros((H, W, 3), np.float32)
        if t < 3.6:  # photo stage
            a_in = ease(t / 0.6)
            sweep = ease((t - 0.9) / 1.1)  # SAM mask sweeps left -> right
            fall = ease((t - 2.3) / 0.9)  # background falls away
            cut = (np.arange(big[0])[None, :] < sweep * big[0]).astype(np.float32)
            m = M * cut
            layer = P * (1 - 0.35 * m[..., None]) + tint * 0.35 * m[..., None]
            layer = layer * (1 - fall * (1 - M[..., None]))  # background to black
            layer += (glow * cut)[..., None] * np.array([255, 225, 120]) * 0.55 * (1 - 0.6 * fall)
            fr[py0: py0 + big[1], px0: px0 + big[0]] = np.clip(layer, 0, 255) * a_in
            out(np.clip(fr, 0, 255).astype(np.uint8))
            continue
        u = ease((t - 3.6) / 1.4)  # glide into the tunnel
        rect = [a + (b - a) * u for a, b in zip(src, dst)]
        if t > 4.6:  # wind on: impulsive start
            s = go(s)
            flow = tunnel_view(s, TUN_SCALE).astype(np.float32) * ease((t - 4.6) / 0.8)
            fr = flow
        fr = ground(fr.astype(np.uint8), u)
        out(paste(fr, img, alpha, rect))
    out.close()


# ------------------------------------------------------------------ heat
def smooth_noise(n, seed, k=6, amp=1.0):
    rng = np.random.default_rng(seed)
    z = np.fft.rfft2(rng.standard_normal((n, n)))
    kx, ky = np.fft.rfftfreq(n, 1 / n)[None, :], np.fft.fftfreq(n, 1 / n)[:, None]
    z *= np.exp(-(kx ** 2 + ky ** 2) / k ** 2)
    f = np.fft.irfft2(z, s=(n, n))
    return amp * f / np.abs(f).max()


def heat_rt(seconds=14):
    b = make("boussinesq")
    n = 512
    p = b.validate({"n": n, "Ra": 3e7, "ic": "rayleigh_taylor", "ic_m": 3, "ic_eps": 0.04, "ic_delta": 0.05})
    s = np.asarray(b.initial(p, 0)).copy()
    x = np.arange(n) * p["L"] / n
    X, Y = np.meshgrid(x, x)
    s[1] = np.tanh(np.sin(Y + 0.05 * smooth_noise(n, 3, 10)) * p["L"] / (2 * np.pi * 0.05))  # seeded interfaces
    step = b.stepper(p)
    s = advance(step, int(1.2 / p["dt"]))(jnp.asarray(s)).block_until_ready()
    k = max(1, round((9.0 / frames(seconds) if not PREVIEW else 0.015) / p["dt"]))
    go, out = advance(step, k), Writer("heat_rt")
    for _ in range(frames(seconds)):
        s = go(s)
        out(colour(tile_x(np.asarray(s[1])), -1.05, 1.05, INFERNO))
    out.close()


def heat_plume(seconds=14):
    b = make("boussinesq")
    n = 512
    y, x = (np.mgrid[0:n, 0:n] + 0.5) / n
    mask = ((x - 0.5) ** 2 + (y - 0.12) ** 2) <= 0.045 ** 2
    p = b.validate({"n": n, "Ra": 2e7, "ic": "zero", "mask": mask, "T_body": 1.0})
    s = b.initial(p, 0).at[1].add(jnp.asarray(0.02 * smooth_noise(n, 1, 20), jnp.float32))
    step = b.stepper(p)
    s = advance(step, int(1.0 / p["dt"]))(s).block_until_ready()
    k = max(1, round((11.0 / frames(seconds) if not PREVIEW else 0.02) / p["dt"]))
    go, out = advance(step, k), Writer("heat_plume")
    r0 = int(0.06 * n)
    rows = slice(r0, r0 + int(n * H / W))
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy, rad = W * 0.5, H - (0.12 * n - r0) * H / (n * H / W), 0.045 * W
    body = np.clip((rad - np.hypot(xx - cx, yy - cy)) / 2, 0, 1)[..., None]
    halo = np.exp(-np.maximum(np.hypot(xx - cx, yy - cy) - rad, 0) / 18)[..., None]
    for _ in range(frames(seconds)):
        s = go(s)
        fr = colour(np.asarray(s[1])[rows][::-1], 0, 0.75, INFERNO).astype(np.float32)
        fr = fr + halo * np.array([255, 150, 60]) * 0.35
        fr = fr * (1 - body) + body * np.array([255, 236, 200])
        out(np.clip(fr, 0, 255).astype(np.uint8))
    out.close()


# ------------------------------------------------------------------ geophysical + patterns
def cyclone(seconds=14):
    f = make("fluid2d")
    n, L = 512, 2 * math.pi
    vort = [(L * 0.62, L * 0.40, 0.55, 7.0), (L * 0.28, L * 0.62, 0.40, -5.0), (L * 0.42, L * 0.22, 0.30, 5.0)]
    p = f.validate({"n": n, "nu": 2e-4, "beta": 2.5, "ic": "vortices", "vortices": vort})
    s = f.initial(p, 0) + 0.12 * f.initial(f.validate({"n": n, "nu": 2e-4, "ic": "random", "ic_k": (8, 16)}), 2)
    step = f.stepper(p)
    k = max(1, round((14.0 / frames(seconds) if not PREVIEW else 0.03) / p["dt"]))
    go, out = advance(step, k), Writer("cyclone")
    for _ in range(frames(seconds)):
        s = go(s)
        out(colour(tile_x(np.asarray(s[0])), 0, 3.2, VORT, soft=True))
    out.close()


def gray_scott(seconds=14):
    g = make("gray_scott")
    n = 384
    p = g.validate({"n": n, "L": 0.32 * n / 64 * 0.75, "F": 0.0367, "k": 0.0649, "dt": 1.0})
    rng = np.random.default_rng(4)
    u, v = np.ones((n, n)), np.zeros((n, n))
    for _ in range(14):
        cx, cy, r = rng.integers(0, n, 2).tolist() + [int(rng.integers(3, 7))]
        u[cy - r: cy + r, cx - r: cx + r], v[cy - r: cy + r, cx - r: cx + r] = 0.5, 0.25
    s = jnp.asarray(np.stack([u, v]) + 0.01 * rng.standard_normal((2, n, n)), jnp.float32)
    step = g.stepper(p)
    total = 22000
    out = Writer("gray_scott")
    nf = frames(seconds)
    done = 0
    for i in range(nf):  # accelerating clock: early growth slow, later filling fast
        target = int(total * ((i + 1) / nf) ** 1.6)
        s, done = run_steps(step, s, max(target - done, 0)), max(target, done)
        out(colour(tile_x(np.asarray(s[1])), 0.0, 0.38, MAKO))
    out.close()


def cahn_hilliard(seconds=12):
    c = make("cahn_hilliard")
    n = 384
    p = c.validate({"n": n, "L": 32.0 * n / 64, "dt": 0.03})
    s = c.initial(p, 7)
    step = c.stepper(p)
    nf, done = frames(seconds), 0
    out = Writer("cahn_hilliard")
    for i in range(nf):  # geometric clock: coarsening looks steady
        target = int(20 * (4000 / 20) ** (i / max(nf - 1, 1)))
        s, done = run_steps(step, s, max(target - done, 0)), max(target, done)
        out(colour(tile_x(np.asarray(s[0])), -1.05, 1.05, PHASE))
    out.close()


# ------------------------------------------------------------------ AI vs truth
CH_RUN = ROOT / "runs" / "assistant" / "20261104-075206-build-a-lightweight-nca-surrogate-for-ca" / "checkpoints" / "model_nca"


def ai_vs_truth(seconds=12):
    from kolmo.surrogates import train
    sysm = make("cahn_hilliard")  # the run's data spec is the library Cahn-Hilliard (n 64, L 32, dt 0.03)
    p = sysm.validate({})
    net = train.load(str(CH_RUN))[0]
    nstep, rstep = net.stepper(p), sysm.stepper(p)
    s0 = sysm.initial(p, 205)  # a held-out test seed of that study
    ref, ai = s0, s0
    side, gap = 560, 60
    y0 = (H - side) // 2 + 20
    xs = [(W - 3 * side - 2 * gap) // 2 + i * (side + gap) for i in range(3)]
    from PIL import Image, ImageDraw, ImageFont
    try:
        f_ = ImageFont.truetype("segoeui.ttf", 28)
    except OSError:
        f_ = ImageFont.load_default()
    im = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(im)
    labels = ("Solver (ground truth)", "Neural surrogate (NCA, 23k weights)", "|difference|")
    for x, t in zip(xs, labels):
        tw = d.textlength(t, font=f_)
        d.text((x + (side - tw) / 2, y0 - 50), t, font=f_, fill=(205, 210, 222))
    text_layer = np.asarray(im)
    nf, total = frames(seconds), 200
    traj_r, traj_a = [np.asarray(s0[0])], [np.asarray(s0[0])]
    for _ in range(total):  # 200 model steps, shown with linear in-between frames (smooth motion)
        ref, ai = rstep(ref), nstep(ai)
        traj_r.append(np.asarray(ref[0]))
        traj_a.append(np.asarray(ai[0]))
    out = Writer("ai_vs_truth")
    for i in range(nf):
        q = i / max(nf - 1, 1) * total
        j, w = min(int(q), total - 1), q - min(int(q), total - 1)
        r = (1 - w) * traj_r[j] + w * traj_r[j + 1]
        a = (1 - w) * traj_a[j] + w * traj_a[j + 1]
        fr = np.zeros((H, W, 3), np.uint8)
        m = max(float(np.abs(r).max()), 1e-6)
        for x, f_, lut, lo, hi in ((xs[0], r, PHASE, -m, m), (xs[1], a, PHASE, -m, m), (xs[2], np.abs(a - r), INFERNO, 0, m)):
            fr[y0: y0 + side, x: x + side] = colour(f_[::-1], lo, hi, lut, (side, side))
        fr[:y0] = np.maximum(fr[:y0], text_layer[:y0])
        out(fr)
    out.close()


CLIPS = {f.__name__: f for f in (car_reveal, car_tunnel, heat_rt, heat_plume, cyclone, gray_scott, cahn_hilliard,
                                 ai_vs_truth)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help=" | ".join(CLIPS))
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()
    PREVIEW = a.preview
    (OUT / "stills").mkdir(parents=True, exist_ok=True)
    for name in a.names or CLIPS:
        CLIPS[name]()
