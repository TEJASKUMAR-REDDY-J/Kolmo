"""3D CFD clip: two vortex rings collide head-on (incompressible Navier-Stokes, pseudo-spectral, JAX), volume-rendered
by ray marching with an orbiting camera. 1920x1080, 60 fps -> video/clips/rings3d.mp4.

    python tools/render_3d.py            # full clip
    python tools/render_3d.py --check    # solver self-check only (divergence, energy decay)

Solver: velocity form du/dt = P[u x w] + nu lap u on a periodic (2 pi)^3 box, 2/3 dealiasing, Lawson RK4 (diffusion
integrated exactly), P = Leray projection. Rings: Gaussian-core vortex tubes with opposite circulation facing each
other along z, with a small azimuthal wobble so they break up into small rings after the collision."""
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
from jax.scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "video" / "clips"
W, H, FPS = 1920, 1080, 60
N, L = 96, 2 * math.pi
NU = 1.5e-3


# ------------------------------------------------------------------ solver
def wavenumbers(n):
    k = np.fft.fftfreq(n, 1 / n)
    kz = np.fft.rfftfreq(n, 1 / n)
    KX, KY, KZ = np.meshgrid(k, k, kz, indexing="ij")
    K = np.stack([KX, KY, KZ]).astype(np.float32)
    k2 = (K ** 2).sum(0)
    keep = (np.abs(KX) < n / 3) & (np.abs(KY) < n / 3) & (np.abs(KZ) < n / 3)
    return jnp.asarray(K), jnp.asarray(k2), jnp.asarray(np.where(k2 == 0, 1, k2)), jnp.asarray(keep.astype(np.float32))


K, K2, K2S, DEAL = wavenumbers(N)
fft = lambda f: jnp.fft.rfftn(f, axes=(1, 2, 3))
ifft = lambda f: jnp.fft.irfftn(f, s=(N, N, N), axes=(1, 2, 3))


def project(fh):
    return fh - K * (K * fh).sum(0) / K2S


def curl_hat(uh):
    kx, ky, kz = K
    ux, uy, uz = uh
    return 1j * jnp.stack([ky * uz - kz * uy, kz * ux - kx * uz, kx * uy - ky * ux])


def rhs(uh):
    u, w = ifft(uh), ifft(curl_hat(uh))
    return project(fft(jnp.cross(u, w, axis=0))) * DEAL


@jax.jit
def step(uh, dt):
    """Lawson RK4: v = exp(nu k^2 t) u."""
    e1, e2 = jnp.exp(-NU * K2 * dt / 2), jnp.exp(-NU * K2 * dt)
    a = rhs(uh)
    b = rhs(e1 * (uh + dt / 2 * a))
    c = rhs(e1 * uh + dt / 2 * b)
    d = rhs(e2 * uh + dt * e1 * c)
    return e2 * uh + dt / 6 * (e2 * a + 2 * e1 * (b + c) + d)


def rings():
    """Vorticity of two rings (radius R, core a, circulation +-G) at z = L/2 -+ s, axis z, then u = curl^-1 w."""
    x = np.arange(N) * L / N
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    w = np.zeros((3, N, N, N))
    R, a, G, s = 1.15, 0.22, 2.4, 1.05
    rng = np.random.default_rng(1)
    for z0, sign in ((L / 2 - s, 1.0), (L / 2 + s, -1.0)):
        dx, dy, dz = X - L / 2, Y - L / 2, Z - z0
        rho = np.sqrt(dx ** 2 + dy ** 2) + 1e-9
        phi = np.arctan2(dy, dx)
        wob = R * (1 + sum(0.02 * rng.normal() * np.cos(m * phi + rng.uniform(0, 6.3)) for m in range(3, 9)))
        core = G / (math.pi * a * a) * np.exp(-((rho - wob) ** 2 + dz ** 2) / (a * a))
        w[0] += sign * core * (-dy / rho)
        w[1] += sign * core * (dx / rho)
    wh = jnp.asarray(np.fft.rfftn(w, axes=(1, 2, 3)), jnp.complex64)
    kx, ky, kz = K
    wx, wy, wz = project(wh)
    uh = 1j * jnp.stack([ky * wz - kz * wy, kz * wx - kx * wz, kx * wy - ky * wx]) / K2S  # u = curl w / k^2
    return (uh * DEAL).astype(jnp.complex64)


def energy(uh):
    return float(0.5 * jnp.mean(ifft(uh) ** 2) * 3)


def simulate(t_end=8.0, snaps=240):
    uh = rings()
    u_max = float(jnp.abs(ifft(uh)).max())
    dt = 0.35 * (L / N) / max(u_max, 1e-6)
    every = t_end / snaps
    sub = max(1, math.ceil(every / dt))
    dt = every / sub
    print(f"3D NS {N}^3, nu {NU}, dt {dt:.4f}, {snaps * sub} steps", flush=True)
    vols, e0, t0 = [], energy(uh), time.perf_counter()
    mag = jax.jit(lambda uh: jnp.sqrt((ifft(curl_hat(uh)) ** 2).sum(0)))
    for i in range(snaps):
        for _ in range(sub):
            uh = step(uh, dt)
        vols.append(np.asarray(mag(uh), np.float16))
        if i % 40 == 0:
            print(f"  t {(i + 1) * every:5.2f}  energy/E0 {energy(uh) / e0:.4f}  {time.perf_counter() - t0:.0f} s", flush=True)
    return vols


# ------------------------------------------------------------------ volume renderer
RW, RH, NS = 640, 360, 210  # render size (upsampled to 1080p), samples per ray


def ramp(points):
    x = np.linspace(0, 1, 256)
    return np.stack([np.interp(x, [p for p, _ in points], [c[i] for _, c in points]) for i in range(3)], 1) / 255


CMAP = jnp.asarray(ramp([(0, (20, 40, 120)), (0.3, (40, 120, 255)), (0.55, (90, 230, 255)), (0.8, (255, 210, 140)),
                         (1, (255, 255, 255))]), jnp.float32)


@jax.jit
def render(vol, eye, target, scale):
    """Emission-absorption ray march through the box [0, N)^3 (index units, periodic volume, clipped to one box)."""
    fwd = target - eye
    fwd = fwd / jnp.linalg.norm(fwd)
    right = jnp.cross(fwd, jnp.array([0.0, 0.0, 1.0]))
    right = right / jnp.linalg.norm(right)
    up = jnp.cross(right, fwd)
    fov = 0.62
    sx = (jnp.arange(RW) + 0.5) / RW * 2 - 1
    sy = (jnp.arange(RH) + 0.5) / RH * 2 - 1
    d = fwd[None, None] + fov * (sx[None, :, None] * right + (-sy[:, None, None] * RH / RW) * up)
    d = d / jnp.linalg.norm(d, axis=-1, keepdims=True)
    dist = jnp.linalg.norm(target - eye)
    ts = jnp.linspace(dist - 0.9 * N, dist + 0.9 * N, NS)
    p = eye + d[..., None, :] * ts[:, None]  # (RH, RW, NS, 3)
    inside = jnp.all((p >= 0) & (p <= N - 1), -1)
    v = map_coordinates(vol, [p[..., 0], p[..., 1], p[..., 2]], order=1, mode="wrap")
    x = jnp.clip(v / scale, 0, 1)
    alpha = jnp.where(inside, 1 - jnp.exp(-2.2 * (1.8 * N / NS) * jnp.clip((x - 0.12) / 0.88, 0, 1) ** 1.6), 0.0)
    h = jnp.clip((p[..., 2] - 0.3 * N) / (0.4 * N), 0, 1)[..., None]  # lower ring warm, upper ring cool
    tint = (1 - h) * jnp.array([1.0, 0.45, 0.16]) + h * jnp.array([0.2, 0.62, 1.0])
    col = tint * (0.22 + 0.85 * x[..., None]) + 0.75 * x[..., None] ** 3  # white-hot cores
    trans = jnp.cumprod(jnp.concatenate([jnp.ones_like(alpha[..., :1]), 1 - alpha[..., :-1]], -1), -1)
    return ((trans * alpha)[..., None] * col).sum(-2)  # (RH, RW, 3), premultiplied, ~0..1+


def box_edges(img, eye, target):
    """Thin glowing edges of the simulation box, projected with the same camera."""
    fwd = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    c = np.array([[i, j, k] for i in (0, N - 1) for j in (0, N - 1) for k in (0, N - 1)], float)

    def proj(q):
        r = q - eye
        z = r @ fwd
        return (r @ right / z / 0.62 * 0.5 + 0.5) * W, (0.5 - r @ up / z / 0.62 * 0.5 * W / H) * H

    layer = np.zeros_like(img)
    for a in range(8):
        for b in range(a + 1, 8):
            if np.sum(c[a] != c[b]) == 1:
                (x0, y0), (x1, y1) = proj(c[a]), proj(c[b])
                cv2.line(layer, (int(x0 * 16), int(y0 * 16)), (int(x1 * 16), int(y1 * 16)), (70, 90, 130), 1,
                         cv2.LINE_AA, shift=4)
    return np.maximum(img, layer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--seconds", type=float, default=14.0)
    a = ap.parse_args()
    if a.check:
        uh = rings()
        div = float(jnp.abs(ifft(1j * (K * uh).sum(0)[None])).max())
        e0 = energy(uh)
        for _ in range(20):
            uh = step(uh, 0.01)
        e1 = energy(uh)
        assert div < 1e-3 and 0.9 * e0 < e1 < e0, (div, e0, e1)
        print("ok: divergence", div, "energy", e0, "->", e1)
        return
    vols = simulate()
    scale = float(np.percentile(np.asarray(vols[0], np.float32), 99.95)) * 0.8
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stills").mkdir(exist_ok=True)
    path = OUT / "rings3d.mp4"
    ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt",
                           "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                           "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
    nf = int(a.seconds * FPS)
    target = np.array([N / 2, N / 2, N / 2])
    t0 = time.perf_counter()
    for i in range(nf):
        q = i / (nf - 1) * (len(vols) - 1)
        j = min(int(q), len(vols) - 2)
        w = q - j
        vol = jnp.asarray((1 - w) * vols[j].astype(np.float32) + w * vols[j + 1].astype(np.float32))
        ang = math.radians(30 + 75 * i / nf)
        elev = math.radians(18 + 10 * math.sin(math.pi * i / nf))
        r = (1.75 - 0.25 * i / nf) * N  # slow push-in
        eye = target + r * np.array([math.cos(ang) * math.cos(elev), math.sin(ang) * math.cos(elev), math.sin(elev)])
        img = np.asarray(render(vol, jnp.asarray(eye, jnp.float32), jnp.asarray(target, jnp.float32), scale))
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)
        glow = cv2.GaussianBlur(img, (0, 0), 14) * 0.55  # soft bloom
        img = 1 - np.exp(-1.25 * (img + glow))  # filmic tone map
        bg = np.linspace(0.035, 0.0, H)[:, None, None] * np.array([0.5, 0.6, 1.0])
        frame = (np.clip(img + bg * (1 - img), 0, 1) * 255).astype(np.uint8)
        frame = box_edges(frame, eye, target)
        ff.stdin.write(frame.tobytes())
        if i in (FPS, nf // 2, nf - 2):
            cv2.imwrite(str(OUT / "stills" / f"rings3d_{i}.png"), frame[..., ::-1])
        if i % 120 == 0:
            print(f"  frame {i}/{nf}  {time.perf_counter() - t0:.0f} s", flush=True)
    ff.stdin.close()
    ff.wait()
    print(f"{path.name}: {nf / FPS:.1f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
