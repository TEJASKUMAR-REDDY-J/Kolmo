"""Surrogate networks (Flax linen) and the Surrogate wrapper that turns a trained network into a core.Stepper.

I/O of every network: `net.apply(params, z, c)` with
  z: (B, C, H, W) float32, the state in NORMALISED units, z = (s - mean[c]) / std[c] per channel (train-set stats);
  c: (B, P) float32, normalised conditioning parameters (p - cond_mean) / cond_std, P may be 0;
returns z_next (B, C, H, W) = z + f(z, c): a residual update in normalised units. The last layer is zero-initialised,
so an untrained network is the persistence model z_next = z.
Parameter conditioning: each of the P parameters becomes one constant input plane (its normalised value), concatenated
with the state channels. Position: `pos` harmonics add planes sin/cos(k x), sin/cos(k y) for k = 1..pos (x, y in
[0, 2 pi) over the periodic box); they break translation equivariance on purpose, needed when forcing is fixed in space
(Kolmogorov forcing sin(4 y) -> pos >= 4). Input channels to the network: C + P + 4 * pos.
One network step = one surrogate step = `spacing` solver steps (stored in the checkpoint metadata)."""
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import tracers as _flax_tracers

# ponytail: Flax 0.8.4 reads Trace.level, which JAX 0.7 removed; comparing the trace objects themselves keeps its
# "module used across a transform boundary" check working. Drop this shim when Flax is upgraded.
_flax_tracers.trace_level = lambda trace: trace


def _planes(z, c, pos):
    B, _, H, W = z.shape
    parts = [z, jnp.broadcast_to(c[:, :, None, None], (B, c.shape[1], H, W))]
    if pos:
        k = jnp.arange(1, pos + 1, dtype=z.dtype)[:, None]
        x = k * jnp.arange(W, dtype=z.dtype)[None] * (2 * np.pi / W)  # (pos, W)
        y = k * jnp.arange(H, dtype=z.dtype)[None] * (2 * np.pi / H)  # (pos, H)
        px = jnp.concatenate([jnp.sin(x), jnp.cos(x)])[:, None, :].repeat(H, 1)
        py = jnp.concatenate([jnp.sin(y), jnp.cos(y)])[:, :, None].repeat(W, 2)
        parts.append(jnp.broadcast_to(jnp.concatenate([px, py])[None], (B, 4 * pos, H, W)))
    return jnp.moveaxis(jnp.concatenate(parts, 1), 1, -1)  # (B, H, W, C + P + 4 pos) for flax


class ResCNN(nn.Module):
    """`depth` periodic 'same' convolutions (kernel x kernel, `width` channels, GELU), then a zero-init conv to C."""
    width: int = 32
    depth: int = 4
    kernel: int = 3
    pos: int = 0

    @nn.compact
    def __call__(self, z, c):
        h, p = _planes(z, c, self.pos), self.kernel // 2
        wrap = lambda a: jnp.pad(a, ((0, 0), (p, p), (p, p), (0, 0)), mode="wrap")
        for _ in range(self.depth):
            h = nn.gelu(nn.Conv(self.width, (self.kernel, self.kernel), padding="VALID")(wrap(h)))
        dz = nn.Conv(z.shape[1], (self.kernel, self.kernel), padding="VALID", kernel_init=nn.initializers.zeros)(wrap(h))
        return z + jnp.moveaxis(dz, -1, 1)


class SpectralConv(nn.Module):
    """Learned complex multiplier on the lowest `modes` x `modes` Fourier modes (both signs of ky), others zeroed."""
    width: int
    modes: int

    @nn.compact
    def __call__(self, h):  # (B, H, W, width)
        B, H, W, C = h.shape
        m = self.modes
        init = lambda key, shape: jax.random.normal(key, shape) / (C * self.width)
        w = self.param("w", init, (2, 2, m, m, C, self.width))  # (block, re/im, ...)
        w = w[:, 0] + 1j * w[:, 1]
        f = jnp.fft.rfft2(h, axes=(1, 2))
        out = jnp.zeros((B, H, W // 2 + 1, self.width), f.dtype)
        out = out.at[:, :m, :m].set(jnp.einsum("bxyi,xyio->bxyo", f[:, :m, :m], w[0]))
        out = out.at[:, -m:, :m].set(jnp.einsum("bxyi,xyio->bxyo", f[:, -m:, :m], w[1]))
        return jnp.fft.irfft2(out, s=(H, W), axes=(1, 2)).astype(h.dtype)


class FNO(nn.Module):
    """Fourier neural operator: 1x1 lift to `width`, `depth` blocks h <- GELU(SpectralConv(h) + 1x1(h)), zero-init
    1x1 projection to C. Resolution-independent weights; `modes` must be <= min(H, W) / 2."""
    width: int = 24
    modes: int = 12
    depth: int = 4
    pos: int = 0

    @nn.compact
    def __call__(self, z, c):
        h = nn.Dense(self.width)(_planes(z, c, self.pos))
        for _ in range(self.depth):
            h = nn.gelu(SpectralConv(self.width, self.modes)(h) + nn.Dense(self.width)(h))
        dz = nn.Dense(z.shape[1], kernel_init=nn.initializers.zeros)(nn.gelu(nn.Dense(2 * self.width)(h)))
        return z + jnp.moveaxis(dz, -1, 1)


ARCHS = {"cnn": ResCNN, "fno": FNO}


def build(arch: dict) -> nn.Module:
    """arch = {"kind": "cnn" | "fno", **constructor kwargs}."""
    a = dict(arch)
    kind = a.pop("kind")
    if kind not in ARCHS:
        raise ValueError(f"unknown model kind {kind!r}; known: {sorted(ARCHS)}")
    return ARCHS[kind](**a)


@dataclass
class Surrogate:
    """A trained network plus everything needed to apply it to physical states."""
    net: nn.Module
    params: dict
    mean: np.ndarray  # (C,) physical units
    std: np.ndarray  # (C,)
    cond_names: tuple  # conditioning parameter names, in channel order
    cond_mean: np.ndarray  # (P,)
    cond_std: np.ndarray  # (P,)
    spacing: int  # solver steps per surrogate step

    def cond(self, params: dict) -> np.ndarray:
        """(P,) normalised conditioning vector for a parameter dict."""
        raw = np.array([float(params[k]) for k in self.cond_names], np.float32)
        return (raw - self.cond_mean) / self.cond_std

    def stepper(self, params: dict):
        """Jitted physical (C, H, W) -> physical (C, H, W), advancing spacing * dt. Never alters `params`."""
        c = jnp.asarray(self.cond(params))[None]
        mean, std = jnp.asarray(self.mean)[:, None, None], jnp.asarray(self.std)[:, None, None]
        return jax.jit(lambda s: self.net.apply(self.params, ((s - mean) / std)[None], c)[0] * std + mean)
