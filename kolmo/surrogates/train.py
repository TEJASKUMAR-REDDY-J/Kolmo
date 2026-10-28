"""Training: one-step (unroll = 1) or rollout-aware (unroll = K > 1), early stopping on validation rollout error.

Loss (normalised units): mean over k = 1..K of MSE(z_k, target_k), z_k = net(z_{k-1}, c), z_0 = a train state.
pushforward=True puts stop_gradient on every step input, so gradients flow through one network application at a time
but inputs come from the model's own rollout (pushforward trick, Brandstetter et al. 2022); False backpropagates
through the whole unroll. Validation metric: mean over val trajectories and surrogate steps 1..val_horizon of the
relative L2 error ||s_pred - s_ref|| / ||s_ref|| in physical units, rolled out from step 0 (non-finite -> 1e3).
Determinism: numpy Generator(seed) picks windows, PRNGKey(seed) initialises weights; same config -> same weights.
Checkpoint: <path>.npz (flattened params, keys joined by '/') + <path>.json (metadata, see save())."""
import json
import os
import time
from dataclasses import asdict, dataclass, field

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from kolmo.surrogates.models import Surrogate, build

VERSION = "kolmo-surrogates 0.1"


@dataclass
class TrainConfig:
    model: dict = field(default_factory=lambda: {"kind": "cnn"})  # models.build() spec
    unroll: int = 1  # K: 1 = one-step training, >1 = rollout-aware
    pushforward: bool = False  # stop-gradient between unrolled steps
    steps: int = 1000  # optimiser steps (upper bound; early stopping may end sooner)
    batch: int = 16
    lr: float = 1e-3  # Adam, cosine decay to 0 over `steps`, global-norm clip 1
    seed: int = 0
    eval_every: int = 100
    val_horizon: int = 20  # surrogate steps in the validation rollout
    patience: int = 5  # evaluations without improvement before stopping
    max_minutes: float = 20.0  # wall-clock cap

    def validate(self):
        if self.unroll < 1 or self.steps < 1 or self.batch < 1 or self.val_horizon < 1 or self.patience < 1:
            raise ValueError("unroll, steps, batch, val_horizon, patience must be >= 1")
        if not self.lr > 0:
            raise ValueError("lr must be positive")
        build(self.model)
        return self


def _norm(ds, states):
    return (states - ds.mean[:, None, None]) / ds.std[:, None, None]


def surrogate_from(ds, net, params) -> Surrogate:
    return Surrogate(net, params, ds.mean, ds.std, tuple(ds.config.cond), ds.cond_mean, ds.cond_std, ds.config.spacing)


def train(ds, cfg: TrainConfig, verbose=True):
    """Returns (Surrogate with the best-validation params, history list of dicts)."""
    cfg.validate()
    net = build(cfg.model)
    tr, va = ds.splits["train"], ds.splits["val"]
    if cfg.unroll >= tr.states.shape[1]:
        raise ValueError("unroll must be shorter than the trajectories")
    X = jnp.asarray(_norm(ds, tr.states))  # (N, T+1, C, H, W)
    Cn = jnp.asarray((tr.cond - ds.cond_mean) / ds.cond_std)
    H = min(cfg.val_horizon, va.states.shape[1] - 1)
    V0 = jnp.asarray(_norm(ds, va.states[:, 0]))
    Vref = jnp.asarray(va.states[:, 1:H + 1])
    Vc = jnp.asarray((va.cond - ds.cond_mean) / ds.cond_std)
    mean, std = jnp.asarray(ds.mean)[:, None, None], jnp.asarray(ds.std)[:, None, None]

    params = net.init(jax.random.PRNGKey(cfg.seed), X[:1, 0], Cn[:1])
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(optax.cosine_decay_schedule(cfg.lr, cfg.steps)))
    opt_state = opt.init(params)

    # ponytail: Python loops, not lax.scan: on XLA CPU (JAX 0.7.1) convolutions inside a loop ran ~90x slower
    # (measured: 5-layer 64x64 CNN gradient 11.7 s in lax.scan vs 0.13 s straight-line). Compile time grows with K, H.
    def loss_fn(p, z, c, targets):  # targets (B, K, C, H, W)
        losses = []
        for k in range(targets.shape[1]):
            z = net.apply(p, jax.lax.stop_gradient(z) if cfg.pushforward else z, c)
            losses.append(jnp.mean((z - targets[:, k]) ** 2))
        return jnp.mean(jnp.stack(losses))

    @jax.jit
    def update(p, o, z, c, targets):
        loss, g = jax.value_and_grad(loss_fn)(p, z, c, targets)
        u, o = opt.update(g, o, p)
        return optax.apply_updates(p, u), o, loss

    @jax.jit
    def val_error(p):
        z, preds = V0, []
        for _ in range(H):
            z = net.apply(p, z, Vc)
            preds.append(z)
        pred = jnp.stack(preds, 1) * std + mean  # (Nv, H, C, H, W)
        err = jnp.sqrt(((pred - Vref) ** 2).sum((2, 3, 4)) / (Vref ** 2).sum((2, 3, 4)))
        return jnp.where(jnp.isfinite(err), err, 1e3).mean()

    rng = np.random.default_rng(cfg.seed)
    N, T1 = X.shape[:2]
    K = cfg.unroll
    best, best_params, bad, hist, losses = np.inf, params, 0, [], []
    t0 = time.perf_counter()
    for it in range(1, cfg.steps + 1):
        i = rng.integers(0, N, cfg.batch)
        t = rng.integers(0, T1 - K, cfg.batch)
        win = X[i[:, None], t[:, None] + np.arange(K + 1)[None]]  # (B, K+1, C, H, W)
        params, opt_state, loss = update(params, opt_state, win[:, 0], Cn[i], win[:, 1:])
        losses.append(loss)
        if it % cfg.eval_every == 0 or it == cfg.steps:
            v = float(val_error(params))
            minutes = (time.perf_counter() - t0) / 60
            hist.append({"step": it, "train_loss": float(np.mean(jnp.stack(losses))), "val_rollout_rel_l2": v,
                         "minutes": round(minutes, 3)})
            losses = []
            if verbose:
                print(f"  step {it:5d}  loss {hist[-1]['train_loss']:.3e}  val rollout relL2 {v:.4f}  {minutes:.1f} min",
                      flush=True)
            if v < best:
                best, best_params, bad = v, params, 0
            else:
                bad += 1
            if bad >= cfg.patience or minutes > cfg.max_minutes:
                hist[-1]["stopped"] = "patience" if bad >= cfg.patience else "time"
                break
    return surrogate_from(ds, net, best_params), hist


def save(path, sur: Surrogate, ds, cfg: TrainConfig, history, train_minutes=None):
    """<path>.npz params and <path>.json metadata: version, architecture, normalisation, data config + hash,
    train config, history, library versions."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    flat = flax.traverse_util.flatten_dict(sur.params, sep="/")
    np.savez(path + ".npz", **{k: np.asarray(v) for k, v in flat.items()})
    meta = {"version": VERSION, "arch": cfg.model, "spacing": sur.spacing, "fields": list(ds.fields),
            "norm": {"mean": sur.mean.tolist(), "std": sur.std.tolist(), "cond_names": list(sur.cond_names),
                     "cond_mean": sur.cond_mean.tolist(), "cond_std": sur.cond_std.tolist()},
            "data": asdict(ds.config), "data_hash": ds.hash, "train": asdict(cfg), "history": history, "train_minutes": train_minutes,
            "libs": {"jax": jax.__version__, "flax": flax.__version__, "optax": optax.__version__, "numpy": np.__version__}}
    with open(path + ".json", "w") as f:
        json.dump(meta, f, indent=1)


def load(path) -> tuple[Surrogate, dict]:
    with open(path + ".json") as f:
        meta = json.load(f)
    z = np.load(path + ".npz")
    params = flax.traverse_util.unflatten_dict({k: jnp.asarray(z[k]) for k in z.files}, sep="/")
    n = meta["norm"]
    f32 = lambda v: np.asarray(v, np.float32)
    sur = Surrogate(build(meta["arch"]), params, f32(n["mean"]), f32(n["std"]), tuple(n["cond_names"]),
                    f32(n["cond_mean"]), f32(n["cond_std"]), meta["spacing"])
    return sur, meta
