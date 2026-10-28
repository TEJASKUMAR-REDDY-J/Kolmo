"""JEPA-style latent world model: encode once, roll the dynamics forward in latent space, decode only to read out.

Model (all convolutions periodic; NHWC inside, the I/O contract of models.py outside):
  encoder  E: z (B, C, H, W) -> h (B, H/p, W/p, D): non-overlapping p x p patch conv, GELU, 3x3 conv
  predictor P: h, c -> h + f(h, c): residual 3-conv CNN on the latent grid with the conditioning plane and `pos`
              position harmonics (models._planes), last conv zero-initialised (untrained P = latent persistence)
  decoder  G: h -> z: 3x3 conv, GELU, 1x1 conv to C p^2 channels, pixel shuffle back to (B, C, H, W)
Objective per window x_0..x_K (normalised units, train-set stats as models.py), h_0 = E(x_0), h_k = P(h_{k-1}, c):
  L = mean_k ||h_k - sg(E_ema(x_k))||^2                                   JEPA latent prediction (I-JEPA, Assran 2023)
    + w_var * mean_d relu(1 - std_d(E(x)))  +  w_cov * sum_{i!=j} Cov_ij(E(x))^2 / D   VICReg (Bardes et al. 2022)
    + w_rec * mean_k ||G(E(x_k)) - x_k||^2                                 small autoencoding term (states comparable)
    + mean_k ||G(sg(h_k)) - x_k||^2                                        trains G only, on predicted latents
Ablation (w_jepa=0, pixel_grad=True): same network trained by decoded-rollout pixel loss, i.e. a latent but not JEPA model.
E_ema is an exponential moving average of E (rate `ema`), never differentiated; std/Cov are over batch x latent grid,
per latent channel d. rec_grad=False stops the reconstruction gradient at the encoder (pure JEPA, G is a probe).
Rollout (`LatentSurrogate.stepper`): the latent state is carried between calls and decoded for output; it is
re-encoded only when the input is not the stepper's own previous output (first step, or after a solver intervention
in kolmo.hybrid). mode="reencode" re-encodes every step (a pixel-space loop through the latent model).
Checkpoint: the same <path>.npz + <path>.json as train.save, arch {"kind": "latent", ...}; importing this module
registers "latent" in models.ARCHS, so train.load then returns a plain Surrogate (re-encode mode). load() here returns
the LatentSurrogate (latent rollout). Run: python -m kolmo.surrogates.latent benchmarks/long_horizon.json OUT_DIR"""
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from kolmo.surrogates import models
from kolmo.surrogates.models import PeriodicConv, Surrogate, _planes
from kolmo.surrogates.train import TrainConfig, _norm, save


class LatentWM(nn.Module):
    latent: int = 24  # D, latent channels
    patch: int = 4  # p, latent grid = (H/p, W/p)
    width: int = 24  # predictor / decoder hidden channels
    pos: int = 0
    out: int = 1  # C, state channels

    def setup(self):
        self.e1 = nn.Conv(self.latent, (self.patch, self.patch), strides=self.patch, padding="VALID")
        self.e2 = PeriodicConv(self.latent)
        self.p1, self.p2 = PeriodicConv(self.width), PeriodicConv(self.width)
        self.p3 = PeriodicConv(self.latent, zero=True)
        self.d1 = PeriodicConv(self.width)
        self.d2 = nn.Conv(self.out * self.patch ** 2, (1, 1))

    def encode(self, z):  # (B, C, H, W) -> (B, h, w, D)
        return self.e2(nn.gelu(self.e1(jnp.moveaxis(z, 1, -1))))

    def predict(self, h, c):
        g = _planes(jnp.moveaxis(h, -1, 1), c, self.pos)
        return h + self.p3(nn.gelu(self.p2(nn.gelu(self.p1(g)))))

    def decode(self, h):  # (B, h, w, D) -> (B, C, H, W)
        y = self.d2(nn.gelu(self.d1(h)))
        B, hh, ww, _ = y.shape
        p, C = self.patch, self.out
        y = y.reshape(B, hh, ww, p, p, C).transpose(0, 5, 1, 3, 2, 4)
        return y.reshape(B, C, hh * p, ww * p)

    def __call__(self, z, c):  # models.py contract: one step, re-encoding
        return self.decode(self.predict(self.encode(z), c))


models.ARCHS["latent"] = LatentWM


@dataclass
class LatentConfig(TrainConfig):
    ema: float = 0.99  # target-encoder EMA rate
    w_var: float = 1.0  # VICReg variance hinge weight
    w_cov: float = 0.04  # VICReg covariance weight (VICReg 25:25:1 ratio of invariance:variance:covariance)
    w_rec: float = 0.1  # autoencoding weight
    rec_grad: bool = True  # False: reconstruction does not shape the encoder (pure JEPA + probe decoder)
    w_jepa: float = 1.0  # latent prediction weight (ablation: 0 together with pixel_grad=True)
    pixel_grad: bool = False  # ablation: decoded-rollout pixel loss trains P and E too (no stop-gradient)


@dataclass
class LatentSurrogate(Surrogate):
    mode: str = "latent"  # "latent" | "reencode"

    def stepper(self, params: dict):
        c = jnp.asarray(self.cond(params))[None]
        mean, std = jnp.asarray(self.mean)[:, None, None], jnp.asarray(self.std)[:, None, None]
        a = lambda m, *x: self.net.apply(self.params, *x, method=m)
        enc = jax.jit(lambda s: a(LatentWM.encode, ((s - mean) / std)[None]))
        fwd = jax.jit(lambda h: (lambda h2: (h2, a(LatentWM.decode, h2)[0] * std + mean))(a(LatentWM.predict, h, c)))
        last = [None, None]  # (previous output array, its latent)

        # ponytail: carries the latent by object identity of the last output; a copied/modified input re-encodes.
        def step(s):
            h = last[1] if self.mode == "latent" and s is last[0] else enc(s)
            last[1], last[0] = fwd(h)
            return last[0]
        return step


def n_params(params):
    return int(sum(np.size(x) for x in jax.tree_util.tree_leaves(params)))


def _vicreg(h):
    x = h.reshape(-1, h.shape[-1])
    x = x - x.mean(0)
    std = jnp.sqrt(x.var(0) + 1e-4)
    cov = x.T @ x / (x.shape[0] - 1)
    off = cov - jnp.diag(jnp.diag(cov))
    return jnp.mean(jax.nn.relu(1.0 - std)), jnp.sum(off ** 2) / x.shape[1], jnp.mean(std)


def train(ds, cfg: LatentConfig, verbose=True):
    """Returns (LatentSurrogate with best-validation params, history). Validation = models.py baseline metric: relative
    L2 in physical units over val_horizon decoded latent-rollout steps from step 0."""
    cfg.validate()
    C = ds.splits["train"].states.shape[2]
    net = models.build({**cfg.model, "out": C})
    tr, va = ds.splits["train"], ds.splits["val"]
    if cfg.unroll >= tr.states.shape[1]:
        raise ValueError("unroll must be shorter than the trajectories")
    X = jnp.asarray(_norm(ds, tr.states))
    Cn = jnp.asarray((tr.cond - ds.cond_mean) / ds.cond_std)
    H = min(cfg.val_horizon, va.states.shape[1] - 1)
    V0 = jnp.asarray(_norm(ds, va.states[:, 0]))
    Vref = jnp.asarray(va.states[:, 1:H + 1])
    Vc = jnp.asarray((va.cond - ds.cond_mean) / ds.cond_std)
    mean, std = jnp.asarray(ds.mean)[:, None, None], jnp.asarray(ds.std)[:, None, None]
    params = net.init(jax.random.PRNGKey(cfg.seed), X[:1, 0], Cn[:1])
    target = params
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(optax.cosine_decay_schedule(cfg.lr, cfg.steps)))
    opt_state = opt.init(params)
    A = lambda p, m, *x: net.apply(p, *x, method=m)
    sg = jax.lax.stop_gradient

    # ponytail: Python loop over K, not lax.scan (train.py: XLA CPU convs in loops ~90x slower)
    def loss_fn(p, tp, win, c):  # win (B, K+1, C, H, W)
        B, K1 = win.shape[:2]
        flat = win.reshape((B * K1,) + win.shape[2:])
        e = A(p, LatentWM.encode, flat)
        tgt = sg(A(tp, LatentWM.encode, flat)).reshape((B, K1) + e.shape[1:])
        rec = jnp.mean((A(p, LatentWM.decode, e if cfg.rec_grad else sg(e)) - flat) ** 2)
        h, pred, prec = e.reshape((B, K1) + e.shape[1:])[:, 0], [], []
        for k in range(1, K1):
            h = A(p, LatentWM.predict, h, c)
            pred.append(jnp.mean((h - tgt[:, k]) ** 2))
            prec.append(jnp.mean((A(p, LatentWM.decode, h if cfg.pixel_grad else sg(h)) - win[:, k]) ** 2))
        v, cv, s = _vicreg(e)
        jepa = jnp.mean(jnp.stack(pred))
        total = cfg.w_jepa * jepa + cfg.w_var * v + cfg.w_cov * cv + cfg.w_rec * rec + jnp.mean(jnp.stack(prec))
        return total, jnp.stack([jepa, v, cv, rec, s])

    @jax.jit
    def update(p, tp, o, win, c):
        (loss, parts), g = jax.value_and_grad(loss_fn, has_aux=True)(p, tp, win, c)
        u, o = opt.update(g, o, p)
        p = optax.apply_updates(p, u)
        tp = jax.tree_util.tree_map(lambda t, q: cfg.ema * t + (1 - cfg.ema) * q, tp, p)
        return p, tp, o, loss, parts

    @jax.jit
    def val_error(p):
        h, preds = A(p, LatentWM.encode, V0), []
        for _ in range(H):
            h = A(p, LatentWM.predict, h, Vc)
            preds.append(A(p, LatentWM.decode, h))
        pred = jnp.stack(preds, 1) * std + mean
        err = jnp.sqrt(((pred - Vref) ** 2).sum((2, 3, 4)) / (Vref ** 2).sum((2, 3, 4)))
        return jnp.where(jnp.isfinite(err), err, 1e3).mean()

    rng = np.random.default_rng(cfg.seed)
    N, T1 = X.shape[:2]
    K = cfg.unroll
    best, best_params, bad, hist, losses, parts_acc = np.inf, params, 0, [], [], []
    t0 = time.perf_counter()
    for it in range(1, cfg.steps + 1):
        i = rng.integers(0, N, cfg.batch)
        t = rng.integers(0, T1 - K, cfg.batch)
        win = X[i[:, None], t[:, None] + np.arange(K + 1)[None]]
        params, target, opt_state, loss, parts = update(params, target, opt_state, win, Cn[i])
        losses.append(loss)
        parts_acc.append(parts)
        if it % cfg.eval_every == 0 or it == cfg.steps:
            v = float(val_error(params))
            minutes = (time.perf_counter() - t0) / 60
            pm = np.asarray(jnp.stack(parts_acc)).mean(0)
            hist.append({"step": it, "train_loss": float(np.mean(jnp.stack(losses))), "val_rollout_rel_l2": v,
                         "minutes": round(minutes, 3), **dict(zip(("jepa", "var", "cov", "rec", "latent_std"),
                                                                  map(float, pm)))})
            losses, parts_acc = [], []
            if verbose:
                h = hist[-1]
                print(f"  step {it:5d}  loss {h['train_loss']:.3e} jepa {h['jepa']:.2e} rec {h['rec']:.2e} "
                      f"std {h['latent_std']:.2f}  val rollout relL2 {v:.4f}  {minutes:.1f} min", flush=True)
            if v < best:
                best, best_params, bad = v, params, 0
            else:
                bad += 1
            if bad >= cfg.patience or minutes > cfg.max_minutes:
                hist[-1]["stopped"] = "patience" if bad >= cfg.patience else "time"
                break
    sur = LatentSurrogate(net, best_params, ds.mean, ds.std, tuple(ds.config.cond), ds.cond_mean, ds.cond_std,
                          ds.config.spacing)
    return sur, hist


def load(path, mode="latent") -> tuple[LatentSurrogate, dict]:
    """Checkpoint written by train.save (arch kind "latent") -> LatentSurrogate."""
    from kolmo.surrogates import train as T
    sur, meta = T.load(path)
    return LatentSurrogate(**vars(sur), mode=mode), meta


def main(config_path, out_dir, **overrides):
    """Train one latent model on the benchmark's dataset and evaluate it (latent and re-encode rollouts)."""
    from kolmo import evaluate as ev
    from kolmo.surrogates import data
    with open(config_path) as f:
        cfg = json.load(f)
    dcfg = data.DataConfig(**cfg["data"])
    ds = data.build(dcfg, "runs/cache")
    lc = LatentConfig(**{"model": {"kind": "latent", "latent": 24, "patch": 4, "width": 24, "pos": 4},
                         "unroll": 4, "steps": 2000, "batch": 8, "lr": 1e-3, "eval_every": 200, "val_horizon": 20,
                         "patience": 5, "max_minutes": 20, **overrides})
    lc.model = {**lc.model, "out": ds.splits["train"].states.shape[2]}
    os.makedirs(out_dir, exist_ok=True)
    ck = os.path.join(out_dir, "latent")
    t = time.perf_counter()
    sur, hist = train(ds, lc)
    save(ck, sur, ds, lc, hist, train_minutes=(time.perf_counter() - t) / 60)
    print(f"params: {n_params(sur.params)}", flush=True)
    system = data.get_system(dcfg.system)
    res = {}
    for mode in ("latent", "reencode"):
        print(f"evaluating mode={mode}", flush=True)
        sur.mode = mode
        res[mode] = ev.evaluate(system, ds, sur, steps=cfg.get("eval_steps"), threshold=cfg.get("threshold", 0.5),
                                verbose=True)
        res[mode]["training"] = {"history": hist, "n_params": n_params(sur.params), "config": asdict(lc)}
    ev.write_json(res, os.path.join(out_dir, "results.json"))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    kw = dict(a.split("=", 1) for a in sys.argv[3:])
    main(sys.argv[1], sys.argv[2], **{k: json.loads(v) for k, v in kw.items()})
