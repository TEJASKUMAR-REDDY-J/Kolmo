"""Fast tests of the surrogate pipeline on the exact Diffusion system (n=16). Run: python -m pytest tests -q"""
import json
import os

import jax.numpy as jnp
import numpy as np
import pytest

from kolmo import diagnose, evaluate, experiments, hybrid, rollout
from kolmo.surrogates import data, models, train
from kolmo.systems.diffusion import Diffusion

P = {"n": 16, "kappa": 0.2, "dt": 0.05}


def small_cfg(**kw):
    base = dict(system="diffusion", params=P, steps=12, spacing=2, train_seeds=[0, 1, 2, 3], val_seeds=[10],
                test_seeds=[20, 21], vary={"kappa": [0.1, 0.2]}, heldout={"kappa": [0.4]}, cond=["kappa"])
    return data.DataConfig(**{**base, **kw})


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    return data.build(small_cfg(), str(tmp_path_factory.mktemp("cache")), verbose=False)


def test_data_splits_cache_and_stats(ds, tmp_path):
    assert ds.splits["train"].states.shape == (8, 13, 1, 16, 16)  # 4 seeds x 2 kappa
    assert ds.splits["heldout"].states.shape == (2, 13, 1, 16, 16)
    assert {p["kappa"] for p in ds.splits["heldout"].params} == {0.4}
    tr = ds.splits["train"].states
    assert np.allclose(ds.mean, tr.mean(), atol=1e-6) and np.allclose(ds.std, tr.std(), rtol=1e-5)
    # trajectory spacing: step 1 equals two solver steps
    sysm = Diffusion(); p = ds.splits["test"].params[0]
    s0 = ds.splits["test"].states[0, 0]
    st = sysm.stepper(p)
    assert np.allclose(ds.splits["test"].states[0, 1], st(st(jnp.asarray(s0))), atol=1e-6)
    a = data.build(small_cfg(steps=3), str(tmp_path), verbose=False)
    b = data.build(small_cfg(steps=3), str(tmp_path), verbose=False)  # from cache
    assert len(os.listdir(tmp_path)) == 1 and np.array_equal(a.splits["val"].states, b.splits["val"].states)
    with pytest.raises(ValueError):
        small_cfg(val_seeds=[0]).validate()
    with pytest.raises(ValueError):
        small_cfg(heldout={"kappa": [0.2]}).validate()


@pytest.mark.parametrize("arch", [{"kind": "cnn", "width": 8, "depth": 2}, {"kind": "fno", "width": 8, "modes": 4, "depth": 2, "pos": 2}])
def test_models_shapes_identity_init(arch):
    import jax
    net = models.build(arch)
    z, c = jnp.ones((3, 2, 16, 16)), jnp.ones((3, 1))
    params = net.init(jax.random.PRNGKey(0), z, c)
    out = net.apply(params, z, c)
    assert out.shape == z.shape and np.allclose(out, z)  # zero-init last layer = persistence


def test_cnn_periodic_equivariance():
    import jax
    net = models.build({"kind": "cnn", "width": 8, "depth": 2})
    z = jax.random.normal(jax.random.PRNGKey(1), (1, 1, 16, 16))
    params = net.init(jax.random.PRNGKey(0), z, jnp.zeros((1, 0)))
    params = jax.tree_util.tree_map(lambda w: w + 0.1, params)  # make last layer non-zero
    f = lambda x: net.apply(params, x, jnp.zeros((1, 0)))
    assert np.allclose(f(jnp.roll(z, 5, -1)), jnp.roll(f(z), 5, -1), atol=1e-5)


def test_train_improves_deterministic_and_checkpoint_roundtrip(ds, tmp_path):
    cfg = train.TrainConfig(model={"kind": "cnn", "width": 8, "depth": 2}, steps=60, batch=4, lr=3e-3,
                            eval_every=20, val_horizon=5, patience=10)
    sur, hist = train.train(ds, cfg, verbose=False)
    sur2, hist2 = train.train(ds, cfg, verbose=False)
    assert [h["val_rollout_rel_l2"] for h in hist] == [h["val_rollout_rel_l2"] for h in hist2]
    va = ds.splits["val"]
    persistence = np.mean([evaluate.rel_l2(np.repeat(va.states[i, :1], 6, 0), va.states[i, :6])[1:].mean()
                           for i in range(len(va.seeds))])
    assert min(h["val_rollout_rel_l2"] for h in hist) < persistence
    train.save(str(tmp_path / "m"), sur, ds, cfg, hist)
    sur3, meta = train.load(str(tmp_path / "m"))
    s = jnp.asarray(ds.splits["test"].states[0, 0])
    p = ds.splits["test"].params[0]
    assert np.array_equal(sur.stepper(p)(s), sur3.stepper(p)(s))
    assert meta["arch"]["kind"] == "cnn" and meta["norm"]["cond_names"] == ["kappa"] and meta["spacing"] == 2
    rk = train.TrainConfig(**{**vars(cfg), "unroll": 3, "pushforward": True, "steps": 20})
    assert np.isfinite(train.train(ds, rk, verbose=False)[1][-1]["train_loss"])


def test_rollout_failures_and_timing():
    s0 = jnp.ones((1, 8, 8))
    tr = rollout.rollout(lambda s: s * 3.0, s0, 5)
    assert tr.states.shape == (6, 1, 8, 8) and tr.compile_seconds >= 0 and tr.seconds >= 0
    assert [e["reason"] for e in tr.events] == ["blowup"] and tr.events[0]["step"] == 3  # 27 > 10
    st = np.ones((4, 1, 2, 2)); st[2, 0, 0, 0] = np.nan
    ev = rollout.detect_failures(st, ref_energy=1.0, energy_tol=0.1)
    assert {(e["reason"], e["step"]) for e in ev} == {("nonfinite", 2), ("energy_drift", 2)}


def test_hybrid_modes_cost_and_events(ds):
    sysm, sp = Diffusion(), ds.splits["test"]
    p = dict(sp.params[0])
    p_before = dict(p)
    fake = models.Surrogate(None, None, ds.mean, ds.std, ("kappa",), ds.cond_mean, ds.cond_std, 2)
    fake.stepper = lambda params: (lambda s: s * 1.01)  # a slightly wrong surrogate
    ref = hybrid.run(hybrid.HybridPolicy("reference"), sysm, p, fake, sp.states[0, 0], 12)
    assert np.allclose(ref.states, sp.states[0], atol=1e-5)
    tr = hybrid.run(hybrid.HybridPolicy(every=4), sysm, p, fake, sp.states[0, 0], 12)
    assert [i + 1 for i, s in enumerate(tr.source) if s == "reference"] == [4, 8, 12]
    assert hybrid.cost(tr, 2) == {"solver_steps": 6, "fraction_of_reference": 0.25, "interventions": 3}
    assert all(e["reason"] == "every" for e in tr.events if e["kind"] == "intervention")
    allres = hybrid.run(hybrid.HybridPolicy(residual_tol=0.0), sysm, p, fake, sp.states[0, 0], 4)
    assert allres.source == ["reference"] * 4 and allres.events[0]["value"] > 0
    en = hybrid.run(hybrid.HybridPolicy(energy_tol=1e-6), sysm, p, fake, sp.states[0, 0], 3)
    assert {e["reason"] for e in en.events if e["kind"] == "intervention"} == {"energy"}
    assert p == p_before


def test_metric_definitions():
    r = np.random.default_rng(0).normal(size=(5, 2, 4, 4))
    assert np.allclose(evaluate.rel_l2(r, r), 0) and np.allclose(evaluate.rel_l2(2 * r, r), 1)
    assert np.allclose(evaluate.correlation(3 * r + 1, r), 1)
    assert evaluate.rel_l2(r, r, (-2, -1)).shape == (5, 2)
    assert evaluate.time_to_failure([0, 0.1, 0.6, 0.2]) == 2 and evaluate.time_to_failure([0, 0.1]) is None
    assert evaluate.time_to_failure([0, np.inf]) == 1
    assert evaluate.clean({"a": np.array([1.0, np.nan]), "b": np.int64(3)}) == {"a": [1.0, None], "b": 3}


def test_diagnose_rules():
    t = np.arange(41)
    err = 0.01 * np.exp(0.1 * t); err[0] = 0
    split = {"threshold": 0.5, "steps": 40, "n_traj": 2, "rel_l2": {"median": err.tolist()},
             "first_step_only": {"median": (0.01 * np.exp(0.02 * t)).tolist()},
             "time_to_failure": {"per_traj": [40, None], "censored": 1, "median_observed": 40.0, "min": 40},
             "nonphysical_steps": 0, "energy_drift_final": 0.2, "energy_drift_max_abs": 0.2,
             "residual_ratio_median": 50.0, "horizons": {"1": err[1], "10": err[10]}}
    held = {**split, "horizons": {"1": err[1] * 5, "10": err[10] * 5}}
    r = {"splits": {"test": split, "heldout": held}, "hybrid": []}
    f = diagnose.diagnose({"m": r})
    growth = [x for x in f if "doubles every" in x["fact"]][0]
    assert abs(growth["evidence"]["rate_per_step"] - 0.1) < 1e-6 and "repeated errors" in growth["interpretation"]
    text = " ".join(x["interpretation"] for x in f)
    assert "energy injection" in text and "distribution shift" in text and "physically inconsistent" in text
    assert "m" in diagnose.to_markdown(f)


def test_experiment_end_to_end(tmp_path):
    cfg = {"name": "smoke", "out_dir": str(tmp_path),
           "data": {"system": "diffusion", "params": P, "steps": 10, "spacing": 2, "train_seeds": [0, 1, 2],
                    "val_seeds": [10], "test_seeds": [20], "vary": {"kappa": [0.1, 0.2]}, "heldout": {"kappa": [0.4]},
                    "cond": ["kappa"]},
           "models": {"one_step": {"model": {"kind": "cnn", "width": 8, "depth": 2}, "steps": 20, "batch": 4, "eval_every": 10},
                      "rollout": {"model": {"kind": "fno", "width": 8, "modes": 4, "depth": 2}, "unroll": 3,
                                  "steps": 20, "batch": 4, "eval_every": 10}},
           "hybrid": [{"mode": "surrogate"}, {"every": 2}, {"residual_tol": 0.5}]}
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg))
    out, res, findings = experiments.run(experiments.load_config(str(path)), verbose=False)
    for f in ("config.json", "env.json", "results.json", "diagnosis.json", "report.md", "error_vs_horizon.png",
              "energy_vs_time.png", "hybrid_tradeoff.png", "checkpoints/one_step.npz", "checkpoints/rollout.json"):
        assert os.path.exists(os.path.join(out, f)), f
    assert set(res["one_step"]["splits"]) == {"test", "heldout"} and findings
    assert res["one_step"]["hybrid"][1]["cost_fraction"] == 0.5
    with pytest.raises(ValueError):
        experiments.ExperimentConfig(**{**cfg, "models": {"x": {"bogus": 1}}}).parsed()
