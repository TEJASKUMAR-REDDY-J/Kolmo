"""Fast tests of the JEPA latent world model on the Diffusion system (n=16). Run: python -m pytest tests/test_latent.py -q"""
import numpy as np

from kolmo import evaluate, hybrid
from kolmo.rollout import rollout
from kolmo.surrogates import data, latent, train


def test_latent_train_rollout_checkpoint_hybrid(tmp_path):
    ds = data.build(data.DataConfig(system="diffusion", params={"n": 16, "kappa": 0.2, "dt": 0.05}, steps=12, spacing=2,
                                    train_seeds=[0, 1, 2, 3], val_seeds=[10], test_seeds=[20], vary={"kappa": [0.1, 0.2]},
                                    cond=["kappa"]), str(tmp_path), verbose=False)
    cfg = latent.LatentConfig(model={"kind": "latent", "latent": 8, "patch": 2, "width": 8, "pos": 0}, unroll=2,
                              steps=200, batch=4, lr=3e-3, eval_every=50, val_horizon=5, patience=5)
    sur, hist = latent.train(ds, cfg, verbose=False)
    assert hist[-1]["latent_std"] > 0.3  # VICReg variance term keeps the latent from collapsing
    assert hist[-1]["val_rollout_rel_l2"] < hist[0]["val_rollout_rel_l2"] or hist[0]["val_rollout_rel_l2"] < 0.5
    sp = ds.splits["test"]
    p, ref = sp.params[0], sp.states[0]
    # latent rollout differs from re-encoding every step, and is deterministic
    a = rollout(sur.stepper(p), ref[0], 6).states
    assert np.array_equal(a, rollout(sur.stepper(p), ref[0], 6).states)
    sur.mode = "reencode"
    b = rollout(sur.stepper(p), ref[0], 6).states
    assert np.isfinite(a).all() and not np.array_equal(a[2:], b[2:]) and np.allclose(a[1], b[1], atol=1e-5)
    # checkpoint: train.save format; train.load gives a re-encode Surrogate, latent.load the latent one
    train.save(str(tmp_path / "lat"), sur, ds, cfg, hist)
    s2, meta = train.load(str(tmp_path / "lat"))
    assert meta["arch"]["kind"] == "latent"
    assert np.allclose(rollout(s2.stepper(p), ref[0], 6).states, b, atol=1e-5)
    s3, _ = latent.load(str(tmp_path / "lat"))
    assert np.allclose(rollout(s3.stepper(p), ref[0], 6).states, a, atol=1e-5)
    # evaluate + hybrid accept it; a re-anchored hybrid run is still finite
    res = evaluate.evaluate(data.get_system("diffusion"), ds, s3, splits=("test",), steps=6,
                            policies=[hybrid.HybridPolicy(every=2)], verbose=False)
    assert res["splits"]["test"]["n_traj"] == 2 and res["hybrid"][0]["cost_fraction"] == 0.5
