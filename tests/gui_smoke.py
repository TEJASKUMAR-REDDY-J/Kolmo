"""GUI smoke test: start the window, run a short Diffusion reference rollout, check the panels update, save screenshots.

    python tests/gui_smoke.py          (sets QT_QPA_PLATFORM=offscreen unless already set)

Screenshots go to runs/ui/ (gitignored)."""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":  # the offscreen platform has no font database of its own
    os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication  # noqa: E402

from kolmo.app.window import MainWindow  # noqa: E402

OUT = ROOT / "runs" / "ui"


def wait(app, cond, timeout=300.0):
    t0 = time.time()
    while not cond():
        app.processEvents()
        time.sleep(0.01)
        if time.time() - t0 > timeout:
            raise TimeoutError("condition not met in time")


def shot(win, name):
    win.grab().save(str(OUT / name))
    print("saved", OUT / name)


def select_system(win, name):
    win.system.setCurrentIndex(next(i for i, (_, s, _) in enumerate(win._systems) if s is not None and s.name == name))


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # findings contain ≤, ‖, × (Windows console default is cp1252)
    OUT.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.resize(1680, 1000)
    win.show()
    app.processEvents()
    select_system(win, "diffusion")
    app.processEvents()
    assert win.current_system().name == "diffusion"
    assert win.ref_view.image is None and win.pill.text() == "Idle"
    shot(win, "01_idle.png")

    # reference rollout: panels and charts update; prediction and error stay honestly empty
    win.steps.setValue(300)
    win.run()
    wait(app, lambda: win.runner is None)
    assert win.pill.text() == "Finished", (win.pill.text(), win.run_error.text())
    assert win.ref_view.image is not None
    assert win.cand_view.image is None and win.err_view.image is None
    assert win.xs == list(range(301)), len(win.xs)
    var = win.diag_ref["variance"]
    assert len(var) == 301 and var[-1] < var[0], "diffusion must dissipate variance"
    t = win.timing["reference"]
    assert t.per_step, "steady-state step time must be measured"
    print(f"diffusion reference: {len(win.xs) - 1} steps, variance {var[0]:.4g} -> {var[-1]:.4g}, "
          f"{1e3 * t.per_step:.3f} ms/step steady, first call {t.first_call:.2f} s")
    shot(win, "02_diffusion_reference.png")

    # pause mid-run: stepping stops while the UI stays live; reset stops the run
    win.steps.setValue(200000)
    win.run()
    wait(app, lambda: len(win.xs) > 50)
    win.toggle_pause()
    for _ in range(20):  # let in-flight batches land
        app.processEvents()
        time.sleep(0.01)
    n = len(win.xs)
    for _ in range(30):
        app.processEvents()
        time.sleep(0.01)
    assert len(win.xs) == n and win.pill.text() == "Paused", (n, len(win.xs), win.pill.text())
    shot(win, "03_paused.png")
    win.reset()
    assert win.runner is None and win.pill.text() == "Idle"

    # surrogate mode without a checkpoint argument: honest empty state, a run refuses with the reason
    win.mode_group.button(1).click()
    if win.ckpt.currentData() is None:
        assert "No surrogate checkpoint" in win.cand_view.empty
        win.steps.setValue(50)
        win.run()
        wait(app, lambda: win.runner is None)
        assert win.pill.text() == "Error" and "checkpoint" in win.run_error.text(), win.run_error.text()
        shot(win, "04_surrogate_no_checkpoint.png")
    else:  # a diffusion checkpoint exists (e.g. from kolmo.experiments): surrogate and hybrid end to end
        print("checkpoint:", win.ckpt.currentData())
        win._use_training_params()
        win.steps.setValue(60)
        win.run()
        wait(app, lambda: win.runner is None)
        assert win.pill.text() == "Finished", (win.pill.text(), win.run_error.text())
        assert win.cand_view.image is not None and win.err_view.image is not None and len(win.err) == 61
        print(f"surrogate: rel L2 step 1 {win.err[1]:.3g}, step 60 {win.err[-1]:.3g}; speedup card "
              f"{win.m_speed.value} ({win.m_speed.caption}); horizon {win.m_horizon.value}")
        print("diagnosis:", win.findings.toPlainText()[:600].replace("\n", " | "))
        shot(win, "04_surrogate.png")
        win.mode_group.button(2).click()
        win.pol_every.setValue(10)
        win.run()
        wait(app, lambda: win.runner is None)
        assert win.pill.text() == "Finished", (win.pill.text(), win.run_error.text())
        assert len(win.err_chart.vlines) >= 5, win.err_chart.vlines
        print(f"hybrid every 10: rel L2 max {max(win.err):.3g}, interventions {len(win.err_chart.vlines)}")
        print("diagnosis:", win.findings.toPlainText()[:400].replace("\n", " | "))
        shot(win, "04_hybrid.png")
        win.side_tabs.button(1).click()
        shot(win, "04_hybrid_assistant.png")
        win.side_tabs.button(0).click()
        win.nav.button(1).click()
        shot(win, "07_experiments.png")
        win.nav.button(0).click()
    win.mode_group.button(0).click()

    # Fluid 2D reference, Kolmogorov-forced defaults, when the solver module is present
    if any(s is not None and s.name == "fluid2d" for _, s, _ in win._systems):
        select_system(win, "fluid2d")
        win.steps.setValue(300)
        win.run()
        wait(app, lambda: win.runner is None)
        assert win.pill.text() == "Finished", (win.pill.text(), win.run_error.text())
        e = win.diag_ref["energy"]
        t = win.timing["reference"]
        print(f"fluid2d reference: {len(win.xs) - 1} steps, energy {e[0]:.4g} -> {e[-1]:.4g}, "
              f"{1e3 * t.per_step:.3f} ms/step steady, first call {t.first_call:.2f} s")
        shot(win, "05_fluid_reference.png")

        # flow past a built-in cylinder: free stream on, forcing off, Re from U_inf and the body size
        w = win.param_widgets
        w["A"].setValue(0.0)
        w["U_inf"].setValue(1.0)
        w["Re"].setValue(100.0)
        win.obstacle.setCurrentIndex(1)
        win._update_thumb()
        assert win._current_mask() is not None
        win.steps.setValue(400)
        win.run()
        wait(app, lambda: win.runner is None)
        assert win.pill.text() == "Finished", (win.pill.text(), win.run_error.text())
        assert "drag" in win.diag_ref and win.ref_view.overlay is not None
        print(f"fluid2d cylinder: cd at step 400 = {win.diag_ref['cd'][-1]:.3g}, header: {win.run_sub.text()}")
        win.diag_key.setCurrentText("cd")
        shot(win, "06_fluid_cylinder.png")
    win.close()
    print("gui smoke ok")


if __name__ == "__main__":
    main()
