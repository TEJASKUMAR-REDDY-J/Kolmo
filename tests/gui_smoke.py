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
    for _ in range(3):  # queued show events of newly added widgets
        QApplication.processEvents()
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
    # PDE library system with two fields (auto-form from describe(), field selector)
    gs = next((i for i, (lbl, s_, _) in enumerate(win._systems) if s_ is not None and "gray" in lbl.lower()), None)
    if gs is not None:
        win.system.setCurrentIndex(gs)
        win.mode_group.button(0).click()
        win.steps.setValue(200)
        win.run()
        wait(app, lambda: win.runner is None)
        assert win.pill.text() == "Finished", (win.pill.text(), win.run_error.text())
        assert win.field_sel.count() == 2 and win.field_sel.isVisible()
        win.field_sel.setCurrentIndex(1)
        print(f"gray-scott: {len(win.xs) - 1} steps, params {[sp.key for sp in win.specs]}")
        shot(win, "11_pde_gray_scott.png")

    # Fluid 2D viscosity: Re or nu, never both
    select_system(win, "fluid2d")
    assert win.excl_sel.isVisible() and win.excl_sel.count() == 2
    win.excl_sel.setCurrentIndex([win.excl_sel.itemData(i) for i in range(2)].index("Re"))
    p = win._params()
    assert p["nu"] is None and p["Re"] is not None and not win.param_widgets["nu"].isVisible()

    # Train new (REAL tiny run): diffusion CNN, 60 optimiser steps, progress into the Training panel, registered
    win.nav.button(3).click()
    win.trn_sys.setCurrentIndex(win.trn_sys.findData("diffusion"))
    win.trn_kind.setCurrentIndex(win.trn_kind.findData("cnn"))
    win.trn_budget.setValue(60)
    win._train()
    wait(app, lambda: win.trn_btn.isEnabled(), 900)
    print("train:", win.trn_note.text(), "| summary:", win.train_summary.text(), "| loss points:", len(win.tr["loss"]))
    assert win.tr["loss"] and "registered" in win.trn_note.text(), win.trn_note.text()
    win.nav.button(0).click()
    win.pages.widget(0).verticalScrollBar().setValue(10 ** 6)
    shot(win, "12_training_real.png")
    win.pages.widget(0).verticalScrollBar().setValue(0)

    # Models page: registry + disk checkpoints
    win.nav.button(3).click()
    app.processEvents()
    print(f"models page: {win.mdl_list.count()} models listed")
    shot(win, "08_models.png")
    win.nav.button(0).click()

    # zen mode: user toggle, agent zen:on honoured, ignored after the user turned it off during the task
    win.set_zen(True)
    wait(app, lambda: not win.nav_card.isVisible() and win.side_card.width() == 520, 5)
    assert win.zen and not win.setup_card.isVisible()
    shot(win, "09_zen.png")
    win._agent_busy = True  # an agent task is running
    win.set_zen(False)  # the user leaves zen during the task
    win.set_zen(True, source="agent")
    assert not win.zen, "agent zen:on must be ignored after the user turned zen off during the task"
    win.set_zen(True)  # the user re-enables: agent events count again
    win.set_zen(False, source="agent")
    assert not win.zen
    win.set_zen(True, source="agent")
    assert win.zen
    win._agent_busy = False
    win.set_zen(False)
    wait(app, lambda: win.nav_card.isVisible() and win.nav_card.width() == 214, 5)

    # agent activity + training panel driven by contract events (TEST INPUT: synthetic events, not measurements)
    win.act_box.setVisible(True)
    win._reset_training()
    win._agent_event({"type": "plan", "steps": ["Validate config", "Train surrogate", "Evaluate rollout"]}, "q")
    win._agent_event({"type": "tool", "name": "validate_config", "ok": True, "summary": "ok"}, "q")
    for k, (loss, val) in enumerate([(1.0, 0.9), (0.5, 0.6), (0.3, 0.5), (0.25, 0.51), (0.22, 0.505), (0.2, 0.502)], 1):
        win._agent_event({"type": "progress", "phase": "train", "step": 100 * k, "total": 1000, "loss": loss,
                          "val_rollout_error": val, "eta_s": 30.0 * (10 - k)}, "q")
    assert "plateaued 3" in win.train_summary.text(), win.train_summary.text()
    assert win._step_states == ["done", "running", "pending"], win._step_states
    win.pages.widget(0).verticalScrollBar().setValue(10 ** 6)  # scroll the centre to the training card
    shot(win, "10_agent_training_SYNTHETIC_TEST_EVENTS.png")
    win.close()
    print("gui smoke ok")


if __name__ == "__main__":
    main()
