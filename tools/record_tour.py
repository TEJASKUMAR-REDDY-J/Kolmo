"""Record dashboard run-through shots of the real app (on screen, real fonts) as 1080p30 MP4s with an animated cursor,
click ripples and human-speed typing. Each shot is its own file in video/tour/.

    python tools/record_tour.py                 # every shot
    python tools/record_tour.py tunnel models   # some shots
Shots: tunnel, plume, rt, cyclone, gray_scott, experiments, models, assistant"""
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PySide6.QtCore import QPoint, QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kolmo.app.window import MainWindow  # noqa: E402

OUT = ROOT / "video" / "tour"
W, H, FPS = 1920, 1080, 30
LOGICAL = (1600, 900)  # x1.25 display scaling -> 2000x1125 grab -> 1920x1080
ACCENT = (167, 139, 250)
ARROW = np.array([(0, 0), (0, 23), (6, 18), (10, 27), (14, 25), (10, 16), (17, 16)], np.float32) * 1.15


class Recorder:
    def __init__(self, win):
        self.win, self.ff, self.n = win, None, 0
        self.cur, self.move = np.array([W * 0.62, H * 0.56]), None
        self.ripple, self.lapse = None, 1.0

    def start(self, name, lapse=1.0):
        OUT.mkdir(parents=True, exist_ok=True)
        self.path = OUT / f"{name}.mp4"
        self.ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
                                    "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                                    "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                    str(self.path)], stdin=subprocess.PIPE)
        self.t0, self.n, self.lapse = time.perf_counter(), 0, lapse

    def stop(self):
        if self.ff:
            self.ff.stdin.close()
            self.ff.wait()
            print(f"{self.path.name}: {self.n / FPS:.1f} s", flush=True)
        self.ff = None

    def set_lapse(self, lapse):  # keep the timeline continuous when switching speed
        now = time.perf_counter()
        self.t0 = now - self.n / FPS * lapse
        self.lapse = lapse

    def frame(self):
        img = self.win.grab().toImage().convertToFormat(QImage.Format_RGB888)
        w, h = img.width(), img.height()
        a = np.frombuffer(img.constBits(), np.uint8, img.bytesPerLine() * h).reshape(h, img.bytesPerLine())
        a = cv2.resize(a[:, : w * 3].reshape(h, w, 3), (W, H), interpolation=cv2.INTER_AREA)
        now = time.perf_counter()
        if self.move:
            p0, p1, t0, d = self.move
            u = min((now - t0) / d, 1.0)
            u = u * u * (3 - 2 * u)
            self.cur = p0 + (p1 - p0) * u
        if self.ripple and now - self.ripple < 0.45:
            k = (now - self.ripple) / 0.45
            layer = a.copy()
            cv2.circle(layer, tuple(int(v) for v in self.cur), int(8 + 30 * k), ACCENT, 3, cv2.LINE_AA)
            a = cv2.addWeighted(layer, 1 - k, a, k, 0)
        pts = (ARROW + self.cur).astype(np.int32)
        cv2.fillPoly(a, [pts + 2], (0, 0, 0), cv2.LINE_AA)  # soft shadow
        cv2.fillPoly(a, [pts], (255, 255, 255), cv2.LINE_AA)
        cv2.polylines(a, [pts], True, (20, 20, 30), 1, cv2.LINE_AA)
        return a

    def tick(self):
        if not self.ff:
            return
        need = int((time.perf_counter() - self.t0) / self.lapse * FPS) + 1
        if need > self.n:
            f = self.frame().tobytes()
            for _ in range(min(need - self.n, 4)):  # duplicate at most a few frames after a slow grab
                self.ff.stdin.write(f)
            self.n = need if need - self.n <= 4 else self.n + 4


def main(names):
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(*LOGICAL)
    win.move(0, 0)
    win.show()
    rec = Recorder(win)
    sx = W / win.width()

    def at(widget, dx=0.5, dy=0.5):
        r = widget.rect()
        p = widget.mapTo(win, QPoint(int(r.width() * dx), int(r.height() * dy)))
        return np.array([p.x() * sx, p.y() * sx])

    def move(target, d=0.65):
        rec.move = (rec.cur.copy(), target if isinstance(target, np.ndarray) else at(target), time.perf_counter(), d)
        yield d + 0.05

    def click(widget, act=None, d=0.65):
        yield from move(widget, d)
        rec.ripple = time.perf_counter()
        yield 0.12
        (act or widget.click)()
        yield 0.35

    def type_into(edit, text, cps=15):
        yield from click(edit, edit.setFocus)
        for ch in text:
            edit.setText(edit.text() + ch)
            yield random.uniform(0.5, 1.4) / cps

    def until(cond, limit):
        t = time.perf_counter()
        while not cond() and time.perf_counter() - t < limit:
            yield 0.2

    def nav(i):
        yield from click(win.nav.button(i))

    def scroll_sim(to=1.0, d=1.0):
        """Smooth-scroll the Simulate page so the flow panels are in view."""
        sa = win.pages.widget(0)
        bar = sa.verticalScrollBar()
        top = win.ref_view.mapTo(sa.widget(), QPoint(0, 0)).y() - 150  # the run card's header, just above the fields
        a, b, n = bar.value(), max(0, min(bar.maximum(), top)), max(1, int(d * 30))
        for k in range(n):
            u = (k + 1) / n
            bar.setValue(int(a + (b - a) * u * u * (3 - 2 * u)))
            yield 1 / 30

    def scenario(key, seconds, zen_at=None, warm=0.0, name=None):
        if warm:  # develop the flow off camera, then roll
            win._run_scenario(win.scenario.findData(key))
            yield 1.0
            yield from scroll_sim(d=0.1)
            yield warm
            rec.start(name)
            yield 0.3
        else:
            rec.start(name)
            yield from nav(0)
            yield 0.8
            yield from click(win.scenario, lambda: win._run_scenario(win.scenario.findData(key)))
            yield from scroll_sim()
        t = time.perf_counter()
        yield from move(at(win.ref_view, 0.7, 0.6), 1.2)
        if zen_at:
            yield zen_at
            yield from click(win.zen_btn)
            yield 0.4
            yield from scroll_sim(d=0.5)
        yield max(0.0, seconds - (time.perf_counter() - t))
        if win.zen:
            yield from click(win.zen_btn)
        win.reset()

    def shots():
        yield 2.0
        if "tunnel" in names:  # warm SAM 3.1 off camera so the take shows a ~10 s segmentation, not the model load
            from kolmo.geometry import photo_dialog as pd
            from kolmo.geometry.segment import get_segmenter
            if pd._SEGMENTER is None:
                pd._SEGMENTER = get_segmenter("auto")
            pd._SEGMENTER.segment(win.geo["rgb"], "car")
            win.geo.update(mask=None, point=None)
            win.photo_view.show_rgb(win._geo_overlay())
            win.geo_text.clear()
            win.geo_go.setEnabled(False)
            win._geo_preview()
            win.geo_status.setText("Drop a photo or choose one, then say what the object is.")
            rec.start("01_photo_wind_tunnel")
            yield 1.0
            yield from nav(4)
            yield 1.2
            yield from type_into(win.geo_text, "car")
            yield 0.4
            yield from click(win.geo_seg)
            yield from move(at(win.photo_view, 0.62, 0.45), 1.2)
            yield from until(lambda: win.geo["mask"] is not None, 90)
            yield 2.0
            yield from click(win.geo_ground, d=0.8)
            yield 0.8
            yield from click(win.geo_ground)
            yield 1.0
            yield from click(win.geo_go)
            yield 0.6
            yield from scroll_sim()
            yield from move(at(win.ref_view, 0.75, 0.55), 1.2)
            yield 12.0
            yield from click(win.zen_btn)
            yield 0.4
            yield from scroll_sim(d=0.5)
            yield 12.0
            yield from click(win.zen_btn)
            yield 1.0
            rec.stop()
            win.reset()
        for key, name, secs, warm in (("plume", "02_heated_plumes", 24, 75), ("rt", "03_rayleigh_taylor", 20, 30),
                                      ("cyclone", "04_cyclones", 18, 8), ("gray_scott", "05_gray_scott", 18, 40)):
            if key in names:
                yield from scenario(key, secs, zen_at=secs * 0.4, warm=warm, name=name)
                rec.stop()
        if "experiments" in names:
            rec.start("06_experiments")
            yield from nav(1)
            yield 2.5
            sa = win.pages.currentWidget().findChildren(type(win.pages.widget(0)))
            bar = next((s.verticalScrollBar() for s in sa if s.verticalScrollBar().maximum() > 0), None)
            if bar:
                yield from move(at(win.pages.currentWidget(), 0.5, 0.7), 0.8)
                for k in range(90):
                    bar.setValue(int(bar.maximum() * (k + 1) / 90))
                    yield 1 / 30
                yield 1.5
                for k in range(45):
                    bar.setValue(int(bar.maximum() * (1 - (k + 1) / 45)))
                    yield 1 / 30
            yield 1.5
            rec.stop()
        if "models" in names:
            rec.start("07_models_ai_vs_solver")
            yield from nav(3)
            yield 1.5
            lst = win.mdl_list
            for i in (2, 1, 0):
                if i < lst.count():
                    r = lst.visualItemRect(lst.item(i))
                    p = lst.viewport().mapTo(win, r.center())
                    rec.move = (rec.cur.copy(), np.array([p.x() * sx, p.y() * sx]), time.perf_counter(), 0.5)
                    yield 0.55
                    rec.ripple = time.perf_counter()
                    lst.setCurrentRow(i)
                    yield 1.4
            yield from click(win.mdl_run)
            yield 0.6
            yield from scroll_sim()
            yield from move(at(win.err_view, 0.5, 0.5), 1.4)
            yield 18.0
            rec.stop()
            win.reset()
        if "assistant" in names:
            rec.start("08_assistant_autonomous")
            yield 1.0
            yield from type_into(win.chat_input, "Build a lightweight NCA surrogate for Gray-Scott spot patterns, compare it "
                                                 "with an FNO over 100 steps, and tell me which keeps the spots stable.", 24)
            yield 0.5
            send = [b for b in win.findChildren(type(win.zen_btn)) if b.text() == "➤"][0]
            yield from click(send)
            yield 6.0
            if win._pending:
                yield from type_into(win.chat_input, "yes", 6)
                yield from click(send)
            yield 4.0
            rec.set_lapse(12.0)  # the long autonomous run as a time-lapse
            yield from until(lambda: not win._agent_busy, 40 * 60)
            rec.set_lapse(1.0)
            yield 6.0
            yield from move(at(win.chat_log, 0.5, 0.6), 1.0)
            yield 4.0
            rec.stop()
        app.quit()

    gen = shots()
    state = {"next": time.perf_counter()}

    def step():
        rec.tick()
        if time.perf_counter() >= state["next"]:
            try:
                state["next"] = time.perf_counter() + next(gen)
            except StopIteration:
                timer.stop()

    timer = QTimer()
    timer.timeout.connect(step)
    timer.start(8)
    app.exec()
    rec.stop()


if __name__ == "__main__":
    args = sys.argv[1:] or ["tunnel", "plume", "rt", "cyclone", "gray_scott", "experiments", "models", "assistant"]
    main(set(args))
