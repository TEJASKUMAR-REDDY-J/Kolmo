"""Kolmo main window: animated fabric backdrop, frosted-glass cards. Left: navigation. Centre (Simulate): reference |
prediction | error fields with transport controls, metric cards, error-over-rollout and diagnostic charts. Right:
Setup (what to run) and the offline rule-based Assistant. All computation runs in worker threads.
Shortcuts: Space = run / pause / resume, R = reset, E = error panel."""
import dataclasses
import html
import math
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import (QAbstractAnimation, QEasingCurve, QParallelAnimationGroup, QPointF, QPropertyAnimation,
                            QRectF, QSettings, Qt, QThread, QTimer, QVariantAnimation, Signal)
from PySide6.QtGui import (QColor, QConicalGradient, QImage, QKeySequence, QPainter, QPen,
                           QPixmap, QPolygonF, QShortcut)
from PySide6.QtWidgets import (QAbstractSpinBox, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMenu,
                               QProgressBar, QPushButton,
                               QScrollArea, QSizePolicy, QSlider, QSpinBox, QStackedWidget, QTextBrowser, QVBoxLayout,
                               QWidget)

from kolmo.app import backend
from kolmo.app.assistant import RuleBased, plan, training_summary
from kolmo.assistant.agent import Agent
from kolmo.app.widgets import (ACCENT, ACCENT2, CAND, COMPARE, DIVERGING, EVENT, FAIL, FAINT, MUTED, OK, REF,
                               SEQUENTIAL, SYMBOL_FONT, TEXT, Backdrop, ColorBar, FieldView, Glass, LineChart,
                               MetricCard, card, fmt, heading, mask_image, sci)

MODES = (("reference", "Reference"), ("surrogate", "Surrogate"), ("hybrid", "Hybrid"))
OBSTACLES = (("none", "None"), ("circle", "Circle"), ("square", "Square"), ("photo", "From photo…"),
             ("tunnel", "Photo wind tunnel"))
NAV = (("simulate", "◈", "Simulate"), ("experiments", "⚗", "Experiments"), ("assistant", "✦", "Assistant"),
       ("models", "⬡", "Models"), ("geometry", "◐", "Geometry"), ("reports", "▤", "Reports"))
AUTO = "auto"
GLASS_INPUT = "rgba(255,255,255,0.07)"

QSS = f"""
* {{ font-family: "Segoe UI"; font-size: 12px; color: {TEXT}; }}
QMainWindow {{ background: #0b0820; }}
QLabel, QCheckBox, QScrollArea, QScrollArea > QWidget > QWidget, QStackedWidget, #page {{ background: transparent; border: none; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#note {{ color: {MUTED}; font-size: 11px; }}
QLabel#faint {{ color: {FAINT}; font-size: 11px; }}
QLabel#section {{ color: {FAINT}; font-size: 10px; font-weight: 700; letter-spacing: 1.4px; padding-top: 10px; }}
QLabel#error {{ color: {FAIL}; }}
QLabel#counter {{ font-family: "Cascadia Mono"; font-size: 13px; color: {MUTED}; }}
QLabel#pill {{ border-radius: 11px; padding: 3px 12px; font-size: 11px; font-weight: 600; }}
QLabel#badge {{ background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.14); border-radius: 14px;
    padding: 5px 14px; }}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background: {GLASS_INPUT}; border: 1px solid rgba(255,255,255,0.13);
    border-radius: 9px; padding: 5px 10px; min-height: 18px; selection-background-color: {ACCENT}; }}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{ border-color: rgba(176,123,255,0.55); }}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{ border-color: {ACCENT2}; }}
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QPushButton:disabled, QLabel:disabled {{ color: {FAINT}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: #1a1342; border: 1px solid rgba(255,255,255,0.15); padding: 4px;
    selection-background-color: rgba(124,92,255,0.45); outline: none; }}
QPushButton {{ background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.13); border-radius: 10px;
    padding: 7px 14px; }}
QPushButton:hover {{ background: rgba(255,255,255,0.12); border-color: rgba(176,123,255,0.55); }}
QPushButton:pressed {{ background: rgba(124,92,255,0.25); }}
QPushButton#primary {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7c5cff, stop:1 #b07bff);
    border: 1px solid rgba(255,255,255,0.35); color: white; font-weight: 600; font-size: 13px; padding: 9px 14px; }}
QPushButton#primary:hover {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8d70ff, stop:1 #c08fff); }}
QPushButton#primary:pressed {{ background: #6a4cf0; }}
QPushButton#primary:disabled {{ background: rgba(124,92,255,0.25); color: rgba(255,255,255,0.45); }}
QPushButton#round {{ border-radius: 19px; min-width: 38px; max-width: 38px; min-height: 38px; max-height: 38px; padding: 0;
    font-family: "{SYMBOL_FONT}"; font-size: 14px; }}
QPushButton#roundPrimary {{ border-radius: 21px; min-width: 42px; max-width: 42px; min-height: 42px; max-height: 42px;
    padding: 0; font-family: "{SYMBOL_FONT}"; font-size: 15px; color: white; border: 1px solid rgba(255,255,255,0.35);
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #7c5cff, stop:1 #b07bff); }}
QPushButton#roundPrimary:hover {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #8d70ff, stop:1 #c08fff); }}
QPushButton#roundPrimary:disabled {{ background: rgba(124,92,255,0.25); }}
QPushButton#nav {{ text-align: left; padding: 11px 14px; border-radius: 13px; font-size: 14px; background: transparent;
    border: 1px solid transparent; }}
QPushButton#nav:hover {{ background: rgba(255,255,255,0.07); }}
QPushButton#nav:checked {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(124,92,255,0.55),
    stop:1 rgba(176,123,255,0.30)); border: 1px solid rgba(200,170,255,0.55); font-weight: 600; }}
QPushButton#seg {{ border-radius: 9px; padding: 6px 0; background: transparent; border: 1px solid transparent; color: {MUTED}; }}
QPushButton#seg:hover {{ color: {TEXT}; }}
QPushButton#seg:checked {{ background: rgba(124,92,255,0.40); border-color: rgba(200,170,255,0.5); color: {TEXT};
    font-weight: 600; }}
QPushButton#chip {{ border-radius: 12px; padding: 4px 12px; font-size: 11px; color: {MUTED}; }}
QPushButton#chip:checked {{ background: rgba(124,92,255,0.35); border-color: rgba(200,170,255,0.5); color: {TEXT}; }}
QPushButton#link {{ background: transparent; border: none; color: {ACCENT2}; padding: 0; text-align: left; }}
QPushButton#link:hover {{ color: {TEXT}; }}
QTextBrowser {{ background: transparent; border: none; }}
QListWidget {{ background: transparent; border: none; outline: none; }}
QListWidget::item {{ padding: 9px 10px; border-radius: 10px; margin: 2px 0; }}
QListWidget::item:hover {{ background: rgba(255,255,255,0.06); }}
QListWidget::item:selected {{ background: rgba(124,92,255,0.35); color: white; }}
QProgressBar {{ background: rgba(255,255,255,0.10); border: none; border-radius: 3px; }}
QProgressBar::chunk {{ border-radius: 3px; background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7c5cff, stop:1 #d29bff); }}
QCheckBox {{ color: {MUTED}; spacing: 6px; }}
QSlider::groove:horizontal {{ height: 5px; border-radius: 2px; background: rgba(255,255,255,0.12); }}
QSlider::sub-page:horizontal {{ border-radius: 2px; background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7c5cff,
    stop:1 #d29bff); }}
QSlider::handle:horizontal {{ width: 16px; height: 16px; margin: -6px 0; border-radius: 8px; background: white;
    border: 2px solid {ACCENT2}; }}
QSlider::handle:horizontal:disabled {{ background: rgba(255,255,255,0.3); border-color: rgba(255,255,255,0.2); }}
QMenu {{ background: #1a1342; border: 1px solid rgba(255,255,255,0.15); padding: 6px; border-radius: 10px; }}
QMenu::item {{ padding: 6px 18px; border-radius: 6px; }}
QMenu::item:selected {{ background: rgba(124,92,255,0.4); }}
QToolTip {{ background: #1a1342; color: {TEXT}; border: 1px solid rgba(255,255,255,0.18); padding: 4px; }}
QScrollBar:vertical {{ background: transparent; width: 8px; }}
QScrollBar::handle:vertical {{ background: rgba(255,255,255,0.15); border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
"""


def arrow_rule():
    """Combo-box arrow as a tiny PNG in the temp folder (QSS cannot draw a triangle itself)."""
    path = Path(tempfile.gettempdir()) / "kolmo_combo_arrow_v2.png"
    img = QImage(10, 6, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(MUTED))
    p.drawPolygon(QPolygonF([QPointF(0, 0), QPointF(10, 0), QPointF(5, 6)]))
    p.end()
    img.save(str(path))
    return f"QComboBox::down-arrow {{ image: url({path.as_posix()}); width: 10px; height: 6px; }}"


def label(text, name=None, wrap=False, squeeze=False):
    """squeeze: never widens its layout (long text is clipped instead of growing the window)."""
    w = QLabel(text)
    if name:
        w.setObjectName(name)
    w.setWordWrap(wrap)
    if squeeze:
        w.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        w.setMinimumWidth(0)
    return w


def spin(lo, hi, decimals=0, special=None, width=112):
    w = QSpinBox() if decimals == 0 else QDoubleSpinBox()
    if decimals:
        w.setDecimals(decimals)
        w.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
    w.setRange(lo, hi)
    if special:
        w.setSpecialValueText(special)  # shown when the value equals the minimum
    w.setButtonSymbols(QAbstractSpinBox.NoButtons)
    w.setFixedWidth(width)
    return w


def button(text, name=None, tip="", slot=None):
    b = QPushButton(text)
    if name:
        b.setObjectName(name)
    b.setToolTip(tip)
    b.setCursor(Qt.PointingHandCursor)
    if slot:
        b.clicked.connect(slot)
    return b


class Logo(QWidget):
    """Brand mark: a violet swirl (conical gradient rings)."""

    def __init__(self, size=46):
        super().__init__()
        self.setFixedSize(size, size)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = QPointF(self.width() / 2, self.height() / 2)
        g = QConicalGradient(c, 90)
        for pos, col in ((0, "#5b8cff"), (0.35, "#b07bff"), (0.7, "#7c5cff"), (1, "#5b8cff")):
            g.setColorAt(pos, QColor(col))
        for i, (r, wdt) in enumerate(((19, 5), (12.5, 4), (6.5, 3))):
            p.setPen(QPen(g, wdt, Qt.SolidLine, Qt.RoundCap))
            p.drawArc(QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r), (40 + 70 * i) * 16, 290 * 16)


class Task(QThread):
    """Run fn() off the UI thread; emits ok(result) or err(message)."""
    ok, err = Signal(object), Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.ok.emit(self.fn())
        except (ValueError, RuntimeError) as e:
            self.err.emit(str(e))
        except Exception:  # noqa: BLE001
            self.err.emit(traceback.format_exc(limit=3))


def _disc(n, x, y, r):
    yy, xx = (np.mgrid[0:n, 0:n] + 0.5) / n
    return (xx - x) ** 2 + (yy - y) ** 2 <= r * r


# one-click demos: (label, system, params (incl. ones without a widget, e.g. ic / mask), steps, stride, field)
SCENARIOS = {
    "plume": ("Heated plume (buoyant convection)", "boussinesq",
              lambda: {"n": 256, "Ra": 2e7, "Pr": 1.0, "ic": "zero", "T_body": 1.0, "mask": _disc(256, 0.5, 0.12, 0.06)},
              4000, 8, "temperature"),
    "rt": ("Rayleigh–Taylor mushrooms (heavy over light)", "boussinesq",
           lambda: {"n": 256, "Ra": 1e7, "Pr": 1.0, "ic": "rayleigh_taylor", "ic_m": 3, "ic_eps": 0.05, "ic_delta": 0.05},
           2500, 4, "temperature"),
    "cyclone": ("Cyclones on a β-plane (drift north-west)", "fluid2d",
                lambda: {"n": 256, "nu": 3e-4, "beta": 2.5, "ic": "vortices",
                         "vortices": [(3.9, 2.5, 0.55, 7.0), (1.75, 3.9, 0.4, -5.0), (2.6, 1.4, 0.3, 5.0)]},
                3000, 4, "vorticity"),
    "gray_scott": ("Gray–Scott self-replicating spots", "gray_scott",
                   lambda: {"n": 192, "L": 0.96, "F": 0.0367, "k": 0.0649, "dt": 1.0,
                           "ic": {"u": "const:1+spots:-0.5:0.03:10+noise:0.01", "v": "spots:0.25:0.03:10"}},
                   2500, 16, "v"),
    "cahn": ("Cahn–Hilliard phase separation", "cahn_hilliard", lambda: {"n": 256, "L": 128.0, "dt": 0.03}, 2500, 4, "u"),
}


class PhotoView(QLabel):
    """Photo panel: drop an image file on it; a click on the photo reports (x, y) in photo pixels."""
    dropped, clicked = Signal(str), Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(420, 340)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"background: rgba(0,0,0,0.30); border: 1px dashed rgba(255,255,255,0.22); "
                           f"border-radius: 16px; color: {MUTED}; font-size: 15px;")
        self.setText("Drop a photo here\n\nside views work best: a car, a plane, a wing, a mug…")
        self.rgb, self.scale, self.off = None, 1.0, (0.0, 0.0)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        self.dropped.emit(e.mimeData().urls()[0].toLocalFile())

    def show_rgb(self, rgb):
        self.rgb = np.ascontiguousarray(rgb)
        h, w = rgb.shape[:2]
        self.scale = min((self.width() - 20) / w, (self.height() - 20) / h)
        self.off = ((self.width() - w * self.scale) / 2, (self.height() - h * self.scale) / 2)
        qi = QImage(self.rgb.data, w, h, self.rgb.strides[0], QImage.Format_RGB888).copy()
        self.setPixmap(QPixmap.fromImage(qi).scaled(int(w * self.scale), int(h * self.scale), Qt.KeepAspectRatio,
                                                    Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.rgb is not None:
            self.show_rgb(self.rgb)

    def mousePressEvent(self, e):
        if self.rgb is None:
            return
        x, y = (e.position().x() - self.off[0]) / self.scale, (e.position().y() - self.off[1]) / self.scale
        if 0 <= x < self.rgb.shape[1] and 0 <= y < self.rgb.shape[0]:
            self.clicked.emit(x, y)


class Runner(QThread):
    """Iterates a backend.LiveRun; sends frames in batches at most ~30 times a second so drawing never limits stepping."""
    started_run = Signal(object)
    frames = Signal(object)  # (list[Frame], {name: Timing})
    done = Signal(object)  # {"timing", "summary" (dict or message str), "stopped"}
    failed = Signal(str)

    def __init__(self, cfg):
        super().__init__()
        self.cfg, self.paused, self.stopped = cfg, False, False

    def run(self):
        try:
            run = backend.LiveRun(self.cfg)
            self.started_run.emit(run)
            timing = lambda: {k: dataclasses.replace(v) for k, v in run.timing.items()}
            batch, last = [], 0.0
            for fr in run:
                while self.paused and not self.stopped:
                    time.sleep(0.03)
                if self.stopped:
                    break
                batch.append(fr)
                now = time.perf_counter()
                if now - last > 1 / 30:
                    self.frames.emit((batch, timing()))
                    batch, last = [], now
            if batch:
                self.frames.emit((batch, timing()))
            try:
                summary = backend.summarize(run, self.stopped)
            except backend.Unavailable as e:
                summary = str(e)
            self.done.emit({"timing": timing(), "summary": summary, "stopped": self.stopped})
        except (ValueError, backend.Unavailable) as e:
            self.failed.emit(str(e))
        except Exception:  # noqa: BLE001 - shown in the UI instead of a silent dead thread
            self.failed.emit(traceback.format_exc(limit=4))


class TrainTask(QThread):
    """fn(progress_callback) off the UI thread; progress dicts arrive on the UI thread."""
    progress, ok, err = Signal(object), Signal(object), Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.ok.emit(self.fn(self.progress.emit))
        except (ValueError, RuntimeError) as e:
            self.err.emit(str(e))
        except Exception:  # noqa: BLE001
            self.err.emit(traceback.format_exc(limit=3))


class AgentTask(QThread):
    """Runs the research agent off the UI thread; every agent event arrives as a signal on the UI thread."""
    event = Signal(object)

    def __init__(self, agent, question, confirmed, should_stop):
        super().__init__()
        self.agent, self.question, self.confirmed, self.should_stop = agent, question, confirmed, should_stop

    def run(self):
        try:
            backend.agent_run(self.agent, self.question, self.confirmed, self.event.emit, self.should_stop)
        except Exception as e:  # noqa: BLE001 - reported in the chat, never silent
            self.event.emit({"type": "answer", "status": "error", "text": f"{type(e).__name__}: {e}"})
            self.event.emit({"type": "done"})


def row(*widgets, spacing=8, stretch_first=False):
    h = QHBoxLayout()
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(spacing)
    for i, w in enumerate(widgets):
        if w is None:
            h.addStretch(1)
        elif isinstance(w, int):
            h.addSpacing(w)
        else:
            h.addWidget(w, 1 if (stretch_first and i == 0) else 0)
    return h


def scroller(widget):
    s = QScrollArea()
    s.setWidgetResizable(True)
    s.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    s.setWidget(widget)
    return s


class MainWindow(QMainWindow):
    geo_progress = Signal(str)  # SAM 3.1 progress, emitted from a worker thread

    def __init__(self):
        super().__init__()
        self.tunnel, self.extra = None, {}  # extra: scenario params without a widget, for one system
        self.setWindowTitle("Kolmo · Physics-AI Research Platform")
        self.resize(1680, 1000)
        self.setStyleSheet(QSS + arrow_rule())
        self.settings = QSettings("Kolmo", "Kolmo")
        self.runner, self._threads, self.live = None, [], None
        self.photo = None  # {"path", "mask", "meta"} of a photo obstacle
        self.specs, self.param_widgets = [], {}
        self.summary, self.compare, self.timing = None, [], {}
        self.assistant = RuleBased(Agent(out_dir=str(backend.ROOT / "runs")))  # live only if .env KOLMO_LLM_LIVE=1 + key
        self.zen, self._zen_blocked, self._agent_task, self._cancel, self._last_frame = False, False, None, False, None
        self.tr = {"steps": [], "loss": [], "vsteps": [], "val": []}
        self._pending, self._agent_busy = None, False  # question awaiting run confirmation; one agent task at a time
        self._systems = backend.systems()
        self.threshold, self.drift_tol = backend.failure_threshold(), backend.drift_tol()

        self.backdrop = Backdrop()
        self.backdrop.enabled = self.settings.value("animate", True, type=bool)
        self.setCentralWidget(self.backdrop)
        root = QVBoxLayout(self.backdrop)
        root.setContentsMargins(22, 16, 22, 20)
        root.setSpacing(16)
        root.addLayout(self._build_top())
        body = QHBoxLayout()
        body.setSpacing(18)
        body.addWidget(self._build_nav())
        self.pages = QStackedWidget()
        self.pages.addWidget(scroller(self._build_simulate()))  # only the centre scrolls
        self.pages.addWidget(self._build_experiments())
        self.pages.addWidget(self._build_models())
        self.pages.addWidget(self._build_geometry())
        self.geo_progress.connect(lambda m: self.geo_status.setText(m))
        self.pages.addWidget(self._build_reports())
        body.addWidget(self.pages, 1)
        body.addWidget(self._build_side())
        root.addLayout(body, 1)

        for key, fn in (("Space", self._space), ("R", self.reset), ("E", lambda: self.err_toggle.toggle()),
                        ("Z", lambda: self.set_zen(not self.zen))):
            QShortcut(QKeySequence(key), self, activated=fn)
        self._on_system()
        self.reset()

    # ------------------------------------------------------------ top bar + nav
    def _build_top(self):
        brand = QVBoxLayout()
        brand.setSpacing(0)
        t = label("Kolmo")
        t.setStyleSheet("font-size: 26px; font-weight: 700;")
        brand.addWidget(t)
        brand.addWidget(label("Physics-AI Research Platform", "muted"))
        top = QHBoxLayout()
        top.setSpacing(12)
        top.addWidget(Logo())
        top.addLayout(brand)
        top.addStretch(1)
        self.scenario = QComboBox()
        self.scenario.addItem("▶  Try a scenario…", None)
        self.scenario.addItem("Car in a wind tunnel (photo + SAM 3.1)", "car")
        for key, sc in SCENARIOS.items():
            self.scenario.addItem(sc[0], key)
        self.scenario.setToolTip("Load a ready-made demo and run it")
        self.scenario.setStyleSheet("font-size: 14px; padding: 7px 14px; border-radius: 12px;")
        self.scenario.setMaximumWidth(340)
        self.scenario.activated.connect(self._run_scenario)
        top.addWidget(self.scenario)
        self.system = QComboBox()
        self.system.setMinimumWidth(210)
        self.system.setToolTip("Physical system")
        for i, (lbl, s, why) in enumerate(self._systems):
            self.system.addItem(lbl)
            if s is None:  # listed but disabled, with the reason, rather than hidden
                self.system.model().item(i).setEnabled(False)
                self.system.setItemData(i, f"Not available: {why}", Qt.ToolTipRole)
        self.system.setCurrentIndex(next(i for i, (_, s, _) in enumerate(self._systems) if s is not None))
        self.system.currentIndexChanged.connect(self._on_system)
        self.system.setStyleSheet("font-size: 14px; padding: 7px 14px; border-radius: 12px;")
        top.addWidget(self.system)
        badge = label(f"<span style='color:{OK}'>●</span>&nbsp; {backend.device()}", "badge")
        badge.setToolTip("Computation runs on this machine through JAX; no network services are used.")
        top.addWidget(badge)
        gear = button("⚙", "round", "Settings")
        menu = QMenu(gear)
        self.act_anim = menu.addAction("Animated background")
        self.act_anim.setCheckable(True)
        self.act_anim.setChecked(self.backdrop.enabled)
        self.act_anim.toggled.connect(self._set_anim)
        menu.addSeparator()
        menu.addAction("Shortcuts: Space run/pause · R reset · E error panel · Z zen").setEnabled(False)
        gear.setMenu(menu)
        gear.setStyleSheet("QPushButton::menu-indicator { image: none; }")
        self.zen_btn = button("◐  Zen", "chip", "Zen mode (Z): dim the background, hide secondary panels, "
                              "enlarge the view and the chat")
        self.zen_btn.setCheckable(True)
        self.zen_btn.toggled.connect(lambda on: self.set_zen(on))
        top.addWidget(self.zen_btn)
        top.addWidget(gear)
        return top

    def _set_anim(self, on):
        self.backdrop.enabled = on
        self.settings.setValue("animate", on)

    def _build_nav(self):
        c, v = card((12, 14, 12, 14), spacing=4)
        c.setMinimumWidth(214)
        c.setMaximumWidth(214)
        self.nav_card = c
        self.nav = QButtonGroup(self)
        for i, (key, icon, text) in enumerate(NAV):
            b = button(f"{icon}   {text}", "nav")
            b.setCheckable(True)
            b.setStyleSheet(f"font-family: '{SYMBOL_FONT}', 'Segoe UI';")
            self.nav.addButton(b, i)
            v.addWidget(b)
        self.nav.button(0).setChecked(True)
        self.nav.idClicked.connect(self._on_nav)
        v.addStretch(1)
        v.addWidget(label("Offline · local compute\nNumbers shown are measured\nin this session or read from\nrun files.", "faint"))
        return c

    def _on_nav(self, i):
        key = NAV[i][0]
        if key == "assistant":  # the assistant is always visible on the right: focus it
            self.nav.button(self.pages.currentIndex() if self.pages.currentIndex() < 1 else 0).setChecked(True)
            self.chat_input.setFocus()
            return
        page = {"simulate": 0, "experiments": 1, "models": 2, "geometry": 3, "reports": 4}[key]
        if page == 1:
            self._fill_experiments()
        elif page == 2:
            self._fill_models()
        elif page == 4:
            self._fill_reports()
        self.pages.setCurrentIndex(page)

    def _build_simulate(self):
        page = QWidget()
        page.setObjectName("page")
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 6, 0)
        v.setSpacing(16)
        self.setup_card = self._build_setup()
        v.addWidget(self.setup_card)

        fields, fv = card((20, 16, 20, 14), spacing=10)
        fields.setMinimumHeight(500)
        self.run_title = label("", squeeze=True)
        self.run_title.setStyleSheet("font-size: 17px; font-weight: 600;")
        self.pill = label("Idle", "pill")
        self.field_sel = QComboBox()
        self.field_sel.setToolTip("Field shown in the panels")
        self.field_sel.currentIndexChanged.connect(lambda _: self._last_frame and self._show_fields(self._last_frame))
        self.err_toggle = button("Error panel", "chip", "Show |prediction − reference| (E)")
        self.err_toggle.setCheckable(True)
        self.err_toggle.toggled.connect(lambda on: self.err_view.setVisible(on))
        fv.addLayout(row(self.run_title, self.field_sel, self.err_toggle, self.pill, stretch_first=True))
        self.run_sub = label("", "faint", squeeze=True)
        self.run_sub.setTextInteractionFlags(Qt.TextSelectableByMouse)
        fv.addWidget(self.run_sub)
        panels = QHBoxLayout()
        panels.setSpacing(18)
        self.ref_view = FieldView("Reference", "(numerical)", bar=False)
        self.cand_view = FieldView("Surrogate", "(neural)", bar=False)
        self.cbar = ColorBar("ω")
        self.err_view = FieldView("Absolute error", "|Δ|")
        for w in (self.ref_view, self.cand_view, self.cbar, self.err_view):
            panels.addWidget(w)
        self.err_view.setVisible(False)
        fv.addLayout(panels, 1)
        self.play = button("▶", "roundPrimary", "Run (Space)", self._space)
        self.stop_btn = button("⟲", "round", "Reset (R)", self.reset)
        self.scrub = QSlider(Qt.Horizontal)
        self.scrub.setToolTip("Scrub through the recorded steps; drag to the end to follow the live run")
        self.scrub.valueChanged.connect(self._on_scrub)
        self.counter = label("", "counter")
        fv.addLayout(row(self.play, self.stop_btn, 6, self.scrub, 10, self.counter, spacing=10))
        v.addWidget(fields)

        self.metrics_box = QWidget()
        metrics = QHBoxLayout(self.metrics_box)
        metrics.setContentsMargins(0, 0, 0, 0)
        metrics.setSpacing(14)
        self.m_err = MetricCard("◎", "Relative error", "‖prediction − reference‖ / ‖reference‖ at the shown step")
        self.m_drift = MetricCard("⚖", "Energy drift", "mean square of the prediction / of the reference − 1 "
                                                        "(kolmo.rollout.mean_square), at the shown step")
        self.m_speed = MetricCard("⚡", "Speedup", "steady-state reference ms/step ÷ prediction ms/step, measured now")
        self.m_horizon = MetricCard("∿", "Stability horizon", f"first step with relative error > {self.threshold:g}")
        for m in (self.m_err, self.m_drift, self.m_speed, self.m_horizon):
            metrics.addWidget(m)
        v.addWidget(self.metrics_box)

        charts = QHBoxLayout()
        charts.setSpacing(14)
        ec, ev = card((20, 14, 16, 12), spacing=4)
        ec.setMinimumHeight(300)
        self.clear_cmp = button("Clear comparison", "link", "Forget earlier runs in this chart", self._clear_compare)
        ev.addLayout(row(heading("Error over rollout"), None, self.clear_cmp))
        self.err_chart = LineChart(log=True, x_label="surrogate step")
        self.err_chart.hlines = [(self.threshold, f"failure threshold {self.threshold:g}", FAIL)]
        ev.addWidget(self.err_chart, 1)
        charts.addWidget(ec, 3)
        dc, dv = card((20, 14, 16, 12), spacing=4)
        self.diag_card = dc
        self.diag_key = QComboBox()
        self.diag_key.setMinimumWidth(130)
        self.diag_key.setToolTip("Quantity from system.diagnostics")
        self.diag_key.currentTextChanged.connect(self._bind_diag)
        self.diag_log = QCheckBox("log")
        self.diag_log.toggled.connect(lambda on: (setattr(self.diag_chart, "log", on), self.diag_chart.update()))
        dv.addLayout(row(heading("Diagnostic"), None, self.diag_key, self.diag_log))
        self.diag_chart = LineChart(x_label="surrogate step")
        dv.addWidget(self.diag_chart, 1)
        charts.addWidget(dc, 2)
        v.addLayout(charts)
        v.addWidget(self._build_training())
        return page

    def _build_side(self):
        c, v = card((16, 14, 16, 16), spacing=10)
        c.setMinimumWidth(380)
        c.setMaximumWidth(380)
        self.side_card = c
        v.addWidget(self._build_assistant(), 1)
        return c

    def _build_setup(self):
        """Setup card: parameters (auto-form from the system), obstacle and execution side by side, then Run."""
        c, outer = card((20, 14, 20, 16), spacing=6)
        self.system_note = label("", "faint", squeeze=True)
        outer.addLayout(row(heading("Setup"), 12, self.system_note, stretch_first=False))
        outer.itemAt(outer.count() - 1).layout().setStretch(2, 1)
        h = QHBoxLayout()
        h.setSpacing(26)
        outer.addLayout(h)
        self.config_box = QWidget()
        cb = QHBoxLayout(self.config_box)
        cb.setContentsMargins(0, 0, 0, 0)
        cb.setSpacing(26)
        h.addWidget(self.config_box, 1)

        a = QVBoxLayout()
        a.setSpacing(6)
        self.excl_sel = QComboBox()
        self.excl_sel.setToolTip("Viscosity is set by one of the pair; the other follows from it")
        self.excl_sel.currentIndexChanged.connect(lambda _: self._apply_exclusive())
        a.addLayout(row(label("PARAMETERS", "section"), None, self.excl_sel))
        self.params_grid = QGridLayout()
        self.params_grid.setHorizontalSpacing(8)
        self.params_grid.setVerticalSpacing(6)
        self.params_grid.setColumnStretch(0, 1)
        self.params_grid.setColumnStretch(2, 1)
        a.addLayout(self.params_grid)
        self.seed = spin(0, 2 ** 31 - 1, width=100)
        self.seed.setToolTip("Seed of system.initial: same seed, same initial condition")
        a.addLayout(row(label("Random seed"), None, self.seed))
        a.addStretch(1)
        cb.addLayout(a, 3)

        b = QVBoxLayout()
        b.setSpacing(6)
        b.addWidget(label("OBSTACLE", "section"))
        self.obstacle = QComboBox()
        for _, t in OBSTACLES:
            self.obstacle.addItem(t)
        self.obstacle.activated.connect(self._on_obstacle)
        self.obstacle_size = spin(0.02, 0.6, 2, width=64)
        self.obstacle_size.setValue(0.2)
        self.obstacle_size.setToolTip("Diameter / side (photo: streamwise length) as a fraction of the domain side")
        self.obstacle_size.valueChanged.connect(self._update_thumb)
        b.addLayout(row(self.obstacle, self.obstacle_size, stretch_first=True))
        self.thumb = QLabel()
        self.thumb.setFixedSize(44, 44)
        self.thumb.setStyleSheet("background: rgba(0,0,0,0.35); border: 1px solid rgba(255,255,255,0.15); border-radius: 8px;")
        self.obstacle_note = label("", "note", wrap=True)
        hh = row(self.thumb, self.obstacle_note, spacing=10)
        hh.setStretch(1, 1)
        b.addLayout(hh)
        b.addWidget(label("EXECUTION", "section"))
        segbox = Glass(radius=11, level=1)
        seg = QHBoxLayout(segbox)
        seg.setContentsMargins(3, 3, 3, 3)
        seg.setSpacing(2)
        self.mode_group = QButtonGroup(self)
        tips = ("Reference solver only", "Surrogate network rollout vs the reference",
                "Surrogate with solver interventions (kolmo.hybrid)")
        for i, ((_, t), tip) in enumerate(zip(MODES, tips)):
            bt = button(t, "seg", tip)
            bt.setCheckable(True)
            self.mode_group.addButton(bt, i)
            seg.addWidget(bt)
        self.mode_group.button(0).setChecked(True)
        self.mode_group.idClicked.connect(lambda _: self._on_mode())
        b.addWidget(segbox)
        self.exec_box = QWidget()
        g = QVBoxLayout(self.exec_box)
        g.setContentsMargins(0, 2, 0, 0)
        g.setSpacing(5)
        self.ckpt = QComboBox()
        self.ckpt.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.ckpt.setMinimumContentsLength(12)
        self.ckpt.setToolTip("Surrogate checkpoint (Models page: registry with metrics)")
        self.ckpt.currentIndexChanged.connect(lambda _: self._on_mode())
        self.ckpt_browse = button("…", tip="Choose a checkpoint (<name>.json next to <name>.npz)", slot=self._browse_ckpt)
        self.ckpt_browse.setFixedWidth(36)
        g.addLayout(row(self.ckpt, self.ckpt_browse, stretch_first=True))
        self.ckpt_note = label("", "note", wrap=True)
        g.addWidget(self.ckpt_note)
        self.use_train = button("Use the training parameters", "link", slot=self._use_training_params)
        g.addWidget(self.use_train)
        self.policy_box = QWidget()
        pg = QGridLayout(self.policy_box)
        pg.setContentsMargins(0, 2, 0, 0)
        pg.setColumnStretch(0, 1)
        pg.setVerticalSpacing(5)
        self.pol_every = spin(0, 100000, special="off", width=80)
        self.pol_res = spin(0, 100, 4, special="off", width=80)
        self.pol_energy = spin(0, 100, 3, special="off", width=80)
        for i, (t, wdg, tip) in enumerate((
                ("Re-anchor every", self.pol_every, "solver steps at steps N, 2N, ... (surrogate steps)"),
                ("Residual tolerance", self.pol_res, "intervene when the physics residual of a proposal exceeds this"),
                ("Energy tolerance", self.pol_energy, "intervene when |E/E₀ − 1| of a proposal exceeds this; "
                                                      "non-finite proposals always trigger the solver")), 0):
            lab = label(t)
            lab.setToolTip(tip)
            wdg.setToolTip(tip)
            pg.addWidget(lab, i, 0)
            pg.addWidget(wdg, i, 1)
        g.addWidget(self.policy_box)
        b.addWidget(self.exec_box)
        self.mode_note = label("", "note", wrap=True)
        b.addWidget(self.mode_note)
        b.addStretch(1)
        cb.addLayout(b, 2)

        r = QVBoxLayout()
        r.setSpacing(8)
        r.addWidget(label("RUN", "section"))
        self.steps = spin(1, 1_000_000, width=90)
        self.steps.setValue(400)
        self.steps.setToolTip("Surrogate steps (1 step = the checkpoint's spacing in solver steps; 1 in reference mode)")
        r.addLayout(row(label("Steps"), None, self.steps))
        self.stride = QComboBox()
        for k in (1, 2, 4, 8, 16):
            self.stride.addItem(f"{k}×", k)
        self.stride.setToolTip("Solver steps per shown frame (reference mode): higher plays long rollouts faster")
        r.addLayout(row(label("Speed"), None, self.stride))
        self.run_btn = button("Run", "primary", "Run (Space)", self.run)
        r.addWidget(self.run_btn)
        self.run_error = label("", "error", wrap=True)
        self.run_error.setTextInteractionFlags(Qt.TextSelectableByMouse)
        r.addWidget(self.run_error)
        r.addStretch(1)
        r.addWidget(button("Save config…", slot=self._save_config))
        r.addWidget(button("Open config…", slot=self._open_config))
        side = QWidget()
        side.setLayout(r)
        side.setFixedWidth(170)
        h.addWidget(side)
        return c

    def _build_assistant(self):
        w = QWidget()
        w.setObjectName("page")
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(10)
        head = heading("Research Assistant", 16)
        tag = label(self.assistant.name, "pill")
        tag.setStyleSheet(f"color: {MUTED}; border: 1px solid rgba(255,255,255,0.2);")
        tag.setToolTip("Short commands: fixed rules. Questions: an agent that validates, runs and explains a real "
                       "experiment; numbers not found in tool results are replaced by [unverified]. "
                       + ("Live language model (.env KOLMO_LLM_LIVE=1)." if self.assistant.name.startswith("live")
                          else "Offline scripted planner: no language model, no network."))
        v.addLayout(row(head, None, tag))
        self.findings = QTextBrowser()
        self.findings.setOpenLinks(False)
        v.addWidget(self.findings, 3)
        # agent activity: plan steps with live state, tool chips, progress
        self.act_box, av = card((14, 10, 14, 10), spacing=5, radius=14, level=1)
        self.agent_state = label("", "note", squeeze=True)
        self.report_btn = button("Open report", "link", "Open the report of the last agent run", lambda: self._open_report(self._last_report))
        self.report_btn.setVisible(False)
        av.addLayout(row(heading("Agent activity", 13), 10, self.agent_state, self.report_btn, stretch_first=False))
        av.itemAt(0).layout().setStretch(2, 1)
        self.steps_box = QVBoxLayout()
        self.steps_box.setSpacing(3)
        av.addLayout(self.steps_box)
        self.tool_chips = label("", wrap=True)
        av.addWidget(self.tool_chips)
        self.agent_bar = QProgressBar()
        self.agent_bar.setTextVisible(False)
        self.agent_bar.setFixedHeight(6)
        self.agent_bar.setVisible(False)
        av.addWidget(self.agent_bar)
        self.act_box.setVisible(False)
        v.addWidget(self.act_box)
        self._step_rows, self._step_states, self._spin_i = [], [], 0
        self._spinner = QTimer(self, interval=160, timeout=self._spin)
        self._last_report = None
        pl, plv = card((14, 10, 14, 10), spacing=4, radius=14, level=1)
        self.plan_card = pl
        plv.addWidget(heading("Plan", 13))
        self.plan_label = label("", "note", wrap=True)
        plv.addWidget(self.plan_label)
        v.addWidget(pl)
        self.exp_btn = button("Run experiment", "primary", "Runs the current Setup (Space)", self.run)
        v.addWidget(self.exp_btn)
        self.chat_log = QTextBrowser()
        self.chat_log.setMinimumHeight(110)
        v.addWidget(self.chat_log, 2)
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Ask: run, explain, plan, train, or a research question…")
        self.chat_input.returnPressed.connect(self._ask)
        send = button("➤", "roundPrimary", "Send", self._ask)
        v.addLayout(row(self.chat_input, send, stretch_first=True))
        return w

    def _ask(self):
        text = self.chat_input.text().strip()
        if not text:
            return
        self.chat_input.clear()
        if self._agent_busy:
            self._say(text, "Still working on the previous question; please wait.")
            return
        answer, action = self.assistant.reply(text, {"summary": self.summary, "mode": self.mode(), "pending": self._pending,
                                                     "has_checkpoint": bool(self.ckpt.currentData())})
        self._say(text, answer)
        if action == "run":
            self.run()
        elif action == "reset":
            self.reset()
        elif action in ("research", "confirm"):
            question, confirmed = (self._pending, True) if action == "confirm" else (text, False)
            self._pending, self._agent_busy = None, True
            self._start_agent(question, confirmed)
        else:
            self._pending = None

    def _say(self, text, answer):
        self.chat_log.append((f"<div style='color:{ACCENT2}; margin-top:6px'><b>You</b>&nbsp; {html.escape(text)}</div>"
                              if text else "") + f"<div style='color:{TEXT}'>{html.escape(answer).replace(chr(10), '<br>')}</div>")

    def _agent_done(self, question, r):
        self._agent_busy = False
        if r["status"] == "confirm":
            self._pending = question
            self._say("", r["text"] + " Reply yes to run, anything else to cancel.")
            return
        self._say("", r["text"] + (f"\n\nSaved: {r['report']}" if r.get("report") else ""))
        if r.get("report"):
            self._fill_experiments()

    # ------------------------------------------------------------ agent activity
    def _start_agent(self, question, confirmed):
        self._cancel = False
        self._set_steps([])
        self.tool_chips.setText("")
        self.agent_bar.setVisible(False)
        self.report_btn.setVisible(False)
        self.act_box.setVisible(True)
        self.agent_state.setText("working…")
        task = AgentTask(self.assistant.agent, question, confirmed, lambda: self._cancel)
        self._agent_task = task
        task.event.connect(lambda ev, t=task, q=question: self._agent_event(ev, q) if t is self._agent_task else None)
        self._keep(task)
        task.start()
        self.train_cancel.setEnabled(backend.can_cancel_agent(self.assistant.agent))

    def _set_steps(self, steps):
        while self.steps_box.count():
            it = self.steps_box.takeAt(0)
            if it.layout():
                while it.layout().count():
                    w = it.layout().takeAt(0).widget()
                    if w:
                        w.setParent(None)
        self._step_rows, self._step_states = [], []
        for text in steps:
            icon, lab = label("○"), label(text, "note", wrap=True)
            icon.setFixedWidth(18)
            icon.setStyleSheet(f"font-family: '{SYMBOL_FONT}'; color: {FAINT};")
            h = row(icon, lab, spacing=6)
            h.setStretch(1, 1)
            self.steps_box.addLayout(h)
            self._step_rows.append((icon, lab))
            self._step_states.append("pending")
        if steps:
            self._step_state(0, "running")

    def _step_state(self, i, state):
        if not 0 <= i < len(self._step_rows):
            return
        self._step_states[i] = state
        icon, lab = self._step_rows[i]
        glyph, col = {"pending": ("○", FAINT), "running": ("◐", ACCENT2), "done": ("✓", OK), "failed": ("✕", FAIL)}[state]
        icon.setText(glyph)
        icon.setStyleSheet(f"font-family: '{SYMBOL_FONT}'; color: {col};")
        lab.setStyleSheet(f"color: {TEXT if state == 'running' else MUTED};")
        if "running" in self._step_states:
            self._spinner.start()
        else:
            self._spinner.stop()

    def _spin(self):
        self._spin_i = (self._spin_i + 1) % 4
        for (icon, _), st in zip(self._step_rows, self._step_states):
            if st == "running":
                icon.setText("◐◓◑◒"[self._spin_i])

    def _advance(self, ok=True):
        i = next((k for k, st in enumerate(self._step_states) if st == "running"), None)
        if i is None:
            return
        self._step_state(i, "done" if ok else "failed")
        if ok and i + 1 < len(self._step_states):
            self._step_state(i + 1, "running")

    def _agent_event(self, ev, question):
        t = ev.get("type")
        if t == "plan":
            self._set_steps([str(x) for x in ev.get("steps", [])])
        elif t == "status":
            self.agent_state.setText(str(ev.get("text", "")))
        elif t == "tool":
            ok = bool(ev.get("ok", True))
            col = OK if ok else FAIL
            chip = (f"<span style='background: rgba(255,255,255,0.08); color:{col}; border-radius:8px'>&nbsp;"
                    f"{'✓' if ok else '✕'} {html.escape(str(ev.get('name')))}&nbsp;</span>&nbsp; ")
            self.tool_chips.setText(self.tool_chips.text() + chip)
            self.tool_chips.setToolTip(str(ev.get("summary", "")))
            self._advance(ok)
        elif t == "progress":
            self._on_progress(ev)
        elif t == "result":
            m = ev.get("metrics") or {}
            if m:
                self._say("", "Result: " + ", ".join(f"{k} = {fmt(v) if isinstance(v, float) else v}" for k, v in m.items()))
            if ev.get("report"):
                self._last_report = ev["report"]
                self.report_btn.setVisible(True)
        elif t == "zen":
            self.set_zen(bool(ev.get("on")), source="agent")
        elif t == "error":
            self._say("", f"Error: {ev.get('text', '')}")
            i = next((k for k, st in enumerate(self._step_states) if st == "running"), None)
            if i is not None:
                self._step_state(i, "failed")
        elif t == "answer":
            if ev.get("report"):
                self._last_report = ev["report"]
                self.report_btn.setVisible(True)
            self._agent_done(question, ev)
        elif t == "done":
            for i, st in enumerate(self._step_states):
                if st == "running":
                    self._step_state(i, "done")
            self._agent_busy = False
            self.agent_state.setText("done")
            self.train_cancel.setEnabled(False)

    # ------------------------------------------------------------ training panel
    def _build_training(self):
        c, v = card((20, 14, 20, 14), spacing=6)
        c.setMinimumHeight(290)
        self.train_card = c
        self.phase_chips = {}
        chips = []
        for ph in ("data", "train", "eval"):
            lab = label(ph, "pill")
            lab.setToolTip({"data": "generating reference trajectories", "train": "optimising the network",
                            "eval": "rolling out and scoring"}[ph])
            self.phase_chips[ph] = lab
            chips.append(lab)
        self.train_eta = label("", "counter")
        self.train_eta.setMinimumWidth(190)
        self.train_eta.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.train_cancel = button("Cancel", tip="Ask the running training to stop at its next check",
                                   slot=lambda: (setattr(self, "_cancel", True), self.train_cancel.setEnabled(False)))
        self.train_cancel.setEnabled(False)
        v.addLayout(row(heading("Training"), 12, *chips, None, self.train_eta, self.train_cancel))
        self.train_summary = label("", "muted", squeeze=True)
        v.addWidget(self.train_summary)
        self.train_bar = QProgressBar()
        self.train_bar.setTextVisible(False)
        self.train_bar.setFixedHeight(6)
        v.addWidget(self.train_bar)
        h = QHBoxLayout()
        h.setSpacing(14)
        self.loss_chart = LineChart(log=True, x_label="optimiser step")
        self.val_chart = LineChart(log=True, x_label="optimiser step")
        h.addWidget(self.loss_chart)
        h.addWidget(self.val_chart)
        v.addLayout(h, 1)
        self._reset_training()
        return c

    def _reset_training(self):
        self.tr = {"steps": [], "loss": [], "vsteps": [], "val": []}
        for ph, lab in self.phase_chips.items():
            lab.setStyleSheet(f"color: {FAINT}; border: 1px solid rgba(255,255,255,0.15);")
        self.train_eta.setText("")
        self.train_bar.setValue(0)
        self.train_summary.setText("No training running. Progress (loss, validation rollout error, ETA) appears here when "
                                   "the assistant trains a model.")
        self.loss_chart.clear("Training loss per optimiser step")
        self.val_chart.clear("Validation rollout error per evaluation")

    def _on_progress(self, d):
        """Training progress dict (docs/contracts.md): phase, step, total, loss, val_rollout_error, eta_s."""
        ph = d.get("phase")
        for name, lab in self.phase_chips.items():
            on = name == ph
            lab.setStyleSheet(f"color: {TEXT if on else FAINT}; border: 1px solid {ACCENT2 if on else 'rgba(255,255,255,0.15)'};"
                              + (" background: rgba(124,92,255,0.35);" if on else ""))
        step, total = d.get("step"), d.get("total")
        if step is not None and total:
            self.train_bar.setRange(0, int(total))
            self.train_bar.setValue(int(step))
            self.agent_bar.setVisible(True)
            self.agent_bar.setRange(0, int(total))
            self.agent_bar.setValue(int(step))
        eta = d.get("eta_s")
        self.train_eta.setText(("" if step is None else f"{step}/{total}   ") +
                               (f"ETA {int(eta // 60)}:{int(eta % 60):02d}" if isinstance(eta, (int, float)) else ""))
        if ph == "train" and d.get("loss") is not None and step is not None:
            self.tr["steps"].append(step)
            self.tr["loss"].append(float(d["loss"]))
        if d.get("val_rollout_error") is not None and step is not None:
            self.tr["vsteps"].append(step)
            self.tr["val"].append(float(d["val_rollout_error"]))
        self.loss_chart.xmax = self.val_chart.xmax = total or 1
        self.loss_chart.series = {"train loss": (ACCENT2, self.tr["steps"], self.tr["loss"], False, 2.0)}
        self.val_chart.series = {"val rollout error": (REF, self.tr["vsteps"], self.tr["val"], False, 2.0)}
        self.loss_chart.update()
        self.val_chart.update()
        self.train_summary.setText(f"{ph or ''} · " + training_summary(self.tr["val"], self.tr["loss"]))

    # ------------------------------------------------------------ zen mode
    def set_zen(self, on, source="user"):
        """Zen: dim the backdrop, hide secondary panels, widen the view and the chat (250 ms). An agent's zen-on is
        ignored after the user turned zen off during its task, until the user turns zen on again."""
        if source == "agent":
            if on and self._zen_blocked:
                return
        else:
            self._zen_blocked = (not on) and self._agent_busy
        if on == self.zen:
            return
        self.zen = on
        self.zen_btn.blockSignals(True)
        self.zen_btn.setChecked(on)
        self.zen_btn.blockSignals(False)
        for w in (self.setup_card, self.metrics_box, self.diag_card):
            w.setVisible(not on)
        if not on:
            self.nav_card.setVisible(True)
        group = QParallelAnimationGroup(self)
        for w, prop, a, b in ((self.nav_card, b"minimumWidth", 214, 0), (self.nav_card, b"maximumWidth", 214, 0),
                              (self.side_card, b"minimumWidth", 380, 520), (self.side_card, b"maximumWidth", 380, 520)):
            an = QPropertyAnimation(w, prop, group)
            an.setDuration(250)
            an.setEasingCurve(QEasingCurve.OutCubic)
            an.setStartValue(a if on else b)
            an.setEndValue(b if on else a)
            group.addAnimation(an)
        dim = QVariantAnimation(group)
        dim.setDuration(250)
        dim.setStartValue(0.0 if on else 1.0)
        dim.setEndValue(1.0 if on else 0.0)
        dim.valueChanged.connect(self.backdrop.set_dim)
        group.addAnimation(dim)
        group.finished.connect(lambda: self.nav_card.setVisible(not self.zen))
        group.start(QAbstractAnimation.DeleteWhenStopped)

    def _open_report(self, path):
        if not path or not Path(path).exists():
            return
        d = QDialog(self)
        d.setWindowTitle(Path(path).name)
        d.resize(860, 720)
        d.setStyleSheet(QSS + "QDialog { background: #120c30; }")
        lay = QVBoxLayout(d)
        tb = QTextBrowser()
        tb.setOpenLinks(False)
        tb.setSearchPaths([str(Path(path).parent)])
        tb.setMarkdown(Path(path).read_text(encoding="utf-8"))
        lay.addWidget(tb)
        d.show()
        self._report_dialog = d

    # ------------------------------------------------------------ other pages (read from run files, never invented)
    def _page(self, title, subtitle):
        c, v = card((24, 18, 24, 18), spacing=10)
        v.addWidget(heading(title, 20))
        v.addWidget(label(subtitle, "muted", wrap=True))
        return c, v

    def _build_experiments(self):
        c, v = self._page("Experiments", "Every study Kolmo has run, scored on held-out test trajectories: how long each "
                                         "model's rollout stays within the failure threshold, its error and its speed.")
        self.exp_lead = QHBoxLayout()
        self.exp_lead.setSpacing(12)
        v.addLayout(self.exp_lead)
        self.exp_box = QVBoxLayout()
        self.exp_box.setSpacing(14)
        inner = QWidget()
        inner.setObjectName("page")
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 4, 8, 4)
        lay.addLayout(self.exp_box)
        lay.addStretch(1)
        v.addWidget(scroller(inner), 1)
        return c

    @staticmethod
    def _verdict(ttf):
        """(text, colour) for steps to failure."""
        if not isinstance(ttf, (int, float)) or ttf != ttf:
            return "not measured", MUTED
        if ttf >= 50:
            return "stable", OK
        if ttf >= 10:
            return "short range", ACCENT2
        return "fails early", FAIL

    @staticmethod
    def _chip(text, colour):
        lab = QLabel(text)
        lab.setStyleSheet(f"color: {colour}; background: rgba(255,255,255,0.06); border: 1px solid {colour}55; "
                          "border-radius: 10px; padding: 2px 10px; font-size: 11px; font-weight: 600;")
        return lab

    def _fill_experiments(self):
        for lay in (self.exp_box, self.exp_lead):
            while lay.count():
                it = lay.takeAt(0)
                if it.widget():
                    it.widget().deleteLater()
        exps = backend.experiments()
        if not exps:
            self.exp_box.addWidget(label("No experiment runs yet: ask the assistant, or run "
                                         "python -m kolmo.experiments run <config.json>.", "muted", wrap=True))
            return
        best = []
        for e in exps:
            for r in e["rows"]:
                if r.get("split") == "test" and isinstance(r.get("ttf_median"), (int, float)):
                    best.append((r["ttf_median"], r["model"], e["title"]))
        for rank, (ttf, model, title) in enumerate(sorted(best, key=lambda x: -x[0])[:4]):
            bc, bv = card((16, 12, 16, 12), spacing=2, radius=14, level=1)
            bv.addWidget(label(["BEST", "2ND", "3RD", "4TH"][rank], "section"))
            num = QLabel(f"{ttf:g}")
            num.setStyleSheet(f"font-size: 28px; font-weight: 700; color: {self._verdict(ttf)[1]};")
            bv.addWidget(num)
            bv.addWidget(label("steps before failure", "note"))
            who = label(f"<b>{html.escape(model)}</b> · {html.escape(title)}", "muted", wrap=True)
            who.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            bv.addWidget(who)
            self.exp_lead.addWidget(bc, 1)
        cols = ("Model", "Error @1", "Error @10", "Stays accurate", "Speed", "Verdict")
        for e in exps:
            ec, ev = card((18, 14, 18, 14), spacing=8, radius=16, level=1)
            report = e.get("report")
            title = heading(e["title"], 14)
            title.setWordWrap(True)
            title.setMinimumWidth(10)
            title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            hdr = [title, self._chip(f"{e['n_findings']} findings", MUTED)]
            if report:
                hdr.append(button("Open report", "chip", "Open the run report", lambda _=False, r=report: self._open_report(r)))
            ev.addLayout(row(*hdr, stretch_first=True))
            g = QGridLayout()
            g.setHorizontalSpacing(18)
            g.setVerticalSpacing(6)
            for j, h_ in enumerate(cols):
                g.addWidget(label(h_.upper(), "section"), 0, j)
            rows = [r for r in e["rows"] if r.get("split") == "test"] or e["rows"]
            rows = sorted(rows, key=lambda r: -(r.get("ttf_median") or 0))
            for i, r in enumerate(rows, 1):
                rt = e["runtime"].get(r["model"], {})
                sp = rt.get("speedup")
                ttf = r.get("ttf_median")
                txt, col = self._verdict(ttf)
                vals = (f"<b>{html.escape(str(r['model']))}</b>", fmt(r.get("relL2@1")) if r.get("relL2@1") is not None else "–",
                        fmt(r.get("relL2@10")) if r.get("relL2@10") is not None else "–",
                        f"{ttf:g} steps" if isinstance(ttf, (int, float)) else "–",
                        f"{sp:.2f}×" if isinstance(sp, (int, float)) else "–")
                for j, t in enumerate(vals):
                    g.addWidget(label(t), i, j)
                g.addWidget(self._chip(txt, col), i, len(vals), Qt.AlignLeft)
            g.setColumnStretch(0, 2)
            ev.addLayout(g)
            self.exp_box.addWidget(ec)

    def _build_models(self):
        c, v = self._page("Models", "Trained surrogates: the model registry (runs/registry.json) plus checkpoints on disk "
                                    "not registered yet. Metrics are the experiment's test-set numbers.")
        self.mdl_filter = QComboBox()
        self.mdl_filter.addItem("All systems", None)
        for lbl, s_, _ in self._systems:
            if s_ is not None:
                self.mdl_filter.addItem(lbl, s_.name)
        self.mdl_filter.currentIndexChanged.connect(lambda _: self._fill_models())
        self.mdl_hide = QCheckBox("Hide models that fail early (< 10 steps)")
        self.mdl_hide.setChecked(True)
        self.mdl_hide.toggled.connect(lambda _: self._fill_models())
        v.addLayout(row(label("System"), self.mdl_filter, 16, self.mdl_hide, None))
        h = QHBoxLayout()
        h.setSpacing(18)
        self.mdl_list = QListWidget()
        self.mdl_list.currentRowChanged.connect(self._on_model)
        h.addWidget(self.mdl_list, 3)
        d = QVBoxLayout()
        d.setSpacing(8)
        self.mdl_info = QTextBrowser()
        d.addWidget(self.mdl_info, 1)
        self.mdl_steps = spin(1, 100000, width=90)
        self.mdl_steps.setValue(100)
        self.mdl_run = button("Run vs reference", "primary", "Roll out this model next to the reference solver",
                              self._run_model)
        self.mdl_report = button("Open report", slot=lambda: self._open_report(self._model()["report"]))
        self.mdl_register = button("Register", tip="Add this checkpoint to the model registry", slot=self._register_model)
        d.addLayout(row(label("Steps"), None, self.mdl_steps))
        d.addWidget(self.mdl_run)
        d.addLayout(row(self.mdl_report, self.mdl_register))
        h.addLayout(d, 2)
        v.addLayout(h, 1)
        tr, tv = card((16, 12, 16, 12), spacing=6, radius=14, level=1)
        self.trn_sys = QComboBox()
        for lbl, s_, _ in self._systems:
            if s_ is not None:
                self.trn_sys.addItem(lbl, s_.name)
        self.trn_kind = QComboBox()
        for k in backend.TRAIN_KINDS + ("latent",):
            self.trn_kind.addItem(k, k)
        self.trn_kind.setToolTip("latent: " + backend.TRAIN_HELP_LATENT)
        self.trn_budget = spin(10, 100000, width=90)
        self.trn_budget.setValue(300)
        self.trn_budget.setToolTip("Optimiser steps (upper bound; early stopping may end sooner)")
        self.trn_btn = button("Train", "primary", "Generate reference data with the Simulate parameters of this system, "
                              "fit, save and register; progress in the Training panel", self._train)
        self.trn_note = label("Uses the Simulate parameters when the system matches, else the system defaults.", "note", wrap=True)
        tv.addLayout(row(heading("Train new", 13), 10, label("System"), self.trn_sys, label("Kind"), self.trn_kind,
                         label("Budget"), self.trn_budget, None, self.trn_btn))
        tv.addWidget(self.trn_note)
        v.addWidget(tr)
        self._models = []
        return c

    def _fill_models(self):
        try:
            self._models = backend.models(self.mdl_filter.currentData())
        except Exception as e:  # noqa: BLE001
            self._models = []
            self.mdl_info.setHtml(self._html_note(f"Cannot read the registry: {e}"))
        ttf = lambda e: (e.get("metrics") or {}).get("steps_to_failure")
        if self.mdl_hide.isChecked():
            self._models = [e for e in self._models if not (isinstance(ttf(e), (int, float)) and ttf(e) < 10)]
        self._models.sort(key=lambda e: -(ttf(e) if isinstance(ttf(e), (int, float)) else -1))
        self.mdl_list.clear()
        for e in self._models:
            t = ttf(e)
            verdict, colour = self._verdict(t)
            it = QListWidgetItem(f"{e['name']}   ·   {e['kind']}   ·   {e['system']}"
                                 + (f"   ·   accurate for {t:g} steps ({verdict})" if isinstance(t, (int, float)) else "")
                                 + ("" if e.get("registered", True) else "   ·   not registered"))
            it.setForeground(QColor(colour))
            it.setToolTip(str(e.get("path")))
            self.mdl_list.addItem(it)
        if self._models:
            self.mdl_list.setCurrentRow(0)
        else:
            self.mdl_info.setHtml(self._html_note(backend.TRAIN_HELP))
        self._on_model(self.mdl_list.currentRow())

    def _train(self):
        name, kind, budget = self.trn_sys.currentData(), self.trn_kind.currentData(), self.trn_budget.value()
        s_ = next(x for _, x, _ in self._systems if x is not None and x.name == name)
        params = self._params() if self.current_system().name == name else {}
        self._cancel = False
        self._reset_training()
        self.trn_btn.setEnabled(False)
        self.train_cancel.setEnabled(True)
        self.trn_note.setText(f"Training {kind} on {name} for up to {budget} steps… (Training panel in Simulate)")
        task = TrainTask(lambda prog: backend.train_model(s_, params, kind, budget, prog, lambda: self._cancel))
        task.progress.connect(self._on_progress)
        task.ok.connect(self._train_done)
        task.err.connect(self._train_failed)
        self._train_task = task
        self._keep(task)
        task.start()

    def _train_done(self, entry):
        self.trn_btn.setEnabled(True)
        self.train_cancel.setEnabled(False)
        hist = entry.get("history") or [{}]
        val = hist[-1].get("val_rollout_rel_l2")
        stopped = hist[-1].get("stopped")
        msg = (f"Saved and registered {entry.get('id', entry.get('path'))}"
               + (f"; last val rollout error {val:.3g}" if isinstance(val, (int, float)) else "")
               + (f" (stopped: {stopped})" if stopped else ""))
        self.trn_note.setText(msg)
        self.train_summary.setText("done · " + self.train_summary.text().split(" · ", 1)[-1])
        self._say("", msg)
        if self.pages.currentIndex() == 2:
            self._fill_models()
        if self.trn_sys.currentData() == self.current_system().name:
            self._refresh_ckpts()

    def _train_failed(self, msg):
        self.trn_btn.setEnabled(True)
        self.train_cancel.setEnabled(False)
        self.trn_note.setText(f"Training failed: {msg}")

    def _model(self):
        i = self.mdl_list.currentRow()
        return self._models[i] if 0 <= i < len(self._models) else None

    def _on_model(self, _):
        e = self._model()
        for b in (self.mdl_run, self.mdl_report, self.mdl_register):
            b.setEnabled(e is not None)
        if e is None:
            return
        self.mdl_report.setEnabled(bool(e.get("report")))
        self.mdl_register.setVisible(not e.get("registered", True))
        m = e.get("metrics") or {}
        rows = "".join(f"<tr><td style='color:{MUTED}; padding:2px 12px 2px 0'>{html.escape(k)}</td><td>"
                       f"{html.escape(fmt(v) if isinstance(v, float) else str(v))}</td></tr>" for k, v in m.items()
                       if v is not None)
        try:
            desc = backend.describe_checkpoint(e["path"]).replace("\n", "<br>")
        except (OSError, ValueError, KeyError):
            desc = ""
        self.mdl_info.setHtml(f"<h3>{html.escape(e['name'])}</h3><div style='color:{FAINT}'>{html.escape(str(e['id']))}"
                              f" · {html.escape(str(e.get('created', '')))}</div><div style='color:{MUTED}; margin:6px 0'>"
                              f"{desc}</div><table>{rows or '<tr><td>no test metrics recorded</td></tr>'}</table>")

    def _run_model(self):
        e = self._model()
        if e is None:
            return
        idx = next((i for i, (_, s_, _) in enumerate(self._systems) if s_ is not None and s_.name == e["system"]), None)
        if idx is None:
            self.mdl_info.append(self._html_note(f"System {e['system']} is not available."))
            return
        self.system.setCurrentIndex(idx)
        i = self.ckpt.findData(e["path"])
        if i < 0:
            self.ckpt.insertItem(0, e["name"], e["path"])
            i = 0
        self.ckpt.setCurrentIndex(i)
        self._use_training_params()
        self.mode_group.button(1).setChecked(True)
        self._on_mode()
        self.steps.setValue(self.mdl_steps.value())
        self.nav.button(0).setChecked(True)
        self.pages.setCurrentIndex(0)
        self.run()

    def _register_model(self):
        e = self._model()
        try:
            backend.register_model(e["path"])
        except Exception as ex:  # noqa: BLE001
            self.mdl_info.append(self._html_note(f"Register failed: {ex}"))
            return
        self._fill_models()

    # ------------------------------------------------------------ geometry: photo -> Meta SAM 3.1 -> wind tunnel
    def _build_geometry(self):
        c, v = self._page("Photo wind tunnel", "Drop a photo, say what the object is: Meta SAM 3.1 (running locally) cuts "
                                               "it out, Kolmo levels it, scales it and puts it in a virtual wind tunnel "
                                               "with the flow from left to right.")
        self.geo = {"path": None, "rgb": None, "mask": None, "point": None}
        self.photo_view = PhotoView()
        self.photo_view.dropped.connect(self._geo_load)
        self.photo_view.clicked.connect(self._geo_click)
        side = QVBoxLayout()
        side.setSpacing(10)
        self.geo_text = QLineEdit()
        self.geo_text.setPlaceholderText("What is it?  e.g. car, airplane, wing, mug")
        self.geo_text.returnPressed.connect(self._geo_segment)
        self.geo_seg = button("Find it with SAM 3.1", "primary", "Segment the object named above (or click it in the photo)",
                              self._geo_segment)
        self.geo_status = label("Drop a photo or choose one.", "muted", wrap=True)
        self.tunnel_view = QLabel()
        self.tunnel_view.setFixedSize(340, 200)
        self.tunnel_view.setAlignment(Qt.AlignCenter)
        self.tunnel_view.setStyleSheet("background: rgba(0,0,0,0.45); border: 1px solid rgba(255,255,255,0.12); "
                                       "border-radius: 12px;")
        self.geo_size = QDoubleSpinBox()
        self.geo_size.setRange(0.12, 0.5)
        self.geo_size.setSingleStep(0.02)
        self.geo_size.setValue(0.32)
        self.geo_size.setToolTip("Object length as a fraction of the tunnel")
        self.geo_flip = QCheckBox("Flip (front faces right)")
        self.geo_ground = QCheckBox("Road under it")
        self.geo_ground.setChecked(True)
        self.geo_res = QComboBox()
        for n in (192, 256, 384):
            self.geo_res.addItem(f"{n}²", n)
        self.geo_res.setCurrentIndex(1)
        self.geo_wind = QDoubleSpinBox()
        self.geo_wind.setRange(0.25, 3.0)
        self.geo_wind.setSingleStep(0.25)
        self.geo_wind.setValue(1.0)
        for w in (self.geo_size, self.geo_flip, self.geo_ground, self.geo_res):
            (w.toggled if isinstance(w, QCheckBox) else w.valueChanged if isinstance(w, QDoubleSpinBox)
             else w.currentIndexChanged).connect(lambda *_: self._geo_preview())
        self.geo_go = button("Run wind tunnel  ▶", "primary", "Open Simulate and run the flow past this object", self._tunnel_run)
        self.geo_go.setEnabled(False)
        side.addWidget(label("OBJECT", "section"))
        side.addWidget(self.geo_text)
        side.addLayout(row(button("Choose photo…", slot=self._geo_choose), self.geo_seg))
        side.addWidget(self.geo_status)
        side.addWidget(label("TUNNEL", "section"))
        side.addWidget(self.tunnel_view)
        side.addLayout(row(label("Size"), self.geo_size, label("Wind"), self.geo_wind, None))
        side.addLayout(row(self.geo_ground, self.geo_flip, None))
        side.addLayout(row(label("Grid"), self.geo_res, None, self.geo_go))
        side.addStretch(1)
        h = QHBoxLayout()
        h.setSpacing(18)
        h.addWidget(self.photo_view, 3)
        h.addLayout(side, 2)
        v.addLayout(h, 1)
        sample = Path(__file__).resolve().parents[2] / "data" / "samples" / "car_side.png"
        if sample.exists():
            QTimer.singleShot(0, lambda: self._geo_load(str(sample)))
        return c

    def _geo_choose(self):
        path, _ = QFileDialog.getOpenFileName(self, "Photo of an object", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if path:
            self._geo_load(path)

    def _geo_load(self, path):
        import cv2
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            self.geo_status.setText(f"Cannot read {Path(path).name} as an image.")
            return
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        s = 1400 / max(rgb.shape[:2])
        if s < 1:
            rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        self.geo.update(path=path, rgb=rgb, mask=None, point=None)
        cached = Path(path).with_name(Path(path).stem + "_mask.npy")
        if cached.exists():  # a mask SAM made earlier for this photo
            m = np.load(cached)
            if m.shape == rgb.shape[:2]:
                self.geo["mask"] = m
        self.geo_go.setEnabled(self.geo["mask"] is not None)
        self.photo_view.show_rgb(self._geo_overlay())
        self.geo_status.setText(f"{Path(path).name}: type what it is, then Find it." if self.geo["mask"] is None else
                                f"{Path(path).name}: segmented earlier by SAM 3.1. Adjust and run, or Find it again.")
        if not self.geo_text.text() and "car" in Path(path).stem.lower():
            self.geo_text.setText("car")
        self._geo_preview()

    def _geo_click(self, x, y):
        if self.geo["rgb"] is not None:
            self.geo["point"] = (x, y)
            self._geo_segment()

    def _geo_overlay(self):
        import cv2
        img, m = self.geo["rgb"].copy(), self.geo["mask"]
        if m is not None:
            img[m] = (0.6 * img[m] + 0.4 * np.array([70, 150, 255])).astype(np.uint8)
            edge = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(np.float32)
            glow = np.clip(cv2.GaussianBlur(edge, (0, 0), 4) * 2.5 + edge, 0, 1)[..., None]
            img = (img * (1 - glow) + glow * np.array([255, 225, 110])).astype(np.uint8)
        if self.geo["point"] is not None:
            cv2.circle(img, tuple(int(v) for v in self.geo["point"]), 7, (60, 255, 140), -1, cv2.LINE_AA)
        return img

    def _geo_segment(self):
        if self.geo["rgb"] is None:
            self.geo_status.setText("Choose or drop a photo first.")
            return
        text = self.geo_text.text().strip() or None
        prompt = {"text": text, "point": self.geo["point"]} if (text or self.geo["point"]) else None
        rgb = self.geo["rgb"]

        def work():
            from kolmo.geometry import photo_dialog as pd
            from kolmo.geometry.segment import get_segmenter
            if pd._SEGMENTER is None:
                pd._SEGMENTER = get_segmenter("auto")
            seg = pd._SEGMENTER
            if hasattr(seg, "progress"):
                seg.progress = self.geo_progress.emit
            t0 = time.perf_counter()
            mask = seg.segment(rgb, prompt)
            return mask, seg.name, time.perf_counter() - t0, getattr(seg, "last_info", {}).get("score")

        task = Task(work)
        task.ok.connect(self._geo_done)
        task.err.connect(lambda m: (self.geo_status.setText(f"Segmentation failed: {m}"), self.geo_seg.setEnabled(True)))
        self._keep(task)
        self.geo_seg.setEnabled(False)
        self.geo_status.setText("Running Meta SAM 3.1 on this computer… (the first run loads the model, about a minute)")
        task.start()

    def _geo_done(self, res):
        mask, name, secs, score = res
        self.geo_seg.setEnabled(True)
        if not np.asarray(mask).any():
            self.geo_status.setText("Nothing found: try another word, or click on the object.")
            return
        self.geo["mask"] = np.asarray(mask, bool)
        if self.geo["path"]:
            try:
                np.save(Path(self.geo["path"]).with_name(Path(self.geo["path"]).stem + "_mask.npy"), self.geo["mask"])
            except OSError:
                pass
        self.photo_view.show_rgb(self._geo_overlay())
        who = "Meta SAM 3.1" if name == "sam3" else "GrabCut (SAM 3.1 not installed)"
        conf = f", confidence {score:.2f}" if isinstance(score, (int, float)) else ""
        self.geo_status.setText(f"{who}: found it in {secs:.1f} s{conf}. Re-prompt with text or a click to refine.")
        self.geo_go.setEnabled(True)
        self._geo_preview()

    def _tunnel(self, n=None):
        from kolmo.geometry.mask import tunnel_grid
        n = n or self.geo_res.currentData()
        return tunnel_grid(self.geo["mask"], n=n, chord=self.geo_size.value(), x=0.16,
                           ground=0.3 if self.geo_ground.isChecked() else None, ground_cells=max(2, n // 80),
                           flip=self.geo_flip.isChecked())

    def _tunnel_sprite(self):
        """Photo cut-out (RGBA QImage, oriented like the grid) for drawing over the obstacle cells."""
        from kolmo.geometry.mask import orient
        m = self.geo["mask"]
        x0, y0, x1, y1 = orient(m)[2]
        rgba = np.dstack([self.geo["rgb"][y0:y1, x0:x1], (m[y0:y1, x0:x1] * 255).astype(np.uint8)])
        if self.geo_flip.isChecked():
            rgba = rgba[:, ::-1]
        rgba = np.ascontiguousarray(rgba)
        return QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format_RGBA8888).copy()

    def _geo_preview(self):
        if self.geo.get("mask") is None:
            self.tunnel_view.setText("The tunnel preview appears here")
            return
        try:
            grid, meta = self._tunnel(160)
        except ValueError as e:
            self.tunnel_view.setText(str(e))
            return
        img = np.full(grid.shape + (3,), (6, 8, 22), np.uint8)
        img[grid] = (70, 74, 92)
        qi = QImage(np.ascontiguousarray(img[::-1]).data, 160, 160, 480, QImage.Format_RGB888).copy()
        W_, H_ = self.tunnel_view.width() - 8, self.tunnel_view.height() - 8
        pm = QPixmap(W_, H_)
        pm.fill(QColor(6, 8, 22))
        p = QPainter(pm)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        r0 = 0.22 if self.geo_ground.isChecked() else 0.5 - 0.5 * H_ / W_  # show a 16:9-ish band around the object
        src = QRectF(0, (1 - r0 - H_ / W_) * 160, 160, H_ / W_ * 160)
        p.drawImage(QRectF(0, 0, W_, H_), qi, src)
        c0, rr, w, h = meta["box"]
        sx = W_ / 160
        top = (160 - (rr + h)) * sx - src.top() * sx
        p.drawImage(QRectF(c0 * sx, top, w * sx, h * sx), self._tunnel_sprite())
        p.setPen(QPen(QColor(120, 190, 255, 160), 2))
        for k in range(4):  # wind arrows
            y = H_ * (0.18 + 0.12 * k)
            p.drawLine(QPointF(8, y), QPointF(30, y))
            p.drawLine(QPointF(30, y), QPointF(24, y - 4))
            p.drawLine(QPointF(30, y), QPointF(24, y + 4))
        p.end()
        self.tunnel_view.setPixmap(pm)

    def _run_scenario(self, i):
        key = self.scenario.itemData(i)
        self.scenario.setCurrentIndex(0)
        if key is None:
            return
        if key == "car":
            self.nav.button(4).setChecked(True)
            self._on_nav(4)
            if self.geo.get("mask") is not None:
                self._tunnel_run()
            return
        label_, name, make_params, steps, stride, field = SCENARIOS[key]
        if self.runner is not None:
            self.reset()
        idx = next((j for j, (_, s, _) in enumerate(self._systems) if s is not None and s.name == name), None)
        if idx is None:
            return
        self.system.setCurrentIndex(idx)
        self._on_system()
        params = make_params()
        cfg = self.config()
        cfg["params"].update({k: v for k, v in params.items() if k in self.param_widgets})
        cfg.update(mode="reference", steps=steps, stride=stride, obstacle={"kind": "none"})
        self.apply_config(cfg)
        self.extra = {"_system": name, **{k: v for k, v in params.items() if k not in self.param_widgets}}
        if field in self.current_system().fields:
            self.field_sel.setCurrentIndex(list(self.current_system().fields).index(field))
        self.nav.button(0).setChecked(True)
        self._on_nav(0)
        self.run()

    def _tunnel_run(self):
        if self.geo.get("mask") is None:
            return
        n = self.geo_res.currentData()
        try:
            grid, meta = self._tunnel(n)
        except ValueError as e:
            self.geo_status.setText(str(e))
            return
        if self.runner is not None:
            self.reset()
        self.tunnel = {"grid": grid, "meta": meta, "sprite": self._tunnel_sprite()}
        idx = next(i for i, (_, s, _) in enumerate(self._systems) if s is not None and s.name == "fluid2d")
        self.system.setCurrentIndex(idx)
        self._on_system()
        cfg = self.config()
        U = self.geo_wind.value()
        cfg["params"].update(n=n, U_inf=U, nu=0.4 * U / n, Re=None, sponge=0.08, beta=0.0, A=0.0, mu=0.0)
        cfg.update(mode="reference", steps=6000, stride=4, obstacle={"kind": "tunnel", "size": self.geo_size.value()})
        self.apply_config(cfg)
        self.nav.button(0).setChecked(True)
        self._on_nav(0)
        self.run()

    def _build_reports(self):
        c, v = self._page("Reports", "report.md files written by kolmo.experiments.")
        self.rep_list = QComboBox()
        self.rep_list.currentIndexChanged.connect(self._show_report)
        v.addWidget(self.rep_list)
        self.rep_view = QTextBrowser()
        v.addWidget(self.rep_view, 1)
        return c

    def _fill_reports(self):
        self.rep_list.blockSignals(True)
        self.rep_list.clear()
        for e in backend.experiments():
            if e["report"]:
                self.rep_list.addItem(e["name"], e["report"])
        self.rep_list.blockSignals(False)
        self._show_report()

    def _show_report(self, *_):
        path = self.rep_list.currentData()
        if not path:
            self.rep_view.setHtml(f"<p style='color:{MUTED}'>No reports yet. Run "
                                  "<code>python -m kolmo.experiments run &lt;config.json&gt;</code>.</p>")
            return
        self.rep_view.setSearchPaths([str(Path(path).parent)])
        self.rep_view.setMarkdown(Path(path).read_text(encoding="utf-8"))

    # ------------------------------------------------------------ state helpers
    def current_system(self):
        return self._systems[self.system.currentIndex()][1]

    def system_label(self):
        return self._systems[self.system.currentIndex()][0]

    def mode(self):
        return MODES[self.mode_group.checkedId()][0]

    def _cand_name(self):
        return "Hybrid" if self.mode() == "hybrid" else "Surrogate"

    def _on_system(self):
        s = self.current_system()
        if self.extra.get("_system") != s.name:
            self.extra = {}
        while self.params_grid.count():
            w = self.params_grid.takeAt(0).widget()
            if w:
                w.setParent(None)  # off screen now; deleteLater alone leaves it drawn until the next event loop pass
                w.deleteLater()
        self.specs, self.param_widgets, self.param_labels = backend.param_specs(s), {}, {}
        d = backend.defaults(s)
        for i, sp in enumerate(self.specs):
            unit = f"  <span style='color:{FAINT}'>{sp.unit}</span>" if sp.unit else ""
            lab = label(sp.label + unit)
            lab.setToolTip(sp.tip)
            if sp.choices:
                w = QComboBox()
                w.setFixedWidth(100)
                if sp.auto:
                    w.addItem(AUTO, AUTO)
                for ch in sp.choices:
                    w.addItem(str(ch).replace("_", " "), ch)
            else:
                w = spin(sp.lo, sp.hi, sp.decimals, AUTO if sp.auto else None, width=100)
            w.setToolTip(sp.tip)
            self._set_value(w, d.get(sp.key))
            self.param_labels[sp.key] = lab
            self.params_grid.addWidget(lab, i // 2, (i % 2) * 2)
            self.params_grid.addWidget(w, i // 2, (i % 2) * 2 + 1)
            self.param_widgets[sp.key] = w
        self.system_note.setText(backend.describe(s))
        pair = [k for k in backend.exclusive(s) if k in self.param_widgets]
        self.excl_sel.blockSignals(True)
        self.excl_sel.clear()
        for k in pair:
            self.excl_sel.addItem(f"set by {'ν' if k == 'nu' else k}", k)
        self.excl_sel.setVisible(len(pair) == 2)
        self.excl_sel.blockSignals(False)
        self._apply_exclusive()
        self.system_note.setToolTip(backend.describe(s))
        self.field_sel.blockSignals(True)
        self.field_sel.clear()
        self.field_sel.addItems(list(s.fields))
        self.field_sel.setVisible(len(s.fields) > 1)
        self.field_sel.blockSignals(False)
        self.cbar.symbol = {"vorticity": "ω", "temperature": "T"}.get(s.fields[0], s.fields[0][:1])
        self.photo, self.compare = None, []  # curves of another system are not comparable
        self._refresh_ckpts()
        self._update_obstacle_ui()
        self._on_mode()

    @staticmethod
    def _set_value(w, v):
        if isinstance(w, QComboBox):
            v = AUTO if v is None else v
            i = w.findData(v)
            if i < 0:
                w.addItem(str(v), v)
                i = w.count() - 1
            w.setCurrentIndex(i)
        else:
            w.setValue(w.minimum() if v is None else v)

    def _apply_exclusive(self):
        pair = [self.excl_sel.itemData(i) for i in range(self.excl_sel.count())]
        for k in pair:
            on = k == self.excl_sel.currentData()
            self.param_widgets[k].setVisible(on)
            self.param_labels[k].setVisible(on)

    def _params(self):
        out = {}
        for sp in self.specs:
            w = self.param_widgets[sp.key]
            v = w.currentData() if isinstance(w, QComboBox) else w.value()
            out[sp.key] = None if sp.auto and (v == AUTO or (not isinstance(w, QComboBox) and v == w.minimum())) else v
        for i in range(self.excl_sel.count()):  # only the selected member of an exclusive pair is sent
            if self.excl_sel.itemData(i) != self.excl_sel.currentData():
                out[self.excl_sel.itemData(i)] = None
        return out

    def _update_obstacle_ui(self):
        ok = backend.supports_obstacle(self.current_system())
        for w in (self.obstacle, self.obstacle_size):
            w.setEnabled(ok)
        if not ok:
            self.obstacle.setCurrentIndex(0)
        self._update_thumb()

    def _obstacle_kind(self):
        return OBSTACLES[self.obstacle.currentIndex()][0]

    def _on_obstacle(self):
        if self._obstacle_kind() == "tunnel" and self.tunnel is None:
            self.obstacle.setCurrentIndex(0)
            self.nav.button(4).setChecked(True)
            self._on_nav(4)
            return
        if self._obstacle_kind() == "photo":
            from kolmo.geometry.photo_dialog import ask  # SAM 3.1 text/click prompts, cached embedding, live preview
            res = ask(self, int(self._params().get("n") or 128), self.obstacle_size.value())
            if res:
                self._photo_done(*res)
            else:
                self.obstacle.setCurrentIndex(0)
        self._update_thumb()

    def _segment_photo(self, path):
        n, size = int(self._params().get("n") or 128), self.obstacle_size.value()
        self.photo = None
        task = Task(lambda: backend.photo_obstacle(path, n, size))
        task.ok.connect(lambda res: self._photo_done(path, *res))
        task.err.connect(self._photo_failed)
        self._keep(task)
        task.start()
        self.run_btn.setEnabled(False)
        self.obstacle_note.setText("Segmenting the photo…")
        self.geo_note.setText("Segmenting the photo…")

    def _photo_done(self, path, mask, meta):
        self.photo = {"path": path, "mask": mask, "meta": meta}
        self.run_btn.setEnabled(self.runner is None)
        self._update_thumb()
        self.geo_preview.setPixmap(QPixmap.fromImage(mask_image(mask, TEXT)).scaled(300, 300, Qt.KeepAspectRatio))
        self.geo_note.setText(f"{Path(path).name}\nsegmenter: {meta.get('segmenter')} "
                              f"({meta.get('segment_seconds', 0):.1f} s)\n{meta['area_cells']} cells "
                              f"({100 * meta['area_fraction']:.1f}% of the domain), chord {meta['chord_cells']} cells, "
                              f"frontal height {meta['frontal_height_cells']} cells on a {meta['n']}² grid.\n"
                              "Now the obstacle in Simulate (set U∞ > 0 for flow past it).")

    def _photo_failed(self, msg):
        self.photo = None
        self.run_btn.setEnabled(self.runner is None)
        self.obstacle.setCurrentIndex(0)
        self._update_thumb()
        self.obstacle_note.setText(f"Photo failed: {msg}")
        self.geo_note.setText(f"Photo failed: {msg}")

    def _current_mask(self):
        kind = self._obstacle_kind()
        if kind == "none" or not backend.supports_obstacle(self.current_system()):
            return None
        if kind == "photo":
            return self.photo["mask"] if self.photo else None
        if kind == "tunnel":
            return self.tunnel["grid"] if self.tunnel else None
        return backend.builtin_obstacle(kind, int(self._params().get("n") or 64), self.obstacle_size.value())

    def _update_thumb(self):
        mask = self._current_mask()
        self.thumb.setVisible(mask is not None)
        if mask is None:
            if self._obstacle_kind() != "photo":
                self.obstacle_note.setText("No obstacle." if backend.supports_obstacle(self.current_system()) else
                                           f"{self.system_label()} takes no obstacle.")
            return
        self.thumb.setPixmap(QPixmap.fromImage(mask_image(mask, TEXT)).scaled(58, 58, Qt.KeepAspectRatio))
        meta = (self.photo or {}).get("meta", {}) if self._obstacle_kind() == "photo" else {}
        src = f"{meta['segmenter']}, " if meta.get("segmenter") else ""
        self.obstacle_note.setText(f"{src}{int(mask.sum())} cells ({100 * mask.mean():.1f}%) at x = 0.25 L. "
                                   "Grey in the panels; needs U∞ > 0.")

    def _refresh_ckpts(self):
        self.ckpt.blockSignals(True)
        self.ckpt.clear()
        for lbl, stem in backend.checkpoints(self.current_system()):
            self.ckpt.addItem(lbl, stem)
            self.ckpt.setItemData(self.ckpt.count() - 1, stem, Qt.ToolTipRole)
        if self.ckpt.count() == 0:
            self.ckpt.addItem("none found", None)
        self.ckpt.blockSignals(False)

    def _browse_ckpt(self):
        path, _ = QFileDialog.getOpenFileName(self, "Surrogate checkpoint", str(backend.ROOT / "runs"),
                                              "Kolmo checkpoint (*.json)")
        if path:
            stem = str(Path(path).with_suffix(""))
            self.ckpt.insertItem(0, Path(stem).name, stem)
            self.ckpt.setCurrentIndex(0)

    def _use_training_params(self):
        stem = self.ckpt.currentData()
        if not stem:
            return
        tp = backend.training_params(stem)
        for sp in self.specs:
            if sp.key in tp:
                self._set_value(self.param_widgets[sp.key], tp[sp.key])
        self._on_mode()

    def _ckpt_status(self):
        """(note text, mismatches) for the selected checkpoint."""
        stem = self.ckpt.currentData()
        if not stem:
            return backend.TRAIN_HELP, []
        try:
            text, tp = backend.describe_checkpoint(stem), backend.training_params(stem)
        except (OSError, ValueError, KeyError) as e:
            return f"Cannot read checkpoint: {e}", []
        cur = self._params()
        miss = [f"{k} {cur[k]} (trained {tp[k]})" for k in cur if k in tp and cur[k] is not None and cur[k] != tp[k]
                and not (isinstance(cur[k], float) and np.isclose(cur[k], float(tp[k]), rtol=1e-6))]
        return text, miss

    def _on_mode(self):
        m = self.mode()
        self.exec_box.setVisible(m != "reference")
        self.stride.setEnabled(m == "reference")
        self.cand_view.setVisible(m != "reference")
        self.policy_box.setVisible(m == "hybrid")
        note, miss = self._ckpt_status() if m != "reference" else ("", [])
        self.ckpt_note.setText(note + (f"\n! outside training: {', '.join(miss)}" if miss else ""))
        self.use_train.setVisible(m != "reference" and bool(self.ckpt.currentData()) and bool(miss))
        self.mode_note.setText({
            "reference": "Reference solver only: nothing to compare, so the prediction panel and error metrics stay "
                         "empty.",
            "surrogate": "The surrogate rolls out alone from the same initial state; the reference runs alongside "
                         "to measure the error.",
            "hybrid": "kolmo.hybrid runs the surrogate with solver interventions from the current hybrid state; the "
                      "result is replayed next to the live reference.",
        }[m])
        cand = self._cand_name()
        self.cand_view.title, self.cand_view.qualifier = cand, "(neural + solver)" if m == "hybrid" else "(neural)"
        self.err_view.title = "Absolute error"
        self.err_toggle.setChecked(m != "reference")
        if self.runner is None:  # a changed setup invalidates what is on screen
            self.reset()

    # ------------------------------------------------------------ config
    def config(self):
        kind = self._obstacle_kind()
        obst = {"kind": kind, "size": self.obstacle_size.value()}
        if kind == "photo" and self.photo:
            obst["path"] = self.photo["path"]
        params = self._params()
        if self.extra.get("_system") == self.current_system().name:
            params.update({k: v for k, v in self.extra.items() if k != "_system"})
        if kind == "tunnel" and self.tunnel:
            obst["grid"] = self.tunnel["grid"]
            params["ic"] = "zero"  # the wind starts from rest
        policy = {"every": self.pol_every.value(), "residual_tol": self.pol_res.value() or None,
                  "energy_tol": self.pol_energy.value() or None}
        return {"system": self.current_system().name, "params": params, "seed": self.seed.value(),
                "obstacle": obst, "mode": self.mode(), "checkpoint": self.ckpt.currentData(),
                "policy": policy, "steps": self.steps.value(), "stride": self.stride.currentData()}

    def apply_config(self, cfg):
        idx = next((i for i, (_, s, _) in enumerate(self._systems) if s is not None and s.name == cfg["system"]), None)
        if idx is None:
            raise ValueError(f"system {cfg['system']!r} is not available here")
        self.system.setCurrentIndex(idx)
        self._on_system()
        for sp in self.specs:
            if sp.key in cfg.get("params", {}):
                self._set_value(self.param_widgets[sp.key], cfg["params"][sp.key])
        self.seed.setValue(int(cfg.get("seed", 0)))
        self.steps.setValue(int(cfg.get("steps", 400)))
        self.stride.setCurrentIndex(max(0, self.stride.findData(int(cfg.get("stride", 1)))))
        pol = cfg.get("policy") or {}
        self.pol_every.setValue(int(pol.get("every") or 0))
        self.pol_res.setValue(float(pol.get("residual_tol") or 0))
        self.pol_energy.setValue(float(pol.get("energy_tol") or 0))
        if cfg.get("checkpoint"):
            i = self.ckpt.findData(cfg["checkpoint"])
            if i < 0:
                self.ckpt.insertItem(0, Path(cfg["checkpoint"]).name, cfg["checkpoint"])
                i = 0
            self.ckpt.setCurrentIndex(i)
        self.mode_group.button([m for m, _ in MODES].index(cfg.get("mode", "reference"))).setChecked(True)
        obst = cfg.get("obstacle") or {}
        self.obstacle_size.setValue(float(obst.get("size", 0.2)))
        if backend.supports_obstacle(self.current_system()):
            kind = obst.get("kind", "none")
            self.obstacle.setCurrentIndex([k for k, _ in OBSTACLES].index(kind if kind != "tunnel" or self.tunnel else "none"))
            if obst.get("kind") == "photo" and obst.get("path"):
                self._segment_photo(obst["path"])
        self._update_thumb()
        self._on_mode()

    def _save_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save config", str(backend.ROOT / "kolmo_config.json"),
                                              "Kolmo config (*.json)")
        if path:
            backend.save_config(self.config(), path)

    def _open_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open config", str(backend.ROOT), "Kolmo config (*.json)")
        if path:
            try:
                self.apply_config(backend.load_config(path))
                self.run_error.setText("")
            except (ValueError, KeyError, OSError) as e:
                self.run_error.setText(f"Could not open config: {e}")

    # ------------------------------------------------------------ run control
    def _keep(self, thread):
        """Hold a reference until the thread ends (a QThread destroyed while running aborts the process)."""
        self._threads = [t for t in self._threads if t.isRunning()] + [thread]

    def _space(self):
        if self.runner:
            self.toggle_pause()
        elif self.run_btn.isEnabled():
            self.run()

    def run(self):
        self.reset()
        cfg = self.config()
        tun = cfg["obstacle"].get("kind") == "tunnel" and self.tunnel is not None
        n = int(cfg["params"].get("n") or 256)
        low = int(n * ((self.tunnel["meta"]["ground"] or 0.42) - 0.015)) if tun else 0
        sprite = (self.tunnel["sprite"], self.tunnel["meta"]["box"], low) if tun else None
        for view in (self.ref_view, self.cand_view):
            view.sprite = sprite
        self.cand_view.setVisible(cfg["mode"] != "reference")  # nothing to compare: hide the empty panel
        if tun:
            self.err_toggle.setChecked(False)
        if tun:  # fixed scale: the impulsive start would otherwise set it far above the wake's vorticity
            m = 5.0 * float(cfg["params"].get("U_inf") or 1.0)
            self.scale, self.scale_fixed = ("div", -m, m), True
        r = Runner(cfg)
        # each slot ignores a runner that was reset away (its queued signals can still arrive)
        for sig, slot in ((r.started_run, self._on_started), (r.frames, self._on_frames), (r.done, self._on_done),
                          (r.failed, self._on_failed)):
            sig.connect(lambda x, r=r, slot=slot: slot(x) if r is self.runner else None)
        self.runner = r
        self._keep(r)
        self._set_running(True)
        self._set_status("Starting", ACCENT2)
        self.counter.setText(f"step 0 / {cfg['steps']}")
        self.err_chart.xmax = self.diag_chart.xmax = cfg["steps"]
        self.findings.setHtml(self._html_note("Compiling and starting…"))
        r.start()

    def _set_running(self, on):
        self.config_box.setEnabled(not on)
        self.run_btn.setEnabled(not on)
        self.exp_btn.setEnabled(not on)
        self.play.setText("❚❚" if on else "▶")
        self.play.setToolTip("Pause (Space)" if on else "Run (Space)")
        self.backdrop.hold = on  # keep the CPU for the solver while it runs: measured timings stay clean

    def toggle_pause(self):
        if not self.runner:
            return
        self.runner.paused = not self.runner.paused
        self.play.setText("▶" if self.runner.paused else "❚❚")
        self.play.setToolTip("Resume (Space)" if self.runner.paused else "Pause (Space)")
        self._set_status("Paused" if self.runner.paused else "Running", MUTED if self.runner.paused else ACCENT2)

    def reset(self):
        if self.runner:
            self.runner.stopped = True
            self.runner = None
        self.xs, self.err, self.diag_ref, self.diag_cand = [], [], {}, {}
        self.scale = self.run_obstacle = self.live = self._last_frame = None
        self.scale_fixed = False
        self.spacing, self.summary, self.timing, self.horizon = 1, None, {}, None
        self._set_running(False)
        self.run_error.setText("")
        self._set_status("Idle", MUTED)
        self.counter.setText("")
        self.scrub.blockSignals(True)
        self.scrub.setRange(0, 0)
        self.scrub.blockSignals(False)
        self.scrub.setEnabled(False)
        self.diag_key.blockSignals(True)
        self.diag_key.clear()
        self.diag_key.blockSignals(False)
        self.findings.setHtml(self._html_note("Measured facts and rule-based findings (kolmo.diagnose) appear here "
                                              "when a run finishes. Ask: run, explain, plan, train."))
        self.plan_label.setText("Plans come from kolmo.diagnose after a surrogate or hybrid run.")
        self._update_header()
        self._empty_states()
        self._update_metrics(None)

    def _empty_states(self):
        m = self.mode()
        for v in (self.ref_view, self.cand_view, self.err_view):
            v.subtitle = ""
        self.cbar.vmin = None
        self.cbar.update()
        self.ref_view.clear("Press ▶ or Space to run the reference solver.")
        self.err_chart.vlines = []
        self.diag_chart.clear("Run to plot the system's diagnostics per step.")
        if m == "reference":
            self.cand_view.clear("Reference mode: no prediction runs.\n\nChoose Surrogate or Hybrid in Setup to compare.")
            self.err_view.clear("The error needs a prediction to compare against the reference.")
            msg = "Reference mode: no prediction, so no error to plot."
        else:
            missing = backend.TRAIN_HELP if self.ckpt.currentData() is None else None
            self.cand_view.clear(missing or "Press ▶ to roll out the prediction.")
            self.err_view.clear("The error appears once the prediction runs.")
            msg = missing or "Run to plot the relative error per step."
        self.err_chart.clear(msg + ("\nEarlier runs stay here for comparison." if not self.compare else ""))
        self._bind_err()

    def _update_header(self, run=None):
        self.run_title.setText(f"{self.system_label()}  ·  {dict(MODES)[self.mode()]}")
        if run is None:
            self.run_sub.setText("Not running · set up the run on the right")
            return
        p, s = run.params, run.system
        keys = [sp.key for sp in backend.param_specs(s)] + list(backend.header_keys(s))
        keys = list(dict.fromkeys(k for k in keys if k in p and isinstance(p[k], (int, float, str)) and k != "mask"))
        show = lambda v: fmt(v) if isinstance(v, float) else str(v)
        parts = [f"{k} {show(p[k])}" for k in keys] + [f"seed {run.seed}"]
        if run.spacing > 1:
            parts.append(f"1 step = {run.spacing} solver steps")
        if run.obstacle is not None:
            parts.append(f"obstacle {int(run.obstacle.sum())} cells")
        self.run_sub.setText("   ·   ".join(parts))

    def _set_status(self, text, colour):
        self.pill.setText(text)
        self.pill.setStyleSheet(f"color: {colour}; border: 1px solid {colour}; background: rgba(255,255,255,0.05);")

    def _on_started(self, run):
        self.live = run
        self._update_header(run)
        self.run_obstacle, self.spacing = run.obstacle, run.spacing
        self._set_status("Hybrid rollout…" if run.mode == "hybrid" else "Running", ACCENT2)
        self.findings.setHtml(self._html_note("Running kolmo.hybrid.run with this policy, then replaying it…"
                                              if run.mode == "hybrid" else "Running…"))

    def _on_frames(self, payload):
        frames, self.timing = payload
        if self.pill.text() not in ("Paused", "Running"):
            self._set_status("Running", ACCENT2)
        for fr in frames:
            self.xs.append(fr.step)
            if fr.rel_error is not None:
                self.err.append(fr.rel_error)
                if self.horizon is None and fr.step >= 1 and not fr.rel_error <= self.threshold:
                    self.horizon = fr.step
            for k, v in fr.diag_reference.items():
                self.diag_ref.setdefault(k, []).append(v)
            if fr.candidate is not None:
                for k in self.diag_ref:
                    self.diag_cand.setdefault(k, []).append(fr.diag_candidate.get(k, float("nan")))
            for e in fr.events:
                self.err_chart.vlines.append((fr.step, EVENT if e.get("kind") == "intervention" else FAIL))
        last = frames[-1]
        if self.diag_key.count() == 0 and last.diag_reference:
            keys = list(last.diag_reference)
            self.diag_key.addItems(keys)  # the first item binds the chart via currentTextChanged
            pref = next((k for k in ("energy", "variance") if k in keys), None)
            if pref:
                self.diag_key.setCurrentText(pref)
        follow = self.scrub.value() >= self.scrub.maximum()
        self.scrub.blockSignals(True)
        self.scrub.setRange(0, last.step)
        if follow:
            self.scrub.setValue(last.step)
        self.scrub.blockSignals(False)
        if follow:
            self._show_fields(last)
        self._bind_err()
        self.err_chart.update()
        self.diag_chart.update()
        self._update_metrics(last if follow else None)

    def _on_scrub(self, k):
        """Show recorded step k (states kept by the LiveRun); the end of the slider follows the live run."""
        run = self.live
        if run is None or not run.keep or k >= len(run.ref_states):
            return
        ref = run.ref_states[k]
        cand = run.cand_states[k] if k < len(run.cand_states) else None
        src = ""
        if run.hybrid_traj is not None and k >= 1:
            src = run.hybrid_traj.source[k - 1]
        elif cand is not None and k >= 1:
            src = "surrogate"
        err = self.err[k] if k < len(self.err) else None
        fr = backend.Frame(k, k * run.spacing * float(run.params["dt"]), ref, cand, src, err, {}, {}, [])
        self._show_fields(fr)
        self._update_metrics(fr)

    def _show_fields(self, fr):
        self._last_frame = fr
        fi = max(0, self.field_sel.currentIndex())
        name = self.current_system().fields[fi]
        self.cbar.symbol = {"vorticity": "ω", "temperature": "T"}.get(name, name[:2])
        r, c = fr.reference[fi], None if fr.candidate is None else fr.candidate[fi]
        rr = r if self.run_obstacle is None or self.run_obstacle.shape != r.shape else r[~self.run_obstacle]
        lo, hi = float(np.nanmin(rr)), float(np.nanmax(rr))  # the fluid only: a hot body would wash the plume out
        if self.scale is None:
            self.scale = ("seq" if 0 <= lo < hi or name == "temperature" else "div", lo, hi)  # heat: fire colours  # signed or all-zero -> diverging
        kind, slo, shi = self.scale
        # from the reference only, and it only widens: decay shows as fading, a blown-up prediction saturates
        if name == "temperature" and lo > -0.25 * max(shi, hi, 1e-9):
            lo = slo  # ignore small undershoots (spectral ripples next to a heated body): background stays black
        if not getattr(self, "scale_fixed", False):
            self.scale = kind, min(slo, lo), max(shi, hi)
        if kind == "div":
            m = max(abs(self.scale[1]), abs(self.scale[2]), 1e-30)
            vmin, vmax, lut = -m, m, DIVERGING
        else:
            vmin, vmax, lut = self.scale[1], self.scale[2], SEQUENTIAL
        self.cbar.set_range(vmin, vmax, lut)
        self.ref_view.subtitle = f"{name}  ·  step {fr.step}  ·  x →  y ↑"
        if self.run_obstacle is not None:  # soften the spectral ripples next to obstacles (display only)
            from scipy.ndimage import gaussian_filter
            r = gaussian_filter(np.asarray(r, np.float32), 0.8, mode="wrap")
        self.ref_view.set_field(r, vmin, vmax, lut, self.run_obstacle)
        self.counter.setText(f"step {fr.step} / {self.err_chart.xmax}     t = {fr.t:.4g}")
        if c is not None:
            src = {"": "initial state", "reference": "solver step (intervention)",
                   "surrogate": "network step"}.get(fr.source, fr.source)
            self.cand_view.subtitle = f"{name}  ·  step {fr.step}  ·  {src}"
            self.cand_view.set_field(c, vmin, vmax, lut, self.run_obstacle)
            e = np.abs(c - r)
            emax = float(np.nanmax(e)) if np.isfinite(e).any() else 0.0
            self.err_view.subtitle = f"step {fr.step}  ·  own scale, 0 to this frame's max"
            self.err_view.set_field(e, 0.0, emax, SEQUENTIAL, self.run_obstacle)

    def _update_metrics(self, fr):
        """Metric cards from measurements only; pending captions otherwise."""
        m = self.mode()
        pending = "needs Surrogate or Hybrid mode" if m == "reference" else (
            "train a checkpoint first" if self.ckpt.currentData() is None else "run to measure")
        if fr is not None and fr.rel_error is not None:
            self.m_err.set(sci(fr.rel_error), f"at step {fr.step}", FAIL if not fr.rel_error <= self.threshold else OK)
            r, c = fr.reference.astype(np.float64), fr.candidate.astype(np.float64)
            with np.errstate(all="ignore"):
                d = float(np.mean(c * c) / np.mean(r * r) - 1)
            ok = abs(d) <= self.drift_tol
            self.m_drift.set(("+" if d > 0 else "−" if d < 0 else "") + sci(abs(d)) if math.isfinite(d) else "∞",
                             f"{'✓ within' if ok else '! beyond'} ±{self.drift_tol:g}", OK if ok else EVENT)
        elif fr is None and not self.err:
            self.m_err.set("—", pending)
            self.m_drift.set("—", pending)
        t = self.timing
        ref_ms = t.get("reference").per_step if t.get("reference") else None
        cand_ms = t.get("candidate").per_step if t.get("candidate") else None
        if ref_ms and cand_ms:
            sp = ref_ms / cand_ms
            self.m_speed.set(f"{sp:.2f}×", f"{1e3 * ref_ms:.3g} vs {1e3 * cand_ms:.3g} ms/step", OK if sp > 1 else EVENT)
        elif ref_ms:
            self.m_speed.set("—", f"reference {1e3 * ref_ms:.3g} ms/step")
        else:
            self.m_speed.set("—", pending)
        if self.horizon is not None:
            dt = self.spacing * float(self.live.params["dt"]) if self.live else None
            self.m_horizon.set(f"step {self.horizon}", f"t = {self.horizon * dt:.4g} · error > {self.threshold:g}"
                               if dt else f"error > {self.threshold:g}", EVENT)
        elif self.err:
            self.m_horizon.set(f"> {len(self.err) - 1}", f"✓ error ≤ {self.threshold:g} so far", OK)
        else:
            self.m_horizon.set("—", pending)

    def _bind_err(self):
        s = {}
        for i, (name, xs, ys) in enumerate(self.compare):
            s[name] = (COMPARE[i % len(COMPARE)], xs, ys, True, 1.6)
        if self.err:
            s[self._run_name()] = (CAND, self.xs, self.err, False, 2.4)
        self.err_chart.series = s
        self.clear_cmp.setVisible(bool(self.compare))

    def _run_name(self):
        run = self.live
        if run is None:
            return self._cand_name().lower()
        ck = Path(str(run.cfg.get("checkpoint"))).name
        return f"{run.policy.name} · {ck}" if run.policy is not None else f"{run.mode} · {ck}"

    def _clear_compare(self):
        self.compare = []
        self._bind_err()
        self.err_chart.update()

    def _bind_diag(self, key):
        if not key:
            return
        s = {"reference": (REF, self.xs, self.diag_ref.setdefault(key, []), False, 2.0)}
        if self.mode() != "reference":
            s[self._cand_name().lower()] = (CAND, self.xs, self.diag_cand.setdefault(key, []), True, 2.0)
        self.diag_chart.series = s
        self.diag_chart.update()

    @staticmethod
    def _html_note(text):
        return f"<div style='color:{MUTED}'>{html.escape(text)}</div>"

    def _render_summary(self, s):
        if isinstance(s, str):
            return self._html_note(s)
        sec = lambda t: f"<div style='color:{FAINT}; font-size:10px; font-weight:700; margin-top:8px'>{t}</div>"
        out = []
        if s["facts"]:
            out.append(sec("MEASURED IN THIS RUN"))
            out += [f"<div style='margin:3px 0'><span style='color:{OK}'>✓</span>&nbsp; {html.escape(f)}</div>"
                    for f in s["facts"]]
        col = {"critical": FAIL, "warning": EVENT, "info": REF}
        if s["findings"]:
            out.append(sec("RULE-BASED FINDINGS · kolmo.diagnose"))
        for f in s["findings"]:
            out.append(f"<div style='margin-top:7px'><span style='color:{col.get(f.get('level'), MUTED)}'>●</span>&nbsp;"
                       f"<b>{html.escape(str(f.get('level', '')))}</b>&nbsp; {html.escape(str(f.get('fact', '')))}</div>"
                       f"<div style='color:{MUTED}'>Hypothesis: {html.escape(str(f.get('interpretation', '')))}</div>"
                       f"<div style='color:{FAINT}; font-size:10px'>{html.escape(str(f.get('source', '')))}</div>")
        out += [f"<div style='color:{FAINT}; margin-top:6px'>{html.escape(n)}</div>" for n in s["notes"]]
        return "".join(out) or self._html_note("Nothing to report: no failure detected, and the diagnosis rules apply "
                                               "to surrogate runs.")

    def _on_done(self, res):
        self.runner = None
        self._set_running(False)
        self._set_status("Stopped" if res["stopped"] else "Finished", MUTED if res["stopped"] else OK)
        self.timing, self.summary = res["timing"], res["summary"]
        self.scrub.setEnabled(bool(self.live and self.live.keep))
        self._update_metrics(None)
        self.findings.setHtml(self._render_summary(self.summary))
        steps = plan(self.summary)
        self.plan_label.setText("\n".join(f"{i}. {x}" for i, x in enumerate(steps, 1)) if steps else
                                "No suggestions: kolmo.diagnose had nothing to recommend for this run.")
        if self.err and not res["stopped"]:  # keep this curve for comparison with the next runs
            self.compare = (self.compare + [(self._run_name(), list(self.xs), list(self.err))])[-4:]

    def _on_failed(self, msg):
        self.runner = None
        self._set_running(False)
        self._set_status("Error", FAIL)
        self.counter.setText("")
        self.run_error.setText(msg)
        self.findings.setHtml(self._html_note("The run did not start; see the message under Run in Setup."))

    def closeEvent(self, e):
        if self.runner:
            self.runner.stopped = True
        for t in self._threads:
            t.wait(5000)
        super().closeEvent(e)


def main(argv=None):
    import sys
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("Kolmo")
    w = MainWindow()
    w.show()
    return app.exec()
