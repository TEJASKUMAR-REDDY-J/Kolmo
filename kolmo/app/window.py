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
from PySide6.QtCore import QPointF, QRectF, QSettings, Qt, QThread, Signal
from PySide6.QtGui import (QColor, QConicalGradient, QImage, QKeySequence, QPainter, QPen,
                           QPixmap, QPolygonF, QShortcut)
from PySide6.QtWidgets import (QAbstractSpinBox, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMenu, QPushButton,
                               QScrollArea, QSizePolicy, QSlider, QSpinBox, QStackedWidget, QTextBrowser, QVBoxLayout,
                               QWidget)

from kolmo.app import backend
from kolmo.app.assistant import RuleBased, plan
from kolmo.assistant.agent import Agent
from kolmo.app.widgets import (ACCENT, ACCENT2, CAND, COMPARE, DIVERGING, EVENT, FAIL, FAINT, MUTED, OK, REF,
                               SEQUENTIAL, SYMBOL_FONT, TEXT, Backdrop, ColorBar, FieldView, Glass, LineChart,
                               MetricCard, card, fmt, heading, mask_image, sci)

MODES = (("reference", "Reference"), ("surrogate", "Surrogate"), ("hybrid", "Hybrid"))
OBSTACLES = (("none", "None"), ("circle", "Circle"), ("square", "Square"), ("photo", "From photo…"))
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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kolmo · Physics-AI Research Platform")
        self.resize(1680, 1000)
        self.setStyleSheet(QSS + arrow_rule())
        self.settings = QSettings("Kolmo", "Kolmo")
        self.runner, self._threads, self.live = None, [], None
        self.photo = None  # {"path", "mask", "meta"} of a photo obstacle
        self.specs, self.param_widgets = [], {}
        self.summary, self.compare, self.timing = None, [], {}
        self.assistant = RuleBased(Agent(out_dir=str(backend.ROOT / "runs")))  # live only if .env KOLMO_LLM_LIVE=1 + key
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
        self.pages.addWidget(self._build_simulate())
        self.pages.addWidget(self._build_experiments())
        self.pages.addWidget(self._build_models())
        self.pages.addWidget(self._build_geometry())
        self.pages.addWidget(self._build_reports())
        body.addWidget(self.pages, 1)
        body.addWidget(self._build_side())
        root.addLayout(body, 1)

        for key, fn in (("Space", self._space), ("R", self.reset), ("E", lambda: self.err_toggle.toggle())):
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
        menu.addAction("Shortcuts: Space run/pause · R reset · E error panel").setEnabled(False)
        gear.setMenu(menu)
        gear.setStyleSheet("QPushButton::menu-indicator { image: none; }")
        top.addWidget(gear)
        return top

    def _set_anim(self, on):
        self.backdrop.enabled = on
        self.settings.setValue("animate", on)

    def _build_nav(self):
        c, v = card((12, 14, 12, 14), spacing=4)
        c.setFixedWidth(214)
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
        if key == "assistant":
            self.side_tabs.button(1).click()
            self.nav.button(0).setChecked(True)
            self.pages.setCurrentIndex(0)
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

    # ------------------------------------------------------------ simulate page
    def _build_simulate(self):
        page = QWidget()
        page.setObjectName("page")
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(16)

        fields, fv = card((20, 16, 20, 14), spacing=10)
        self.run_title = label("", squeeze=True)
        self.run_title.setStyleSheet("font-size: 17px; font-weight: 600;")
        self.pill = label("Idle", "pill")
        self.err_toggle = button("Error panel", "chip", "Show |prediction − reference| (E)")
        self.err_toggle.setCheckable(True)
        self.err_toggle.toggled.connect(lambda on: self.err_view.setVisible(on))
        fv.addLayout(row(self.run_title, self.err_toggle, self.pill, stretch_first=True))
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
        fv.addLayout(row(self.play, self.stop_btn, 6, self.scrub, 10, self.counter, spacing=10, stretch_first=False))
        v.addWidget(fields, 5)

        metrics = QHBoxLayout()
        metrics.setSpacing(14)
        self.m_err = MetricCard("◎", "Relative error", "‖prediction − reference‖ / ‖reference‖ at the shown step")
        self.m_drift = MetricCard("⚖", "Energy drift", "mean square of the prediction / of the reference − 1 "
                                                        "(kolmo.rollout.mean_square), at the shown step")
        self.m_speed = MetricCard("⚡", "Speedup", "steady-state reference ms/step ÷ prediction ms/step, measured now")
        self.m_horizon = MetricCard("∿", "Stability horizon", f"first step with relative error > {self.threshold:g}")
        for m in (self.m_err, self.m_drift, self.m_speed, self.m_horizon):
            metrics.addWidget(m)
        v.addLayout(metrics)

        charts = QHBoxLayout()
        charts.setSpacing(14)
        ec, ev = card((20, 14, 16, 12), spacing=4)
        self.clear_cmp = button("Clear comparison", "link", "Forget earlier runs in this chart", self._clear_compare)
        ev.addLayout(row(heading("Error over rollout"), None, self.clear_cmp))
        self.err_chart = LineChart(log=True, x_label="surrogate step")
        self.err_chart.hlines = [(self.threshold, f"failure threshold {self.threshold:g}", FAIL)]
        ev.addWidget(self.err_chart, 1)
        charts.addWidget(ec, 3)
        dc, dv = card((20, 14, 16, 12), spacing=4)
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
        v.addLayout(charts, 3)
        return page

    # ------------------------------------------------------------ right column: setup + assistant
    def _build_side(self):
        c, v = card((16, 14, 16, 16), spacing=10)
        c.setFixedWidth(356)
        self.side_tabs = QButtonGroup(self)
        seg = QHBoxLayout()
        seg.setSpacing(4)
        for i, t in enumerate(("Setup", "Assistant")):
            b = button(t, "seg")
            b.setCheckable(True)
            self.side_tabs.addButton(b, i)
            seg.addWidget(b)
        self.side_tabs.button(0).setChecked(True)
        box = Glass(radius=12, level=1)
        box.setLayout(seg)
        seg.setContentsMargins(4, 4, 4, 4)
        v.addWidget(box)
        self.side = QStackedWidget()
        self.side.addWidget(scroller(self._build_setup()))
        self.side.addWidget(self._build_assistant())
        self.side_tabs.idClicked.connect(self.side.setCurrentIndex)
        v.addWidget(self.side, 1)
        return c

    def _build_setup(self):
        w = QWidget()
        w.setObjectName("page")
        v = QVBoxLayout(w)
        v.setContentsMargins(2, 0, 8, 0)
        v.setSpacing(7)
        self.config_box = QWidget()
        c = QVBoxLayout(self.config_box)
        c.setContentsMargins(0, 0, 0, 0)
        c.setSpacing(7)
        v.addWidget(self.config_box)

        self.system_note = label("", "note", wrap=True)
        c.addWidget(self.system_note)
        c.addWidget(label("PARAMETERS", "section"))
        self.params_grid = QGridLayout()
        self.params_grid.setHorizontalSpacing(8)
        self.params_grid.setVerticalSpacing(6)
        self.params_grid.setColumnStretch(0, 1)
        c.addLayout(self.params_grid)
        self.seed = spin(0, 2 ** 31 - 1)
        self.seed.setToolTip("Seed of system.initial: same seed, same initial condition")
        c.addLayout(row(label("Random seed"), None, self.seed))

        c.addWidget(label("OBSTACLE", "section"))
        self.obstacle = QComboBox()
        for _, t in OBSTACLES:
            self.obstacle.addItem(t)
        self.obstacle.activated.connect(self._on_obstacle)
        self.obstacle_size = spin(0.02, 0.6, 2, width=70)
        self.obstacle_size.setValue(0.2)
        self.obstacle_size.setToolTip("Diameter / side (photo: streamwise length) as a fraction of the domain side")
        self.obstacle_size.valueChanged.connect(self._update_thumb)
        c.addLayout(row(self.obstacle, self.obstacle_size, stretch_first=True))
        self.thumb = QLabel()
        self.thumb.setFixedSize(60, 60)
        self.thumb.setStyleSheet("background: rgba(0,0,0,0.35); border: 1px solid rgba(255,255,255,0.15); border-radius: 8px;")
        self.obstacle_note = label("", "note", wrap=True)
        h = row(self.thumb, self.obstacle_note, spacing=10)
        h.setStretch(1, 1)
        c.addLayout(h)

        c.addWidget(label("EXECUTION", "section"))
        segbox = Glass(radius=11, level=1)
        seg = QHBoxLayout(segbox)
        seg.setContentsMargins(3, 3, 3, 3)
        seg.setSpacing(2)
        self.mode_group = QButtonGroup(self)
        tips = ("Reference solver only", "Surrogate network rollout vs the reference",
                "Surrogate with solver interventions (kolmo.hybrid)")
        for i, ((_, t), tip) in enumerate(zip(MODES, tips)):
            b = button(t, "seg", tip)
            b.setCheckable(True)
            self.mode_group.addButton(b, i)
            seg.addWidget(b)
        self.mode_group.button(0).setChecked(True)
        self.mode_group.idClicked.connect(lambda _: self._on_mode())
        c.addWidget(segbox)
        self.exec_box = QWidget()
        g = QVBoxLayout(self.exec_box)
        g.setContentsMargins(0, 4, 0, 0)
        g.setSpacing(6)
        self.ckpt = QComboBox()
        self.ckpt.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.ckpt.setMinimumContentsLength(12)
        self.ckpt.currentIndexChanged.connect(lambda _: self._on_mode())
        self.ckpt_browse = button("…", tip="Choose a checkpoint (<name>.json next to <name>.npz)", slot=self._browse_ckpt)
        self.ckpt_browse.setFixedWidth(36)
        g.addWidget(label("Surrogate checkpoint", "muted"))
        g.addLayout(row(self.ckpt, self.ckpt_browse, stretch_first=True))
        self.ckpt_note = label("", "note", wrap=True)
        g.addWidget(self.ckpt_note)
        self.use_train = button("Use the training parameters", "link", slot=self._use_training_params)
        g.addWidget(self.use_train)
        self.policy_box = QWidget()
        pg = QGridLayout(self.policy_box)
        pg.setContentsMargins(0, 6, 0, 0)
        pg.setColumnStretch(0, 1)
        pg.setVerticalSpacing(6)
        pg.addWidget(label("Hybrid policy", "muted"), 0, 0, 1, 2)
        self.pol_every = spin(0, 100000, special="off")
        self.pol_res = spin(0, 100, 4, special="off")
        self.pol_energy = spin(0, 100, 3, special="off")
        for i, (t, wdg, tip) in enumerate((
                ("Re-anchor every", self.pol_every, "solver steps at steps N, 2N, ... (surrogate steps)"),
                ("Residual tolerance", self.pol_res, "intervene when the physics residual of a proposal exceeds this"),
                ("Energy tolerance", self.pol_energy, "intervene when |E/E₀ − 1| of a proposal exceeds this")), 1):
            lab = label(t)
            lab.setToolTip(tip)
            wdg.setToolTip(tip)
            pg.addWidget(lab, i, 0)
            pg.addWidget(wdg, i, 1)
        pg.addWidget(label("Non-finite proposals always trigger the solver.", "note", wrap=True), 4, 0, 1, 2)
        g.addWidget(self.policy_box)
        c.addWidget(self.exec_box)
        self.mode_note = label("", "note", wrap=True)
        c.addWidget(self.mode_note)

        v.addWidget(label("RUN", "section"))
        self.steps = spin(1, 1_000_000)
        self.steps.setValue(400)
        self.steps.setToolTip("Surrogate steps (1 step = the checkpoint's spacing in solver steps; 1 in reference mode)")
        v.addLayout(row(label("Steps"), None, self.steps))
        self.run_btn = button("Run", "primary", "Run (Space)", self.run)
        v.addWidget(self.run_btn)
        self.run_error = label("", "error", wrap=True)
        self.run_error.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(self.run_error)
        v.addLayout(row(button("Save config…", slot=self._save_config), button("Open config…", slot=self._open_config)))
        v.addStretch(1)
        return w

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
        pl, plv = card((14, 10, 14, 10), spacing=4, radius=14, level=1)
        plv.addWidget(heading("Plan", 13))
        self.plan_label = label("", "note", wrap=True)
        plv.addWidget(self.plan_label)
        v.addWidget(pl)
        self.exp_btn = button("Run experiment", "primary", "Runs the current Setup (Space)", self.run)
        v.addWidget(self.exp_btn)
        self.chat_log = QTextBrowser()
        self.chat_log.setMaximumHeight(130)
        v.addWidget(self.chat_log, 1)
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Ask: run, explain, plan, train…")
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
            task = Task(lambda: self.assistant.agent.ask(question, confirmed, log=lambda *_: None))
            task.ok.connect(lambda r, q=question: self._agent_done(q, r))
            task.err.connect(lambda m: self._agent_done(question, {"status": "error", "text": m}))
            self._keep(task)
            task.start()
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

    # ------------------------------------------------------------ other pages (read from run files, never invented)
    def _page(self, title, subtitle):
        c, v = card((24, 18, 24, 18), spacing=10)
        v.addWidget(heading(title, 20))
        v.addWidget(label(subtitle, "muted", wrap=True))
        return c, v

    def _build_experiments(self):
        c, v = self._page("Experiments", "Experiment runs written by  python -m kolmo.experiments run <config.json>  "
                                         "(runs/<name>-<hash>/). Numbers are read from their results.json.")
        self.exp_view = QTextBrowser()
        v.addWidget(self.exp_view, 1)
        return c

    def _fill_experiments(self):
        exps = backend.experiments()
        if not exps:
            self.exp_view.setHtml(f"<p style='color:{MUTED}'>No experiment runs yet. Run one with "
                                  "<code>python -m kolmo.experiments run &lt;config.json&gt;</code>.</p>")
            return
        out = []
        for e in exps:
            rows = e["rows"]
            keys = [k for k in (rows[0] if rows else {}) if k not in ("model",)]
            head = "".join(f"<th style='padding:3px 10px; color:{MUTED}; text-align:left'>{html.escape(k)}</th>" for k in keys)
            body = "".join("<tr><td style='padding:3px 10px'><b>" + html.escape(str(r["model"])) + "</b></td>" +
                           "".join(f"<td style='padding:3px 10px'>{html.escape(fmt(r[k]) if isinstance(r[k], float) else str(r[k]))}</td>"
                                   for k in keys) + "</tr>" for r in rows)
            rt = "; ".join(f"{m}: speed-up {fmt(x['speedup'])}× ({fmt(1e3 * x['reference_s_per_step'])} vs "
                           f"{fmt(1e3 * x['surrogate_s_per_step'])} ms/step)" for m, x in e["runtime"].items() if x.get("speedup"))
            out.append(f"<h3 style='margin-bottom:2px'>{html.escape(e['name'])}</h3>"
                       f"<div style='color:{FAINT}'>{html.escape(e['dir'])} · {e['n_findings']} rule-based findings</div>"
                       f"<table style='margin-top:6px'><tr><th style='padding:3px 10px; color:{MUTED}; text-align:left'>model</th>"
                       f"{head}</tr>{body}</table><div style='color:{MUTED}; margin-top:4px'>{html.escape(rt)}</div>")
        self.exp_view.setHtml("".join(out))

    def _build_models(self):
        c, v = self._page("Models", "Surrogate checkpoints found under runs/ and checkpoints/ (kolmo.surrogates.train "
                                    "format), with what they were trained on.")
        self.models_view = QTextBrowser()
        v.addWidget(self.models_view, 1)
        return c

    def _fill_models(self):
        cks = backend.all_checkpoints()
        if not cks:
            self.models_view.setHtml(f"<p style='color:{MUTED}'>{html.escape(backend.TRAIN_HELP)}</p>")
            return
        self.models_view.setHtml("".join(
            f"<h3 style='margin-bottom:2px'>{html.escape(lbl)} <span style='color:{MUTED}; font-weight:400'>· "
            f"{html.escape(sys_)}</span></h3><div style='color:{FAINT}'>{html.escape(stem)}</div>"
            f"<div style='color:{MUTED}'>{html.escape(backend.describe_checkpoint(stem)).replace(chr(10), '<br>')}</div>"
            for sys_, lbl, stem in cks))

    def _build_geometry(self):
        c, v = self._page("Geometry", "Photo → object mask (SAM 3 when installed, else OpenCV GrabCut) → obstacle grid "
                                      "for Fluid 2D. The mask becomes the Simulate obstacle.")
        self.geo_preview = QLabel()
        self.geo_preview.setFixedSize(320, 320)
        self.geo_preview.setStyleSheet("background: rgba(0,0,0,0.35); border: 1px solid rgba(255,255,255,0.15); "
                                       "border-radius: 14px;")
        self.geo_preview.setAlignment(Qt.AlignCenter)
        self.geo_note = label("No photo yet.", "muted", wrap=True)
        v.addLayout(row(button("Choose photo…", "primary", slot=self._geo_pick), None))
        h = row(self.geo_preview, self.geo_note, spacing=18)
        h.setStretch(1, 1)
        v.addLayout(h)
        v.addStretch(1)
        return c

    def _geo_pick(self):
        s = next((s for _, s, _ in self._systems if s is not None and backend.supports_obstacle(s)), None)
        if s is None:
            self.geo_note.setText("No system with obstacles is available.")
            return
        if not backend.supports_obstacle(self.current_system()):
            self.system.setCurrentIndex(next(i for i, (_, x, _) in enumerate(self._systems) if x is s))
        self.obstacle.setCurrentIndex(3)
        self._on_obstacle()

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
        while self.params_grid.count():
            w = self.params_grid.takeAt(0).widget()
            if w:
                w.setParent(None)  # off screen now; deleteLater alone leaves it drawn until the next event loop pass
                w.deleteLater()
        self.specs, self.param_widgets = backend.param_specs(s), {}
        d = backend.defaults(s)
        for i, sp in enumerate(self.specs):
            unit = f"  <span style='color:{FAINT}'>{sp.unit}</span>" if sp.unit else ""
            lab = label(sp.label + unit)
            lab.setToolTip(sp.tip)
            if sp.choices:
                w = QComboBox()
                w.setFixedWidth(112)
                if sp.auto:
                    w.addItem(AUTO, AUTO)
                for ch in sp.choices:
                    w.addItem(str(ch).replace("_", " "), ch)
            else:
                w = spin(sp.lo, sp.hi, sp.decimals, AUTO if sp.auto else None)
            w.setToolTip(sp.tip)
            self._set_value(w, d.get(sp.key))
            self.params_grid.addWidget(lab, i, 0)
            self.params_grid.addWidget(w, i, 1)
            self.param_widgets[sp.key] = w
        self.system_note.setText(backend.describe(s))
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

    def _params(self):
        out = {}
        for sp in self.specs:
            w = self.param_widgets[sp.key]
            v = w.currentData() if isinstance(w, QComboBox) else w.value()
            out[sp.key] = None if sp.auto and (v == AUTO or (not isinstance(w, QComboBox) and v == w.minimum())) else v
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
        policy = {"every": self.pol_every.value(), "residual_tol": self.pol_res.value() or None,
                  "energy_tol": self.pol_energy.value() or None}
        return {"system": self.current_system().name, "params": self._params(), "seed": self.seed.value(),
                "obstacle": obst, "mode": self.mode(), "checkpoint": self.ckpt.currentData(),
                "policy": policy, "steps": self.steps.value()}

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
            self.obstacle.setCurrentIndex([k for k, _ in OBSTACLES].index(obst.get("kind", "none")))
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
        self.scale = self.run_obstacle = self.live = None
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
        name = self.current_system().fields[0]
        r, c = fr.reference[0], None if fr.candidate is None else fr.candidate[0]
        lo, hi = float(np.nanmin(r)), float(np.nanmax(r))
        if self.scale is None:
            self.scale = ("seq" if 0 <= lo < hi else "div", lo, hi)  # signed or all-zero -> diverging
        kind, slo, shi = self.scale
        # from the reference only, and it only widens: decay shows as fading, a blown-up prediction saturates
        self.scale = kind, min(slo, lo), max(shi, hi)
        if kind == "div":
            m = max(abs(self.scale[1]), abs(self.scale[2]), 1e-30)
            vmin, vmax, lut = -m, m, DIVERGING
        else:
            vmin, vmax, lut = self.scale[1], self.scale[2], SEQUENTIAL
        self.cbar.set_range(vmin, vmax, lut)
        self.ref_view.subtitle = f"{name}  ·  step {fr.step}  ·  x →  y ↑"
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
        self.side_tabs.button(0).click()
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
