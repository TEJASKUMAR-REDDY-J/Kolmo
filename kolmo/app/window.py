"""Kolmo main window. One screen: left = what to run; centre = reference | prediction | error; bottom = error and
diagnostics vs step, measured runtimes, and the post-run diagnosis. All computation runs in worker threads."""
import dataclasses
import html
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap, QPolygonF
from PySide6.QtWidgets import (QAbstractSpinBox, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFrame, QGridLayout, QHBoxLayout, QLabel, QMainWindow, QPushButton, QScrollArea,
                               QSizePolicy, QSpinBox, QSplitter, QTextBrowser, QVBoxLayout, QWidget)

from kolmo.app import backend
from kolmo.app.widgets import (ACCENT, BG, BORDER, CAND, DIVERGING, EVENT, FAIL, FAINT, MUTED, REF, SEQUENTIAL,
                               SURFACE, SURFACE2, TEXT, FieldView, LineChart, fmt, mask_image)

MODES = (("reference", "Reference"), ("surrogate", "Surrogate"), ("hybrid", "Hybrid"))
OBSTACLES = (("none", "None"), ("circle", "Circle"), ("square", "Square"), ("photo", "From photo…"))
OK_GREEN = "#5fd38d"
AUTO = "auto"

QSS = f"""
* {{ font-family: "Segoe UI"; font-size: 12px; color: {TEXT}; }}
QMainWindow, #centre {{ background: {BG}; }}
#sidebar {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}
QLabel {{ background: transparent; }}
QLabel#brand {{ font-size: 19px; font-weight: 700; }}
QLabel#section {{ color: {FAINT}; font-size: 10px; font-weight: 700; letter-spacing: 1.2px; padding-top: 12px; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#note {{ color: {MUTED}; font-size: 11px; }}
QLabel#error {{ color: {FAIL}; }}
QLabel#runTitle {{ font-size: 17px; font-weight: 600; }}
QLabel#counter {{ font-family: Consolas; font-size: 15px; }}
QLabel#num {{ font-family: Consolas; font-size: 12px; }}
QLabel#pill {{ border: 1px solid {BORDER}; border-radius: 10px; padding: 2px 10px; color: {MUTED}; }}
QLabel#cardTitle {{ font-size: 13px; font-weight: 600; }}
QComboBox, QSpinBox, QDoubleSpinBox {{ background: {SURFACE2}; border: 1px solid {BORDER}; border-radius: 5px;
    padding: 4px 8px; min-height: 18px; selection-background-color: {ACCENT}; }}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: #3a4252; }}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {ACCENT}; }}
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QPushButton:disabled, QLabel:disabled {{ color: {FAINT}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{ background: {SURFACE2}; border: 1px solid {BORDER}; selection-background-color: #2a3550;
    outline: none; }}
QPushButton {{ background: {SURFACE2}; border: 1px solid {BORDER}; border-radius: 5px; padding: 6px 12px; }}
QPushButton:hover {{ border-color: #3a4252; background: #222733; }}
QPushButton#primary {{ background: {ACCENT}; border-color: {ACCENT}; color: #0a1222; font-weight: 600; }}
QPushButton#primary:hover {{ background: #74adff; }}
QPushButton#primary:disabled {{ background: #26314a; border-color: #26314a; color: {FAINT}; }}
QPushButton#seg {{ border-radius: 0; padding: 5px 0; }}
QPushButton#seg:checked {{ background: #22314d; border-color: {ACCENT}; font-weight: 600; }}
QPushButton#link {{ background: transparent; border: none; color: {ACCENT}; padding: 0; text-align: left; }}
QPushButton#link:hover {{ text-decoration: underline; }}
QFrame#card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px; }}
QTextBrowser {{ background: transparent; border: none; }}
QCheckBox {{ color: {MUTED}; spacing: 5px; }}
QSplitter::handle {{ background: {BG}; }}
QToolTip {{ background: {SURFACE2}; color: {TEXT}; border: 1px solid {BORDER}; }}
QScrollBar:vertical {{ background: transparent; width: 8px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
"""


def arrow_rule():
    """Combo-box arrow as a tiny PNG in the temp folder (QSS cannot draw a triangle itself)."""
    path = Path(tempfile.gettempdir()) / "kolmo_combo_arrow.png"
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
    """squeeze: the label never widens its layout (long text is clipped instead of growing the window)."""
    w = QLabel(text)
    if name:
        w.setObjectName(name)
    w.setWordWrap(wrap)
    if squeeze:
        w.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        w.setMinimumWidth(0)
    return w


def spin(lo, hi, decimals=0, special=None, width=110):
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


class Card(QFrame):
    def __init__(self, title, *right):
        super().__init__()
        self.setObjectName("card")
        v = QVBoxLayout(self)
        v.setContentsMargins(14, 10, 14, 12)
        v.setSpacing(6)
        head = QHBoxLayout()
        self.title = label(title, "cardTitle", squeeze=True)
        head.addWidget(self.title, 1)
        for w in right:
            head.addWidget(w)
        v.addLayout(head)
        self.body = v


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kolmo")
        self.resize(1600, 980)
        self.setStyleSheet(QSS + arrow_rule())
        self.runner, self._threads = None, []
        self.photo = None  # {"path", "mask", "meta"} of a photo obstacle
        self.specs, self.param_widgets = [], {}
        self._systems = backend.systems()
        self.threshold = backend.failure_threshold()

        root = QWidget()
        h = QHBoxLayout(root)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        h.addWidget(self._build_sidebar())
        h.addWidget(self._build_centre(), 1)
        self.setCentralWidget(root)
        self._on_system()
        self.reset()

    # ------------------------------------------------------------ sidebar
    def _build_sidebar(self):
        side = QWidget()
        side.setObjectName("sidebar")
        side.setFixedWidth(296)
        outer = QVBoxLayout(side)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)
        inner = QWidget()
        scroll.setWidget(inner)
        v = QVBoxLayout(inner)
        v.setContentsMargins(18, 16, 18, 16)
        v.setSpacing(6)
        brand = QHBoxLayout()
        brand.addWidget(label("Kolmo", "brand"))
        brand.addStretch(1)
        brand.addWidget(label("solver · surrogate · hybrid", "note"))
        v.addLayout(brand)

        self.config_box = QWidget()
        c = QVBoxLayout(self.config_box)
        c.setContentsMargins(0, 0, 0, 0)
        c.setSpacing(6)
        v.addWidget(self.config_box)

        c.addWidget(label("SYSTEM", "section"))
        self.system = QComboBox()
        for i, (lbl, s, why) in enumerate(self._systems):
            self.system.addItem(lbl)
            if s is None:  # listed but disabled, with the reason, rather than hidden
                self.system.model().item(i).setEnabled(False)
                self.system.setItemData(i, f"Not available: {why}", Qt.ToolTipRole)
        self.system.setCurrentIndex(next(i for i, (_, s, _) in enumerate(self._systems) if s is not None))
        self.system.currentIndexChanged.connect(self._on_system)
        c.addWidget(self.system)
        self.system_note = label("", "note", wrap=True)
        c.addWidget(self.system_note)

        c.addWidget(label("PARAMETERS", "section"))
        self.params_grid = QGridLayout()
        self.params_grid.setHorizontalSpacing(8)
        self.params_grid.setVerticalSpacing(6)
        self.params_grid.setColumnStretch(0, 1)
        c.addLayout(self.params_grid)

        c.addWidget(label("INITIAL CONDITION", "section"))
        g = QGridLayout()
        g.setColumnStretch(0, 1)
        self.seed = spin(0, 2 ** 31 - 1)
        g.addWidget(label("Random seed"), 0, 0)
        g.addWidget(self.seed, 0, 1)
        c.addLayout(g)

        c.addWidget(label("OBSTACLE", "section"))
        self.obstacle = QComboBox()
        for _, t in OBSTACLES:
            self.obstacle.addItem(t)
        self.obstacle.activated.connect(self._on_obstacle)
        c.addWidget(self.obstacle)
        g = QGridLayout()
        g.setColumnStretch(0, 1)
        self.obstacle_size = spin(0.02, 0.6, 2)
        self.obstacle_size.setValue(0.2)
        self.obstacle_size.valueChanged.connect(self._update_thumb)
        self.obstacle_size_label = label(f"Size  <span style='color:{MUTED}'>× domain side</span>")
        g.addWidget(self.obstacle_size_label, 0, 0)
        g.addWidget(self.obstacle_size, 0, 1)
        c.addLayout(g)
        th = QHBoxLayout()
        self.thumb = QLabel()
        self.thumb.setFixedSize(64, 64)
        self.thumb.setStyleSheet(f"background:{BG}; border:1px solid {BORDER}; border-radius:4px;")
        th.addWidget(self.thumb)
        self.obstacle_note = label("", "note", wrap=True)
        th.addWidget(self.obstacle_note, 1)
        c.addLayout(th)

        c.addWidget(label("EXECUTION", "section"))
        seg = QHBoxLayout()
        seg.setSpacing(0)
        self.mode_group = QButtonGroup(self)
        for i, (_, t) in enumerate(MODES):
            b = QPushButton(t)
            b.setObjectName("seg")
            b.setCheckable(True)
            self.mode_group.addButton(b, i)
            seg.addWidget(b)
        self.mode_group.button(0).setChecked(True)
        self.mode_group.idClicked.connect(lambda _: self._on_mode())
        c.addLayout(seg)
        self.exec_box = QWidget()
        g = QGridLayout(self.exec_box)
        g.setContentsMargins(0, 4, 0, 0)
        g.setColumnStretch(0, 1)
        g.setVerticalSpacing(6)
        self.ckpt = QComboBox()
        self.ckpt.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.ckpt.setMinimumContentsLength(10)
        self.ckpt.currentIndexChanged.connect(lambda _: self._on_mode())
        self.ckpt_browse = QPushButton("…")
        self.ckpt_browse.setFixedWidth(30)
        self.ckpt_browse.setStyleSheet("padding: 0;")
        self.ckpt_browse.setToolTip("Choose a checkpoint file (<name>.json next to <name>.npz)")
        self.ckpt_browse.clicked.connect(self._browse_ckpt)
        ck = QHBoxLayout()
        ck.setSpacing(4)
        ck.addWidget(self.ckpt, 1)
        ck.addWidget(self.ckpt_browse)
        g.addWidget(label("Surrogate checkpoint", "muted"), 0, 0, 1, 2)
        g.addLayout(ck, 1, 0, 1, 2)
        self.ckpt_note = label("", "note", wrap=True)
        g.addWidget(self.ckpt_note, 2, 0, 1, 2)
        self.use_train = QPushButton("Use the training parameters")
        self.use_train.setObjectName("link")
        self.use_train.clicked.connect(self._use_training_params)
        g.addWidget(self.use_train, 3, 0, 1, 2)
        self.policy_box = QWidget()
        pg = QGridLayout(self.policy_box)
        pg.setContentsMargins(0, 6, 0, 0)
        pg.setColumnStretch(0, 1)
        pg.addWidget(label("Hybrid policy", "muted"), 0, 0, 1, 2)
        self.pol_every = spin(0, 100000, special="off")
        self.pol_res = spin(0, 100, 4, special="off")
        self.pol_energy = spin(0, 100, 3, special="off")
        for i, (t, w, tip) in enumerate((
                ("Re-anchor every", self.pol_every, "solver steps at steps N, 2N, ... (surrogate steps)"),
                ("Residual tolerance", self.pol_res, "intervene when the physics residual of a proposal exceeds this"),
                ("Energy tolerance", self.pol_energy, "intervene when |E/E₀ − 1| of a proposal exceeds this")), 1):
            lab = label(t)
            lab.setToolTip(tip)
            w.setToolTip(tip)
            pg.addWidget(lab, i, 0)
            pg.addWidget(w, i, 1)
        pg.addWidget(label("Non-finite proposals always trigger the solver.", "note", wrap=True), 4, 0, 1, 2)
        g.addWidget(self.policy_box, 4, 0, 1, 2)
        c.addWidget(self.exec_box)
        self.mode_note = label("", "note", wrap=True)
        c.addWidget(self.mode_note)

        v.addWidget(label("RUN", "section"))
        g = QGridLayout()
        g.setColumnStretch(0, 1)
        self.steps = spin(1, 1_000_000)
        self.steps.setValue(400)
        g.addWidget(label("Steps"), 0, 0)
        g.addWidget(self.steps, 0, 1)
        v.addLayout(g)
        row = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self.run)
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset)
        row.addWidget(self.run_btn, 2)
        row.addWidget(self.pause_btn, 1)
        row.addWidget(self.reset_btn, 1)
        v.addLayout(row)
        self.run_error = label("", "error", wrap=True)
        self.run_error.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(self.run_error)

        v.addWidget(label("CONFIG", "section"))
        row = QHBoxLayout()
        save, open_ = QPushButton("Save config…"), QPushButton("Open config…")
        save.clicked.connect(self._save_config)
        open_.clicked.connect(self._open_config)
        row.addWidget(save)
        row.addWidget(open_)
        v.addLayout(row)
        v.addStretch(1)
        return side

    # ------------------------------------------------------------ centre
    def _build_centre(self):
        centre = QWidget()
        centre.setObjectName("centre")
        v = QVBoxLayout(centre)
        v.setContentsMargins(16, 12, 16, 16)
        v.setSpacing(10)

        head = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(1)
        self.run_title = label("", "runTitle", squeeze=True)
        self.run_sub = label("", "muted", squeeze=True)
        self.run_sub.setTextInteractionFlags(Qt.TextSelectableByMouse)
        titles.addWidget(self.run_title)
        titles.addWidget(self.run_sub)
        head.addLayout(titles, 1)
        self.counter = label("", "counter")
        head.addWidget(self.counter)
        head.addSpacing(14)
        self.pill = label("Idle", "pill")
        head.addWidget(self.pill)
        v.addLayout(head)

        split = QSplitter(Qt.Vertical)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(10)
        v.addWidget(split, 1)

        panels = QWidget()
        ph = QHBoxLayout(panels)
        ph.setContentsMargins(0, 0, 0, 0)
        ph.setSpacing(10)
        self.ref_view = FieldView("Reference solver")
        self.cand_view = FieldView("Surrogate prediction")
        self.err_view = FieldView("Absolute error")
        for w in (self.ref_view, self.cand_view, self.err_view):
            ph.addWidget(w)
        split.addWidget(panels)

        bottom = QWidget()
        bh = QHBoxLayout(bottom)
        bh.setContentsMargins(0, 0, 0, 0)
        bh.setSpacing(10)
        self.err_chart = LineChart(log=True)
        self.err_chart.hlines = [(self.threshold, f"failure threshold {self.threshold:g}", FAIL)]
        err_card = Card("Relative L2 error vs step  ·  ‖prediction − reference‖ / ‖reference‖")
        err_card.body.addWidget(self.err_chart, 1)
        bh.addWidget(err_card, 5)

        self.diag_key = QComboBox()
        self.diag_key.setMinimumWidth(130)
        self.diag_key.currentTextChanged.connect(self._bind_diag)
        self.diag_log = QCheckBox("log")
        self.diag_log.toggled.connect(lambda on: (setattr(self.diag_chart, "log", on), self.diag_chart.update()))
        self.diag_chart = LineChart()
        diag_card = Card("Diagnostic vs step", self.diag_key, self.diag_log)
        diag_card.body.addWidget(self.diag_chart, 1)
        bh.addWidget(diag_card, 5)

        side = Card("Runtime")
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(3)
        for j, t in enumerate(("", "ms / step", "first call s")):
            grid.addWidget(label(t, "note"), 0, j, alignment=Qt.AlignRight if j else Qt.AlignLeft)
        self.rt = {}
        for i, key in enumerate(("reference", "candidate"), 1):
            name = label("Reference solver" if key == "reference" else "Surrogate")
            a, b = label("—", "num"), label("—", "num")
            a.setAlignment(Qt.AlignRight)
            b.setAlignment(Qt.AlignRight)
            grid.addWidget(name, i, 0)
            grid.addWidget(a, i, 1)
            grid.addWidget(b, i, 2)
            self.rt[key] = (name, a, b)
        self.speedup = label("", "muted")
        grid.addWidget(self.speedup, 3, 0, 1, 3)
        side.body.addLayout(grid)
        self.rt_note = label("", "note", wrap=True)
        self.rt_note.setStyleSheet(f"color: {FAINT};")
        side.body.addWidget(self.rt_note)
        side.body.addSpacing(4)
        side.body.addWidget(label("Diagnosis", "cardTitle"))
        self.findings = QTextBrowser()
        self.findings.setOpenLinks(False)
        side.body.addWidget(self.findings, 1)
        bh.addWidget(side, 4)
        split.addWidget(bottom)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([580, 330])
        return centre

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
                w.setFixedWidth(110)
                if sp.auto:
                    w.addItem(AUTO, AUTO)
                for ch in sp.choices:
                    w.addItem(str(ch).replace("_", " "), ch)
            else:
                w = spin(sp.lo, sp.hi, sp.decimals, AUTO if sp.auto else None)
            w.setToolTip(sp.tip)
            self._set_value(w, d.get(sp.key), sp)
            self.params_grid.addWidget(lab, i, 0)
            self.params_grid.addWidget(w, i, 1)
            self.param_widgets[sp.key] = w
        self.system_note.setText(backend.describe(s))
        self.photo = None
        self._refresh_ckpts()
        self._update_obstacle_ui()
        self._on_mode()

    @staticmethod
    def _set_value(w, v, sp):
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
        for w in (self.obstacle, self.obstacle_size, self.obstacle_size_label):
            w.setEnabled(ok)
        if not ok:
            self.obstacle.setCurrentIndex(0)
        self._update_thumb()

    def _obstacle_kind(self):
        return OBSTACLES[self.obstacle.currentIndex()][0]

    def _on_obstacle(self):
        if self._obstacle_kind() == "photo":
            path, _ = QFileDialog.getOpenFileName(self, "Obstacle from photo", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
            if path:
                self._segment_photo(path)
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

    def _photo_done(self, path, mask, meta):
        self.photo = {"path": path, "mask": mask, "meta": meta}
        self.run_btn.setEnabled(self.runner is None)
        self._update_thumb()

    def _photo_failed(self, msg):
        self.photo = None
        self.run_btn.setEnabled(self.runner is None)
        self.obstacle.setCurrentIndex(0)
        self._update_thumb()
        self.obstacle_note.setText(f"Photo failed: {msg}")

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
        self.thumb.setPixmap(QPixmap.fromImage(mask_image(mask, TEXT)).scaled(62, 62, Qt.KeepAspectRatio))
        meta = (self.photo or {}).get("meta", {}) if self._obstacle_kind() == "photo" else {}
        src = f"{meta['segmenter']}, " if meta.get("segmenter") else ""
        self.obstacle_note.setText(f"{src}{int(mask.sum())} cells ({100 * mask.mean():.1f}% of the domain) "
                                   f"at x = 0.25 L. Grey in the panels; flow along +x needs U∞ > 0.")

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
                self._set_value(self.param_widgets[sp.key], tp[sp.key], sp)
        self._on_mode()

    def _ckpt_status(self):
        """(note text, mismatch list) for the selected checkpoint."""
        stem = self.ckpt.currentData()
        if not stem:
            return backend.TRAIN_HELP, []
        try:
            text = backend.describe_checkpoint(stem)
            tp = backend.training_params(stem)
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
        self.ckpt_note.setText(note + (f"\n⚠ outside training: {', '.join(miss)}" if miss else ""))
        self.use_train.setVisible(m != "reference" and bool(self.ckpt.currentData()) and bool(miss))
        self.mode_note.setText({
            "reference": "Reference solver only: nothing to compare, so the prediction and error panels stay empty.",
            "surrogate": "The surrogate rolls out alone from the same initial state; the reference runs alongside "
                         "to measure the error.",
            "hybrid": "kolmo.hybrid runs the surrogate with solver interventions from the current hybrid state; the "
                      "result is then replayed next to the live reference.",
        }[m])
        cand = self._cand_name()
        self.cand_view.title = f"{cand} prediction"
        self.err_view.title = f"Absolute error  |{cand.lower()} − reference|"
        self.rt["candidate"][0].setText(cand)
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
                self._set_value(self.param_widgets[sp.key], cfg["params"][sp.key], sp)
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
        self._set_status("Starting", ACCENT)
        self.counter.setText(f"step 0 / {cfg['steps']}")
        self.err_chart.xmax = self.diag_chart.xmax = cfg["steps"]
        self.findings.setHtml(self._html_note("Compiling and starting…"))
        r.start()

    def _set_running(self, on):
        self.config_box.setEnabled(not on)
        self.run_btn.setEnabled(not on)
        self.pause_btn.setEnabled(on)
        self.pause_btn.setText("Pause")

    def toggle_pause(self):
        if not self.runner:
            return
        self.runner.paused = not self.runner.paused
        self.pause_btn.setText("Resume" if self.runner.paused else "Pause")
        self._set_status("Paused" if self.runner.paused else "Running", MUTED if self.runner.paused else ACCENT)

    def reset(self):
        if self.runner:
            self.runner.stopped = True
            self.runner = None
        self.xs, self.err, self.diag_ref, self.diag_cand = [], [], {}, {}
        self.scale = self.run_obstacle = None
        self.spacing = 1
        self._set_running(False)
        self.run_error.setText("")
        self._set_status("Idle", MUTED)
        self.counter.setText("")
        self.diag_key.blockSignals(True)
        self.diag_key.clear()
        self.diag_key.blockSignals(False)
        self._set_runtime({})
        self.rt_note.setText("")
        self.findings.setHtml(self._html_note("Measured facts and rule-based findings appear when a run finishes."))
        self._update_header()
        self._empty_states()

    def _empty_states(self):
        m = self.mode()
        self.ref_view.subtitle = self.cand_view.subtitle = self.err_view.subtitle = ""
        self.ref_view.clear("Press Run to start the reference solver.")
        self.err_chart.vlines = []
        if m == "reference":
            self.cand_view.clear("Reference mode: no prediction runs.\n\nChoose Surrogate or Hybrid to compare.")
            self.err_view.clear("The error needs a prediction to compare against the reference.")
            self.err_chart.clear("Reference mode: no prediction, so no error to plot.")
        else:
            missing = backend.TRAIN_HELP if self.ckpt.currentData() is None else None
            self.cand_view.clear(missing or "Press Run to roll out the prediction.")
            self.err_view.clear("The error appears once the prediction runs.")
            self.err_chart.clear(missing or "Run to plot the relative error per step.")
        self.diag_chart.clear("Run to plot the system's diagnostics per step.")

    def _update_header(self, run=None):
        self.run_title.setText(f"{self.system_label()}  ·  {dict(MODES)[self.mode()]}")
        if run is None:
            self.run_sub.setText("Not running")
            return
        p, s = run.params, run.system
        keys = [sp.key for sp in backend.param_specs(s)] + [k for k in backend.header_keys(s)]
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
        self.pill.setStyleSheet(f"color: {colour}; border-color: {colour};")

    def _on_started(self, run):
        self._update_header(run)
        self.run_obstacle, self.spacing = run.obstacle, run.spacing
        self._set_status("Hybrid rollout…" if run.mode == "hybrid" else "Running", ACCENT)
        self.findings.setHtml(self._html_note("Running kolmo.hybrid.run with this policy, then replaying it…"
                                              if run.mode == "hybrid" else "Running…"))

    def _on_frames(self, payload):
        frames, timing = payload
        if self.pill.text() not in ("Paused", "Running"):
            self._set_status("Running", ACCENT)
        for fr in frames:
            self.xs.append(fr.step)
            if fr.rel_error is not None:
                self.err.append(fr.rel_error)
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
        self._show_fields(last)
        self.counter.setText(f"step {last.step} / {self.err_chart.xmax}    t = {last.t:.4g}")
        if self.err:
            self.err_chart.series = {f"{self._cand_name().lower()} vs reference": (CAND, self.xs, self.err, False)}
        self.err_chart.update()
        self.diag_chart.update()
        self._set_runtime(timing)

    def _show_fields(self, fr):
        name = self.current_system().fields[0]
        r, c = fr.reference[0], None if fr.candidate is None else fr.candidate[0]
        lo, hi = float(np.nanmin(r)), float(np.nanmax(r))
        if self.scale is None:
            self.scale = ("div" if lo < 0 < hi else "seq", lo, hi)
        kind, slo, shi = self.scale
        # from the reference only, and it only widens: decay shows as fading, a blown-up prediction saturates
        self.scale = kind, min(slo, lo), max(shi, hi)
        if kind == "div":
            m = max(abs(self.scale[1]), abs(self.scale[2]), 1e-30)
            vmin, vmax, lut = -m, m, DIVERGING
        else:
            vmin, vmax, lut = self.scale[1], self.scale[2], SEQUENTIAL
        where = f"step {fr.step}  ·  x →  y ↑"
        self.ref_view.subtitle = f"{name}  ·  {where}" + ("  ·  scale shared →" if c is not None else "")
        self.ref_view.set_field(r, vmin, vmax, lut, self.run_obstacle)
        if c is not None:
            src = {"": "initial state", "reference": "this step by the solver (intervention)",
                   "surrogate": "this step by the network"}.get(fr.source, fr.source)
            self.cand_view.subtitle = f"{name}  ·  step {fr.step}  ·  {src}"
            self.cand_view.set_field(c, vmin, vmax, lut, self.run_obstacle)
            e = np.abs(c - r)
            emax = float(np.nanmax(e)) if np.isfinite(e).any() else 0.0
            self.err_view.subtitle = f"|Δ {name}|  ·  step {fr.step}  ·  scale 0 to this frame's max"
            self.err_view.set_field(e, 0.0, emax, SEQUENTIAL, self.run_obstacle)

    def _bind_diag(self, key):
        if not key:
            return
        s = {"reference": (REF, self.xs, self.diag_ref.setdefault(key, []), False)}
        if self.mode() != "reference":
            s[self._cand_name().lower()] = (CAND, self.xs, self.diag_cand.setdefault(key, []), True)
        self.diag_chart.series = s
        self.diag_chart.update()

    def _set_runtime(self, timing):
        ms = {}
        for key, (_, a, b) in self.rt.items():
            t = timing.get(key)
            ms[key] = t.per_step * 1e3 if t and t.per_step is not None else None
            a.setText("—" if ms[key] is None else f"{ms[key]:.3f}")
            b.setText("—" if not t or not t.first_call else f"{t.first_call:.2f}")
        self.speedup.setText(f"Steady-state speed-up  {ms['reference'] / ms['candidate']:.2f}×"
                             if ms["reference"] and ms["candidate"] else "")
        if timing:
            per = f"1 step = {self.spacing} solver steps (fused). " if self.spacing > 1 else ""
            hyb = ("Hybrid time is kolmo.hybrid.run's own (policy checks included). " if self.mode() == "hybrid" else "")
            self.rt_note.setText(f"{per}{hyb}Measured in this run on this machine; first call = compilation + 1 step, "
                                 "excluded from ms / step.")

    @staticmethod
    def _html_note(text):
        return f"<div style='color:{MUTED}'>{html.escape(text)}</div>"

    def _render_summary(self, s):
        if isinstance(s, str):
            return self._html_note(s)
        out = []
        if s["facts"]:
            out.append(f"<div style='color:{FAINT}; font-size:10px; font-weight:700'>MEASURED IN THIS RUN</div>")
            out += [f"<div style='margin:2px 0'>{html.escape(f)}</div>" for f in s["facts"]]
        col = {"critical": FAIL, "warning": CAND, "info": REF}
        if s["findings"]:
            out.append(f"<div style='color:{FAINT}; font-size:10px; font-weight:700; margin-top:8px'>"
                       f"RULE-BASED FINDINGS (kolmo.diagnose)</div>")
        for f in s["findings"]:
            out.append(f"<div style='margin-top:6px'><span style='color:{col.get(f.get('level'), MUTED)}'>●</span>&nbsp;"
                       f"<b>{html.escape(str(f.get('level', '')))}</b>&nbsp; {html.escape(str(f.get('fact', '')))}</div>"
                       f"<div style='color:{MUTED}'>Interpretation (hypothesis): {html.escape(str(f.get('interpretation', '')))}</div>"
                       f"<div style='color:{MUTED}'>Next: {html.escape(str(f.get('next', '')))}</div>"
                       f"<div style='color:{FAINT}; font-size:10px'>{html.escape(str(f.get('source', '')))}</div>")
        out += [f"<div style='color:{FAINT}; margin-top:6px'>{html.escape(n)}</div>" for n in s["notes"]]
        if not out:
            out.append(self._html_note("Nothing to report: no failure detected and no rule applies to a reference run."))
        return "".join(out)

    def _on_done(self, res):
        self.runner = None
        self._set_running(False)
        self._set_status("Stopped" if res["stopped"] else "Finished", MUTED if res["stopped"] else OK_GREEN)
        self._set_runtime(res["timing"])
        self.findings.setHtml(self._render_summary(res["summary"]))

    def _on_failed(self, msg):
        self.runner = None
        self._set_running(False)
        self._set_status("Error", FAIL)
        self.counter.setText("")
        self.run_error.setText(msg)
        self.findings.setHtml(self._html_note("The run did not start; see the message under Run."))

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
