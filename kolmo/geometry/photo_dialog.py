"""PySide6 dialog: photo -> SAM 3.1 (text prompt and/or a click on the object) -> obstacle grid.

    from kolmo.geometry.photo_dialog import ask
    res = ask(parent, n=128, chord=0.25)   # None if cancelled, else (path, grid_y_up (n, n) bool, meta)

The segmenter is shared across dialogs, so SAM 3.1 loads once per app session and re-prompting the same photo
reuses its cached image embedding. Falls back to GrabCut when SAM 3 is not installed."""
import time

import cv2
import numpy as np
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout)

from .mask import mask_to_grid
from .segment import get_segmenter

_SEGMENTER = None
VIEW = 560  # preview size in px


class _Work(QThread):
    progress, ok, err = Signal(str), Signal(object), Signal(str)

    def __init__(self, image, prompt, n, chord):
        super().__init__()
        self.image, self.prompt, self.n, self.chord = image, prompt, n, chord

    def run(self):
        global _SEGMENTER
        try:
            if _SEGMENTER is None:
                _SEGMENTER = get_segmenter("auto")
            if hasattr(_SEGMENTER, "progress"):
                _SEGMENTER.progress = self.progress.emit
            t = time.perf_counter()
            mask = _SEGMENTER.segment(self.image, self.prompt)
            grid, meta = mask_to_grid(mask, n=self.n, chord=self.chord)
            meta.update(segmenter=_SEGMENTER.name, segment_seconds=time.perf_counter() - t,
                        prompt=str(self.prompt), **getattr(_SEGMENTER, "last_info", {}))
            self.ok.emit((mask, grid, meta))
        except Exception as e:  # noqa: BLE001  shown in the dialog
            self.err.emit(f"{type(e).__name__}: {e}")


def _pixmap(rgb):
    h, w = rgb.shape[:2]
    return QPixmap.fromImage(QImage(np.ascontiguousarray(rgb).data, w, h, 3 * w, QImage.Format_RGB888).copy())


class PhotoDialog(QDialog):
    def __init__(self, parent, path, n, chord):
        super().__init__(parent)
        self.setWindowTitle("Obstacle from photo")
        self.path, self.n, self.chord, self.point, self.result, self.work = path, n, chord, None, None, None
        self.image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        self.scale = VIEW / max(self.image.shape[:2])
        self.view, self.grid_view = QLabel(), QLabel()
        self.view.setAlignment(Qt.AlignCenter)
        self.view.mousePressEvent = self._click
        self.grid_view.setFixedSize(220, 220)
        self.text = QLineEdit()
        self.text.setPlaceholderText("What is it? e.g. car, mug, wing (optional if you click the object)")
        self.status = QLabel("Type what the object is and/or click on it, then Segment.")
        self.status.setWordWrap(True)
        self.go, self.use = QPushButton("Segment"), QPushButton("Use as obstacle")
        self.use.setEnabled(False)
        self.go.clicked.connect(self._segment)
        self.text.returnPressed.connect(self._segment)
        self.use.clicked.connect(self.accept)
        side = QVBoxLayout()
        side.addWidget(self.grid_view)
        side.addWidget(self.status, 1)
        top = QHBoxLayout()
        top.addWidget(self.view, 1)
        top.addLayout(side)
        bar = QHBoxLayout()
        for w in (self.text, self.go, self.use):
            bar.addWidget(w)
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addLayout(bar)
        self._draw()

    def _draw(self, mask=None):
        img = self.image.copy()
        if mask is not None:
            img[mask] = (0.55 * img[mask] + 0.45 * np.array([255, 60, 60])).astype(np.uint8)
            cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cs, -1, (255, 230, 0), max(2, int(2 / self.scale)))
        if self.point is not None:
            cv2.circle(img, tuple(int(v) for v in self.point), max(6, int(8 / self.scale)), (0, 255, 120), -1)
        self.view.setPixmap(_pixmap(cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)))

    def _click(self, ev):
        pm = self.view.pixmap()
        ox, oy = (self.view.width() - pm.width()) / 2, (self.view.height() - pm.height()) / 2
        x, y = (ev.position().x() - ox) / self.scale, (ev.position().y() - oy) / self.scale
        if 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]:
            self.point = (x, y)
            self._draw(self.result[0] if self.result else None)
            self._segment()

    def _segment(self):
        if self.work is not None and self.work.isRunning():
            return
        text = self.text.text().strip() or None
        prompt = {"text": text, "point": self.point} if (text or self.point) else None
        self.go.setEnabled(False)
        self.use.setEnabled(False)
        self.status.setText("Working…")
        self.work = _Work(self.image, prompt, self.n, self.chord)
        self.work.progress.connect(self.status.setText)
        self.work.ok.connect(self._done)
        self.work.err.connect(self._failed)
        self.work.start()

    def _done(self, res):
        mask, grid, meta = self.result = res
        self._draw(mask)
        self.grid_view.setPixmap(_pixmap(np.repeat(np.where(grid, 235, 30).astype(np.uint8)[..., None], 3, 2))
                                 .scaled(220, 220, Qt.KeepAspectRatio, Qt.FastTransformation))
        self.status.setText(f"{meta['segmenter']} · {meta['segment_seconds']:.1f} s\n{meta['area_cells']} cells, chord "
                            f"{meta['chord_cells']}, frontal height {meta['frontal_height_cells']} on {meta['n']}².\n"
                            "Re-prompt (text or click) reuses the cached photo encoding.")
        self.go.setEnabled(True)
        self.use.setEnabled(True)

    def _failed(self, msg):
        self.status.setText(f"Failed: {msg}")
        self.go.setEnabled(True)


def ask(parent=None, n=128, chord=0.25, path=None):
    """Pick a photo (unless `path` is given), segment interactively. Returns (path, grid y-up, meta) or None."""
    if path is None:
        path, _ = QFileDialog.getOpenFileName(parent, "Obstacle from photo", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if not path:
            return None
    d = PhotoDialog(parent, path, n, chord)
    if d.exec() != QDialog.Accepted or d.result is None:
        return None
    _, grid, meta = d.result
    return path, np.flipud(grid), meta  # solver/display rows are y-up (as kolmo.app.backend.photo_obstacle)
