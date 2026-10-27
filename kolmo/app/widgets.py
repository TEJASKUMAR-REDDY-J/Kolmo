"""Drawing widgets: field panels (QImage + numpy colour LUT) and a small line chart (QPainter). No matplotlib."""
import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

# ---------------------------------------------------------------- palette (dark)
BG, SURFACE, SURFACE2, BORDER = "#0e1014", "#15181e", "#1c2028", "#272c36"
TEXT, MUTED, FAINT = "#e6e8ec", "#8c95a4", "#5b6372"
ACCENT = "#5b9dff"
REF, CAND, FAIL, EVENT, OBST = "#8fb8ff", "#ffb35c", "#ff6b6b", "#b892ff", "#3b414d"
UI_FONT, MONO_FONT = "Segoe UI", "Consolas"


def font(px, weight=QFont.Normal, mono=False):
    f = QFont(MONO_FONT if mono else UI_FONT)
    f.setPixelSize(px)
    f.setWeight(weight)
    return f


# ---------------------------------------------------------------- colour maps

def _lut(anchors):
    """[(pos, '#rrggbb')] -> (256, 3) uint8, linear in sRGB between anchors."""
    pos = np.array([a[0] for a in anchors])
    rgb = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for _, c in anchors], float)
    x = np.linspace(0, 1, 256)
    return np.stack([np.interp(x, pos, rgb[:, i]) for i in range(3)], 1).round().astype(np.uint8)


# Diverging with a dark centre (after Crameri's "berlin"): zero is near-black, sign is hue, magnitude is lightness.
DIVERGING = _lut([(0, "#9eb0ff"), (0.15, "#5b9ee0"), (0.3, "#2a6a99"), (0.42, "#16303f"), (0.5, "#111111"),
                  (0.58, "#3b130a"), (0.7, "#7a2a18"), (0.85, "#c0614f"), (1, "#ffadad")])
# Sequential for magnitudes (after "inferno"): 0 is black.
SEQUENTIAL = _lut([(0, "#000004"), (0.14, "#1f0c48"), (0.29, "#550f6d"), (0.43, "#88226a"), (0.57, "#ba3655"),
                   (0.71, "#e35933"), (0.86, "#f98e09"), (1, "#fcffa4")])
_TABLES = {id(DIVERGING): [0xFF000000 | (r << 16) | (g << 8) | b for r, g, b in DIVERGING.tolist()],
           id(SEQUENTIAL): [0xFF000000 | (r << 16) | (g << 8) | b for r, g, b in SEQUENTIAL.tolist()]}


def to_image(a, vmin, vmax, lut):
    """2D float array [row = y] -> indexed QImage with y pointing up (last row at the top), values clipped to
    [vmin, vmax], non-finite values drawn at the top of the scale. The LUT lookup is done by Qt's colour table."""
    a = np.nan_to_num(np.asarray(a, np.float32)[::-1], nan=vmax, posinf=vmax, neginf=vmin)
    idx = np.ascontiguousarray(np.clip((a - vmin) * (255.0 / max(vmax - vmin, 1e-30)), 0, 255).astype(np.uint8))
    img = QImage(idx.data, idx.shape[1], idx.shape[0], idx.strides[0], QImage.Format_Indexed8)
    img.setColorTable(_TABLES[id(lut)])
    return img.copy()  # detach from the numpy buffer


def mask_image(mask, rgb=OBST):
    """Bool mask [row = y] -> ARGB image (y up, like to_image), opaque where True, transparent elsewhere."""
    c = QColor(rgb)
    out = np.zeros(mask.shape, np.uint32)
    out[np.asarray(mask, bool)[::-1]] =0xFF000000 | (c.red() << 16) | (c.green() << 8) | c.blue()
    out = np.ascontiguousarray(out)
    return QImage(out.data, out.shape[1], out.shape[0], out.strides[0], QImage.Format_ARGB32).copy()


def fmt(v):
    if v == 0:
        return "0"
    a = abs(v)
    return f"{v:.3g}" if 1e-3 <= a < 1e4 else f"{v:.2e}"


# ---------------------------------------------------------------- field panel

class FieldView(QWidget):
    """One labelled field: big title, subtitle, square image, colour bar with its numeric range. With no data it shows
    an honest empty-state message instead."""

    def __init__(self, title):
        super().__init__()
        self.title, self.subtitle, self.empty = title, "", "No data yet."
        self.image = self.overlay = None
        self.vmin = self.vmax = 0.0
        self.lut = DIVERGING
        self.setMinimumSize(220, 260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_field(self, a, vmin, vmax, lut, mask=None):
        self.image, self.vmin, self.vmax, self.lut = to_image(a, vmin, vmax, lut), vmin, vmax, lut
        self.overlay = None if mask is None else mask_image(mask)
        self.update()

    def clear(self, message):
        self.image = self.overlay = None
        self.empty = message
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(SURFACE))
        pad = 14
        p.setPen(QColor(TEXT))
        p.setFont(font(15, QFont.DemiBold))
        p.drawText(QRectF(pad, pad - 2, w - 2 * pad, 22), Qt.AlignLeft | Qt.AlignVCenter, self.title)
        p.setPen(QColor(MUTED))
        p.setFont(font(11))
        p.drawText(QRectF(pad, pad + 20, w - 2 * pad, 16), Qt.AlignLeft | Qt.AlignVCenter, self.subtitle)
        top, bar = pad + 44, 34
        side = max(10, min(w - 2 * pad, h - top - bar - pad))
        box = QRectF((w - side) / 2, top, side, side)
        if self.image is None:
            p.setPen(QPen(QColor(BORDER), 1, Qt.DashLine))
            p.setBrush(QColor(BG))
            p.drawRoundedRect(box, 6, 6)
            p.setPen(QColor(MUTED))
            p.setFont(font(12))
            p.drawText(box.adjusted(24, 24, -24, -24), Qt.AlignCenter | Qt.TextWordWrap, self.empty)
            return
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.drawImage(box, self.image)
        if self.overlay is not None:
            p.setRenderHint(QPainter.SmoothPixmapTransform, False)
            p.drawImage(box, self.overlay)
        p.setPen(QPen(QColor(BORDER), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(box)
        # colour bar
        cb = QRectF(box.left(), box.bottom() + 10, box.width(), 7)
        strip = np.ascontiguousarray(self.lut[None])
        bar_img = QImage(strip.data, 256, 1, strip.strides[0], QImage.Format_RGB888).copy()
        p.drawImage(cb, bar_img)
        p.setFont(font(10, mono=True))
        p.setPen(QColor(MUTED))
        lab = QRectF(cb.left(), cb.bottom() + 2, cb.width(), 14)
        p.drawText(lab, Qt.AlignLeft | Qt.AlignVCenter, fmt(self.vmin))
        p.drawText(lab, Qt.AlignRight | Qt.AlignVCenter, fmt(self.vmax))
        if self.vmin < 0 < self.vmax:
            p.drawText(lab, Qt.AlignHCenter | Qt.AlignVCenter, "0")


# ---------------------------------------------------------------- line chart

def nice_ticks(lo, hi, n=5):
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=10 * mag)
    first = math.ceil(lo / step) * step
    return [first + i * step for i in range(int((hi - first) / step + 1e-9) + 1)]


_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _decimate(x, y, buckets):
    """Keep the min and max of y per pixel column, so spikes survive while drawing a bounded number of points."""
    if len(x) <= 2 * buckets:
        return x, y
    edges = np.searchsorted(x, np.linspace(x[0], x[-1], buckets + 1)[:-1])
    edges = np.unique(edges)
    lo, hi = np.minimum.reduceat(y, edges), np.maximum.reduceat(y, edges)
    return np.repeat(x[edges], 2), np.stack([lo, hi], 1).ravel()


class LineChart(QWidget):
    """Lines vs step. series: name -> (colour, xs, ys, dashed); xs/ys are lists the owner appends to.
    hlines: [(y, label, colour)]; vlines: [(x, colour)] (e.g. hybrid interventions)."""

    def __init__(self, log=False):
        super().__init__()
        self.log = log
        self.series, self.hlines, self.vlines = {}, [], []
        self.xmax = 1
        self.empty = "No data yet."
        self.y_label = ""
        self.setMinimumSize(240, 150)

    def clear(self, message=None):
        self.series, self.vlines = {}, []
        if message is not None:
            self.empty = message
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(SURFACE))
        w, h = self.width(), self.height()
        legend_h = 18
        L, R, T, B = 56, 12, 6 + legend_h, 22
        plot = QRectF(L, T, w - L - R, h - T - B)
        data = []
        for name, (col, xs, ys, dashed) in self.series.items():
            n = min(len(xs), len(ys))
            x, y = np.asarray(xs[:n], float), np.asarray(ys[:n], float)
            ok = np.isfinite(y) & ((y > 0) if self.log else True)
            if ok.any():
                data.append((name, col, x[ok], y[ok], dashed))
        # legend
        p.setFont(font(11))
        lx = w - R
        for name, col, *_ in reversed(data):
            tw = p.fontMetrics().horizontalAdvance(name)
            lx -= tw
            p.setPen(QColor(MUTED))
            p.drawText(QRectF(lx, 2, tw + 2, legend_h), Qt.AlignVCenter, name)
            lx -= 20
            p.setPen(QPen(QColor(col), 2))
            p.drawLine(QPointF(lx, 2 + legend_h / 2), QPointF(lx + 14, 2 + legend_h / 2))
            lx -= 14
        p.setPen(QPen(QColor(BORDER), 1))
        p.setBrush(QColor(BG))
        p.drawRect(plot)
        if not data:
            p.setPen(QColor(MUTED))
            p.setFont(font(12))
            p.drawText(plot.adjusted(16, 8, -16, -8), Qt.AlignCenter | Qt.TextWordWrap, self.empty)
            return
        ys_all = np.concatenate([d[3] for d in data] + [np.array([y for y, *_ in self.hlines], float)])
        ys_all = ys_all[ys_all > 0] if self.log else ys_all
        lo, hi = float(ys_all.min()), float(ys_all.max())
        if self.log:
            lo, hi = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
            hi = max(hi, lo + 1)
        else:
            if hi == lo:
                lo, hi = lo - (abs(lo) or 1) * 0.05, hi + (abs(hi) or 1) * 0.05
            pad = 0.06 * (hi - lo)
            lo, hi = lo - pad, hi + pad
        x1 = max(self.xmax, max(float(d[2][-1]) for d in data), 1)
        X = lambda x: plot.left() + plot.width() * np.asarray(x, float) / x1
        Y = lambda y: plot.bottom() - plot.height() * ((np.log10(y) if self.log else np.asarray(y, float)) - lo) / (hi - lo)
        # grid + labels
        p.setFont(font(10, mono=True))
        if self.log:
            step = max(1, int(math.ceil((hi - lo) / 6)))
            yt = [10.0 ** e for e in range(int(lo), int(hi) + 1, step)]
            ylab = [f"10{str(int(round(math.log10(v)))).translate(_SUP)}" for v in yt]
        else:
            yt = nice_ticks(lo, hi)
            ylab = [fmt(v) for v in yt]
        for v, s in zip(yt, ylab):
            yy = float(Y(v))
            p.setPen(QPen(QColor(BORDER), 1))
            p.drawLine(QPointF(plot.left(), yy), QPointF(plot.right(), yy))
            p.setPen(QColor(MUTED))
            p.drawText(QRectF(0, yy - 8, L - 6, 16), Qt.AlignRight | Qt.AlignVCenter, s)
        for v in nice_ticks(0, x1, 6):
            xx = float(X(v))
            p.setPen(QPen(QColor(BORDER), 1))
            p.drawLine(QPointF(xx, plot.top()), QPointF(xx, plot.bottom()))
            p.setPen(QColor(MUTED))
            p.drawText(QRectF(xx - 40, plot.bottom() + 3, 80, 16), Qt.AlignHCenter | Qt.AlignTop, f"{v:g}")
        p.setClipRect(plot)
        for x, col in self.vlines:
            p.setPen(QPen(QColor(col), 1, Qt.DotLine))
            xx = float(X(x))
            p.drawLine(QPointF(xx, plot.top()), QPointF(xx, plot.bottom()))
        for y, label, col in self.hlines:
            yy = float(Y(y))
            p.setPen(QPen(QColor(col), 1.2, Qt.DashLine))
            p.drawLine(QPointF(plot.left(), yy), QPointF(plot.right(), yy))
            p.setFont(font(10))
            p.drawText(QRectF(plot.left() + 6, yy - 16, plot.width(), 14), Qt.AlignLeft | Qt.AlignBottom, label)
        for name, col, x, y, dashed in data:
            x, y = _decimate(x, y, max(int(plot.width()), 1))
            poly = QPolygonF([QPointF(a, b) for a, b in zip(X(x).tolist(), Y(y).tolist())])
            p.setPen(QPen(QColor(col), 1.6, Qt.DashLine if dashed else Qt.SolidLine))
            p.drawPolyline(poly)
