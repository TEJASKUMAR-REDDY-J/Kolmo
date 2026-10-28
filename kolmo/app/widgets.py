"""Drawing widgets: animated fabric backdrop, frosted-glass cards, field panels (QImage + numpy colour LUT), colour
bar, metric cards and a small line chart (QPainter). No matplotlib."""
import math

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPolygonF, QRegion
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget

# ---------------------------------------------------------------- palette (violet night)
BG = "#0b0820"
TEXT, MUTED, FAINT = "#eeeaff", "#aaa3d6", "#6f6a9c"
ACCENT, ACCENT2 = "#7c5cff", "#b07bff"
REF, CAND, FAIL, EVENT, OK, OBST = "#6fd3ff", "#c38bff", "#ff6b81", "#ffd166", "#5fe0a0", "#4a4470"
COMPARE = ("#e8e6ff", "#5ad1e6", "#ffb35c", "#ff8bd1")  # earlier runs in the error chart
GRID = QColor(255, 255, 255, 22)
UI_FONT, MONO_FONT, SYMBOL_FONT = "Segoe UI", "Cascadia Mono", "Segoe UI Symbol"


def font(px, weight=QFont.Normal, mono=False, family=None):
    f = QFont(family or (MONO_FONT if mono else UI_FONT))
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


# Diverging with a dark zero: sign is hue (blue < 0 < red), magnitude is lightness.
DIVERGING = _lut([(0, "#d2e4ff"), (0.14, "#6aa2ff"), (0.3, "#2457e0"), (0.44, "#151f63"), (0.5, "#0c0b26"),
                  (0.56, "#3e0f35"), (0.7, "#c0213f"), (0.86, "#ff5a5f"), (1, "#ffd8cc")])
# Sequential for magnitudes (after "inferno"): 0 is black.
SEQUENTIAL = _lut([(0, "#000004"), (0.14, "#1f0c48"), (0.29, "#550f6d"), (0.43, "#88226a"), (0.57, "#ba3655"),
                   (0.71, "#e35933"), (0.86, "#f98e09"), (1, "#fcffa4")])
_TABLES = {id(t): [0xFF000000 | (r << 16) | (g << 8) | b for r, g, b in t.tolist()] for t in (DIVERGING, SEQUENTIAL)}


def to_image(a, vmin, vmax, lut):
    """2D float array [row = y] -> indexed QImage with y pointing up, values clipped to [vmin, vmax], non-finite
    values drawn at the top of the scale. The LUT lookup is Qt's colour table (no per-pixel RGB work in numpy)."""
    a = np.nan_to_num(np.asarray(a, np.float32)[::-1], nan=vmax, posinf=vmax, neginf=vmin)
    idx = np.ascontiguousarray(np.clip((a - vmin) * (255.0 / max(vmax - vmin, 1e-30)), 0, 255).astype(np.uint8))
    img = QImage(idx.data, idx.shape[1], idx.shape[0], idx.strides[0], QImage.Format_Indexed8)
    img.setColorTable(_TABLES[id(lut)])
    return img.copy()  # detach from the numpy buffer


def mask_image(mask, rgb=OBST):
    """Bool mask [row = y] -> ARGB image (y up, like to_image), opaque where True."""
    c = QColor(rgb)
    out = np.zeros(mask.shape, np.uint32)
    out[np.asarray(mask, bool)[::-1]] = 0xFF000000 | (c.red() << 16) | (c.green() << 8) | c.blue()
    out = np.ascontiguousarray(out)
    return QImage(out.data, out.shape[1], out.shape[0], out.strides[0], QImage.Format_ARGB32).copy()


def fmt(v):
    if v == 0:
        return "0"
    a = abs(v)
    return f"{v:.3g}" if 1e-3 <= a < 1e4 else f"{v:.2e}"


_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(v, digits=2):
    """1.7 × 10⁻³ style for metric cards."""
    if v is None:
        return "—"
    if not math.isfinite(v):
        return "∞"
    if v == 0 or 1e-2 <= abs(v) < 1e4:
        return f"{v:.{digits + 1}g}"
    e = int(math.floor(math.log10(abs(v))))
    return f"{v / 10 ** e:.{digits - 1}f} × 10{str(e).translate(_SUP)}"


# ---------------------------------------------------------------- backdrop + glass

class Backdrop(QWidget):
    """Animated 'fabric': slow drifting light ribbons over an indigo gradient, computed with numpy at 96 x 56 and
    smoothly upscaled; a 4x smaller copy is the frosted backdrop of every Glass card.
    Cost control: 6 fps; a frame repaints only the gaps between cards; cards re-sample the blurred backdrop every
    5 s (one full repaint); paused when switched off, held (during runs), hidden, minimised or inactive."""
    W, H, FPS, BLUR_EVERY = 96, 56, 6, 30
    RIBBONS = (((0.30, 0.10, 2.1, 0.0, 0.07), (124, 92, 255)), ((0.62, 0.13, 1.6, 2.0, 0.09), (176, 90, 230)),
               ((0.85, 0.08, 2.7, 4.1, 0.06), (70, 110, 255)), ((0.12, 0.07, 3.3, 1.3, 0.05), (220, 110, 255)))
    instance = None

    def __init__(self):
        super().__init__()
        Backdrop.instance = self
        self.t, self.frame, self.blur_version = 0.0, 0, 0
        self.enabled, self.hold = True, False
        self.x = np.linspace(0, 1, self.W, dtype=np.float32)
        self.y = np.linspace(0, 1, self.H, dtype=np.float32)[:, None]
        g = (0.55 * self.x[None] + 0.45 * self.y)[..., None]
        self.vign = (1 - 0.35 * ((self.x[None] - 0.5) ** 2 + (self.y - 0.45) ** 2))[..., None]
        self.base = ((1 - g) * np.array([10, 8, 34], np.float32) + g * np.array([28, 12, 58], np.float32)) * self.vign
        self.timer = QTimer(self, interval=1000 // self.FPS, timeout=self._tick)
        self.timer.start()
        self._render()

    def _tick(self):
        w = self.window()
        if not self.enabled or self.hold or not self.isVisible() or w.isMinimized() or not w.isActiveWindow():
            return
        self.t += 1.0 / self.FPS
        self.frame += 1
        self._render()
        if self.frame % self.BLUR_EVERY == 0:
            self.blur_version += 1
            self.update()
        else:
            gaps = QRegion(self.rect())
            for c in self.findChildren(Glass):
                if c.level == 0 and c.isVisible():
                    gaps -= QRegion(QRect(c.mapTo(self, QPoint(0, 0)), c.size()))
            self.update(gaps)

    def _render(self):
        x, y, t = self.x, self.y, self.t * 0.05
        img = self.base.copy()
        for (y0, amp, k, ph, width), col in self.RIBBONS:  # centre line, width and brightness depend on x only
            yc = y0 + amp * np.sin(k * math.pi * x + ph + 6 * t) + 0.04 * np.sin(5.1 * x - 4 * t + ph)
            wd = width * (1 + 0.5 * np.sin(3 * x + 2 * t + ph))
            lum = (0.35 + 0.25 * np.sin(2.3 * x - 3 * t + ph)) * 0.55
            d = (y - yc) / wd
            img += (np.exp(-d * d) * lum)[..., None] * (np.array(col, np.float32) * self.vign)
        a = np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))
        self.image = QImage(a.data, self.W, self.H, a.strides[0], QImage.Format_RGB888).copy()
        b = np.ascontiguousarray(a.reshape(self.H // 4, 4, self.W // 4, 4, 3).mean((1, 3)).astype(np.uint8))
        self.blurred = QImage(b.data, b.shape[1], b.shape[0], b.strides[0], QImage.Format_RGB888).copy()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        sx, sy = self.W / max(self.width(), 1), self.H / max(self.height(), 1)
        for r in e.region():  # scale only the dirty rectangles (the gaps), not the whole frame
            p.drawImage(QRectF(r), self.image, QRectF(r.x() * sx, r.y() * sy, r.width() * sx, r.height() * sy))


class Glass(QFrame):
    """Rounded translucent card: the blurred backdrop under it, a light fill, a top highlight and a hairline border.
    level 0 = card on the backdrop, 1 = inset tile inside a card (no second blur). accent = violet tint.
    The background is cached per (size, position, backdrop blur version), so content repaints stay cheap."""

    def __init__(self, radius=18, level=0, accent=False):
        super().__init__()
        self.radius, self.level, self.accent = radius, level, accent
        self._cache, self._key = None, None

    def paintEvent(self, _):
        bd = Backdrop.instance
        blur = self.level == 0 and bd is not None and bd.width() > 0 and bd.isAncestorOf(self)
        o = self.mapTo(bd, QPoint(0, 0)) if blur else QPoint()
        key = (self.width(), self.height(), o.x(), o.y(), bd.blur_version if blur else -1)
        if key != self._key:
            self._key, self._cache = key, self._render_bg(bd if blur else None, o)
        QPainter(self).drawImage(0, 0, self._cache)

    def _render_bg(self, bd, o):
        img = QImage(self.size(), QImage.Format_ARGB32_Premultiplied)
        img.fill(Qt.transparent)
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        r = QRectF(0, 0, self.width(), self.height()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(r, self.radius, self.radius)
        if bd is not None:
            p.setClipPath(path)
            sx, sy = bd.blurred.width() / bd.width(), bd.blurred.height() / bd.height()
            p.drawImage(r, bd.blurred, QRectF(o.x() * sx, o.y() * sy, self.width() * sx, self.height() * sy))
            p.setClipping(False)
        fill = QLinearGradient(0, 0, 0, self.height())
        if self.accent:
            fill.setColorAt(0, QColor(124, 92, 255, 95))
            fill.setColorAt(1, QColor(124, 92, 255, 45))
        else:
            a = (30, 12) if self.level == 0 else (20, 8)
            fill.setColorAt(0, QColor(255, 255, 255, a[0]))
            fill.setColorAt(1, QColor(255, 255, 255, a[1]))
        p.fillPath(path, fill)
        p.setPen(QPen(QColor(255, 255, 255, 44 if self.level == 0 else 28), 1))
        p.drawPath(path)
        hi = QLinearGradient(0, 0, self.width(), 0)  # inner highlight along the top edge
        hi.setColorAt(0, QColor(255, 255, 255, 0))
        hi.setColorAt(0.5, QColor(255, 255, 255, 70))
        hi.setColorAt(1, QColor(255, 255, 255, 0))
        p.setPen(QPen(hi, 1))
        p.drawLine(QPointF(self.radius, 1.5), QPointF(self.width() - self.radius, 1.5))
        p.end()
        return img


def card(margins=(18, 14, 18, 16), spacing=8, **kw):
    """Glass card with a vertical layout; returns (card, layout)."""
    c = Glass(**kw)
    v = QVBoxLayout(c)
    v.setContentsMargins(*margins)
    v.setSpacing(spacing)
    return c, v


def heading(text, size=15, colour=TEXT):
    lab = QLabel(text)
    lab.setStyleSheet(f"font-size: {size}px; font-weight: 600; color: {colour}; background: transparent;")
    return lab


# ---------------------------------------------------------------- field panel + colour bar

class FieldView(QWidget):
    """One labelled field: bold title + qualifier, subtitle, rounded square image, optional colour bar. With no data
    it shows an honest empty-state message instead."""

    def __init__(self, title, qualifier="", bar=True):
        super().__init__()
        self.title, self.qualifier, self.subtitle, self.empty = title, qualifier, "", "No data yet."
        self.bar = bar
        self.image = self.overlay = None
        self.vmin = self.vmax = 0.0
        self.lut = DIVERGING
        self.setMinimumSize(180, 220)
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
        p.setFont(font(15, QFont.DemiBold))
        p.setPen(QColor(TEXT))
        p.drawText(QRectF(2, 0, w, 22), Qt.AlignLeft | Qt.AlignVCenter, self.title)
        tw = p.fontMetrics().horizontalAdvance(self.title)
        p.setFont(font(13))
        p.setPen(QColor(MUTED))
        p.drawText(QRectF(tw + 8, 0, w, 22), Qt.AlignLeft | Qt.AlignVCenter, self.qualifier)
        p.setFont(font(11))
        p.setPen(QColor(FAINT))
        p.drawText(QRectF(2, 22, w, 16), Qt.AlignLeft | Qt.AlignVCenter, self.subtitle)
        top, bar = 44, (30 if self.bar else 2)
        side = max(10, min(w, h - top - bar))
        box = QRectF((w - side) / 2, top, side, side)
        clip = QPainterPath()
        clip.addRoundedRect(box, 14, 14)
        if self.image is None:
            p.fillPath(clip, QColor(8, 6, 26, 140))
            p.setPen(QPen(QColor(255, 255, 255, 34), 1, Qt.DashLine))
            p.drawPath(clip)
            p.setPen(QColor(MUTED))
            p.setFont(font(12))
            p.drawText(box.adjusted(26, 26, -26, -26), Qt.AlignCenter | Qt.TextWordWrap, self.empty)
            return
        p.save()
        p.setClipPath(clip)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.drawImage(box, self.image)
        if self.overlay is not None:
            p.setRenderHint(QPainter.SmoothPixmapTransform, False)
            p.drawImage(box, self.overlay)
        p.restore()
        p.setPen(QPen(QColor(255, 255, 255, 50), 1))
        p.drawPath(clip)
        if self.bar:
            cb = QRectF(box.left(), box.bottom() + 9, box.width(), 6)
            strip = np.ascontiguousarray(self.lut[None])
            path = QPainterPath()
            path.addRoundedRect(cb, 3, 3)
            p.save()
            p.setClipPath(path)
            p.drawImage(cb, QImage(strip.data, 256, 1, strip.strides[0], QImage.Format_RGB888).copy())
            p.restore()
            p.setFont(font(10, mono=True))
            p.setPen(QColor(MUTED))
            lab = QRectF(cb.left(), cb.bottom() + 1, cb.width(), 14)
            p.drawText(lab, Qt.AlignLeft | Qt.AlignVCenter, fmt(self.vmin))
            p.drawText(lab, Qt.AlignRight | Qt.AlignVCenter, fmt(self.vmax))


class ColorBar(QWidget):
    """Vertical colour bar with symbol and tick values, shared by the reference and prediction panels."""

    def __init__(self, symbol):
        super().__init__()
        self.symbol, self.vmin, self.vmax, self.lut = symbol, None, None, DIVERGING
        self.setFixedWidth(64)

    def set_range(self, vmin, vmax, lut):
        self.vmin, self.vmax, self.lut = vmin, vmax, lut
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QColor(TEXT))
        p.setFont(font(16, QFont.DemiBold, family="Cambria Math"))
        p.drawText(QRectF(0, 40, 24, 22), Qt.AlignCenter, self.symbol)
        bar = QRectF(7, 70, 10, max(20, self.height() - 120))
        strip = np.ascontiguousarray(self.lut[::-1][:, None, :])
        path = QPainterPath()
        path.addRoundedRect(bar, 5, 5)
        p.save()
        p.setClipPath(path)
        p.drawImage(bar, QImage(strip.data, 1, 256, strip.strides[0], QImage.Format_RGB888).copy())
        p.restore()
        p.setPen(QPen(QColor(255, 255, 255, 50), 1))
        p.drawPath(path)
        if self.vmin is None:
            return
        p.setFont(font(10, mono=True))
        for v in nice_ticks(self.vmin, self.vmax, 4):
            yy = bar.bottom() - bar.height() * (v - self.vmin) / max(self.vmax - self.vmin, 1e-30)
            p.setPen(QColor(255, 255, 255, 90))
            p.drawLine(QPointF(bar.right() + 1, yy), QPointF(bar.right() + 4, yy))
            p.setPen(QColor(MUTED))
            p.drawText(QRectF(bar.right() + 6, yy - 7, 42, 14), Qt.AlignLeft | Qt.AlignVCenter, fmt(v))


# ---------------------------------------------------------------- metric card

class MetricCard(Glass):
    """Icon, label, big value, small caption. Values only come from measurements; otherwise a pending caption."""

    def __init__(self, icon, title, tip=""):
        super().__init__(radius=18)
        self.icon, self.title_text = icon, title
        self.value, self.caption, self.caption_colour = "—", "", MUTED
        self.setToolTip(tip)
        self.setMinimumHeight(92)

    def set(self, value, caption="", colour=MUTED):
        self.value, self.caption, self.caption_colour = value, caption, colour
        self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = QRectF(16, (self.height() - 42) / 2, 42, 42)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(124, 92, 255, 55))
        p.drawEllipse(c)
        p.setPen(QColor(ACCENT2))
        p.setFont(font(18, family=SYMBOL_FONT))
        p.drawText(c, Qt.AlignCenter, self.icon)
        x, wd = 72, self.width() - 80
        p.setPen(QColor(MUTED))
        p.setFont(font(12))
        p.drawText(QRectF(x, 12, wd, 18), Qt.AlignLeft | Qt.AlignVCenter, self.title_text)
        p.setPen(QColor(TEXT) if self.value != "—" else QColor(FAINT))
        p.setFont(font(22 if len(self.value) < 11 else 17, QFont.DemiBold))
        p.drawText(QRectF(x, 30, wd, 32), Qt.AlignLeft | Qt.AlignVCenter, self.value)
        p.setPen(QColor(self.caption_colour))
        p.setFont(font(11))
        p.drawText(QRectF(x, 61, wd, 18), Qt.AlignLeft | Qt.AlignVCenter,
                   p.fontMetrics().elidedText(self.caption, Qt.ElideRight, int(wd)))


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


def _decimate(x, y, buckets):
    """Keep the min and max of y per pixel column, so spikes survive while drawing a bounded number of points."""
    if len(x) <= 2 * buckets:
        return x, y
    edges = np.unique(np.searchsorted(x, np.linspace(x[0], x[-1], buckets + 1)[:-1]))
    lo, hi = np.minimum.reduceat(y, edges), np.maximum.reduceat(y, edges)
    return np.repeat(x[edges], 2), np.stack([lo, hi], 1).ravel()


class LineChart(QWidget):
    """Lines vs step on a transparent background with a legend box. series: name -> (colour, xs, ys, dashed, width);
    xs/ys are lists the owner appends to. hlines: [(y, label, colour)]; vlines: [(x, colour)]."""

    def __init__(self, log=False, x_label="step"):
        super().__init__()
        self.log, self.x_label = log, x_label
        self.series, self.hlines, self.vlines = {}, [], []
        self.xmax = 1
        self.empty = "No data yet."
        self.setMinimumSize(240, 130)

    def clear(self, message=None):
        self.series, self.vlines = {}, []
        if message is not None:
            self.empty = message
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        data = []
        for name, (col, xs, ys, dashed, width) in self.series.items():
            n = min(len(xs), len(ys))
            x, y = np.asarray(xs[:n], float), np.asarray(ys[:n], float)
            ok = np.isfinite(y) & ((y > 0) if self.log else True)
            if ok.any():
                data.append((name, col, x[ok], y[ok], dashed, width))
        if not data:
            p.setPen(QColor(MUTED))
            p.setFont(font(12))
            p.drawText(QRectF(0, 0, w, h).adjusted(24, 8, -24, -8), Qt.AlignCenter | Qt.TextWordWrap, self.empty)
            return
        p.setFont(font(11))
        legend_w = min(220, max(p.fontMetrics().horizontalAdvance(d[0]) for d in data) + 54)
        L, R, T, B = 50, 14 + legend_w, 8, 34
        plot = QRectF(L, T, max(10, w - L - R), max(10, h - T - B))
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
        p.setFont(font(10, mono=True))
        if self.log:
            step = max(1, int(math.ceil((hi - lo) / 5)))
            yt = [10.0 ** e for e in range(int(lo), int(hi) + 1, step)]
            ylab = [f"10{str(int(round(math.log10(v)))).translate(_SUP)}" for v in yt]
        else:
            yt = nice_ticks(lo, hi, 4)
            ylab = [fmt(v) for v in yt]
        for v, s in zip(yt, ylab):
            yy = float(Y(v))
            p.setPen(QPen(GRID, 1))
            p.drawLine(QPointF(plot.left(), yy), QPointF(plot.right(), yy))
            p.setPen(QColor(MUTED))
            p.drawText(QRectF(0, yy - 8, L - 8, 16), Qt.AlignRight | Qt.AlignVCenter, s)
        for v in nice_ticks(0, x1, 6):
            xx = float(X(v))
            p.setPen(QColor(MUTED))
            p.drawText(QRectF(xx - 40, plot.bottom() + 4, 80, 14), Qt.AlignHCenter | Qt.AlignTop, f"{v:g}")
        p.setPen(QPen(QColor(255, 255, 255, 70), 1))
        p.drawLine(plot.bottomLeft(), plot.bottomRight())
        p.drawLine(plot.bottomLeft(), plot.topLeft())
        p.setFont(font(11))
        p.setPen(QColor(MUTED))
        p.drawText(QRectF(plot.left(), plot.bottom() + 17, plot.width(), 16), Qt.AlignHCenter, self.x_label)
        p.save()
        p.setClipRect(plot.adjusted(0, -2, 2, 2))
        for x, col in self.vlines:
            p.setPen(QPen(QColor(col), 1, Qt.DotLine))
            xx = float(X(x))
            p.drawLine(QPointF(xx, plot.top()), QPointF(xx, plot.bottom()))
        for y, label, col in self.hlines:
            yy = float(Y(y))
            p.setPen(QPen(QColor(col), 1.2, Qt.DashLine))
            p.drawLine(QPointF(plot.left(), yy), QPointF(plot.right(), yy))
            p.setFont(font(10))
            p.drawText(QRectF(plot.left() + 6, yy - 15, plot.width(), 13), Qt.AlignLeft | Qt.AlignBottom, label)
        for name, col, x, y, dashed, width in data:
            x, y = _decimate(x, y, max(int(plot.width()), 1))
            poly = QPolygonF([QPointF(a, b) for a, b in zip(X(x).tolist(), Y(y).tolist())])
            p.setPen(QPen(QColor(col), width, Qt.DashLine if dashed else Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawPolyline(poly)
        p.restore()
        lg = QRectF(plot.right() + 14, plot.top(), legend_w - 4, 22 * len(data) + 10)
        p.setPen(QPen(QColor(255, 255, 255, 40), 1))
        p.setBrush(QColor(255, 255, 255, 12))
        p.drawRoundedRect(lg, 10, 10)
        p.setFont(font(11))
        for i, (name, col, _x, _y, dashed, _w) in enumerate(data):
            yy = lg.top() + 16 + 22 * i
            p.setPen(QPen(QColor(col), 2, Qt.DashLine if dashed else Qt.SolidLine))
            p.drawLine(QPointF(lg.left() + 10, yy), QPointF(lg.left() + 32, yy))
            p.setPen(QColor(TEXT))
            p.drawText(QRectF(lg.left() + 40, yy - 9, lg.width() - 44, 18), Qt.AlignLeft | Qt.AlignVCenter,
                       p.fontMetrics().elidedText(name, Qt.ElideRight, int(lg.width() - 46)))
