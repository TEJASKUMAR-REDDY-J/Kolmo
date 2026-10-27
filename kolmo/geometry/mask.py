"""Object mask (photo pixels) -> boolean obstacle grid for the periodic 2D flow solver.

Grid convention: (n, n) bool, indexed [row, col] = [y, x]; the flow runs along +x (increasing column);
rows follow the photo (row 0 = top), so np.flipud it if a solver plots y upwards. Positions are fractions of
the domain side and refer to the centre of the object's bounding box."""
import math

import cv2
import numpy as np
from scipy import ndimage


def clean(mask):
    """Largest 8-connected component with its holes filled."""
    mask = np.asarray(mask, bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count < 2:
        raise ValueError("the mask is empty: nothing was segmented")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ndimage.binary_fill_holes(labels == largest)


def principal_angle(mask):
    """Angle in degrees (counter-clockwise as displayed) that turns the mask's long axis onto +x."""
    m = cv2.moments(np.asarray(mask, np.uint8), binaryImage=True)
    # image y points down: an axis tilted CCW (as displayed) by t gives atan2/2 = -t, and rotating by -t undoes it
    return math.degrees(0.5 * math.atan2(2 * m["mu11"], m["mu20"] - m["mu02"]))


def rotate(mask, degrees):
    """Rotate a bool mask counter-clockwise (as displayed) on an expanded canvas, so nothing is cut off."""
    if degrees % 360 == 0:
        return np.asarray(mask, bool)
    h, w = mask.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    W, H = int(math.ceil(h * s + w * c)), int(math.ceil(h * c + w * s))
    M[0, 2] += W / 2 - w / 2
    M[1, 2] += H / 2 - h / 2
    return cv2.warpAffine(mask.astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR) >= 0.5


def mask_to_grid(mask, n=128, chord=0.25, position=(0.25, 0.5), rotate_deg=0.0, flip=False, smooth=0.6):
    """mask: (H, W) bool in photo pixels. chord: streamwise length of the object as a fraction of the domain side
    (chord * n cells). rotate_deg: CCW degrees, or "auto" to put the long axis along the flow. flip: mirror
    left-right (put the object's front upstream). smooth: Gaussian sigma in cells applied to the area-averaged
    coverage before thresholding at 1/2 (removes single-cell jaggies; 0 disables).
    Returns (grid (n, n) bool, meta dict)."""
    obj = clean(mask)
    angle = principal_angle(obj) if rotate_deg == "auto" else float(rotate_deg)
    if flip:
        obj = obj[:, ::-1]
    obj = rotate(obj, angle)
    ys, xs = np.nonzero(obj)
    obj = obj[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    h_px, w_px = obj.shape

    chord_cells = chord * n
    scale = chord_cells / w_px  # cells per photo pixel
    w_c, h_c = max(1, round(w_px * scale)), max(1, round(h_px * scale))
    if w_c >= n - 2 or h_c >= n - 2:
        raise ValueError(f"object of {w_c} x {h_c} cells does not fit an {n} x {n} periodic domain with clearance")
    # area-averaged coverage per cell, padded so smoothing and placement never clip the object
    pad = 2 + int(math.ceil(3 * smooth))
    cover = np.pad(cv2.resize(obj.astype(np.float32), (w_c, h_c), interpolation=cv2.INTER_AREA), pad)
    if smooth > 0:
        cover = ndimage.gaussian_filter(cover, smooth, mode="constant")
    local = cover >= 0.5
    if not local.any():  # thinner than half a cell: keep the best-covered cells rather than lose the object
        local = cover >= cover.max() * 0.999
    local = clean(local)

    ys, xs = np.nonzero(local)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    oy, ox = round(position[1] * n - cy), round(position[0] * n - cx)  # bbox centre -> requested position
    grid = np.zeros((n, n), bool)
    grid[(ys + oy) % n, (xs + ox) % n] = True  # periodic placement

    rows, cols = np.nonzero(grid)
    meta = {
        "n": n,
        "cells_per_pixel": scale,
        "rotation_deg": angle,
        "flipped": bool(flip),
        "smooth_sigma_cells": smooth,
        "source_bbox_px": [int(w_px), int(h_px)],  # after rotation
        "area_cells": int(grid.sum()),
        "area_fraction": float(grid.mean()),
        "chord_cells": int(np.unique(cols).size),  # streamwise extent (columns occupied)
        "frontal_height_cells": int(np.unique(rows).size),  # projected height normal to the flow
        "centre_cells": [float(ox + cx), float(oy + cy)],  # (x, y) of the bbox centre, before wrapping
        "position": list(position),
    }
    return grid, meta
