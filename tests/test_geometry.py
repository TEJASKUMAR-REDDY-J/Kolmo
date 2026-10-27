"""Photo -> obstacle grid on synthetic drawn shapes (no downloads, no SAM 3)."""
import json

import cv2
import numpy as np
import pytest

from kolmo.geometry.mask import clean, mask_to_grid
from kolmo.geometry.segment import GrabCutSegmenter, parse_prompt


def rect(w, h, angle=0.0, size=(600, 800)):
    m = np.zeros(size, np.uint8)
    box = cv2.boxPoints(((size[1] / 2, size[0] / 2), (w, h), -angle))  # -angle: CCW as displayed
    cv2.fillPoly(m, [np.round(box).astype(np.int32)], 1)
    return m.astype(bool)


def bbox(g):
    ys, xs = np.nonzero(g)
    return g[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


def test_rectangle_scale_area_and_placement():
    g, meta = mask_to_grid(rect(400, 100), n=128, chord=0.25, position=(0.25, 0.5))
    assert g.shape == (128, 128) and g.dtype == bool
    assert meta["chord_cells"] == 32 and meta["frontal_height_cells"] == 8
    assert abs(meta["area_cells"] - 256) <= 8
    assert meta["cells_per_pixel"] == pytest.approx(32 / 400, rel=0.01)
    ys, xs = np.nonzero(g)
    assert abs((xs.min() + xs.max()) / 2 - 32) <= 1 and abs((ys.min() + ys.max()) / 2 - 64) <= 1


def test_disk_area():
    m = np.zeros((500, 500), np.uint8)
    cv2.circle(m, (250, 250), 200, 1, -1)
    g, meta = mask_to_grid(m.astype(bool), n=256, chord=0.25)  # diameter 64 cells
    assert meta["area_cells"] == pytest.approx(np.pi * 32 ** 2, rel=0.03)
    assert meta["chord_cells"] == meta["frontal_height_cells"] == 64


def test_clean_keeps_largest_and_fills_holes():
    m = np.zeros((200, 200), np.uint8)
    cv2.circle(m, (100, 100), 60, 1, -1)
    cv2.circle(m, (100, 100), 20, 0, -1)  # hole
    m[5:8, 5:8] = 1  # speck
    c = clean(m.astype(bool))
    assert c[100, 100] and not c[6, 6]
    assert c.sum() == pytest.approx(np.pi * 60 ** 2, rel=0.02)


def test_rotation_explicit_and_auto():
    g, meta = mask_to_grid(rect(200, 100), n=128, chord=0.25, rotate_deg=90)  # long axis now across the flow
    assert meta["chord_cells"] == 32 and abs(meta["frontal_height_cells"] - 64) <= 1
    g, meta = mask_to_grid(rect(400, 100, angle=30), n=128, chord=0.25, rotate_deg="auto")
    assert meta["rotation_deg"] == pytest.approx(-30, abs=1)
    assert meta["chord_cells"] == 32 and abs(meta["frontal_height_cells"] - 8) <= 1


def test_flip_mirrors_streamwise():
    m = np.zeros((300, 600), np.uint8)
    cv2.fillPoly(m, [np.array([[50, 250], [550, 250], [50, 50]], np.int32)], 1)  # blunt face on the left
    a, _ = mask_to_grid(m.astype(bool), n=128)
    b, _ = mask_to_grid(m.astype(bool), n=128, flip=True)
    assert np.array_equal(bbox(b), bbox(a)[:, ::-1])


def test_periodic_wrap_keeps_shape():
    a, ma = mask_to_grid(rect(400, 100), n=128, position=(0.5, 0.5))
    b, mb = mask_to_grid(rect(400, 100), n=128, position=(0.0, 0.0))  # straddles both periodic seams
    assert ma["area_cells"] == mb["area_cells"] and mb["chord_cells"] == 32
    assert np.array_equal(np.roll(b, (64, 64), axis=(0, 1)), a)


def test_rejects_empty_and_oversized():
    with pytest.raises(ValueError):
        mask_to_grid(np.zeros((50, 50), bool))
    with pytest.raises(ValueError):
        mask_to_grid(rect(100, 400), n=64, chord=0.5)  # 32 x 128 cells


def synthetic_photo():
    rng = np.random.default_rng(0)
    img = np.clip(rng.normal([90, 140, 90], 12, (480, 640, 3)), 0, 255).astype(np.uint8)  # textured green
    truth = np.zeros((480, 640), np.uint8)
    cv2.ellipse(truth, (330, 250), (180, 70), 0, 0, 360, 1, -1)
    img[truth.astype(bool)] = (200, 40, 40)
    return img, truth.astype(bool)


@pytest.mark.parametrize("prompt", [None, "car", (330, 250), (120, 160, 540, 340)])
def test_grabcut_finds_object(prompt):
    img, truth = synthetic_photo()
    m = GrabCutSegmenter().segment(img, prompt)
    iou = (m & truth).sum() / (m | truth).sum()
    assert iou > 0.95, iou


def test_pipeline_is_deterministic(tmp_path):
    from kolmo.geometry.__main__ import main
    img, _ = synthetic_photo()
    cv2.imwrite(str(tmp_path / "p.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    outs = []
    for k in range(2):
        main([str(tmp_path / "p.png"), "--prompt", "car", "--segmenter", "grabcut", "--out", str(tmp_path / f"o{k}")])
        outs.append(np.load(tmp_path / f"o{k}" / "mask.npy"))
    assert np.array_equal(*outs) and outs[0].shape == (128, 128)
    meta = json.loads((tmp_path / "o0" / "meta.json").read_text())
    assert meta["chord_cells"] == 32 and (tmp_path / "o0" / "preview.png").exists()


def test_parse_prompt():
    assert parse_prompt("car") == "car" and parse_prompt(None) is None
    assert parse_prompt("10,20") == (10, 20) and parse_prompt("1,2,3,4") == (1, 2, 3, 4)
    with pytest.raises(ValueError):
        parse_prompt("1,2,3")
