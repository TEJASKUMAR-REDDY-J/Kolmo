"""python -m kolmo.geometry photo.jpg --prompt "car" --n 128 --chord 0.25 --out runs/geometry/car

Writes <out>/mask.npy (bool (n, n) obstacle grid), <out>/preview.png (photo with mask overlay | grid) and
<out>/meta.json (scales, area, frontal height, segmenter, timings)."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .mask import mask_to_grid
from .segment import get_segmenter, parse_prompt


def preview(image, mask, grid):
    """Photo with the mask tinted red and outlined, next to the grid (obstacle white, fluid dark) at the same height."""
    over = image.copy()
    over[mask] = (0.55 * over[mask] + 0.45 * np.array([255, 40, 40])).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(over, contours, -1, (255, 255, 0), max(1, image.shape[0] // 300))
    H = over.shape[0]
    g = np.where(grid, 235, 30).astype(np.uint8)
    g = cv2.resize(g, (H, H), interpolation=cv2.INTER_NEAREST)
    g = np.repeat(g[..., None], 3, axis=2)
    return np.concatenate([over, np.full((H, 8, 3), 128, np.uint8), g], axis=1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m kolmo.geometry", description="Photo -> obstacle grid for the 2D solver")
    ap.add_argument("photo")
    ap.add_argument("--prompt", default=None, help='text "car", point "x,y" or box "x0,y0,x1,y1" (photo pixels)')
    ap.add_argument("--n", type=int, default=128, help="grid size n (n x n periodic domain)")
    ap.add_argument("--chord", type=float, default=0.25, help="streamwise object length as a fraction of the domain")
    ap.add_argument("--x", type=float, default=0.25, help="object centre x, fraction of the domain")
    ap.add_argument("--y", type=float, default=0.5, help="object centre y, fraction of the domain")
    ap.add_argument("--rotate", default="0", help='CCW degrees, or "auto" (long axis along the flow)')
    ap.add_argument("--flip", action="store_true", help="mirror left-right (front of the object upstream)")
    ap.add_argument("--smooth", type=float, default=0.6, help="smoothing sigma in cells (0 = off)")
    ap.add_argument("--segmenter", default="auto", choices=["auto", "sam3", "grabcut"])
    ap.add_argument("--out", default="runs/geometry", help="output folder")
    a = ap.parse_args(argv)

    bgr = cv2.imread(a.photo)
    if bgr is None:
        ap.error(f"cannot read image {a.photo}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    seg = get_segmenter(a.segmenter)
    t0 = time.perf_counter()
    mask = seg.segment(image, parse_prompt(a.prompt))
    t_seg = time.perf_counter() - t0
    grid, meta = mask_to_grid(mask, n=a.n, chord=a.chord, position=(a.x, a.y),
                              rotate_deg="auto" if a.rotate == "auto" else float(a.rotate), flip=a.flip, smooth=a.smooth)
    meta.update(photo=str(Path(a.photo).resolve()), prompt=a.prompt, segmenter=seg.name, segment_seconds=round(t_seg, 3),
                mask_pixels=int(mask.sum()), photo_size=[image.shape[1], image.shape[0]])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "mask.npy", grid)
    cv2.imwrite(str(out / "preview.png"), cv2.cvtColor(preview(image, mask, grid), cv2.COLOR_RGB2BGR))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("segmenter", "segment_seconds", "area_cells", "chord_cells",
                                           "frontal_height_cells")}), "->", out)


if __name__ == "__main__":
    main()
