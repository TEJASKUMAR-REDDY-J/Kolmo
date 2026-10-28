# Geometry from a photo

`kolmo.geometry` turns a photo of an object (car, phone, mug, wing...) into a boolean obstacle mask for the
2D flow solver (`kolmo/systems/fluid2d.py` takes a bool `(H, W)` obstacle mask).

```
python -m kolmo.geometry photo.jpg --prompt "car" --n 128 --chord 0.25 --out runs/geometry/car
```

writes to `--out`:

| file | content |
|---|---|
| `mask.npy` | bool `(n, n)` obstacle grid, `True` = solid |
| `preview.png` | photo with the segmentation tinted red and outlined, next to the grid (white = solid) |
| `meta.json` | `cells_per_pixel`, `area_cells`, `area_fraction`, `chord_cells`, `frontal_height_cells`, rotation, segmenter, timings |

## Pipeline

1. **Segment** (`segment.py`). `get_segmenter("auto")` uses SAM 3 when it is installed, otherwise OpenCV GrabCut.
   Prompts: text (`--prompt car`), point (`--prompt 640,380`, photo pixels) or box (`--prompt 100,200,1500,900`).
   GrabCut ignores text and assumes the object fills the middle of the photo (6 % margin); give it a box when the
   background is busy. GrabCut is seeded, so results are deterministic.
2. **Grid** (`mask.py`, `mask_to_grid`):
   - keep the largest 8-connected component and fill its holes (an obstacle is a solid body);
   - optional `--flip` (mirror so the front faces upstream) and `--rotate DEG` (counter-clockwise as displayed;
     `auto` turns the long axis onto the flow direction);
   - crop to the object and scale so its streamwise length is `chord * n` cells (`cells_per_pixel` in the meta);
   - area-average the photo mask into cells (`cv2.INTER_AREA`), Gaussian-smooth the coverage by `--smooth` cells
     (default 0.6) and threshold at 1/2: this removes single-cell jaggies without eroding the shape;
   - place the bounding-box centre at `(--x, --y)` (fractions of the domain; default 0.25, 0.5) with periodic
     wrap-around. Objects that do not fit with 2 cells of clearance raise `ValueError`.

Conventions: grid indexed `[row, col] = [y, x]`, flow along `+x` (increasing column), row 0 at the **top** like the
photo. If the solver's `y` points up, use `np.flipud(mask)`.

Tests: `python -m pytest tests/test_geometry.py` (synthetic drawn shapes: area, scale, placement, rotation, flip,
periodic wrap, GrabCut IoU > 0.95, determinism; ~5 s).

## SAM 3.1 (local, no API)

Used automatically when installed (`get_segmenter("auto")`); otherwise GrabCut. The app's **Simulate > Obstacle >
From photo…** opens `photo_dialog.ask`: pick a photo, type what it is and/or click on it, *Segment*, check the
overlay and grid, *Use as obstacle*. Text and click combine (the instance under the click wins).

SAM 3's requirements would change the base environment (`timm`, `ftfy==6.1.1`, `iopath>=0.1.10`), so it lives in its
own venv. `Sam3Segmenter` starts `sam3_bridge.py` there once and keeps it alive (JSON lines over stdin/stdout): the
model loads once per session and the photo's image embedding is cached, so a re-prompt costs only the prompt.

```
third_party/sam3/              facebookresearch/sam3 source (GitHub archive of main) + 4 small KOLMO PATCHes (below)
third_party/venv-sam3/         torch 2.8.0+cpu, torchvision 0.23.0, numpy 1.26.4, timm, ftfy 6.1.1, iopath, psutil,
                               setuptools<81, openvino 2026.4.1, nncf 3.4.0
third_party/weights/sam3.1/    sam3.1_multiplex.pt (3,502,755,717 bytes) + LICENSE + config.json (facebook/sam3.1)
third_party/weights/sam3.1/ov/ trunk_int8_1008.xml/.bin: OpenVINO int8 image encoder (sam3_bench.py ov-gpu-int8)
```

Setup (after the owner accepted the gated licence and ran `hf auth login`):

```
python -m venv third_party/venv-sam3
third_party/venv-sam3/Scripts/python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
third_party/venv-sam3/Scripts/python -m pip install numpy==1.26.4 "timm>=1.0.17" ftfy==6.1.1 regex "iopath>=0.1.10" huggingface_hub tqdm pillow psutil einops pycocotools "setuptools<81" openvino==2026.4.1 nncf
python -c "from huggingface_hub import hf_hub_download as d; [d('facebook/sam3.1', f, local_dir='third_party/weights/sam3.1') for f in ('LICENSE', 'config.json', 'sam3.1_multiplex.pt')]"
```

The SAM 3.1 checkpoint's detector loads into SAM 3's image model; the only missing keys are the neck's 4th (0.5x)
level, which the image model discards (`scalp=1`). Model: 840.5 M parameters (ViT image encoder 454.0 M, text
encoder 353.7 M, detector/heads ~33 M).

KOLMO PATCHes (CPU-only Windows): `sam3/model/sam3_tracker_utils.py` (Triton import optional),
`sam3/model/position_encoding.py` and `sam3/model/decoder.py` (no hard-coded `device="cuda"`),
`sam3/perflib/fused.py` (the fused bf16 MLP is CUDA-only; on CPU plain fp32), `sam3/model/geometry_encoders.py`
(no `pin_memory` without CUDA).

### Measured (i3-10110U, 2 threads, `truck.jpg` 1800x1200, text prompt, `kolmo/geometry/sam3_bench.py`)

BENCH_TABLE

## Licences and attribution

- **SAM 3 / SAM 3.1 weights and code** are released by Meta under the **SAM License** (custom licence, not MIT/Apache;
  the GitHub `pyproject.toml` classifier says MIT but the repository `LICENSE` file is the SAM License). The licence
  text is kept next to the weights (`third_party/weights/*/LICENSE`) and in `third_party/sam3/LICENSE`.
  Anything Kolmo ships that includes or is derived from these weights must carry a copy of the SAM License and keep
  Meta's attribution notice; downloaded weights are never committed (third_party/ is gitignored).
- **SAM 3D Objects** (`facebook/sam-3d-objects`) is also under the SAM License (`third_party/sam-3d-objects/LICENSE`).
- Cite SAM 3 (arXiv:2511.16719) and SAM 3D (arXiv:2511.16624) in publications that use these outputs.
- The GrabCut fallback is plain OpenCV (Apache 2.0) and has no extra obligations.
