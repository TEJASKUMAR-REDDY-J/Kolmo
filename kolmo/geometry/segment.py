"""Photo -> boolean object mask.

A prompt is one of
  None or text "car"         -> the object to segment (text is used by SAM 3; GrabCut uses a centre prior)
  (x, y)                     -> a point on the object, in image pixels
  (x0, y0, x1, y1)           -> a box around the object, in image pixels

`get_segmenter("auto")` returns SAM 3 when its code and weights are installed (see docs/geometry.md),
otherwise the OpenCV GrabCut fallback, so the pipeline always works."""
import atexit
import hashlib
import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Protocol, Union

import cv2
import numpy as np

Prompt = Union[None, str, tuple]

THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party"
SAM3_CODE = THIRD_PARTY / "sam3"
SAM3_WEIGHTS = THIRD_PARTY / "weights" / "sam3"  # holds sam3.pt + LICENSE from facebook/sam3
# SAM 3.1 first (its detector loads into the image model; verify with sam3_bench.py), else SAM 3
SAM3_CHECKPOINT = next((p for p in (THIRD_PARTY / "weights" / "sam3.1" / "sam3.1_multiplex.pt", SAM3_WEIGHTS / "sam3.pt")
                        if p.exists()), SAM3_WEIGHTS / "sam3.pt")
SAM3_PYTHON = THIRD_PARTY / "venv-sam3" / "Scripts" / "python.exe"
BRIDGE = Path(__file__).with_name("sam3_bridge.py")
OV_ENCODER = THIRD_PARTY / "weights" / "sam3.1" / "ov" / "trunk_1008.xml"  # made by sam3_bench.py ov-gpu (fp16 on iGPU)


class Segmenter(Protocol):
    name: str

    def segment(self, image: np.ndarray, prompt: Prompt = None) -> np.ndarray:
        """image: (H, W, 3) uint8 RGB. Returns a (H, W) bool mask of the object."""


def _check_image(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"expected an (H, W, 3) uint8 RGB image, got {image.shape} {image.dtype}")
    return image


class GrabCutSegmenter:
    """OpenCV GrabCut. Box -> the box is the initial foreground; point -> a box centred on the point plus a sure-
    foreground dot; text/None -> centre prior (the object fills the middle of the photo, margin `margin`)."""
    name = "grabcut"

    def __init__(self, max_side=512, iters=5, margin=0.06):
        self.max_side, self.iters, self.margin = max_side, iters, margin

    def segment(self, image, prompt=None):
        image = _check_image(image)
        if isinstance(prompt, dict):  # {"text", "point", "box"}: GrabCut can only use the geometry
            prompt = prompt.get("box") or prompt.get("point") or prompt.get("text")
        H, W = image.shape[:2]
        s = min(1.0, self.max_side / max(H, W))  # ponytail: GrabCut on a <=512 px copy; edges upsampled, ~1 px error
        small = cv2.resize(image, (max(1, round(W * s)), max(1, round(H * s))), interpolation=cv2.INTER_AREA) if s < 1 else image
        h, w = small.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        point = None
        if isinstance(prompt, (tuple, list)) and len(prompt) == 4:
            x0, y0, x1, y1 = (float(v) * s for v in prompt)
        elif isinstance(prompt, (tuple, list)) and len(prompt) == 2:
            point = (int(prompt[0] * s), int(prompt[1] * s))
            x0, y0, x1, y1 = point[0] - 0.35 * w, point[1] - 0.35 * h, point[0] + 0.35 * w, point[1] + 0.35 * h
        elif prompt is None or isinstance(prompt, str):
            x0, y0, x1, y1 = self.margin * w, self.margin * h, (1 - self.margin) * w, (1 - self.margin) * h
        else:
            raise ValueError(f"unsupported prompt {prompt!r}")
        x0, y0 = int(np.clip(x0, 0, w - 2)), int(np.clip(y0, 0, h - 2))
        x1, y1 = int(np.clip(x1, x0 + 2, w)), int(np.clip(y1, y0 + 2, h))
        mask[y0:y1, x0:x1] = cv2.GC_PR_FGD
        if point is not None:
            cv2.circle(mask, point, max(2, min(h, w) // 40), cv2.GC_FGD, -1)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.setRNGSeed(0)  # GrabCut's GMM init uses OpenCV's RNG: fixed seed -> deterministic masks
        cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), mask, None, bgd, fgd, self.iters, cv2.GC_INIT_WITH_MASK)
        fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
        if s < 1:
            fg = cv2.resize(fg * 255, (W, H), interpolation=cv2.INTER_LINEAR) >= 128
        return fg.astype(bool)


class Sam3Segmenter:
    """Meta SAM 3.1 / SAM 3, run locally. The model lives in its own venv (third_party/venv-sam3) so the base
    environment's numpy/torch stay untouched. A `sam3_bridge.py` server process is started on the first call and
    kept alive: the model loads once (~30 s) and the photo's image embedding is cached, so re-prompting the same
    photo (other text, a click) costs only the prompt (~15-20 s on a 2-core laptop) instead of the encoder too.
    Extra prompt form: a dict {"text": "car", "point": (x, y), "box": (x0, y0, x1, y1)} (any subset).
    `progress(msg)` is called before each slow step. encoder: "torch" | "ov-cpu" | "ov-gpu" | "auto"
    ("auto" = OpenVINO encoder, fp16 on the iGPU, when its converted XML exists, else torch)."""
    name = "sam3"

    def __init__(self, python=SAM3_PYTHON, checkpoint=SAM3_CHECKPOINT, threshold=0.4, encoder="auto", threads=None,
                 progress=None):
        self.python, self.checkpoint = Path(python), Path(checkpoint)
        self.threshold, self.threads, self.progress = threshold, threads, progress or (lambda msg: None)
        self.encoder = encoder if encoder != "auto" else ("ov-gpu" if OV_ENCODER.exists() else "torch")
        self.last_info, self._proc, self._image_key = {}, None, None
        self._lock = threading.Lock()
        self._tmp = tempfile.mkdtemp(prefix="kolmo_sam3_")

    @staticmethod
    def available(python=SAM3_PYTHON, checkpoint=SAM3_CHECKPOINT):
        return Path(python).exists() and Path(checkpoint).exists() and (SAM3_CODE / "sam3").is_dir()

    def _call(self, req):
        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()
        return self._read()

    def _read(self):
        line = self._proc.stdout.readline()
        if not line:
            self.close()
            raise RuntimeError(f"SAM 3 server exited; see {self._tmp}/bridge.log")
        r = json.loads(line)
        if "error" in r:
            raise RuntimeError(f"SAM 3: {r['error']}")
        return r

    def _start(self):
        if not self.available(self.python, self.checkpoint):
            raise RuntimeError("SAM 3 is not installed: see docs/geometry.md (venv, code and checkpoint)")
        self.progress("Loading SAM 3.1 (once, ~30 s)…")
        cfg = {"checkpoint": str(self.checkpoint), "encoder": self.encoder, "ov_xml": str(OV_ENCODER),
               "threads": self.threads}
        log = open(Path(self._tmp) / "bridge.log", "w")
        self._proc = subprocess.Popen([str(self.python), str(BRIDGE), json.dumps(cfg)], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1,
                                      env={**os.environ, "PYTHONPATH": str(SAM3_CODE)})
        atexit.register(self.close)
        self.last_info = self._read()

    def segment(self, image, prompt=None):
        image = _check_image(image)
        req = {"cmd": "prompt", "out": str(Path(self._tmp) / "mask.npy"), "threshold": self.threshold}
        if isinstance(prompt, dict):
            req.update({k: (list(map(float, v)) if k != "text" else v) for k, v in prompt.items() if v is not None})
        elif prompt is None or isinstance(prompt, str):
            req["text"] = prompt or "object"
        elif len(prompt) == 2:
            req["point"] = [float(v) for v in prompt]
        elif len(prompt) == 4:
            req["box"] = [float(v) for v in prompt]
        else:
            raise ValueError(f"unsupported prompt {prompt!r}")
        with self._lock:
            if self._proc is None:
                self._start()
            key = hashlib.sha1(image.tobytes()).hexdigest() + str(image.shape)
            if key != self._image_key:
                self.progress("Encoding the photo (once per photo)…")
                img = Path(self._tmp) / "image.png"
                cv2.imwrite(str(img), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                self._image_key = None
                self.last_info.update(self._call({"cmd": "image", "path": str(img)}))
                self._image_key = key
            self.progress("Segmenting…")
            self.last_info.update(self._call(req))
            return np.load(req["out"]).astype(bool)

    def close(self):
        p, self._proc, self._image_key = self._proc, None, None
        if p is not None and p.poll() is None:
            try:
                p.stdin.write('{"cmd": "quit"}\n')
                p.stdin.flush()
                p.wait(10)
            except Exception:
                p.kill()


def get_segmenter(kind="auto", **kw) -> Segmenter:
    if kind == "grabcut":
        return GrabCutSegmenter(**kw)
    if kind == "sam3" or (kind == "auto" and Sam3Segmenter.available()):
        return Sam3Segmenter(**kw)
    if kind == "auto":
        return GrabCutSegmenter(**kw)
    raise ValueError(f"unknown segmenter {kind!r} (auto | sam3 | grabcut)")


def parse_prompt(text):
    """CLI prompt: "car" -> text, "x,y" -> point, "x0,y0,x1,y1" -> box."""
    if text is None:
        return None
    parts = text.split(",")
    try:
        nums = tuple(float(p) for p in parts)
    except ValueError:
        return text
    if len(nums) not in (2, 4):
        raise ValueError("a numeric prompt is a point 'x,y' or a box 'x0,y0,x1,y1'")
    return nums

