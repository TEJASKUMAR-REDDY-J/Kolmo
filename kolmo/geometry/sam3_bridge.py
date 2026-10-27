"""Runs INSIDE third_party/venv-sam3 (never imported by Kolmo): one SAM 3 segmentation per call.

    venv-sam3/Scripts/python.exe sam3_bridge.py '{"image": "...png", "out": "...npy", "checkpoint": "...pt",
                                                 "text": "car" | "point": [x, y] | "box": [x0, y0, x1, y1],
                                                 "threshold": 0.4, "resolution": null, "dtype": "fp32", "threads": null}'

Writes the best instance mask (bool, photo size) to `out` and prints one JSON line with timings and peak RAM.
PYTHONPATH must contain third_party/sam3 (Sam3Segmenter sets it)."""
import json
import sys
import time

import numpy as np
import psutil
import torch
from PIL import Image


def main(req):
    t0 = time.perf_counter()
    if req.get("threads"):
        torch.set_num_threads(int(req["threads"]))
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    point = "point" in req
    model = build_sam3_image_model(device="cpu", checkpoint_path=req["checkpoint"], load_from_HF=False,
                                   enable_inst_interactivity=point)
    proc = Sam3Processor(model, resolution=req.get("resolution") or 1008, device="cpu",
                         confidence_threshold=req.get("threshold", 0.4))
    img = Image.open(req["image"]).convert("RGB")
    W, H = img.size
    t_load = time.perf_counter() - t0

    bf16 = req.get("dtype") == "bf16"
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        t1 = time.perf_counter()
        state = proc.set_image(img)
        t_image = time.perf_counter() - t1
        t2 = time.perf_counter()
        if point:
            masks, scores, _ = model.predict_inst(state, point_coords=np.array([req["point"]], np.float32),
                                                  point_labels=np.array([1]), multimask_output=True)
            masks, scores = np.asarray(masks) > 0, np.asarray(scores)
        else:
            if "box" in req:
                x0, y0, x1, y1 = req["box"]
                box = [(x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H]  # normalised cxcywh
                state = proc.add_geometric_prompt(box, True, state)
            else:
                state = proc.set_text_prompt(req["text"], state)
            if len(state["scores"]) == 0:  # nothing above threshold: take the best candidate anyway
                state = proc.set_confidence_threshold(0.0, state)
            masks, scores = state["masks"][:, 0].cpu().numpy(), state["scores"].float().cpu().numpy()
        t_prompt = time.perf_counter() - t2
    best = int(np.argmax(scores))
    np.save(req["out"], masks[best].astype(bool))
    mem = psutil.Process().memory_info()
    print(json.dumps({"load_s": round(t_load, 2), "image_s": round(t_image, 2), "prompt_s": round(t_prompt, 2),
                      "score": float(scores[best]), "candidates": int(len(scores)),
                      "peak_rss_gb": round(getattr(mem, "peak_wset", mem.rss) / 1e9, 2), "dtype": req.get("dtype", "fp32")}))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
