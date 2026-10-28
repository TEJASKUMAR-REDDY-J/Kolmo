"""SAM 3 / 3.1 server. Runs INSIDE third_party/venv-sam3 (never imported by Kolmo); Sam3Segmenter starts it once and
keeps it alive, so the model loads once and the image embedding is cached across prompts.

    venv-sam3/Scripts/python.exe sam3_bridge.py '{"checkpoint": "...pt", "encoder": "torch" | "ov-cpu" | "ov-gpu",
                                                 "ov_xml": "...xml", "threads": null}'

Protocol: one JSON object per line on stdin, one JSON reply per line on stdout (first reply: {"ready": ...}).
  {"cmd": "image", "path": "photo.png"}                         -> {"image_s": ...}         (encodes, caches)
  {"cmd": "prompt", "out": "mask.npy", "text": "car", "point": [x, y], "box": [x0, y0, x1, y1], "threshold": 0.4}
        any of text / point / box (pixels). Writes the chosen instance mask (bool, photo size)  -> {"prompt_s", "score"}
        With a point, the instance containing it wins; otherwise the highest score.
  {"cmd": "quit"}
PYTHONPATH must contain third_party/sam3 (Sam3Segmenter sets it). Library chatter goes to stderr."""
import json
import os
import sys
import time

import numpy as np
import psutil
import torch
from PIL import Image

from sam3_bench import OVTrunk, no_random_init, openvino_trunk  # same folder (script dir is on sys.path)


def load(cfg):
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    if cfg.get("threads"):
        torch.set_num_threads(int(cfg["threads"]))
    with no_random_init():  # weights come from the checkpoint; skipping random init saves ~20 s
        model = build_sam3_image_model(device="cpu", checkpoint_path=cfg["checkpoint"], load_from_HF=False)
    enc = cfg.get("encoder", "torch")
    if enc.startswith("ov-"):  # OpenVINO ViT trunk; converted once (~10 min) and cached as XML
        xml = cfg["ov_xml"]
        compiled, _, _ = openvino_trunk(model, "GPU" if enc == "ov-gpu" else "CPU", xml.endswith("_int8_1008.xml"),
                                         os.path.dirname(xml), 1008)
        model.backbone.vision_backbone.trunk = OVTrunk(compiled)
    return model, Sam3Processor(model, resolution=1008, device="cpu", confidence_threshold=0.4)


def prompt(model, proc, state, req):
    proc.reset_all_prompts(state)
    proc.set_confidence_threshold(req.get("threshold", 0.4))
    W, H = state["original_width"], state["original_height"]
    if req.get("text"):
        state = proc.set_text_prompt(req["text"], state)
    if req.get("box") or req.get("point"):
        if "language_features" not in state["backbone_out"]:  # geometry-only prompt (as Sam3Processor does)
            state["backbone_out"].update(model.backbone.forward_text(["visual"], device="cpu"))
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = model._get_dummy_prompt()
        if req.get("box"):
            x0, y0, x1, y1 = req["box"]
            b = torch.tensor([(x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H]).view(1, 1, 4)
            state["geometric_prompt"].append_boxes(b, torch.tensor([True]).view(1, 1))
        if req.get("point"):
            x, y = req["point"]
            state["geometric_prompt"].append_points(torch.tensor([x / W, y / H]).view(1, 1, 2),
                                                    torch.tensor([True]).view(1, 1))
        state = proc._forward_grounding(state)
    if len(state["scores"]) == 0:  # nothing above threshold: take the best candidate anyway
        state = proc.set_confidence_threshold(0.0, state)
    masks, scores = state["masks"][:, 0].cpu().numpy(), state["scores"].float().cpu().numpy()
    best = int(np.argmax(scores))
    if req.get("point"):
        x, y = (int(round(v)) for v in req["point"])
        hit = [i for i in range(len(masks)) if masks[i][min(y, H - 1), min(x, W - 1)]]
        if hit:
            best = max(hit, key=lambda i: scores[i])
    return masks[best], float(scores[best]), len(scores)


def main(cfg):
    out = sys.stdout
    sys.stdout = sys.stderr  # keep stdout for the protocol
    reply = lambda d: (out.write(json.dumps(d) + "\n"), out.flush())
    t = time.perf_counter()
    model, proc = load(cfg)
    reply({"ready": True, "load_s": round(time.perf_counter() - t, 1)})
    state = None
    for line in sys.stdin:
        req = json.loads(line)
        try:
            t = time.perf_counter()
            with torch.inference_mode():
                if req["cmd"] == "quit":
                    break
                if req["cmd"] == "image":
                    state = proc.set_image(Image.open(req["path"]).convert("RGB"))
                    info = {"image_s": round(time.perf_counter() - t, 2)}
                elif req["cmd"] == "prompt":
                    mask, score, n = prompt(model, proc, state, req)
                    np.save(req["out"], mask.astype(bool))
                    info = {"prompt_s": round(time.perf_counter() - t, 2), "score": round(score, 3), "candidates": n}
                else:
                    raise ValueError(f"unknown cmd {req['cmd']!r}")
            mem = psutil.Process().memory_info()
            reply({**info, "peak_rss_gb": round(getattr(mem, "peak_wset", mem.rss) / 1e9, 2)})
        except Exception as e:  # report and keep serving
            reply({"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
