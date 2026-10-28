"""SAM 3 CPU / iGPU speed study. Runs INSIDE third_party/venv-sam3, one configuration per process (clean peak RAM):

    PYTHONPATH=third_party/sam3 third_party/venv-sam3/Scripts/python kolmo/geometry/sam3_bench.py MODE CKPT IMAGE OUTDIR

MODE: fp32 | bf16 | int8 (torch dynamic quantisation of every nn.Linear) | res<N> (input resolution N, e.g. res672) |
      ov-cpu | ov-gpu (OpenVINO image encoder, fp32 on CPU / fp16 on the iGPU) | ov-cpu-int8 | ov-gpu-int8
      (NNCF int8 weight compression of the encoder).  The ViT trunk is what OpenVINO runs; the neck, text encoder,
      detector and mask head stay in torch.
Prints one JSON line (load / encoder / per-prompt seconds, peak RSS, IoU against OUTDIR/ref_fp32.npy) and saves the
mask of the first prompt as OUTDIR/<MODE>.npy."""
import contextlib
import json
import os
import sys
import time

import numpy as np
import psutil
import torch
from PIL import Image

PROMPTS = ["truck", "wheel", "window"]  # first one is the object; the others show the cost of re-prompting


@contextlib.contextmanager
def no_random_init():
    """Skip torch.nn.init during model construction: every weight is overwritten by the checkpoint anyway."""
    names = [n for n in dir(torch.nn.init) if n.endswith("_") and not n.startswith("_")]
    saved = {n: getattr(torch.nn.init, n) for n in names}
    for n in names:
        setattr(torch.nn.init, n, lambda t, *a, **k: t)
    try:
        yield
    finally:
        for n, f in saved.items():
            setattr(torch.nn.init, n, f)


def build(ckpt, inst=False):
    from sam3.model_builder import build_sam3_image_model
    with no_random_init():
        return build_sam3_image_model(device="cpu", checkpoint_path=ckpt, load_from_HF=False,
                                      enable_inst_interactivity=inst)


class OVTrunk(torch.nn.Module):
    """Drop-in for the ViT trunk: runs a compiled OpenVINO model and returns the trunk's feature list."""

    def __init__(self, compiled):
        super().__init__()
        self.compiled = compiled

    def forward(self, x):
        return [torch.from_numpy(self.compiled(x.numpy())[0])]


def openvino_trunk(model, device, int8, outdir, res):
    import openvino as ov
    xml = os.path.join(outdir, f"trunk{'_int8' if int8 else ''}_{res}.xml")
    t = time.perf_counter()
    if not os.path.exists(xml):
        class Last(torch.nn.Module):  # trunk as a registered submodule, so its weights are traced as parameters
            def __init__(self, trunk):
                super().__init__()
                self.trunk = trunk

            def forward(self, x):
                return self.trunk(x)[-1]

        with torch.no_grad():
            ovm = ov.convert_model(Last(model.backbone.vision_backbone.trunk.requires_grad_(False)),
                                   example_input=torch.zeros(1, 3, res, res))
        if int8:
            import nncf
            ovm = nncf.compress_weights(ovm)  # int8 weight-only (asymmetric, per-channel) by default
        ov.save_model(ovm, xml, compress_to_fp16=False)
    convert_s = time.perf_counter() - t
    core = ov.Core()
    core.set_property({"CACHE_DIR": os.path.join(outdir, "ov_cache")})  # compiled kernels reused across runs
    t = time.perf_counter()
    cfg = {"INFERENCE_PRECISION_HINT": "f16"} if device == "GPU" else {"INFERENCE_PRECISION_HINT": "f32"}
    m = core.read_model(xml)
    m.reshape({m.inputs[0]: [1, 3, res, res]})  # static shape: GPU kernels built at compile time (and cached), not on 1st call
    compiled = core.compile_model(m, device, cfg)
    return compiled, convert_s, time.perf_counter() - t


def main(mode, ckpt, image, outdir):
    from sam3.model.sam3_image_processor import Sam3Processor
    os.makedirs(outdir, exist_ok=True)
    info = {"mode": mode, "threads": torch.get_num_threads()}
    t = time.perf_counter()
    model = build(ckpt)
    info["load_s"] = round(time.perf_counter() - t, 1)
    res = int(mode[3:]) if mode.startswith("res") else 1008
    if mode == "int8":
        t = time.perf_counter()
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        info["quantise_s"] = round(time.perf_counter() - t, 1)
    if mode.startswith("ov-"):
        device = "GPU" if "gpu" in mode else "CPU"
        compiled, info["ov_convert_s"], info["ov_compile_s"] = openvino_trunk(model, device, mode.endswith("int8"),
                                                                              outdir, res)
        model.backbone.vision_backbone.trunk = OVTrunk(compiled)
        info["ov_device"] = compiled.get_property("EXECUTION_DEVICES")
    proc = Sam3Processor(model, resolution=res, device="cpu", confidence_threshold=0.4)
    img = Image.open(image).convert("RGB")
    ctx = torch.autocast("cpu", dtype=torch.bfloat16) if mode == "bf16" else contextlib.nullcontext()
    with torch.inference_mode(), ctx:
        enc = []
        for _ in range(2):  # 1st call includes one-off warm-up (allocations, kernel selection)
            t = time.perf_counter()
            state = proc.set_image(img)
            enc.append(round(time.perf_counter() - t, 2))
        info["encoder_s"] = enc
        info["prompt_s"], first = [], None
        for p in PROMPTS:
            proc.reset_all_prompts(state)
            t = time.perf_counter()
            state = proc.set_text_prompt(p, state)
            info["prompt_s"].append(round(time.perf_counter() - t, 2))
            if first is None:
                if len(state["scores"]) == 0:
                    state = proc.set_confidence_threshold(0.0, state)
                    proc.set_confidence_threshold(0.4)
                k = int(torch.argmax(state["scores"]))
                first = state["masks"][k, 0].cpu().numpy()
                info["score"] = round(float(state["scores"][k]), 3)
    np.save(os.path.join(outdir, f"{mode}.npy"), first)
    ref = os.path.join(outdir, "ref_fp32.npy")
    if mode == "fp32" and not os.path.exists(ref):
        np.save(ref, first)
    if os.path.exists(ref):
        r = np.load(ref)
        info["iou_vs_fp32"] = round(float((r & first).sum() / max(1, (r | first).sum())), 4)
    mem = psutil.Process().memory_info()
    info["peak_rss_gb"] = round(getattr(mem, "peak_wset", mem.rss) / 1e9, 2)
    info["mask_pixels"] = int(first.sum())
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:5])
