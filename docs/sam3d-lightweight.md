# SAM 3D Objects on light hardware: study

Target: i3-10110U (2 cores / 4 threads, AVX2, no AVX-512/AMX/bf16 hardware), 24 GB RAM, Intel UHD iGPU.
Goal for Kolmo: photo -> **solid geometry** for the flow solver (no texture needed).

## Status

- Code: `third_party/sam-3d-objects` (GitHub archive of `main`). Read, not run.
- Weights: **not downloaded.** `facebook/sam-3d-objects` access was still pending (`GatedRepoError`) at the last
  check, and the repo's checkpoints total **13.11 GB** (> 5 GB limit; owner approval needed). Sizes from the HF file
  listing (`model_info(..., files_metadata=True)`):

| file | bytes | used at inference |
|---|---:|---|
| `ss_generator.ckpt` (stage 1: coarse shape + pose flow) | 6,690,136,964 | yes |
| `slat_generator.ckpt` (stage 2: structured-latent flow) | 4,906,537,684 | yes (not needed for geometry-only) |
| `slat_decoder_mesh.ckpt` / `.pt` (same model twice) | 363,726,862 / 363,728,714 | one of them, stage 2 |
| `slat_decoder_gs.ckpt`, `slat_decoder_gs_4.ckpt` | 171,476,155 / 170,269,801 | splats, stage 2 |
| `slat_encoder.ckpt`, `ss_encoder.ckpt` | 173,263,986 / 119,085,402 | no (training / encoding only) |
| `ss_decoder.ckpt` (16^3 latent -> 64^3 occupancy) | 147,609,242 | yes |
| **total** | **13,105,834,810** | |

Geometry-only minimum (`ss_generator` + `ss_decoder`): **6.84 GB**. Not counted: MoGe depth model and DINOv2,
fetched by the configs from elsewhere (`torch.hub` / HF), sizes not checked.

Official requirements (`doc/setup.md`): Linux-64, NVIDIA GPU with >= 32 GB VRAM; CUDA-only wheels (`spconv-cu121`,
`xformers`, `kaolin`, `gsplat`, `pytorch3d`, `nvidia-cuda-nvcc`). It will not run on this laptop as shipped.

## Architecture (from the code; `sam3d_objects/pipeline/inference_pipeline*.py`)

1. **Condition**: RGBA image (alpha = the SAM mask) -> DINOv2 tokens (`embedder/dino.py`, default `dinov2_vitb14`; the
   real size is set in the gated yaml) + a **pointmap** from the MoGe monocular depth model
   (`InferencePipelinePointMap.compute_pointmap`).
2. **Stage 1, sparse structure ("ss") generator**: a flow-matching transformer (`SparseStructureFlowModel`, dense
   full attention, Mixture-of-Transformers blocks `MOTModulatedTransformerCrossBlock`, one stream per latent:
   shape + layout/pose). Shape latent: 16^3 = **4096 tokens x 8 channels**. Decoder `ss_decoder` -> **64^3
   occupancy**; `stage1_only=True` returns it as voxel coordinates (`coords / 64 - 0.5`) plus pose (rotation,
   translation, scale).
3. **Stage 2, structured latent ("slat") generator**: sparse flow transformer over the occupied voxels only
   (`SLatFlowModel`, sparse convs via spconv + sparse attention), 8 latent channels per voxel.
4. **Decoders**: `slat_decoder_mesh` (FlexiCubes mesh) and `slat_decoder_gs` (Gaussian splats); then optional
   mesh post-processing, texture baking (1024^2), layout optimisation.

Sampling cost (code defaults, `InferencePipeline.__init__`): **25 Euler steps per stage**; classifier-free guidance
strength 7 (stage 1) and 5 (stage 2) with `cfg_interval=[0, 500]` on a 0-1000 time scale, and CFG runs the backbone
**twice** per guided step (`ClassifierFreeGuidance.inner_forward`). So each stage is 25 to 50 backbone forwards
(exact count depends on `rescale_t` in the gated yaml). Both generators are **shortcut models** (arXiv 2410.12557,
`generator/shortcut/model.py`): `use_stage1_distillation=True` turns CFG off and uses step size `d = 1/steps`
(trained for d in {1, 1/2, ..., 1/128}), i.e. a built-in few-step distilled mode.

What dominates (inference from code + file sizes; **not measured**: no weights):
- Memory: the two generators (6.69 + 4.91 GB on disk; if stored in fp32 that is ~1.7 B and ~1.2 B parameters).
- Compute: the generators x number of forwards. Stage 1 is 4096 dense tokens through the biggest network for
  25-50 forwards; stage 2 scales with the number of occupied voxels. Decoders and DINOv2 run once.
- On CPU the pipeline's `torch.autocast(device_type="cuda", dtype=bfloat16)` is silently disabled, so it would
  run fp32.

## Measured on this laptop (proxy, from SAM 3, see docs/geometry.md)

SAM 3.1's 454 M-parameter ViT image encoder at 1008^2 (5184 tokens) takes **101 s** per image fp32 on 2 threads
(~2 x 454e6 x 5184 = 4.7 TFLOP -> ~46 GFLOP/s effective). Applying that throughput to stage 1 (untested
estimate, assumes ~1.7 B active parameters x 4096 tokens per forward, ~14 TFLOP): **~5 min per forward**, i.e.
**~2-4 h** for 25-50 forwards, **~20 min** with a 4-step shortcut, **~5 min** with 1 step. Peak RAM ~7-10 GB fp32.
This is why the plan below starts with steps and stage selection, not kernels.

## Ranked plan (fastest wins first)

| # | change | expected effect | evidence |
|---|---|---|---|
| 1 | **Stage 1 only** (`stage1_only=True`): keep the 64^3 occupancy + pose; skip stage 2, decoders, texture, layout | removes slat generator (4.9 GB), spconv/kaolin/gsplat/pytorch3d deps and all stage-2 compute; ~half the total time | code path exists (`run(..., stage1_only=True)`); geometry-only is what the solver needs |
| 2 | **Shortcut sampling** (`use_stage1_distillation=True`, 1-4 steps, no CFG) | 25-50 forwards -> 1-4 (**~10-40x**) | shortcut objective + `d=1/steps` path in code; quality at 1-4 steps **untested** |
| 3 | **int8 weight compression of the ss generator** (OpenVINO + NNCF `compress_weights`, or torch dynamic int8 Linear) | 6.7 GB -> ~1.7 GB RAM; on CPU, speed from memory bandwidth | SAM 3 encoder measurements in docs/geometry.md; SAM 3D **untested** |
| 4 | **OpenVINO export of the ss generator** (dense transformer, static shapes: 4096 tokens) on CPU, fp16 on the iGPU | see SAM 3 encoder numbers in docs/geometry.md for the CPU/iGPU ratio on this machine | SAM 3D **untested** |
| 5 | **Cheaper conditioning**: smallest DINOv2 the checkpoint accepts, lower input resolution, cache tokens; skip MoGe by passing `pointmap=` from a cheaper depth model | DINOv2/MoGe run once, so a few % to tens of % | **untested**; changing the encoder needs retraining of the cross-attention |
| 6 | **Distil a small student** for the 2D/3D occupancy only: (photo, mask) -> 32^3/64^3 occupancy, a ~10-50 M param CNN/ViT trained on SAM 3D stage-1 outputs (teacher run on a GPU box) | seconds on CPU, < 200 MB | standard distillation; **untested**; needs a GPU for data generation |
| 7 | fewer tokens: run stage 1 at 8^3 latent (512 tokens) | ~8x fewer tokens per forward | **not possible without retraining** (fixed `pos_emb` size) |

For Kolmo's 2D solver, #1 + #2 (+ #3) is the shortest path; #6 is the only route to interactive speed on this laptop.

## From the 3D output to solver geometry

- **Stage 1 output** is already a voxel grid: `coords` (N x 3 integer, 64^3) -> `occ = zeros((64,)*3, bool);
  occ[tuple(coords.T)] = True`, then apply `rotation/translation/scale` from the pose to put it in camera/world
  frame, fill interior holes (`scipy.ndimage.binary_fill_holes`).
- A **mesh** (stage 2, GLB) -> voxelise with `trimesh.voxel.creation.voxelize` or ray parity per column;
  **splats** -> threshold the summed Gaussian density on a grid.
- **2D slice for the current solver**: choose the flow direction (+x) and the slice normal (z for a top view, y
  for a side view); take the slice through the centroid (or the max-area slice / union projection for a
  conservative frontal area), then hand the 2D bool image to `kolmo.geometry.mask.mask_to_grid` as if it were a
  photo mask (it crops, scales to the chord, smooths and places the object).
- A future 3D solver takes the occupancy volume directly (resample to the solver grid with area/volume averaging
  and a 1/2 threshold, same rule as the 2D path).
