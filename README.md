# modly-trellis2-extension

TRELLIS.2 extension for Modly.

## What works today

- Two runtime nodes are exposed: `trellis-2/generate` and `trellis-2/texture-mesh`
- `trellis-2/generate` is the production-safe **image-to-mesh** compatibility node
- Output is a textured `.glb` mesh generated from a single input image
- A second node is now exposed: `trellis-2/texture-mesh`
- That node is an **image + mesh -> textured mesh** capability backed by the upstream TRELLIS.2 texturing pipeline
- Clean install now completes against current Modly and current extension setup

## What does NOT work yet

- No native TRELLIS text-to-mesh node is exposed
- This phase still does NOT expose native TRELLIS text-to-mesh; prompt-first UX remains `text-to-image -> trellis-2/generate`

If you want a prompt-first UX today, compose it as:

`text-to-image -> trellis-2/generate`

Do NOT treat this extension as native TRELLIS text-to-mesh yet.

## Current workflow shapes

- `trellis-2/generate`: `image -> mesh`
- `trellis-2/texture-mesh`: `image + mesh -> mesh`

For `texture-mesh`, Modly must provide the side mesh input as `params.mesh_path`. Current workflow wiring already passes named mesh inputs that way, so the extension reads the existing mesh directly from disk and re-exports a textured `.glb`.

## Gated Hugging Face dependencies

Install/setup success is NOT the full runtime story. First real runtime use still depends on access to gated upstream repos.

Required gated repos:

- https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
- https://huggingface.co/briaai/RMBG-2.0

You must be logged into Hugging Face in the extension environment and your account must have access approval for those repos.

The TRELLIS weights used by the extension are pulled from:

- `microsoft/TRELLIS.2-4B`

## Practical install/runtime expectations

- `python setup.py ...` prepares the extension venv and native dependencies
- The first model load may still fail if Hugging Face access is missing, even after setup succeeded
- Expect NVIDIA CUDA runtime requirements; this is not a CPU-oriented extension
- Practical target is roughly **24 GB VRAM** for the current image-to-mesh path
- Higher voxel resolutions and larger textures materially increase runtime and memory pressure

## Current capability contract

- Manifest node id stays `generate` for compatibility with the currently working Modly contract
- UI naming is normalized to **Image to Mesh**
- `texture-mesh` reuses the same TRELLIS.2 weight owner metadata so both nodes share one downloaded snapshot
- This repo intentionally does NOT add native `text -> mesh` in Phase 2

## Validation

Lightweight checks used for this extension:

- `python3 -m py_compile generator.py setup.py validate_harden_arm64_native_setup.py`
- `python3 validate_harden_arm64_native_setup.py`

No heavy build is required for these checks.
