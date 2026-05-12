# modly-trellis2-extension

TRELLIS.2 extension for Modly.

## What works today

- Three runtime nodes are exposed: `trellis-2/generate`, `trellis-2/texture-mesh`, and `trellis-2/text-to-mesh`
- `trellis-2/generate` is the production-safe **image-to-mesh** compatibility node
- Output is a textured `.glb` mesh generated from a single input image
- A second node is now exposed: `trellis-2/texture-mesh`
- That node is an **image + mesh -> textured mesh** capability backed by the upstream TRELLIS.2 texturing pipeline
- A third node is now exposed: `trellis-2/text-to-mesh`
- That node is a native **text -> mesh** capability backed by upstream `TrellisTextTo3DPipeline`
- Clean install now completes against current Modly and current extension setup

## Current text input assumption

- Native text generation is grounded on `params.prompt` as the canonical prompt source
- If Modly mirrors native text input into additional keys, the generator also tolerates `params.text` and `params.input_text` as fallbacks

If you want a prompt-first UX with the image family, you can still compose:

`text-to-image -> trellis-2/generate`

## Current workflow shapes

- `trellis-2/generate`: `image -> mesh`
- `trellis-2/texture-mesh`: `image + mesh -> mesh`
- `trellis-2/text-to-mesh`: `text -> mesh`

For `texture-mesh`, Modly must provide the side mesh input as `params.mesh_path`. Current workflow wiring already passes named mesh inputs that way, so the extension reads the existing mesh directly from disk and re-exports a textured `.glb`.

## Gated Hugging Face dependencies

Install/setup success is NOT the full runtime story. First real runtime use still depends on access to gated upstream repos.

Required gated repos:

- https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
- https://huggingface.co/briaai/RMBG-2.0

You must be logged into Hugging Face in the extension environment and your account must have access approval for those repos.

The TRELLIS weights used by the extension are pulled from:

- `microsoft/TRELLIS.2-4B`
- `microsoft/TRELLIS-text-xlarge`

For the native text pipeline, the extension also localizes auxiliary decoder checkpoints referenced from the text pipeline config under the `text-xlarge` owner directory so runtime does not depend on incidental Hugging Face cache state.

## Practical install/runtime expectations

- `python setup.py ...` prepares the extension venv and native dependencies
- The first model load may still fail if Hugging Face access is missing, even after setup succeeded
- Expect NVIDIA CUDA runtime requirements; this is not a CPU-oriented extension
- Practical target is roughly **24 GB VRAM** for the current image-to-mesh path
- Native text-to-mesh uses a separate official TRELLIS family and carries similar high-end CUDA/VRAM expectations
- Higher voxel resolutions and larger textures materially increase runtime and memory pressure

## Current capability contract

- Manifest node id stays `generate` for compatibility with the currently working Modly contract
- UI naming is normalized to **Image to Mesh**
- `texture-mesh` reuses the same TRELLIS.2 weight owner metadata so both nodes share one downloaded snapshot
- `text-to-mesh` uses a separate `text-xlarge` owner and official `microsoft/TRELLIS-text-xlarge` snapshot
- Auxiliary image-family decoders required by the text pipeline are copied under the text owner as localized assets before pipeline load

## Validation

Lightweight checks used for this extension:

- `python3 -m py_compile generator.py setup.py validate_harden_arm64_native_setup.py`
- `python3 validate_harden_arm64_native_setup.py`

No heavy build is required for these checks.
