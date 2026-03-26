"""
TRELLIS.2 extension for Modly.

Reference : https://huggingface.co/microsoft/TRELLIS.2-4B
GitHub    : https://github.com/microsoft/TRELLIS.2

All runtime pure-Python dependencies (easydict, plyfile, einops, trellis2
source) are bundled in vendor/ — no pip install, no internet required at
runtime.

The following compiled CUDA extensions must be pre-installed in the app's venv
(they have C++/CUDA components and cannot be vendored as pure Python):
    o-voxel        — core O-Voxel representation library
    nvdiffrast     — differentiable rasterizer (NVlabs)
    nvdiffrec      — PBR renderer (JeffreyXiang fork)
    cumesh         — CUDA mesh utilities
    flexgemm       — Triton sparse convolution
    flash-attn     — fused attention (or: xformers as fallback)

To rebuild vendor/:
    python build_vendor.py   (run once with the app's venv active)
"""

import io
import os
import sys
import time
import threading
import uuid
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from services.generators.base import BaseGenerator, smooth_progress, GenerationCancelled

_EXTENSION_DIR = Path(__file__).parent
_VENDOR_DIR    = _EXTENSION_DIR / "vendor"


class Trellis2Generator(BaseGenerator):
    MODEL_ID     = "trellis-2"
    DISPLAY_NAME = "TRELLIS.2"
    VRAM_GB      = 24

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def is_downloaded(self) -> bool:
        return (self.model_dir / "pipeline.json").exists()

    def load(self) -> None:
        if self._model is not None:
            return

        if not self.is_downloaded():
            self._auto_download()

        self._setup_env()    # must run before vendor imports so SPARSE_CONV_BACKEND is set
        self._setup_vendor()

        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        print(f"[Trellis2Generator] Loading model from {self.model_dir}...")
        pipe = Trellis2ImageTo3DPipeline.from_pretrained(str(self.model_dir))
        pipe.cuda()

        self._model = pipe
        print("[Trellis2Generator] Loaded on CUDA.")

    def unload(self) -> None:
        super().unload()

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        import o_voxel

        pipeline_type = params.get("pipeline_type", "1024_cascade")
        sparse_steps  = int(params.get("sparse_steps", 12))
        shape_steps   = int(params.get("shape_steps", 12))
        tex_steps     = int(params.get("tex_steps", 12))
        seed          = int(params.get("seed", 42))
        faces         = int(params.get("faces", -1))
        texture_size  = int(params.get("texture_size", 4096))

        target_faces = faces if faces > 0 else 1_000_000

        # Load image
        self._report(progress_cb, 5, "Loading image...")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        self._check_cancelled(cancel_event)

        # Forward pass (three-stage diffusion)
        self._report(progress_cb, 10, "Generating 3D structure...")

        stop_evt = threading.Event()
        if progress_cb:
            t = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 10, 85, "Generating 3D structure...", stop_evt, 5.0),
                daemon=True,
            )
            t.start()

        try:
            outputs = self._model.run(
                image,
                seed=seed,
                preprocess_image=True,
                pipeline_type=pipeline_type,
                sparse_structure_sampler_params={"steps": sparse_steps},
                shape_slat_sampler_params={"steps": shape_steps},
                tex_slat_sampler_params={"steps": tex_steps},
            )
        finally:
            stop_evt.set()

        self._check_cancelled(cancel_event)

        # Simplify mesh (nvdiffrast hard limit: 16,777,216 faces)
        self._report(progress_cb, 87, "Simplifying mesh...")
        mesh = outputs[0]
        mesh.simplify(min(target_faces, 16_777_216))

        self._check_cancelled(cancel_event)

        # Bake PBR textures and export GLB
        self._report(progress_cb, 93, "Baking textures & exporting GLB...")
        glb = o_voxel.postprocess.to_glb(
            vertices          = mesh.vertices,
            faces             = mesh.faces,
            attr_volume       = mesh.attrs,
            coords            = mesh.coords,
            attr_layout       = mesh.layout,
            voxel_size        = mesh.voxel_size,
            aabb              = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target = target_faces,
            texture_size      = texture_size,
            remesh            = True,
            remesh_band       = 1,
            remesh_project    = 0,
            verbose           = False,
        )

        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.glb"
        path = self.outputs_dir / name
        glb.export(str(path), extension_webp=True)

        self._report(progress_cb, 100, "Done")
        return path

    # ------------------------------------------------------------------ #
    # Vendor / env setup
    # ------------------------------------------------------------------ #

    def _setup_vendor(self) -> None:
        if not _VENDOR_DIR.exists():
            raise RuntimeError(
                f"[Trellis2Generator] vendor/ directory not found at {_VENDOR_DIR}.\n"
                "Run 'python build_vendor.py' from the extension directory to build it."
            )

        # Import torch first so it registers its DLL directory on Windows.
        # Compiled CUDA extensions in vendor/ depend on torch DLLs — without
        # this, Windows cannot find them even if the path is correct.
        import torch  # noqa: F401

        # Install large packages that cannot be vendored in git (pre-built wheels,
        # no compilation required — just a pip download).
        self._ensure_spconv(torch)
        self._ensure_opencv()

        vendor_str = str(_VENDOR_DIR)
        if vendor_str not in sys.path:
            sys.path.insert(0, vendor_str)

        try:
            from trellis2.pipelines import Trellis2ImageTo3DPipeline  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                f"[Trellis2Generator] vendor/ is incomplete: {exc}\n"
                "Re-run 'python build_vendor.py' to rebuild it."
            ) from exc

    def _ensure_spconv(self, torch) -> None:
        """Install spconv via pip if not already available (pre-built wheel, no compilation)."""
        try:
            import spconv  # noqa: F401
            return
        except (ImportError, OSError):
            pass

        cuda_tag = "cu" + torch.version.cuda.replace(".", "")
        fallbacks = [cuda_tag, "cu124", "cu122", "cu121", "cu120", "cu118"]
        seen = []
        for tag in fallbacks:
            if tag in seen:
                continue
            seen.append(tag)
            pkg = f"spconv-{tag}"
            print(f"[Trellis2Generator] Installing {pkg} via pip...")
            import subprocess
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"[Trellis2Generator] {pkg} installed successfully.")
                return
        raise RuntimeError(
            "[Trellis2Generator] Could not install spconv for any CUDA version. "
            f"Tried: {seen}"
        )

    def _ensure_opencv(self) -> None:
        """Install opencv-python via pip if not already available (pre-built wheel)."""
        try:
            import cv2  # noqa: F401
            return
        except (ImportError, OSError):
            pass

        print("[Trellis2Generator] Installing opencv-python via pip...")
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "opencv-python"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                "[Trellis2Generator] Failed to install opencv-python:\n" + result.stderr
            )
        print("[Trellis2Generator] opencv-python installed successfully.")

    def _setup_env(self) -> None:
        """Set environment variables required by TRELLIS.2 before first import."""
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # flex_gemm is not available on PyPI — use spconv as conv backend instead.
        os.environ.setdefault("SPARSE_CONV_BACKEND", "spconv")

    # ------------------------------------------------------------------ #
    # UI schema
    # ------------------------------------------------------------------ #

    @classmethod
    def params_schema(cls) -> list:
        return [
            {
                "id":      "pipeline_type",
                "label":   "Resolution",
                "type":    "select",
                "options": [
                    {"value": "512",          "label": "512³  (~3 s)"},
                    {"value": "1024",         "label": "1024³  (~17 s)"},
                    {"value": "1024_cascade", "label": "1024³ Cascade (~17 s)"},
                    {"value": "1536_cascade", "label": "1536³ Cascade (~60 s)"},
                ],
                "default": "1024_cascade",
                "tooltip": "Voxel resolution. Higher = more geometry detail but much slower and more VRAM.",
            },
            {
                "id":      "sparse_steps",
                "label":   "Sparse Structure Steps",
                "type":    "int",
                "default": 12,
                "min":     1,
                "max":     50,
                "tooltip": "Diffusion steps for the sparse structure stage. More steps = better structure.",
            },
            {
                "id":      "shape_steps",
                "label":   "Shape SLAT Steps",
                "type":    "int",
                "default": 12,
                "min":     1,
                "max":     50,
                "tooltip": "Diffusion steps for the shape latent stage. More steps = finer geometry.",
            },
            {
                "id":      "tex_steps",
                "label":   "Texture SLAT Steps",
                "type":    "int",
                "default": 12,
                "min":     1,
                "max":     50,
                "tooltip": "Diffusion steps for the texture latent stage. More steps = sharper textures.",
            },
            {
                "id":      "faces",
                "label":   "Max Faces",
                "type":    "int",
                "default": -1,
                "min":     -1,
                "max":     16777216,
                "tooltip": "Target polygon count after simplification. -1 uses 1,000,000.",
            },
            {
                "id":      "texture_size",
                "label":   "Texture Size",
                "type":    "select",
                "options": [
                    {"value": 2048, "label": "2048"},
                    {"value": 4096, "label": "4096"},
                    {"value": 8192, "label": "8192"},
                ],
                "default": 4096,
                "tooltip": "Resolution of the baked PBR texture atlas (base color, roughness, metallic).",
            },
            {
                "id":      "seed",
                "label":   "Seed",
                "type":    "int",
                "default": 42,
                "min":     0,
                "max":     2147483647,
                "tooltip": "Seed for reproducibility. Click shuffle for a random seed.",
            },
        ]
