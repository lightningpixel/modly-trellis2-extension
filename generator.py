"""
TRELLIS.2 extension for Modly.

Reference : https://huggingface.co/microsoft/TRELLIS.2-4B
GitHub    : https://github.com/microsoft/TRELLIS.2

All pure-Python TRELLIS.2 sources used by the extension are bundled in vendor/.
Native/runtime dependencies are expected to be installed by setup.py into the
extension venv before Modly starts the generator subprocess.

The following core runtime dependencies must be available in the extension
venv (installed by setup.py):
    o-voxel        — core O-Voxel representation library
    nvdiffrast     — differentiable rasterizer (NVlabs)
    cumesh         — CUDA mesh utilities
    xformers / flash-attn — sparse attention backend
    spconv         — sparse convolution backend

Optional renderer dependency:
    nvdiffrec      — deferred renderer stack used only by optional rendering paths

To rebuild vendor/:
    python build_vendor.py   (run once with the app's venv active)
"""

import importlib.util
import io
import os
import sys
import time
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from services.generators.base import BaseGenerator, smooth_progress, GenerationCancelled

_EXTENSION_DIR = Path(__file__).parent
_VENDOR_DIR    = _EXTENSION_DIR / "vendor"
_OPTIONAL_NVDIFFREC_ENV = "MODLY_TRELLIS2_INSTALL_NVDIFFREC"
_NATIVE_VENDOR_OVERLAPS = {"nvdiffrast"}


IMAGE_TO_MESH_PARAMS_SCHEMA = [
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


TEXTURE_MESH_PARAMS_SCHEMA = [
    {
        "id":      "pipeline_type",
        "label":   "Texture Resolution",
        "type":    "select",
        "options": [
            {"value": "512",  "label": "512³"},
            {"value": "1024", "label": "1024³"},
        ],
        "default": "1024",
        "tooltip": "Internal voxel/texturing resolution used by the TRELLIS.2 texturing pipeline. 1024 gives better fidelity but costs more VRAM and time.",
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
        "id":      "texture_size",
        "label":   "Texture Size",
        "type":    "select",
        "options": [
            {"value": 2048, "label": "2048"},
            {"value": 4096, "label": "4096"},
            {"value": 8192, "label": "8192"},
        ],
        "default": 4096,
        "tooltip": "Resolution of the exported texture atlas. Higher values improve detail but increase memory and export time.",
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


@dataclass(frozen=True)
class CapabilityConfig:
    node_id: str
    capability_id: str
    display_name: str
    config_file: str
    download_check: str
    input_kind: str
    output_kind: str
    params_schema: list[dict]


CAPABILITIES: dict[str, CapabilityConfig] = {
    "generate": CapabilityConfig(
        node_id="generate",
        capability_id="image-to-mesh",
        display_name="Image to Mesh",
        config_file="pipeline.json",
        download_check="pipeline.json",
        input_kind="image",
        output_kind="mesh",
        params_schema=IMAGE_TO_MESH_PARAMS_SCHEMA,
    ),
    "texture-mesh": CapabilityConfig(
        node_id="texture-mesh",
        capability_id="texture-mesh",
        display_name="Texture Mesh",
        config_file="texturing_pipeline.json",
        download_check="texturing_pipeline.json",
        input_kind="image",
        output_kind="mesh",
        params_schema=TEXTURE_MESH_PARAMS_SCHEMA,
    ),
}


def filtered_vendor_paths() -> list[str]:
    ignored = sorted(name for name in _NATIVE_VENDOR_OVERLAPS if (_VENDOR_DIR / name).exists())
    if ignored:
        raise RuntimeError(
            "[Trellis2Generator] Vendored native overlap directories are not allowed: "
            + ", ".join(ignored)
            + ". Remove them from vendor/ so native packages resolve only from the extension venv."
        )
    return [str(_VENDOR_DIR)]


def module_spec_origin(module_name: str) -> str | None:
    spec = importlib.util.find_spec(module_name)
    return getattr(spec, "origin", None) if spec is not None else None


class Trellis2Generator(BaseGenerator):
    MODEL_ID     = "trellis-2"
    DISPLAY_NAME = "TRELLIS.2"
    VRAM_GB      = 24

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def is_downloaded(self) -> bool:
        return (self.model_dir / self._capability().download_check).exists()

    def load(self) -> None:
        if self._model is not None:
            return

        if not self.is_downloaded():
            self._auto_download()

        self._setup_env()    # must run before vendor imports so SPARSE_CONV_BACKEND is set
        self._setup_vendor()

        capability = self._capability()
        pipeline_cls = self._resolve_pipeline_class(capability)

        print(f"[Trellis2Generator] Loading {capability.capability_id} model from {self.model_dir}...")
        pipe = pipeline_cls.from_pretrained(str(self.model_dir), config_file=capability.config_file)
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
        capability = self._capability()
        if capability.capability_id == "image-to-mesh":
            return self._generate_image_to_mesh(image_bytes, params, progress_cb, cancel_event)
        if capability.capability_id == "texture-mesh":
            return self._generate_texture_mesh(image_bytes, params, progress_cb, cancel_event)
        raise RuntimeError(f"[Trellis2Generator] Unsupported capability '{capability.capability_id}'.")

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

        self._require_runtime_dependency("spconv", "spconv")
        self._require_runtime_dependency("cv2", "opencv-python-headless")
        self._require_runtime_dependency("o_voxel", "o-voxel")
        self._require_runtime_dependency("cumesh", "CuMesh")
        self._require_runtime_dependency("nvdiffrast", "nvdiffrast", allow_vendor=False)

        if importlib.util.find_spec("xformers") is None and importlib.util.find_spec("flash_attn") is None:
            raise RuntimeError(
                "[Trellis2Generator] Missing attention backend in extension venv. "
                "Install xformers or flash-attn from setup.py."
            )

        for vendor_path in filtered_vendor_paths():
            if vendor_path not in sys.path:
                sys.path.append(vendor_path)

        nvdiffrast_origin = module_spec_origin("nvdiffrast")
        if nvdiffrast_origin and str(_VENDOR_DIR) in nvdiffrast_origin:
            raise RuntimeError(
                "[Trellis2Generator] nvdiffrast resolved from vendor/ instead of the extension venv. "
                "Remove the vendored overlap and reinstall the extension."
            )

        try:
            self._resolve_pipeline_class(self._capability())
        except ImportError as exc:
            raise RuntimeError(
                f"[Trellis2Generator] vendor/ is incomplete: {exc}\n"
                "Re-run 'python build_vendor.py' to rebuild it."
            ) from exc

    def _require_runtime_dependency(self, module_name: str, package_name: str, *, allow_vendor: bool = True) -> None:
        origin = module_spec_origin(module_name)
        if origin is not None:
            if not allow_vendor and str(_VENDOR_DIR) in origin:
                raise RuntimeError(
                    f"[Trellis2Generator] Runtime dependency '{module_name}' resolved from vendor/ instead of the extension venv. "
                    f"Reinstall the extension so setup.py can install '{package_name}' into the venv."
                )
            return

        raise RuntimeError(
            f"[Trellis2Generator] Missing runtime dependency '{module_name}' in the extension venv. "
            f"Reinstall the extension so setup.py can install '{package_name}'."
        )

    def _setup_env(self) -> None:
        """Set environment variables required by TRELLIS.2 before first import."""
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # flex_gemm is not available on PyPI — use spconv as conv backend instead.
        os.environ.setdefault("SPARSE_CONV_BACKEND", "spconv")
        if importlib.util.find_spec("xformers") is not None:
            os.environ.setdefault("ATTN_BACKEND", "xformers")
            os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
        elif importlib.util.find_spec("flash_attn") is not None:
            os.environ.setdefault("ATTN_BACKEND", "flash_attn")
            os.environ.setdefault("SPARSE_ATTN_BACKEND", "flash_attn")

    # ------------------------------------------------------------------ #
    # UI schema
    # ------------------------------------------------------------------ #

    @classmethod
    def params_schema(cls) -> list:
        return cls.capability_params_schema("generate")

    @classmethod
    def capability_params_schema(cls, node_id: str) -> list:
        return CAPABILITIES.get(node_id, CAPABILITIES["generate"]).params_schema

    def _capability(self) -> CapabilityConfig:
        node_id = getattr(self.model_dir, "name", "")
        if node_id in CAPABILITIES:
            return CAPABILITIES[node_id]
        if self.download_check:
            for capability in CAPABILITIES.values():
                if capability.download_check == self.download_check:
                    return capability
        return CAPABILITIES["generate"]

    def _resolve_pipeline_class(self, capability: CapabilityConfig):
        if capability.capability_id == "image-to-mesh":
            from trellis2.pipelines import Trellis2ImageTo3DPipeline
            return Trellis2ImageTo3DPipeline
        if capability.capability_id == "texture-mesh":
            from trellis2.pipelines import Trellis2TexturingPipeline
            return Trellis2TexturingPipeline
        raise RuntimeError(f"[Trellis2Generator] No pipeline is configured for '{capability.capability_id}'.")

    def _resolve_mesh_path(self, params: dict) -> Path:
        mesh_path = params.get("mesh_path")
        if not isinstance(mesh_path, str) or not mesh_path.strip():
            raise RuntimeError(
                "[Trellis2Generator] The 'texture-mesh' capability requires a mesh side-input. "
                "Expected params.mesh_path to point to an existing mesh file."
            )

        candidate_path = Path(mesh_path).expanduser()
        search_paths = [candidate_path]
        if not candidate_path.is_absolute():
            workspace_dir = self._runtime_workspace_dir(params)
            if workspace_dir is not None:
                search_paths.insert(0, workspace_dir / candidate_path)

        resolved_path = next((path for path in search_paths if path.exists() and path.is_file()), None)
        if resolved_path is None:
            searched_locations = ", ".join(str(path) for path in search_paths)
            raise RuntimeError(
                f"[Trellis2Generator] Mesh side-input was not found. Checked: {searched_locations}. "
                "Ensure the workflow provides a valid mesh connection."
            )
        return resolved_path

    def _runtime_workspace_dir(self, params: dict) -> Path | None:
        workspace_candidates = [
            params.get("workspace_dir"),
            getattr(self, "workspace_dir", None),
            getattr(self, "workspace", None),
            getattr(self, "runtime_workspace_dir", None),
        ]

        outputs_dir = getattr(self, "outputs_dir", None)
        if outputs_dir is not None:
            workspace_candidates.append(Path(outputs_dir).parent)

        for candidate in workspace_candidates:
            if candidate is None:
                continue
            candidate_path = Path(candidate).expanduser()
            if candidate_path.exists() and candidate_path.is_dir():
                return candidate_path
        return None

    def _load_input_mesh(self, mesh_path: Path):
        import trimesh

        try:
            mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
        except Exception as exc:
            raise RuntimeError(
                f"[Trellis2Generator] Failed to load mesh side-input '{mesh_path}': {exc}"
            ) from exc

        if isinstance(mesh, trimesh.Scene):
            geometries = [geometry for geometry in mesh.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
            if not geometries:
                raise RuntimeError(
                    f"[Trellis2Generator] Mesh side-input '{mesh_path}' did not contain any mesh geometry."
                )
            mesh = trimesh.util.concatenate(geometries)

        if not isinstance(mesh, trimesh.Trimesh):
            raise RuntimeError(
                f"[Trellis2Generator] Mesh side-input '{mesh_path}' could not be normalized into a trimesh.Trimesh instance."
            )

        return mesh

    def _generate_image_to_mesh(
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
        target_faces  = faces if faces > 0 else 1_000_000

        self._report(progress_cb, 5, "Loading image...")
        image = self._load_image(image_bytes)
        self._check_cancelled(cancel_event)

        self._report(progress_cb, 10, "Generating 3D structure...")
        outputs = self._run_with_smoothed_progress(
            progress_cb,
            start=10,
            end=85,
            label="Generating 3D structure...",
            run=lambda: self._model.run(
                image,
                seed=seed,
                preprocess_image=True,
                pipeline_type=pipeline_type,
                sparse_structure_sampler_params={"steps": sparse_steps},
                shape_slat_sampler_params={"steps": shape_steps},
                tex_slat_sampler_params={"steps": tex_steps},
            ),
        )
        self._check_cancelled(cancel_event)

        self._report(progress_cb, 87, "Simplifying mesh...")
        mesh = outputs[0]
        mesh.simplify(min(target_faces, 16_777_216))
        self._check_cancelled(cancel_event)

        self._report(progress_cb, 93, "Baking textures & exporting GLB...")
        try:
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
        except ModuleNotFoundError as exc:
            if exc.name in {"nvdiffrec", "nvdiffrec_render"}:
                raise RuntimeError(
                    "[Trellis2Generator] Optional renderer dependency 'nvdiffrec' is not installed in the extension venv. "
                    f"Core setup intentionally skips it by default. Re-run setup with {_OPTIONAL_NVDIFFREC_ENV}=1 "
                    "if this renderer path is required."
                ) from exc
            raise

        output_path = self._next_output_path()
        glb.export(str(output_path), extension_webp=True)
        self._report(progress_cb, 100, "Done")
        return output_path

    def _generate_texture_mesh(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        mesh_path = self._resolve_mesh_path(params)
        pipeline_type = str(params.get("pipeline_type", "1024"))
        if pipeline_type not in {"512", "1024"}:
            raise RuntimeError(
                f"[Trellis2Generator] Unsupported texture pipeline_type '{pipeline_type}'. Expected '512' or '1024'."
            )

        tex_steps = int(params.get("tex_steps", 12))
        seed = int(params.get("seed", 42))
        texture_size = int(params.get("texture_size", 4096))
        resolution = int(pipeline_type)

        self._report(progress_cb, 5, "Loading image...")
        image = self._load_image(image_bytes)
        self._check_cancelled(cancel_event)

        self._report(progress_cb, 10, "Loading mesh...")
        mesh = self._load_input_mesh(mesh_path)
        self._check_cancelled(cancel_event)

        self._report(progress_cb, 15, "Generating textures...")
        textured_mesh = self._run_with_smoothed_progress(
            progress_cb,
            start=15,
            end=92,
            label="Generating textures...",
            run=lambda: self._model.run(
                mesh,
                image,
                seed=seed,
                preprocess_image=True,
                resolution=resolution,
                texture_size=texture_size,
                tex_slat_sampler_params={"steps": tex_steps},
            ),
        )
        self._check_cancelled(cancel_event)

        output_path = self._next_output_path()
        self._report(progress_cb, 95, "Exporting textured GLB...")
        textured_mesh.export(str(output_path), file_type="glb")
        self._report(progress_cb, 100, "Done")
        return output_path

    def _load_image(self, image_bytes: bytes):
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def _run_with_smoothed_progress(
        self,
        progress_cb: Optional[Callable[[int, str], None]],
        *,
        start: int,
        end: int,
        label: str,
        run: Callable[[], object],
    ):
        stop_evt = threading.Event()
        if progress_cb:
            thread = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, start, end, label, stop_evt, 5.0),
                daemon=True,
            )
            thread.start()

        try:
            return run()
        finally:
            stop_evt.set()

    def _next_output_path(self) -> Path:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.glb"
        return self.outputs_dir / name
