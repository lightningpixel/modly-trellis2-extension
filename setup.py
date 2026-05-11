"""
TRELLIS.2 extension setup for Modly.

Creates an isolated extension venv and installs the runtime dependencies needed
by generator.py when Modly runs the extension in subprocess mode.

Install success only means the local Python/native runtime is ready. First load
still requires Hugging Face access to the gated runtime dependencies documented
in README.md and to the TRELLIS.2 model snapshot itself.

Accepted invocation forms:

    python setup.py '{"python_exe":"...","ext_dir":"...","gpu_sm":86,"cuda_version":124}'
    python setup.py <python_exe> <ext_dir> <gpu_sm> [cuda_version]
    python setup.py --dry-run-plan [gpu_sm] [cuda_version]
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


FLASH_ATTN_VERSION = "2.7.3"
SPCONV_SOURCE_REPO = "https://github.com/traveller59/spconv.git"
SPCONV_SOURCE_REF = "v2.3.8"
CUMM_SOURCE_REPO = "https://github.com/FindDefinition/cumm.git"
CUMM_SOURCE_REF = "v0.7.11"
NVDIFFRAST_SOURCE_REPO = "https://github.com/NVlabs/nvdiffrast.git"
NVDIFFRAST_SOURCE_REF = "v0.4.0"
NVDIFFREC_SOURCE_REPO = "https://github.com/JeffreyXiang/nvdiffrec.git"
NVDIFFREC_SOURCE_REF = "b296927cc7fd01c2ac1087c8065c4d7248f72da4"
CUMESH_SOURCE_REPO = "https://github.com/JeffreyXiang/CuMesh.git"
# Pinned from CuMesh HEAD validated for this integration on 2026-04-13.
CUMESH_SOURCE_REF = "cf1a2f07304b5fe388ed86a16e4a0474599df914"
MIP_SPLATTING_SOURCE_REPO = "https://github.com/autonomousvision/mip-splatting.git"
# Pinned from mip-splatting HEAD validated for the TRELLIS Gaussian renderer API on 2026-05-11.
MIP_SPLATTING_SOURCE_REF = "dda02ab5ecf45d6edb8c540d9bb65c7e451345a9"
MIP_SPLATTING_DIFF_GAUSSIAN_SUBDIRECTORY = "submodules/diff-gaussian-rasterization"
TRELLIS2_SOURCE_REPO = "https://github.com/microsoft/TRELLIS.2.git"
TRELLIS2_SOURCE_REF = "5565d240c4a494caaf9ece7a554542b76ffa36d3"
TRELLIS_SOURCE_REPO = "https://github.com/microsoft/TRELLIS.git"
TRELLIS_SOURCE_REF = "442aa1e1afb9014e80681d3bf604e8d728a86ee7"
O_VOXEL_SUBDIRECTORY = "o-voxel"
O_VOXEL_SUPPORT_PACKAGES = ("plyfile", "zstandard")
PYTHON_RUNTIME_DEPENDENCIES = (
    "Pillow",
    "numpy",
    "opencv-python-headless",
    "huggingface_hub",
    "transformers>=4.46.0",
    "accelerate",
    "safetensors",
    "imageio",
    "imageio-ffmpeg",
    "easydict",
    "tqdm",
    "trimesh",
    "scipy",
    "scikit-image",
    "kornia",
    "timm",
    "ninja",
    "xatlas",
    "pyvista",
    "pymeshfix",
    "igraph",
)
OPTIONAL_NVDIFFREC_ENV = "MODLY_TRELLIS2_INSTALL_NVDIFFREC"
CUMM_CUDA_DISCOVERY_PATCH_MARKER = "modly_trellis2_cuda_root_override"
CUMM_SUPPORTED_CUDA_ARCHES = frozenset(
    {
        "5.2",
        "6.0",
        "6.1",
        "7.0",
        "7.2",
        "7.5",
        "8.0",
        "8.6",
        "8.7",
        "8.9",
        "9.0",
        "5.2+PTX",
        "6.0+PTX",
        "6.1+PTX",
        "7.0+PTX",
        "7.2+PTX",
        "7.5+PTX",
        "8.0+PTX",
        "8.6+PTX",
        "8.7+PTX",
        "8.9+PTX",
        "9.0+PTX",
    }
)
CUMM_MAX_SUPPORTED_SM = 90
CUMM_MAX_SUPPORTED_ARCH = "9.0"
CUMM_FORWARD_COMPAT_ARCH = "9.0+PTX"


@dataclass(frozen=True)
class PlatformInstallPlan:
    name: str
    attention_backends: tuple[tuple[str, str], ...]
    optional_renderer_default: bool = False


def is_windows() -> bool:
    return platform.system() == "Windows"


def is_linux() -> bool:
    return platform.system() == "Linux"


def machine_arch() -> str:
    return platform.machine().lower()


def platform_label() -> str:
    return f"{platform.system()} {machine_arch()}"


def is_linux_arm64() -> bool:
    return is_linux() and machine_arch() in {"aarch64", "arm64"}


def cuda_arch_string_from_sm(gpu_sm: int) -> str | None:
    if gpu_sm <= 0:
        return None
    major, minor = divmod(gpu_sm, 10)
    return f"{major}.{minor}"


def resolve_cumm_cuda_arch(gpu_sm: int) -> tuple[str | None, str]:
    requested_arch = cuda_arch_string_from_sm(gpu_sm)
    if requested_arch is None:
        return None, "gpu_sm was not provided; upstream CUDA arch autodetection will be used"
    if requested_arch in CUMM_SUPPORTED_CUDA_ARCHES:
        return requested_arch, f"SM {gpu_sm} maps directly to supported cumm arch {requested_arch}"
    if gpu_sm > CUMM_MAX_SUPPORTED_SM:
        return (
            CUMM_FORWARD_COMPAT_ARCH,
            f"SM {gpu_sm} maps to unsupported arch {requested_arch}; clamping to {CUMM_FORWARD_COMPAT_ARCH} because cumm {CUMM_SOURCE_REF} supports up to {CUMM_MAX_SUPPORTED_ARCH} and PTX enables forward compatibility",
        )
    return requested_arch, f"SM {gpu_sm} maps to arch {requested_arch}; no compatibility remap applied"


def plan_platform_install() -> PlatformInstallPlan:
    if is_linux_arm64():
        return PlatformInstallPlan(
            name="linux-arm64",
            attention_backends=(("flash_attn", f"flash-attn=={FLASH_ATTN_VERSION}"),),
            optional_renderer_default=False,
        )

    attention_backends = (("xformers", "xformers"),)
    if not is_windows():
        attention_backends += (("flash_attn", f"flash-attn=={FLASH_ATTN_VERSION}"),)
    return PlatformInstallPlan(
        name=f"{platform.system().lower()}-{machine_arch()}",
        attention_backends=attention_backends,
        optional_renderer_default=True,
    )


def describe_install_plan(gpu_sm: int, cuda_version: int) -> dict[str, object]:
    torch_pkgs, torch_index, cuda_tag = select_torch(gpu_sm, cuda_version)
    plan = plan_platform_install()
    attention_backend_install_args = {
        backend: (["--no-build-isolation"] if attention_backend_needs_no_build_isolation(backend, requirement) else [])
        for backend, requirement in plan.attention_backends
    }
    description = {
        "platform": platform_label(),
        "plan": plan.name,
        "spconv_strategy": "source" if is_linux_arm64() else "prebuilt",
        "attention_backends": [backend for backend, _ in plan.attention_backends],
        "attention_backend_install_args": attention_backend_install_args,
        "torch_packages": torch_pkgs,
        "torch_index": torch_index,
        "cuda_tag": cuda_tag,
        "optional_renderer_default": plan.optional_renderer_default,
    }
    if is_linux_arm64():
        _, source_build_diagnostics = source_build_env_overrides(gpu_sm=gpu_sm, cuda_version=cuda_version)
        description["source_build_env"] = {
            key: value for key, value in source_build_diagnostics.items() if key != "cumm_cuda_arch"
        }
        if description["source_build_env"].get("PATH"):
            description["source_build_env"]["PATH"] = (
                f"<extension-venv-bin>{os.pathsep}{description['source_build_env']['PATH']}"
            )
        else:
            description["source_build_env"]["PATH"] = f"<extension-venv-bin>{os.pathsep}${{PATH}}"
        description["cumm_cuda_arch"] = source_build_diagnostics["cumm_cuda_arch"]
    return description


def venv_bin(venv: Path, name: str) -> Path:
    if is_windows():
        suffix = ".exe" if not name.endswith(".exe") else ""
        return venv / "Scripts" / f"{name}{suffix}"
    return venv / "bin" / name


def prepend_directory_to_path(env: dict[str, str], directory: Path) -> dict[str, str]:
    updated_env = env.copy()
    directory_str = str(directory)
    existing_parts = [part for part in updated_env.get("PATH", "").split(os.pathsep) if part]
    updated_env["PATH"] = os.pathsep.join([directory_str, *[part for part in existing_parts if part != directory_str]])
    return updated_env


def prepend_env_path(env: dict[str, str], key: str, *entries: Path) -> None:
    values = [str(entry) for entry in entries if str(entry)]
    if not values:
        return
    existing_parts = [part for part in env.get(key, "").split(os.pathsep) if part]
    env[key] = os.pathsep.join([*values, *[part for part in existing_parts if part not in values]])


def cuda_version_to_toolkit_version(cuda_version: int) -> str | None:
    if cuda_version <= 0:
        return None
    major, minor = divmod(cuda_version, 10)
    return f"{major}.{minor}"


def candidate_cuda_toolkit_roots(cuda_version: int, env: dict[str, str] | None = None) -> list[Path]:
    source_env = env or os.environ
    candidates: list[Path] = []
    for key in ("MODLY_TRELLIS2_CUDA_TOOLKIT_ROOT", "CUDA_HOME", "CUDA_PATH"):
        raw_value = source_env.get(key)
        if raw_value:
            candidates.append(Path(raw_value).expanduser())

    toolkit_version = cuda_version_to_toolkit_version(cuda_version)
    if toolkit_version:
        candidates.append(Path(f"/usr/local/cuda-{toolkit_version}"))
    candidates.append(Path("/usr/local/cuda"))

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(candidate)
    return deduped


def resolve_cuda_toolkit_root(cuda_version: int, env: dict[str, str] | None = None) -> Path | None:
    for candidate in candidate_cuda_toolkit_roots(cuda_version, env=env):
        if candidate.exists():
            return candidate
    return None


def cuda_toolkit_library_dirs(toolkit_root: Path) -> tuple[Path, ...]:
    candidates = [toolkit_root / "lib64"]
    if is_linux_arm64():
        candidates.extend(
            [
                toolkit_root / "targets" / "aarch64-linux" / "lib",
                toolkit_root / "targets" / "sbsa-linux" / "lib",
            ]
        )
    elif is_linux():
        candidates.append(toolkit_root / "targets" / "x86_64-linux" / "lib")
    return tuple(path for path in candidates if path.exists())


def source_build_env_overrides(
    *,
    gpu_sm: int,
    cuda_version: int,
    build_env: dict[str, str] | None = None,
    venv: Path | None = None,
) -> tuple[dict[str, str], dict[str, object]]:
    source_env = dict(build_env or os.environ)
    venv_bin_dir: Path | None = None
    diagnostics: dict[str, object] = {
        "CUMM_DISABLE_JIT": "1",
        "SPCONV_DISABLE_JIT": "1",
    }
    source_env.setdefault("CUMM_DISABLE_JIT", "1")
    source_env.setdefault("SPCONV_DISABLE_JIT", "1")

    requested_arch = cuda_arch_string_from_sm(gpu_sm)
    cuda_arch, arch_reason = resolve_cumm_cuda_arch(gpu_sm)
    diagnostics["cumm_cuda_arch"] = {
        "requested": requested_arch,
        "resolved": cuda_arch,
        "reason": arch_reason,
    }
    if cuda_arch:
        source_env.setdefault("CUMM_CUDA_ARCH_LIST", cuda_arch)
        diagnostics["CUMM_CUDA_ARCH_LIST"] = source_env["CUMM_CUDA_ARCH_LIST"]

    if venv is not None:
        venv_bin_dir = venv_bin(venv, "python").parent
        source_env = prepend_directory_to_path(source_env, venv_bin_dir)
        diagnostics["PATH"] = f"<extension-venv-bin>{os.pathsep}${{PATH}}"

    toolkit_root = resolve_cuda_toolkit_root(cuda_version, env=source_env)
    diagnostics["cuda_toolkit_root_candidates"] = [str(path) for path in candidate_cuda_toolkit_roots(cuda_version, env=source_env)]
    if toolkit_root is None:
        diagnostics["cuda_toolkit_root"] = None
        return source_env, diagnostics

    source_env["CUDA_HOME"] = str(toolkit_root)
    source_env["CUDA_PATH"] = str(toolkit_root)
    source_env["CUDACXX"] = str(toolkit_root / "bin" / "nvcc")
    if venv_bin_dir is not None:
        prepend_env_path(source_env, "PATH", venv_bin_dir, toolkit_root / "bin")
    else:
        prepend_env_path(source_env, "PATH", toolkit_root / "bin")

    include_dir = toolkit_root / "include"
    prepend_env_path(source_env, "CPATH", include_dir)
    prepend_env_path(source_env, "C_INCLUDE_PATH", include_dir)
    prepend_env_path(source_env, "CPLUS_INCLUDE_PATH", include_dir)

    library_dirs = cuda_toolkit_library_dirs(toolkit_root)
    if library_dirs:
        prepend_env_path(source_env, "LIBRARY_PATH", *library_dirs)
        prepend_env_path(source_env, "LD_LIBRARY_PATH", *library_dirs)

    diagnostics["cuda_toolkit_root"] = str(toolkit_root)
    diagnostics["CUDA_HOME"] = source_env["CUDA_HOME"]
    diagnostics["CUDA_PATH"] = source_env["CUDA_PATH"]
    diagnostics["CUDACXX"] = source_env["CUDACXX"]
    diagnostics["PATH"] = (
        f"<extension-venv-bin>{os.pathsep}{toolkit_root / 'bin'}{os.pathsep}${{PATH}}"
        if venv is not None
        else f"{toolkit_root / 'bin'}{os.pathsep}${{PATH}}"
    )
    diagnostics["CPATH"] = f"{include_dir}{os.pathsep}${{CPATH}}"
    diagnostics["C_INCLUDE_PATH"] = f"{include_dir}{os.pathsep}${{C_INCLUDE_PATH}}"
    diagnostics["CPLUS_INCLUDE_PATH"] = f"{include_dir}{os.pathsep}${{CPLUS_INCLUDE_PATH}}"
    if library_dirs:
        joined_libs = os.pathsep.join(str(path) for path in library_dirs)
        diagnostics["LIBRARY_PATH"] = f"{joined_libs}{os.pathsep}${{LIBRARY_PATH}}"
        diagnostics["LD_LIBRARY_PATH"] = f"{joined_libs}{os.pathsep}${{LD_LIBRARY_PATH}}"
    diagnostics["source_build_hotfixes"] = [
        "patch installed cumm/common.py on Linux ARM64 so CUDA include/lib discovery honors CUDA_HOME/CUDA_PATH before /usr/local/cuda"
    ]
    return source_env, diagnostics


def patch_installed_cumm_cuda_discovery(venv: Path) -> None:
    cumm_common = Path(
        subprocess.check_output(
            [str(venv_bin(venv, "python")), "-c", "import cumm.common; print(cumm.common.__file__)"],
            text=True,
        ).strip()
    )
    original = cumm_common.read_text()
    if CUMM_CUDA_DISCOVERY_PATCH_MARKER in original:
        print(f"[setup] cumm CUDA discovery patch already present at {cumm_common}")
        return

    old = """        else:\n            try:\n                nvcc_path = subprocess.check_output([\"which\", \"nvcc\"\n                                                    ]).decode(\"utf-8\").strip()\n                lib = Path(nvcc_path).parent.parent / \"lib\"\n                include = Path(nvcc_path).parent.parent / \"targets/x86_64-linux/include\"\n                if lib.exists() and include.exists():\n                    if (lib / \"libcudart.so\").exists() and (include / \"cuda.h\").exists():\n                        # should be nvidia conda package\n                        _CACHED_CUDA_INCLUDE_LIB = ([include], lib)\n                        return _CACHED_CUDA_INCLUDE_LIB\n            except:\n                pass \n\n            linux_cuda_root = Path(\"/usr/local/cuda\")\n            include = linux_cuda_root / f\"include\"\n            lib64 = linux_cuda_root / f\"lib64\"\n            assert linux_cuda_root.exists(), f\"can't find cuda in {linux_cuda_root} install via cuda installer or conda first.\"\n"""
    new = """        else:\n            try:\n                nvcc_path = subprocess.check_output([\"which\", \"nvcc\"\n                                                    ]).decode(\"utf-8\").strip()\n                linux_cuda_root = Path(nvcc_path).parent.parent\n                include_candidates = [\n                    linux_cuda_root / \"targets/x86_64-linux/include\",\n                    linux_cuda_root / \"targets/aarch64-linux/include\",\n                    linux_cuda_root / \"targets/sbsa-linux/include\",\n                    linux_cuda_root / \"include\",\n                ]\n                lib_candidates = [\n                    linux_cuda_root / \"lib\",\n                    linux_cuda_root / \"lib64\",\n                    linux_cuda_root / \"targets/x86_64-linux/lib\",\n                    linux_cuda_root / \"targets/aarch64-linux/lib\",\n                    linux_cuda_root / \"targets/sbsa-linux/lib\",\n                ]\n                for include in include_candidates:\n                    for lib in lib_candidates:\n                        if (lib / \"libcudart.so\").exists() and (include / \"cuda.h\").exists():\n                            # should be nvidia conda package or an explicitly selected toolkit root\n                            _CACHED_CUDA_INCLUDE_LIB = ([include], lib)\n                            return _CACHED_CUDA_INCLUDE_LIB\n            except:\n                pass \n\n            linux_cuda_roots = []\n            for env_name in (\"CUDA_HOME\", \"CUDA_PATH\"):\n                env_value = os.getenv(env_name)\n                if env_value:\n                    linux_cuda_roots.append(Path(env_value))\n            linux_cuda_roots.append(Path(\"/usr/local/cuda\"))\n            for linux_cuda_root in linux_cuda_roots:\n                include_candidates = [\n                    linux_cuda_root / \"include\",\n                    linux_cuda_root / \"targets/x86_64-linux/include\",\n                    linux_cuda_root / \"targets/aarch64-linux/include\",\n                    linux_cuda_root / \"targets/sbsa-linux/include\",\n                ]\n                lib_candidates = [\n                    linux_cuda_root / \"lib64\",\n                    linux_cuda_root / \"lib\",\n                    linux_cuda_root / \"targets/x86_64-linux/lib\",\n                    linux_cuda_root / \"targets/aarch64-linux/lib\",\n                    linux_cuda_root / \"targets/sbsa-linux/lib\",\n                ]\n                for include in include_candidates:\n                    for lib64 in lib_candidates:\n                        if (lib64 / \"libcudart.so\").exists() and (include / \"cuda.h\").exists():\n                            # modly_trellis2_cuda_root_override: honor explicit CUDA root on Linux ARM64 before /usr/local/cuda\n                            _CACHED_CUDA_INCLUDE_LIB = ([include], lib64)\n                            return _CACHED_CUDA_INCLUDE_LIB\n            linux_cuda_root = Path(\"/usr/local/cuda\")\n            include = linux_cuda_root / f\"include\"\n            lib64 = linux_cuda_root / f\"lib64\"\n            assert linux_cuda_root.exists(), f\"can't find cuda in {linux_cuda_root} install via cuda installer or conda first.\"\n"""
    if old not in original:
        raise RuntimeError(f"Unable to patch cumm CUDA discovery at {cumm_common}; upstream layout changed.")
    cumm_common.write_text(original.replace(old, new, 1))
    print(f"[setup] Patched cumm CUDA discovery at {cumm_common} to honor explicit CUDA toolkit roots on Linux ARM64.")


def run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    print("[setup] $", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, check=True, env=env, cwd=str(cwd) if cwd else None)


def pip(venv: Path, *args: str, env: dict[str, str] | None = None) -> None:
    run([str(venv_bin(venv, "pip")), *args], env=env)


def pip_install(
    venv: Path,
    *packages: str,
    env: dict[str, str] | None = None,
    no_build_isolation: bool = False,
) -> None:
    cmd = ["install"]
    if no_build_isolation:
        cmd.append("--no-build-isolation")
    cmd.extend(packages)
    pip(venv, *cmd, env=env)


def python(venv: Path, *args: str, env: dict[str, str] | None = None) -> None:
    run([str(venv_bin(venv, "python")), *args], env=env)


def native_install_error(package_name: str, attempted_ref: str, exc: Exception) -> RuntimeError:
    return RuntimeError(
        f"Failed to install native dependency '{package_name}'.\n"
        f"Platform: {platform_label()}\n"
        f"Attempted ref/version: {attempted_ref}\n"
        "Checks: verify CUDA toolkit availability, nvcc on PATH, compiler toolchain support, "
        f"and torch/CUDA compatibility for this environment.\nCause: {exc}"
    )


def clone_repo(dest: Path, repo: str, *, ref: str | None = None, recursive: bool = False) -> Path:
    run(["git", "clone", repo, str(dest)])
    if ref:
        run(["git", "checkout", ref], cwd=dest)
    if recursive:
        run(["git", "submodule", "update", "--init", "--recursive"], cwd=dest)
    return dest


def install_from_repo(
    venv: Path,
    tmpdir: Path,
    folder_name: str,
    repo: str,
    *,
    ref: str,
    recursive: bool = False,
    subdirectory: str | None = None,
    env: dict[str, str] | None = None,
    no_deps: bool = False,
) -> None:
    try:
        checkout = clone_repo(tmpdir / folder_name, repo, ref=ref, recursive=recursive)
        package_dir = checkout / subdirectory if subdirectory else checkout
        cmd = ["install", "--no-build-isolation"]
        if no_deps:
            cmd.append("--no-deps")
        cmd.append(str(package_dir))
        pip(venv, *cmd, env=env)
    except (subprocess.CalledProcessError, RuntimeError) as exc:
        raise native_install_error(folder_name, ref, exc) from exc


def install_packages_with_diagnostics(
    venv: Path,
    package_name: str,
    attempted_ref: str,
    *packages: str,
    env: dict[str, str] | None = None,
    no_build_isolation: bool = False,
) -> None:
    try:
        pip_install(venv, *packages, env=env, no_build_isolation=no_build_isolation)
    except subprocess.CalledProcessError as exc:
        raise native_install_error(package_name, attempted_ref, exc) from exc


def attention_backend_needs_no_build_isolation(backend_name: str, requirement: str) -> bool:
    return is_linux_arm64() and backend_name == "flash_attn" and requirement == f"flash-attn=={FLASH_ATTN_VERSION}"


def uninstall_packages(venv: Path, *packages: str) -> None:
    if not packages:
        return
    pip(venv, "uninstall", "-y", *packages)


def smoke_check_spconv(venv: Path, *, env: dict[str, str] | None = None) -> None:
    print("[setup] Verifying spconv import ...")
    python(
        venv,
        "-c",
        "import spconv.pytorch as spconv; print('[setup] spconv import OK:', getattr(spconv, '__version__', 'unknown'))",
        env=env,
    )


def install_prebuilt_spconv(venv: Path, cuda_tag: str) -> None:
    fallbacks = [cuda_tag, "cu128", "cu124", "cu122", "cu121", "cu120", "cu118"]
    tried: list[str] = []
    last_error: subprocess.CalledProcessError | None = None
    for tag in fallbacks:
        pkg = f"spconv-{tag}"
        if pkg in tried:
            continue
        tried.append(pkg)
        try:
            pip(venv, "install", pkg)
            print(f"[setup] Installed {pkg}.")
            smoke_check_spconv(venv)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            print(f"[setup] {pkg} not available for this environment, trying next fallback.")
    raise RuntimeError(
        f"Failed to install native dependency 'spconv'.\n"
        f"Platform: {platform_label()}\n"
        f"Attempted ref/version: {', '.join(tried)}\n"
        "Checks: verify CUDA toolkit availability, nvcc on PATH, compiler toolchain support, "
        "and torch/CUDA compatibility for this environment."
    ) from last_error


def install_spconv_from_source(venv: Path, gpu_sm: int, cuda_version: int, build_env: dict[str, str]) -> None:
    source_env, source_build_diagnostics = source_build_env_overrides(
        gpu_sm=gpu_sm,
        cuda_version=cuda_version,
        build_env=build_env,
        venv=venv,
    )
    cumm_cuda_arch = source_build_diagnostics["cumm_cuda_arch"]
    requested_arch = cumm_cuda_arch["requested"]
    cuda_arch = cumm_cuda_arch["resolved"]
    arch_reason = cumm_cuda_arch["reason"]
    if cuda_arch:
        print(
            f"[setup] Resolved CUMM_CUDA_ARCH_LIST from gpu_sm={gpu_sm}: "
            f"requested={requested_arch} resolved={cuda_arch}."
        )
        print(f"[setup] CUMM arch mapping reason: {arch_reason}")
    else:
        print(f"[setup] {arch_reason}.")

    print("[setup] Linux ARM64 detected. Falling back to source install for cumm + spconv.")
    print(
        "[setup] Forcing non-JIT package builds so cumm/spconv install bundled headers and runtime assets "
        "from the temporary source clones."
    )
    print(f"[setup] Source build PATH begins with: {source_env['PATH'].split(os.pathsep)[0]}")
    if source_build_diagnostics.get("cuda_toolkit_root"):
        print(f"[setup] Steering cumm/spconv source builds to CUDA toolkit root: {source_build_diagnostics['cuda_toolkit_root']}")
        print(f"[setup] CUDACXX={source_env['CUDACXX']}")
        print(f"[setup] CPATH begins with: {source_env['CPATH'].split(os.pathsep)[0]}")
        if source_env.get("LIBRARY_PATH"):
            print(f"[setup] LIBRARY_PATH begins with: {source_env['LIBRARY_PATH'].split(os.pathsep)[0]}")
    else:
        print("[setup] WARNING: no CUDA toolkit root was resolved; source builds will rely on ambient CUDA discovery.")
    uninstall_packages(venv, "spconv", "cumm")
    install_packages_with_diagnostics(
        venv,
        "spconv-build-prereqs",
        "pccm>=0.4.16, ccimport>=0.4.4, pybind11>=2.6.0, fire",
        "pccm>=0.4.16",
        "ccimport>=0.4.4",
        "pybind11>=2.6.0",
        "fire",
        env=source_env,
    )

    with tempfile.TemporaryDirectory(prefix="trellis2-spconv-") as tmp:
        tmpdir = Path(tmp)
        install_from_repo(
            venv,
            tmpdir,
            "cumm",
            CUMM_SOURCE_REPO,
            ref=CUMM_SOURCE_REF,
            env=source_env,
            no_deps=True,
        )
        patch_installed_cumm_cuda_discovery(venv)
        install_from_repo(
            venv,
            tmpdir,
            "spconv",
            SPCONV_SOURCE_REPO,
            ref=SPCONV_SOURCE_REF,
            env=source_env,
            no_deps=True,
        )

    smoke_check_spconv(venv, env=source_env)


def install_spconv(venv: Path, cuda_tag: str, gpu_sm: int, build_env: dict[str, str]) -> None:
    if is_linux_arm64():
        cuda_version = int(cuda_tag[2:]) if cuda_tag.startswith("cu") else 0
        install_spconv_from_source(venv, gpu_sm, cuda_version, build_env)
        return

    install_prebuilt_spconv(venv, cuda_tag)


def install_attention_backend(venv: Path, plan: PlatformInstallPlan) -> str:
    failures: list[str] = []
    for backend_name, requirement in plan.attention_backends:
        try:
            pip_install(
                venv,
                requirement,
                no_build_isolation=attention_backend_needs_no_build_isolation(backend_name, requirement),
            )
            print(f"[setup] Installed {backend_name} attention backend.")
            return backend_name
        except subprocess.CalledProcessError as exc:
            failures.append(str(native_install_error(backend_name, requirement, exc)))
            print(f"[setup] {backend_name} install failed; trying next supported backend.")

    raise RuntimeError(
        "No supported sparse attention backend could be installed for this platform.\n"
        f"Platform: {platform_label()}\n"
        f"Attempted backends: {', '.join(requirement for _, requirement in plan.attention_backends)}\n"
        "Core generation cannot proceed without a supported sparse attention backend.\n\n"
        + "\n\n".join(failures)
    )


def resolve_native_build_env(
    venv: Path,
    *,
    gpu_sm: int,
    cuda_version: int,
    build_env: dict[str, str],
) -> tuple[dict[str, str], dict[str, object] | None]:
    if not is_linux_arm64():
        return build_env, None

    native_env, diagnostics = source_build_env_overrides(
        gpu_sm=gpu_sm,
        cuda_version=cuda_version,
        build_env=build_env,
        venv=venv,
    )
    return native_env, diagnostics


def install_core_native_dependencies(venv: Path, tmpdir: Path, build_env: dict[str, str]) -> None:
    print("[setup] Installing core CUDA/native runtime packages ...")
    install_from_repo(
        venv,
        tmpdir,
        "mip-splatting",
        MIP_SPLATTING_SOURCE_REPO,
        ref=MIP_SPLATTING_SOURCE_REF,
        recursive=True,
        subdirectory=MIP_SPLATTING_DIFF_GAUSSIAN_SUBDIRECTORY,
        env=build_env,
    )
    install_from_repo(
        venv,
        tmpdir,
        "nvdiffrast",
        NVDIFFRAST_SOURCE_REPO,
        ref=NVDIFFRAST_SOURCE_REF,
        env=build_env,
    )
    install_from_repo(
        venv,
        tmpdir,
        "cumesh",
        CUMESH_SOURCE_REPO,
        ref=CUMESH_SOURCE_REF,
        recursive=True,
        env=build_env,
    )
    install_from_repo(
        venv,
        tmpdir,
        "o-voxel",
        TRELLIS2_SOURCE_REPO,
        ref=TRELLIS2_SOURCE_REF,
        recursive=True,
        subdirectory=O_VOXEL_SUBDIRECTORY,
        env=build_env,
        no_deps=True,
    )
    install_packages_with_diagnostics(
        venv,
        "o-voxel-support-packages",
        ", ".join(O_VOXEL_SUPPORT_PACKAGES),
        *O_VOXEL_SUPPORT_PACKAGES,
    )


def should_install_optional_nvdiffrec(plan: PlatformInstallPlan) -> tuple[bool, bool]:
    raw_value = os.environ.get(OPTIONAL_NVDIFFREC_ENV)
    if raw_value is None:
        return plan.optional_renderer_default, False
    enabled = raw_value.strip().lower() in {"1", "true", "yes", "on"}
    return enabled, True


def install_optional_native_dependencies(
    venv: Path,
    tmpdir: Path,
    build_env: dict[str, str],
    plan: PlatformInstallPlan,
) -> None:
    should_install, explicitly_requested = should_install_optional_nvdiffrec(plan)
    if not should_install:
        print(
            "[setup] Skipping optional nvdiffrec renderer install. "
            f"Set {OPTIONAL_NVDIFFREC_ENV}=1 to install it when the optional renderer path is needed."
        )
        return

    print("[setup] Installing optional renderer dependency nvdiffrec ...")
    try:
        install_from_repo(
            venv,
            tmpdir,
            "nvdiffrec",
            NVDIFFREC_SOURCE_REPO,
            ref=NVDIFFREC_SOURCE_REF,
            env=build_env,
        )
    except RuntimeError as exc:
        if explicitly_requested:
            raise
        print(f"[setup] Optional nvdiffrec install failed but core setup can continue: {exc}")


def select_torch(gpu_sm: int, cuda_version: int) -> tuple[list[str], str, str]:
    if gpu_sm >= 100 or cuda_version >= 128:
        return (["torch==2.7.0", "torchvision==0.22.0"], "https://download.pytorch.org/whl/cu128", "cu128")
    if gpu_sm == 0 or gpu_sm >= 70:
        return (["torch==2.6.0", "torchvision==0.21.0"], "https://download.pytorch.org/whl/cu124", "cu124")
    return (["torch==2.5.1", "torchvision==0.20.1"], "https://download.pytorch.org/whl/cu118", "cu118")


def install_python_runtime_dependencies(venv: Path) -> None:
    pip(venv, "install", *PYTHON_RUNTIME_DEPENDENCIES)


def setup(python_exe: str, ext_dir: Path, gpu_sm: int, cuda_version: int = 0) -> None:
    venv = ext_dir / "venv"
    build_env = os.environ.copy()
    build_env.setdefault("CUDAFLAGS", "-allow-unsupported-compiler")
    build_env.setdefault("CMAKE_CUDA_FLAGS", "-allow-unsupported-compiler")
    plan = plan_platform_install()

    print(f"[setup] Platform install plan: {plan.name} ({platform_label()})")
    print(f"[setup] Creating venv at {venv} ...")
    run([python_exe, "-m", "venv", str(venv)])
    pip(venv, "install", "--upgrade", "pip", "setuptools", "wheel")

    torch_pkgs, torch_index, cuda_tag = select_torch(gpu_sm, cuda_version)
    print(f"[setup] Installing PyTorch from {torch_index} ...")
    pip(venv, "install", *torch_pkgs, "--index-url", torch_index)

    print("[setup] Installing Python runtime dependencies ...")
    install_python_runtime_dependencies(venv)

    install_spconv(venv, cuda_tag, gpu_sm, build_env)
    chosen_attention_backend = install_attention_backend(venv, plan)
    print(f"[setup] Selected sparse attention backend: {chosen_attention_backend}")

    native_build_env, native_build_diagnostics = resolve_native_build_env(
        venv,
        gpu_sm=gpu_sm,
        cuda_version=cuda_version,
        build_env=build_env,
    )
    if native_build_diagnostics and native_build_diagnostics.get("cuda_toolkit_root"):
        print(
            "[setup] Steering native source builds (nvdiffrast/cumesh/o-voxel) to CUDA toolkit root: "
            f"{native_build_diagnostics['cuda_toolkit_root']}"
        )
        print(f"[setup] Native source-build CUDACXX={native_build_env['CUDACXX']}")
        print(f"[setup] Native source-build PATH begins with: {native_build_env['PATH'].split(os.pathsep)[0]}")
    elif native_build_diagnostics:
        print("[setup] WARNING: no CUDA toolkit root was resolved for native source builds; ambient CUDA discovery will be used.")

    with tempfile.TemporaryDirectory(prefix="trellis2-setup-") as tmp:
        tmpdir = Path(tmp)
        install_core_native_dependencies(venv, tmpdir, native_build_env)
        install_optional_native_dependencies(venv, tmpdir, native_build_env, plan)

    print("[setup] Done. Extension venv is ready at:", venv)
    print("[setup] Note: first runtime load still requires Hugging Face access for gated dependencies and TRELLIS.2 weights; see README.md.")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--dry-run-plan":
        gpu_sm = int(sys.argv[2]) if len(sys.argv) >= 3 else 0
        cuda_version = int(sys.argv[3]) if len(sys.argv) >= 4 else 0
        print(json.dumps(describe_install_plan(gpu_sm, cuda_version), indent=2))
    elif len(sys.argv) >= 4:
        setup(
            python_exe=sys.argv[1],
            ext_dir=Path(sys.argv[2]),
            gpu_sm=int(sys.argv[3]),
            cuda_version=int(sys.argv[4]) if len(sys.argv) >= 5 else 0,
        )
    elif len(sys.argv) == 2:
        args = json.loads(sys.argv[1])
        setup(
            python_exe=args["python_exe"],
            ext_dir=Path(args["ext_dir"]),
            gpu_sm=int(args.get("gpu_sm", 0)),
            cuda_version=int(args.get("cuda_version", 0)),
        )
    else:
        print("Usage: python setup.py <python_exe> <ext_dir> <gpu_sm> [cuda_version]")
        print('   or: python setup.py \'{"python_exe":"...","ext_dir":"...","gpu_sm":86,"cuda_version":124}\'')
        print("   or: python setup.py --dry-run-plan [gpu_sm] [cuda_version]")
        sys.exit(1)
