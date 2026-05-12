"""
Build the vendor/ directory for the TRELLIS.2 extension.

Run this script once (with the app's venv active) to populate vendor/ with the
pure-Python TRELLIS.2 sources the extension needs at runtime.

Native/runtime packages such as nvdiffrast must come from the extension venv
installed by setup.py so the active CUDA environment stays authoritative.

Usage:
    python build_vendor.py

Requirements (must be run from the app's venv):
    - pip (always available)
    - PyTorch + CUDA (must be available at inference time anyway)
    - MSVC on Windows / gcc on Linux (for compiling CUDA extensions)
"""

import os
import subprocess
import sys
from pathlib import Path

VENDOR       = Path(__file__).parent / "vendor"
TRELLIS2_REF = "5565d240c4a494caaf9ece7a554542b76ffa36d3"
TRELLIS2_ZIP = f"https://github.com/microsoft/TRELLIS.2/archive/{TRELLIS2_REF}.zip"
TRELLIS_REF  = "442aa1e1afb9014e80681d3bf604e8d728a86ee7"
TRELLIS_ZIP  = f"https://github.com/microsoft/TRELLIS/archive/{TRELLIS_REF}.zip"
FLEXICUBES_SUBMODULE_PATH = "trellis/representations/mesh/flexicubes"

# Pure-Python packages to vendor (no compilation needed)
PURE_PACKAGES = [
    "easydict",       # configuration dict used internally by trellis2
    "plyfile",        # PLY mesh format I/O
    "einops",         # tensor reshaping helpers
    "lpips",          # perceptual loss metric
    "trimesh",        # mesh processing
    "tqdm",           # progress bars
    # opencv-python and spconv are too large to vendor in git — installed at runtime via pip
]

UTILS3D_REF = "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

# Compiled CUDA extensions to vendor (require --no-build-isolation to find torch)
# Note: flex_gemm is not on PyPI — spconv is used instead (set via SPARSE_CONV_BACKEND env var)
# Note: nvdiffrast must NOT be vendored; setup.py installs it into the extension venv.
COMPILED_PACKAGES = [
    "cumesh",         # CUDA mesh utilities
]

# spconv fallback versions (newest to oldest) — tried in order until one works
SPCONV_FALLBACK_VERSIONS = ["cu128", "cu124", "cu122", "cu121", "cu120", "cu118"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list, **kwargs):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def vendor_pure_package(package: str, dest: Path) -> None:
    """Install a pure-Python package into vendor/ via pip --target."""
    run([sys.executable, "-m", "pip", "install",
         "--no-deps",
         "--target", str(dest),
         "--upgrade",
         package])
    print(f"  Vendored {package}.")


def vendor_utils3d(dest: Path) -> None:
    """Install the official TRELLIS utils3d package, not the unrelated PyPI homonym."""
    run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(dest), "--upgrade", UTILS3D_REF])
    print("  Vendored official utils3d.")


def vendor_compiled_package(package: str, dest: Path) -> None:
    """Install a compiled package into vendor/ via pip --target --no-build-isolation.

    --no-build-isolation lets the build process find torch in the current
    environment, which is required by CUDA extensions that depend on PyTorch.
    CUDAFLAGS is set to allow unsupported MSVC versions (e.g. VS 2025).
    """
    import os
    env = os.environ.copy()
    env["CUDAFLAGS"] = "-allow-unsupported-compiler"
    env["CMAKE_CUDA_FLAGS"] = "-allow-unsupported-compiler"
    run([sys.executable, "-m", "pip", "install",
         "--no-deps",
         "--no-build-isolation",
         "--target", str(dest),
         "--upgrade",
         package], env=env)
    print(f"  Vendored {package}.")


def vendor_trellis2(dest: Path) -> None:
    """Download TRELLIS.2 source and extract only the trellis2/ package into vendor/."""
    import urllib.request
    import io
    import zipfile

    trellis2_dest = dest / "trellis2"
    if trellis2_dest.exists():
        print("  trellis2/ already present, skipping.")
        return

    print("  Downloading TRELLIS.2 source from GitHub...")
    with urllib.request.urlopen(TRELLIS2_ZIP, timeout=180) as resp:
        data = resp.read()

    archive_root = f"TRELLIS.2-{TRELLIS2_REF}/"
    prefix = f"{archive_root}trellis2/"
    strip  = archive_root

    extracted = 0
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            if not member.startswith(prefix):
                continue
            rel    = member[len(strip):]
            target = dest / rel
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))
                extracted += 1

    if extracted == 0:
        raise RuntimeError(
            f"No files were extracted from the ZIP. "
            f"The expected prefix '{prefix}' was not found.\n"
            "Check that the GitHub archive structure matches and update the "
            "'prefix' variable in vendor_trellis2() if needed."
        )

    print(f"  trellis2/ extracted to {dest} ({extracted} files).")


def vendor_trellis(dest: Path) -> None:
    """Download TRELLIS source and extract the official runtime package into vendor/."""
    import urllib.request
    import io
    import zipfile

    trellis_dest = dest / "trellis"
    if trellis_dest.exists():
        print("  trellis/ already present, refreshing submodule-backed runtime files.")
        sync_trellis_runtime_submodules(dest)
        return

    print("  Downloading TRELLIS source from GitHub...")
    with urllib.request.urlopen(TRELLIS_ZIP, timeout=180) as resp:
        data = resp.read()

    archive_root = f"TRELLIS-{TRELLIS_REF}/"
    allowed_prefixes = [
        f"{archive_root}trellis/__init__.py",
        f"{archive_root}trellis/models/",
        f"{archive_root}trellis/modules/",
        f"{archive_root}trellis/pipelines/",
        f"{archive_root}trellis/renderers/",
        f"{archive_root}trellis/representations/",
        f"{archive_root}trellis/utils/",
    ]

    extracted = 0
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            if not any(member.startswith(prefix) for prefix in allowed_prefixes):
                continue
            rel = member[len(archive_root):]
            target = dest / rel
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))
                extracted += 1

    if extracted == 0:
        raise RuntimeError(
            "No official trellis/ runtime files were extracted from the TRELLIS archive. "
            "Check the pinned archive layout in vendor_trellis()."
        )

    print(f"  trellis/ extracted to {dest} ({extracted} files).")
    sync_trellis_runtime_submodules(dest)


def trellis_submodule_ref(path: str) -> tuple[str, str]:
    """Resolve a TRELLIS submodule URL and pinned commit from GitHub metadata."""
    import json
    import urllib.request

    url = f"https://api.github.com/repos/microsoft/TRELLIS/contents/{path}?ref={TRELLIS_REF}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        metadata = json.load(resp)

    if metadata.get("type") != "submodule":
        raise RuntimeError(f"Expected '{path}' to be a TRELLIS submodule at ref {TRELLIS_REF}.")

    repo_url = metadata.get("submodule_git_url")
    commit = metadata.get("sha")
    if not repo_url or not commit:
        raise RuntimeError(f"Missing submodule metadata for '{path}' at ref {TRELLIS_REF}.")
    return repo_url, commit


def vendor_flexicubes_submodule(dest: Path) -> None:
    """Vendor the FlexiCubes submodule that upstream TRELLIS references for mesh extraction."""
    import io
    import urllib.request
    import zipfile

    repo_url, commit = trellis_submodule_ref(FLEXICUBES_SUBMODULE_PATH)
    archive_url = f"{repo_url[:-4]}/archive/{commit}.zip" if repo_url.endswith(".git") else f"{repo_url}/archive/{commit}.zip"
    package_dest = dest / FLEXICUBES_SUBMODULE_PATH
    package_dest.mkdir(parents=True, exist_ok=True)

    print(f"  Syncing FlexiCubes submodule from {repo_url} @ {commit}...")
    with urllib.request.urlopen(archive_url, timeout=180) as resp:
        data = resp.read()

    archive_root = None
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            if archive_root is None:
                archive_root = member.split("/", 1)[0] + "/"
            if member in {f"{archive_root}flexicubes.py", f"{archive_root}tables.py"}:
                target = package_dest / member[len(archive_root):]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))

    expected = [package_dest / "flexicubes.py", package_dest / "tables.py"]
    missing = [path.name for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(
            "Failed to vendor FlexiCubes runtime files from the pinned TRELLIS submodule: "
            + ", ".join(missing)
        )

    (package_dest / "__init__.py").write_text("from .flexicubes import FlexiCubes\n", encoding="utf-8")
    print(f"  FlexiCubes runtime synced into {package_dest}.")


def sync_trellis_runtime_submodules(dest: Path) -> None:
    """Sync runtime-critical TRELLIS submodules that GitHub source archives omit."""
    vendor_flexicubes_submodule(dest)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Guard: torch must be importable — ensures we're in the right venv.
    try:
        import torch  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "torch is not importable from this Python environment.\n"
            "Run build_vendor.py using the app's venv Python (the one with PyTorch),\n"
            f"not the system Python.\nCurrent interpreter: {sys.executable}"
        )

    print(f"Building vendor/ in {VENDOR}")
    VENDOR.mkdir(parents=True, exist_ok=True)

    # 1. Pure-Python packages
    print("\n[1] Vendoring pure-Python packages...")
    for pkg in PURE_PACKAGES:
        print(f"\n  -> {pkg}")
        try:
            vendor_pure_package(pkg, VENDOR)
        except Exception as exc:
            print(f"  WARNING: failed to vendor {pkg}: {exc}")
            print("  Skipping — it may already be available in the venv.")

    print("\n  -> utils3d (official TRELLIS fork)")
    try:
        vendor_utils3d(VENDOR)
    except Exception as exc:
        print(f"  WARNING: failed to vendor official utils3d: {exc}")
        print("  Skipping — it may already be available in the venv.")

    # 2. Official TRELLIS runtimes
    print("\n[2] Vendoring trellis2 source...")
    vendor_trellis2(VENDOR)
    print("\n[2b] Vendoring trellis source...")
    vendor_trellis(VENDOR)

    # 3. Compiled CUDA extensions
    print("\n[3] Vendoring compiled CUDA extensions...")
    import torch

    failed = []

    # Standard compiled packages
    for pkg in COMPILED_PACKAGES:
        print(f"\n  -> {pkg}")
        try:
            vendor_compiled_package(pkg, VENDOR)
        except Exception as exc:
            print(f"  WARNING: failed to vendor {pkg}: {exc}")
            failed.append(pkg)

    # spconv — try versions from newest to oldest until one works
    cuda_ver = torch.version.cuda  # e.g. "12.8"
    cuda_tag = "cu" + cuda_ver.replace(".", "")
    versions_to_try = [cuda_tag] + [v for v in SPCONV_FALLBACK_VERSIONS if v != cuda_tag]
    spconv_ok = False
    for ver in versions_to_try:
        pkg = f"spconv-{ver}"
        print(f"\n  -> {pkg}")
        try:
            vendor_compiled_package(pkg, VENDOR)
            spconv_ok = True
            break
        except Exception:
            print(f"  Not available, trying next version...")
    if not spconv_ok:
        print("  WARNING: could not vendor any spconv version.")
        failed.append("spconv")

    if failed:
        print(f"\n  The following packages could not be vendored: {failed}")
        print("  Generation may not work without them.")

    print("\n  Native runtime packages such as nvdiffrast must come from setup.py, not vendor/.")

    print("\nDone! vendor/ is ready.")
    print("Commit the vendor/ directory to the extension repository.")
    print("End users still need setup.py to install native runtime packages into the extension venv.")


if __name__ == "__main__":
    main()
