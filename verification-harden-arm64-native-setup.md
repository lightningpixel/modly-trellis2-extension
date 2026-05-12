# Harden ARM64 Native Setup Verification Checklist

- `python3 setup.py --dry-run-plan` shows the resolved platform plan without building anything.
- `python3 validate_harden_arm64_native_setup.py` runs the project-owned lightweight validation suite for install-plan selection, fallback ordering, pinned native refs, optional `nvdiffrec`, and vendor `nvdiffrast` guards.
- Linux ARM64 should report `flash_attn` first and `spconv_strategy: source`.
- Linux ARM64 dry-run diagnostics should also show `attention_backend_install_args.flash_attn = ["--no-build-isolation"]` so `flash-attn` can build against the already-installed extension `torch` without isolated build deps.
- Linux ARM64 dry-run diagnostics should also expose `source_build_env` with `CUMM_DISABLE_JIT=1`, `SPCONV_DISABLE_JIT=1`, and `PATH=<extension-venv-bin>:${PATH}` so temp-clone source builds can discover venv-installed tools like `ninja`.
- Linux ARM64 setup should reuse that same CUDA-toolkit-steered env for later native source builds too, so `nvdiffrast` resolves CUDA 12.8 from `CUDA_HOME`/`CUDA_PATH`/`CUDACXX` instead of falling back to ambient `/usr/local/cuda` 13.0.
- Non-ARM64 should keep the existing `xformers`-first order, with `flash_attn` fallback only where previously supported.
- `nvdiffrec` is deferred on Linux ARM64 by default; other platforms keep the current auto-attempt but no longer block core setup if the optional renderer install fails. Set `MODLY_TRELLIS2_INSTALL_NVDIFFREC=1` to force the optional renderer install.
- `CuMesh` is pinned to `cf1a2f07304b5fe388ed86a16e4a0474599df914` in `setup.py`; update only after validating a new immutable ref.
- `o-voxel` installs from TRELLIS.2 using `--no-deps`, with explicit support packages `plyfile` and `zstandard` installed separately.
- Expected native install failures now mention the package name, platform, attempted ref/version, and toolchain checks (`CUDA`, `nvcc`, compiler, torch/CUDA compatibility).
- `generator.py` must keep vendor imports available while ensuring `nvdiffrast` resolves from the extension venv, not `vendor/`, and it must fail fast if `vendor/nvdiffrast` reappears.
