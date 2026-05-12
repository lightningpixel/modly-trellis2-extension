from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def load_module(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / file_name)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def patched_attr(obj, name, value):
    original = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, original)


@contextmanager
def patched_env(name: str, value: str | None):
    original = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original


@contextmanager
def stubbed_generator_imports():
    original = {name: sys.modules.get(name) for name in [
        "PIL",
        "PIL.Image",
        "services",
        "services.generators",
        "services.generators.base",
    ]}

    pil_module = types.ModuleType("PIL")
    pil_image = types.ModuleType("PIL.Image")
    pil_image.open = lambda *_args, **_kwargs: None
    pil_module.Image = pil_image

    services_module = types.ModuleType("services")
    generators_module = types.ModuleType("services.generators")
    base_module = types.ModuleType("services.generators.base")

    class BaseGenerator:
        pass

    base_module.BaseGenerator = BaseGenerator
    base_module.smooth_progress = lambda *args, **kwargs: None
    base_module.GenerationCancelled = RuntimeError

    sys.modules["PIL"] = pil_module
    sys.modules["PIL.Image"] = pil_image
    sys.modules["services"] = services_module
    sys.modules["services.generators"] = generators_module
    sys.modules["services.generators.base"] = base_module

    try:
        yield
    finally:
        for name, module in original.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@contextmanager
def stubbed_image_feature_extractor_imports():
    module_names = [
        "torch",
        "torch.nn",
        "torch.nn.functional",
        "torchvision",
        "torchvision.transforms",
        "transformers",
        "numpy",
        "PIL",
        "PIL.Image",
    ]
    original = {name: sys.modules.get(name) for name in module_names}

    class FakeTensor:
        def __init__(self, value=None, shape=(1, 2, 3)):
            self.value = value
            self.shape = shape
            self.dtype = "fake-dtype"

        def to(self, _dtype):
            return self

    torch_module = types.ModuleType("torch")
    torch_module.Tensor = FakeTensor
    torch_module.no_grad = lambda: (lambda fn: fn)
    torch_module.from_numpy = lambda value: FakeTensor(value=value)
    torch_module.stack = lambda values: FakeTensor(value=values)
    torch_module.hub = types.SimpleNamespace(load=lambda *args, **kwargs: None)

    torch_nn_module = types.ModuleType("torch.nn")
    torch_nn_functional = types.ModuleType("torch.nn.functional")
    torch_nn_functional.layer_norm = lambda hidden_states, normalized_shape: {
        "hidden_states": getattr(hidden_states, "value", hidden_states),
        "normalized_shape": normalized_shape,
    }
    torch_nn_module.functional = torch_nn_functional
    torch_module.nn = torch_nn_module

    transforms_module = types.ModuleType("torchvision.transforms")

    class Compose:
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, value):
            for transform in self.transforms:
                value = transform(value)
            return value

    class Normalize:
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def __call__(self, value):
            return value

    transforms_module.Compose = Compose
    transforms_module.Normalize = Normalize
    torchvision_module = types.ModuleType("torchvision")
    torchvision_module.transforms = transforms_module

    transformers_module = types.ModuleType("transformers")

    class DINOv3ViTModel:
        @classmethod
        def from_pretrained(cls, _model_name):
            return cls()

    transformers_module.DINOv3ViTModel = DINOv3ViTModel

    numpy_module = types.ModuleType("numpy")
    numpy_module.float32 = "float32"
    numpy_module.array = lambda value: value

    pil_module = types.ModuleType("PIL")
    pil_image = types.ModuleType("PIL.Image")
    pil_image.Image = type("Image", (), {})
    pil_image.LANCZOS = "LANCZOS"
    pil_module.Image = pil_image

    sys.modules["torch"] = torch_module
    sys.modules["torch.nn"] = torch_nn_module
    sys.modules["torch.nn.functional"] = torch_nn_functional
    sys.modules["torchvision"] = torchvision_module
    sys.modules["torchvision.transforms"] = transforms_module
    sys.modules["transformers"] = transformers_module
    sys.modules["numpy"] = numpy_module
    sys.modules["PIL"] = pil_module
    sys.modules["PIL.Image"] = pil_image

    try:
        yield FakeTensor
    finally:
        for name, module in original.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_setup_plan_and_attention() -> None:
    setup = load_module("modly_setup_validation", "setup.py")
    toolkit_root = Path("/usr/local/cuda-12.8")
    include_dir = toolkit_root / "include"
    library_dir = toolkit_root / "lib64"

    with (
        patched_attr(setup, "is_linux_arm64", lambda: True),
        patched_attr(setup, "is_windows", lambda: False),
        patched_attr(setup, "resolve_cuda_toolkit_root", lambda _cuda_version, env=None: toolkit_root),
        patched_attr(setup, "cuda_toolkit_library_dirs", lambda _toolkit_root: (library_dir,)),
    ):
        plan = setup.plan_platform_install()
        install_plan = setup.describe_install_plan(gpu_sm=90, cuda_version=128)
        remapped_install_plan = setup.describe_install_plan(gpu_sm=121, cuda_version=128)
        assert_true(plan.name == "linux-arm64", "ARM64 plan name should be linux-arm64")
        assert_true(plan.attention_backends == (("flash_attn", f"flash-attn=={setup.FLASH_ATTN_VERSION}"),), "ARM64 plan should only allow flash-attn")
        assert_true(plan.optional_renderer_default is False, "ARM64 should defer optional renderer by default")
        assert_true(
            install_plan["attention_backend_install_args"] == {"flash_attn": ["--no-build-isolation"]},
            "ARM64 dry-run plan should show that flash-attn installs without build isolation",
        )
        assert_true(
            install_plan["source_build_env"] == {
                "CUMM_DISABLE_JIT": "1",
                "SPCONV_DISABLE_JIT": "1",
                "CUMM_CUDA_ARCH_LIST": "9.0",
                "PATH": f"<extension-venv-bin>:{toolkit_root / 'bin'}:${{PATH}}",
                "cuda_toolkit_root_candidates": [str(toolkit_root), "/usr/local/cuda"],
                "cuda_toolkit_root": str(toolkit_root),
                "CUDA_HOME": str(toolkit_root),
                "CUDA_PATH": str(toolkit_root),
                "CUDACXX": str(toolkit_root / "bin" / "nvcc"),
                "CPATH": f"{include_dir}:${{CPATH}}",
                "C_INCLUDE_PATH": f"{include_dir}:${{C_INCLUDE_PATH}}",
                "CPLUS_INCLUDE_PATH": f"{include_dir}:${{CPLUS_INCLUDE_PATH}}",
                "LIBRARY_PATH": f"{library_dir}:${{LIBRARY_PATH}}",
                "LD_LIBRARY_PATH": f"{library_dir}:${{LD_LIBRARY_PATH}}",
                "source_build_hotfixes": [
                    "patch installed cumm/common.py on Linux ARM64 so CUDA include/lib discovery honors CUDA_HOME/CUDA_PATH before /usr/local/cuda"
                ],
            },
            "ARM64 dry-run plan should expose the forced non-JIT source build env, CUDA toolkit steering, and resolved arch",
        )
        assert_true(
            install_plan["cumm_cuda_arch"] == {
                "requested": "9.0",
                "resolved": "9.0",
                "reason": "SM 90 maps directly to supported cumm arch 9.0",
            },
            "ARM64 dry-run plan should explain the exact cumm arch mapping for supported SM values",
        )
        assert_true(
            remapped_install_plan["source_build_env"] == {
                "CUMM_DISABLE_JIT": "1",
                "SPCONV_DISABLE_JIT": "1",
                "CUMM_CUDA_ARCH_LIST": "9.0+PTX",
                "PATH": f"<extension-venv-bin>:{toolkit_root / 'bin'}:${{PATH}}",
                "cuda_toolkit_root_candidates": [str(toolkit_root), "/usr/local/cuda"],
                "cuda_toolkit_root": str(toolkit_root),
                "CUDA_HOME": str(toolkit_root),
                "CUDA_PATH": str(toolkit_root),
                "CUDACXX": str(toolkit_root / "bin" / "nvcc"),
                "CPATH": f"{include_dir}:${{CPATH}}",
                "C_INCLUDE_PATH": f"{include_dir}:${{C_INCLUDE_PATH}}",
                "CPLUS_INCLUDE_PATH": f"{include_dir}:${{CPLUS_INCLUDE_PATH}}",
                "LIBRARY_PATH": f"{library_dir}:${{LIBRARY_PATH}}",
                "LD_LIBRARY_PATH": f"{library_dir}:${{LD_LIBRARY_PATH}}",
                "source_build_hotfixes": [
                    "patch installed cumm/common.py on Linux ARM64 so CUDA include/lib discovery honors CUDA_HOME/CUDA_PATH before /usr/local/cuda"
                ],
            },
            "ARM64 dry-run plan should clamp unsupported newer SM values to the PTX fallback while keeping CUDA toolkit steering visible",
        )
        assert_true(
            remapped_install_plan["cumm_cuda_arch"] == {
                "requested": "12.1",
                "resolved": "9.0+PTX",
                "reason": "SM 121 maps to unsupported arch 12.1; clamping to 9.0+PTX because cumm v0.7.11 supports up to 9.0 and PTX enables forward compatibility",
            },
            "ARM64 dry-run plan should expose why unsupported newer SM values are remapped",
        )

    with patched_attr(setup, "is_linux_arm64", lambda: False), patched_attr(setup, "is_windows", lambda: False), patched_attr(setup, "machine_arch", lambda: "x86_64"), patched_attr(setup.platform, "system", lambda: "Linux"):
        plan = setup.plan_platform_install()
        assert_true([name for name, _ in plan.attention_backends] == ["xformers", "flash_attn"], "Non-ARM64 Linux should keep xformers before flash-attn")
        install_plan = setup.describe_install_plan(gpu_sm=90, cuda_version=124)
        assert_true(
            install_plan["attention_backend_install_args"] == {"xformers": [], "flash_attn": []},
            "Non-ARM64 dry-run plan should not force no-build-isolation for attention backends",
        )

        attempted = []

        def fake_pip(_venv, *args, env=None):
            del env
            requirement = args[-1]
            attempted.append(args)
            if requirement == "xformers":
                raise subprocess.CalledProcessError(returncode=1, cmd=["pip", "install", requirement])

        with patched_attr(setup, "pip", fake_pip):
            selected = setup.install_attention_backend(Path("/tmp/venv"), plan)
        assert_true(selected == "flash_attn", "Fallback backend should resolve to flash_attn")
        assert_true(
            attempted == [
                ("install", "xformers"),
                ("install", f"flash-attn=={setup.FLASH_ATTN_VERSION}"),
            ],
            "Attention backend order changed unexpectedly",
        )

    with patched_attr(setup, "is_linux_arm64", lambda: True), patched_attr(setup, "is_windows", lambda: False), patched_attr(setup.platform, "system", lambda: "Linux"), patched_attr(setup, "machine_arch", lambda: "aarch64"):
        plan = setup.plan_platform_install()
        install_plan = setup.describe_install_plan(gpu_sm=121, cuda_version=128)
        assert_true(
            install_plan["attention_backend_install_args"] == {"flash_attn": ["--no-build-isolation"]},
            "ARM64 dry-run plan should keep flash-attn no-build-isolation visible for remapped SM values",
        )

        attempted = []

        def always_fail(_venv, *args, env=None):
            del env
            attempted.append(args)
            raise subprocess.CalledProcessError(returncode=1, cmd=["pip", *args])

        with patched_attr(setup, "pip", always_fail):
            try:
                setup.install_attention_backend(Path("/tmp/venv"), plan)
            except RuntimeError as exc:
                message = str(exc)
            else:
                raise AssertionError("ARM64 backend failure should raise RuntimeError")
        assert_true("Core generation cannot proceed" in message, "ARM64 failure should explain that core generation cannot proceed")
        assert_true("flash-attn==" in message and "Platform: Linux aarch64" in message, "ARM64 failure should include attempted backend and platform")
        assert_true(
            attempted == [("install", "--no-build-isolation", f"flash-attn=={setup.FLASH_ATTN_VERSION}")],
            "ARM64 flash-attn install must disable build isolation",
        )


def test_optional_and_core_native_install_contracts() -> None:
    setup = load_module("modly_setup_validation_optional", "setup.py")
    arm_plan = setup.PlatformInstallPlan(name="linux-arm64", attention_backends=(("flash_attn", "flash-attn==2.7.3"),), optional_renderer_default=False)
    desktop_plan = setup.PlatformInstallPlan(name="linux-x86_64", attention_backends=(("xformers", "xformers"),), optional_renderer_default=True)

    with patched_env(setup.OPTIONAL_NVDIFFREC_ENV, None):
        enabled, explicit = setup.should_install_optional_nvdiffrec(arm_plan)
        assert_true((enabled, explicit) == (False, False), "ARM64 default should skip optional nvdiffrec without explicit opt-in")

        calls = []

        def fake_install_from_repo(*args, **kwargs):
            calls.append((args, kwargs))

        with patched_attr(setup, "install_from_repo", fake_install_from_repo):
            setup.install_optional_native_dependencies(Path("/tmp/venv"), Path("/tmp"), {}, arm_plan)
        assert_true(not calls, "Skipped optional nvdiffrec should not invoke install_from_repo")

    with patched_env(setup.OPTIONAL_NVDIFFREC_ENV, "1"):
        enabled, explicit = setup.should_install_optional_nvdiffrec(arm_plan)
        assert_true((enabled, explicit) == (True, True), "Explicit opt-in should enable optional nvdiffrec")

        def failing_install(*args, **kwargs):
            raise RuntimeError("synthetic nvdiffrec failure")

        with patched_attr(setup, "install_from_repo", failing_install):
            try:
                setup.install_optional_native_dependencies(Path("/tmp/venv"), Path("/tmp"), {}, arm_plan)
            except RuntimeError as exc:
                assert_true("synthetic nvdiffrec failure" in str(exc), "Explicit nvdiffrec failures should propagate")
            else:
                raise AssertionError("Explicit nvdiffrec install failure should not be swallowed")

    core_calls = []
    support_calls = []

    def capture_install_from_repo(_venv, _tmpdir, folder_name, repo, **kwargs):
        core_calls.append((folder_name, repo, kwargs))

    def capture_support_packages(_venv, package_name, attempted_ref, *packages, env=None):
        del env
        support_calls.append((package_name, attempted_ref, packages))

    with patched_attr(setup, "install_from_repo", capture_install_from_repo), patched_attr(setup, "install_packages_with_diagnostics", capture_support_packages):
        setup.install_core_native_dependencies(Path("/tmp/venv"), Path("/tmp"), {})

    assert_true(any(name == "nvdiffrast" and kwargs["ref"] == setup.NVDIFFRAST_SOURCE_REF for name, _, kwargs in core_calls), "Core install must pin nvdiffrast source")
    assert_true(any(name == "cumesh" and kwargs["ref"] == setup.CUMESH_SOURCE_REF and kwargs.get("recursive") is True for name, _, kwargs in core_calls), "Core install must pin recursive CuMesh source")
    assert_true(any(name == "o-voxel" and kwargs["ref"] == setup.TRELLIS2_SOURCE_REF and kwargs.get("no_deps") is True for name, _, kwargs in core_calls), "o-voxel must install from pinned TRELLIS.2 ref with --no-deps")
    assert_true(support_calls == [("o-voxel-support-packages", ", ".join(setup.O_VOXEL_SUPPORT_PACKAGES), setup.O_VOXEL_SUPPORT_PACKAGES)], "o-voxel support packages changed unexpectedly")

    with patched_env(setup.OPTIONAL_NVDIFFREC_ENV, None):
        def fail_optional(*args, **kwargs):
            raise RuntimeError("synthetic optional failure")

        with patched_attr(setup, "install_from_repo", fail_optional):
            setup.install_optional_native_dependencies(Path("/tmp/venv"), Path("/tmp"), {}, desktop_plan)


def test_python_runtime_dependency_contract() -> None:
    setup = load_module("modly_setup_validation_runtime_deps", "setup.py")
    assert_true("kornia" in setup.PYTHON_RUNTIME_DEPENDENCIES, "Clean install contract must include kornia")
    assert_true("timm" in setup.PYTHON_RUNTIME_DEPENDENCIES, "Clean install contract must include timm")
    assert_true("xatlas" in setup.PYTHON_RUNTIME_DEPENDENCIES, "Native text postprocessing must include xatlas")
    assert_true("pyvista" in setup.PYTHON_RUNTIME_DEPENDENCIES, "Native text postprocessing must include pyvista")
    assert_true("pymeshfix" in setup.PYTHON_RUNTIME_DEPENDENCIES, "Native text postprocessing must include pymeshfix")
    assert_true("igraph" in setup.PYTHON_RUNTIME_DEPENDENCIES, "Native text postprocessing must include igraph")

    captured = []

    def fake_pip(_venv, *args, env=None):
        del env
        captured.append(args)

    with patched_attr(setup, "pip", fake_pip):
        setup.install_python_runtime_dependencies(Path("/tmp/venv"))

    assert_true(
        captured == [("install", *setup.PYTHON_RUNTIME_DEPENDENCIES)],
        "Python runtime dependency install flow changed unexpectedly",
    )


def test_native_build_env_steering_for_arm64_source_builds() -> None:
    setup = load_module("modly_setup_validation_native_env", "setup.py")
    toolkit_root = Path("/usr/local/cuda-12.8")
    include_dir = toolkit_root / "include"
    library_dir = toolkit_root / "lib64"

    with (
        patched_attr(setup, "is_linux_arm64", lambda: True),
        patched_attr(setup, "resolve_cuda_toolkit_root", lambda _cuda_version, env=None: toolkit_root),
        patched_attr(setup, "cuda_toolkit_library_dirs", lambda _toolkit_root: (library_dir,)),
    ):
        native_env, diagnostics = setup.resolve_native_build_env(
            Path("/tmp/venv"),
            gpu_sm=121,
            cuda_version=128,
            build_env={"CUSTOM_FLAG": "kept"},
        )

    assert_true(native_env["CUSTOM_FLAG"] == "kept", "Native ARM64 env should preserve caller build env")
    assert_true(native_env["PATH"].split(os.pathsep)[0] == "/tmp/venv/bin", "Native ARM64 env must prepend the extension venv bin to PATH")
    assert_true(native_env["PATH"].split(os.pathsep)[1] == str(toolkit_root / "bin"), "Native ARM64 env must keep the selected CUDA toolkit bin immediately after the extension venv bin")
    assert_true(native_env["CUDA_HOME"] == str(toolkit_root), "Native ARM64 env must export CUDA_HOME")
    assert_true(native_env["CUDA_PATH"] == str(toolkit_root), "Native ARM64 env must export CUDA_PATH")
    assert_true(native_env["CUDACXX"] == str(toolkit_root / "bin" / "nvcc"), "Native ARM64 env must point CUDACXX at the selected toolkit nvcc")
    assert_true(native_env["CPATH"].split(os.pathsep)[0] == str(include_dir), "Native ARM64 env must force the selected CUDA include path")
    assert_true(native_env["LIBRARY_PATH"].split(os.pathsep)[0] == str(library_dir), "Native ARM64 env must force the selected CUDA library path")
    assert_true(diagnostics is not None and diagnostics["cuda_toolkit_root"] == str(toolkit_root), "Native ARM64 diagnostics must expose the resolved toolkit root")

    with patched_attr(setup, "is_linux_arm64", lambda: False):
        passthrough_env, passthrough_diagnostics = setup.resolve_native_build_env(
            Path("/tmp/venv"),
            gpu_sm=121,
            cuda_version=128,
            build_env={"CUSTOM_FLAG": "kept"},
        )
    assert_true(passthrough_env == {"CUSTOM_FLAG": "kept"}, "Non-ARM64 native env resolution should be a passthrough")
    assert_true(passthrough_diagnostics is None, "Non-ARM64 native env resolution should not emit diagnostics")


def test_arm64_spconv_source_build_env() -> None:
    setup = load_module("modly_setup_validation_spconv", "setup.py")
    toolkit_root = Path("/usr/local/cuda-12.8")
    include_dir = toolkit_root / "include"
    library_dir = toolkit_root / "lib64"

    uninstall_calls = []
    prereq_calls = []
    repo_calls = []
    smoke_calls = []
    patch_calls = []

    def capture_uninstall(_venv, *packages):
        uninstall_calls.append(packages)

    def capture_prereqs(_venv, package_name, attempted_ref, *packages, env=None):
        prereq_calls.append((package_name, attempted_ref, packages, dict(env or {})))

    def capture_install_from_repo(_venv, _tmpdir, folder_name, repo, **kwargs):
        repo_calls.append((folder_name, repo, kwargs))

    def capture_smoke(_venv, *, env=None):
        smoke_calls.append(dict(env or {}))

    def capture_patch(venv):
        patch_calls.append(venv)

    build_env = {"CUSTOM_FLAG": "kept"}

    with (
        patched_attr(setup, "uninstall_packages", capture_uninstall),
        patched_attr(setup, "install_packages_with_diagnostics", capture_prereqs),
        patched_attr(setup, "install_from_repo", capture_install_from_repo),
        patched_attr(setup, "smoke_check_spconv", capture_smoke),
        patched_attr(setup, "patch_installed_cumm_cuda_discovery", capture_patch),
        patched_attr(setup, "resolve_cuda_toolkit_root", lambda _cuda_version, env=None: toolkit_root),
        patched_attr(setup, "cuda_toolkit_library_dirs", lambda _toolkit_root: (library_dir,)),
    ):
        setup.install_spconv_from_source(Path("/tmp/venv"), gpu_sm=87, cuda_version=128, build_env=build_env)

    assert_true(uninstall_calls == [("spconv", "cumm")], "ARM64 source fallback must uninstall stale spconv/cumm first")
    assert_true(len(prereq_calls) == 1, "ARM64 source fallback should install build prereqs once")

    prereq_env = prereq_calls[0][3]
    assert_true(prereq_env["CUSTOM_FLAG"] == "kept", "ARM64 source fallback should preserve caller build env")
    assert_true(prereq_env["PATH"].split(os.pathsep)[0] == "/tmp/venv/bin", "ARM64 source fallback must prepend the extension venv bin to PATH")
    assert_true(prereq_env["PATH"].split(os.pathsep)[1] == str(toolkit_root / "bin"), "ARM64 source fallback must keep the CUDA toolkit bin immediately after the extension venv bin on PATH")
    assert_true(prereq_env["CUMM_DISABLE_JIT"] == "1", "ARM64 source fallback must force CUMM_DISABLE_JIT=1")
    assert_true(prereq_env["SPCONV_DISABLE_JIT"] == "1", "ARM64 source fallback must force SPCONV_DISABLE_JIT=1")
    assert_true(prereq_env["CUMM_CUDA_ARCH_LIST"] == "8.7", "ARM64 source fallback should derive CUMM_CUDA_ARCH_LIST from gpu_sm")
    assert_true(prereq_env["CUDA_HOME"] == str(toolkit_root), "ARM64 source fallback must export CUDA_HOME for ccimport/pccm")
    assert_true(prereq_env["CUDA_PATH"] == str(toolkit_root), "ARM64 source fallback must export CUDA_PATH for CUDA discovery")
    assert_true(prereq_env["CUDACXX"] == str(toolkit_root / "bin" / "nvcc"), "ARM64 source fallback must point CUDACXX at the selected toolkit nvcc")
    assert_true(prereq_env["CPATH"].split(os.pathsep)[0] == str(include_dir), "ARM64 source fallback must force CUDA include precedence through CPATH")
    assert_true(prereq_env["C_INCLUDE_PATH"].split(os.pathsep)[0] == str(include_dir), "ARM64 source fallback must force CUDA include precedence through C_INCLUDE_PATH")
    assert_true(prereq_env["CPLUS_INCLUDE_PATH"].split(os.pathsep)[0] == str(include_dir), "ARM64 source fallback must force CUDA include precedence through CPLUS_INCLUDE_PATH")
    assert_true(prereq_env["LIBRARY_PATH"].split(os.pathsep)[0] == str(library_dir), "ARM64 source fallback must force CUDA library precedence through LIBRARY_PATH")
    assert_true(prereq_env["LD_LIBRARY_PATH"].split(os.pathsep)[0] == str(library_dir), "ARM64 source fallback must force CUDA library precedence through LD_LIBRARY_PATH")
    assert_true(patch_calls == [Path("/tmp/venv")], "ARM64 source fallback must patch installed cumm CUDA discovery before building spconv")

    assert_true([name for name, _, _ in repo_calls] == ["cumm", "spconv"], "ARM64 source fallback should install cumm before spconv")
    for name, _, kwargs in repo_calls:
        env = kwargs["env"]
        assert_true(env["PATH"].split(os.pathsep)[0] == "/tmp/venv/bin", f"{name} source install must prepend the extension venv bin to PATH")
        assert_true(env["PATH"].split(os.pathsep)[1] == str(toolkit_root / "bin"), f"{name} source install must preserve the selected CUDA toolkit bin on PATH")
        assert_true(env["CUMM_DISABLE_JIT"] == "1", f"{name} source install must inherit CUMM_DISABLE_JIT=1")
        assert_true(env["SPCONV_DISABLE_JIT"] == "1", f"{name} source install must inherit SPCONV_DISABLE_JIT=1")
        assert_true(env["CUMM_CUDA_ARCH_LIST"] == "8.7", f"{name} source install must inherit the resolved CUDA arch list")
        assert_true(env["CUDA_HOME"] == str(toolkit_root), f"{name} source install must inherit CUDA_HOME")
        assert_true(env["CPATH"].split(os.pathsep)[0] == str(include_dir), f"{name} source install must inherit CUDA include steering")

    assert_true(smoke_calls and smoke_calls[0]["PATH"].split(os.pathsep)[0] == "/tmp/venv/bin", "spconv smoke import should run with the extension venv bin first on PATH")
    assert_true(smoke_calls and smoke_calls[0]["CUMM_DISABLE_JIT"] == "1", "spconv smoke import should run under the same non-JIT env")
    assert_true(smoke_calls and smoke_calls[0]["CPATH"].split(os.pathsep)[0] == str(include_dir), "spconv smoke import should keep the selected CUDA include path")

    uninstall_calls.clear()
    prereq_calls.clear()
    repo_calls.clear()
    smoke_calls.clear()
    patch_calls.clear()

    with (
        patched_attr(setup, "uninstall_packages", capture_uninstall),
        patched_attr(setup, "install_packages_with_diagnostics", capture_prereqs),
        patched_attr(setup, "install_from_repo", capture_install_from_repo),
        patched_attr(setup, "smoke_check_spconv", capture_smoke),
        patched_attr(setup, "patch_installed_cumm_cuda_discovery", capture_patch),
        patched_attr(setup, "resolve_cuda_toolkit_root", lambda _cuda_version, env=None: toolkit_root),
        patched_attr(setup, "cuda_toolkit_library_dirs", lambda _toolkit_root: (library_dir,)),
    ):
        setup.install_spconv_from_source(Path("/tmp/venv"), gpu_sm=121, cuda_version=128, build_env=build_env)

    prereq_env = prereq_calls[0][3]
    assert_true(prereq_env["PATH"].split(os.pathsep)[0] == "/tmp/venv/bin", "Unsupported newer ARM64 SM values must keep the extension venv bin first on PATH")
    assert_true(prereq_env["PATH"].split(os.pathsep)[1] == str(toolkit_root / "bin"), "Unsupported newer ARM64 SM values must keep the selected CUDA toolkit bin next on PATH")
    assert_true(prereq_env["CUMM_CUDA_ARCH_LIST"] == "9.0+PTX", "Unsupported newer ARM64 SM values should clamp to the PTX fallback")
    for name, _, kwargs in repo_calls:
        env = kwargs["env"]
        assert_true(env["PATH"].split(os.pathsep)[0] == "/tmp/venv/bin", f"{name} source install must preserve the extension venv bin PATH precedence")
        assert_true(env["PATH"].split(os.pathsep)[1] == str(toolkit_root / "bin"), f"{name} source install must preserve the selected CUDA toolkit PATH precedence")
        assert_true(env["CUMM_CUDA_ARCH_LIST"] == "9.0+PTX", f"{name} source install must inherit the PTX fallback arch list")
    assert_true(patch_calls == [Path("/tmp/venv")], "Unsupported newer ARM64 SM values must still patch installed cumm CUDA discovery")


def test_patch_installed_cumm_cuda_discovery() -> None:
    setup = load_module("modly_setup_validation_cumm_patch", "setup.py")

    with tempfile.TemporaryDirectory(prefix="trellis2-cumm-patch-") as tmp:
        tmpdir = Path(tmp)
        cumm_dir = tmpdir / "cumm"
        cumm_dir.mkdir()
        cumm_common = cumm_dir / "common.py"
        cumm_common.write_text(
            (
                "import os\n"
                "import subprocess\n"
                "from pathlib import Path\n\n"
                "def sample():\n"
                "        else:\n"
                "            try:\n"
                "                nvcc_path = subprocess.check_output([\"which\", \"nvcc\"\n"
                "                                                    ]).decode(\"utf-8\").strip()\n"
                "                lib = Path(nvcc_path).parent.parent / \"lib\"\n"
                "                include = Path(nvcc_path).parent.parent / \"targets/x86_64-linux/include\"\n"
                "                if lib.exists() and include.exists():\n"
                "                    if (lib / \"libcudart.so\").exists() and (include / \"cuda.h\").exists():\n"
                "                        # should be nvidia conda package\n"
                "                        _CACHED_CUDA_INCLUDE_LIB = ([include], lib)\n"
                "                        return _CACHED_CUDA_INCLUDE_LIB\n"
                "            except:\n"
                "                pass \n\n"
                "            linux_cuda_root = Path(\"/usr/local/cuda\")\n"
                "            include = linux_cuda_root / f\"include\"\n"
                "            lib64 = linux_cuda_root / f\"lib64\"\n"
                "            assert linux_cuda_root.exists(), f\"can't find cuda in {linux_cuda_root} install via cuda installer or conda first.\"\n"
            )
        )

        def fake_check_output(cmd, text=False):
            del text
            assert_true(cmd[0].endswith("/bin/python"), "Patch helper must inspect cumm using the extension venv python")
            return f"{cumm_common}\n"

        with patched_attr(setup.subprocess, "check_output", fake_check_output):
            setup.patch_installed_cumm_cuda_discovery(tmpdir)

        patched = cumm_common.read_text()
        assert_true(setup.CUMM_CUDA_DISCOVERY_PATCH_MARKER in patched, "Patch helper must stamp the cumm CUDA discovery hotfix marker")
        assert_true("for env_name in (\"CUDA_HOME\", \"CUDA_PATH\")" in patched, "Patch helper must teach cumm to honor CUDA_HOME/CUDA_PATH")
        assert_true("targets/aarch64-linux/include" in patched, "Patch helper must add ARM64 CUDA target include discovery")


def test_vendor_precedence_guards() -> None:
    vendor_overlap = ROOT / "vendor" / "nvdiffrast"
    assert_true(not vendor_overlap.exists(), "vendor/nvdiffrast must be absent so it is not import-discoverable")

    with stubbed_generator_imports():
        generator = load_module("modly_generator_validation", "generator.py")
        clean_paths = generator.filtered_vendor_paths()
        assert_true(clean_paths == [str(generator._VENDOR_DIR)], "Filtered vendor paths should expose the vendor root when no native overlap exists")

        with tempfile.TemporaryDirectory(prefix="trellis2-vendor-overlap-") as tmp:
            fake_vendor = Path(tmp)
            (fake_vendor / "nvdiffrast").mkdir()
            with patched_attr(generator, "_VENDOR_DIR", fake_vendor):
                try:
                    generator.filtered_vendor_paths()
                except RuntimeError as exc:
                    assert_true("Vendored native overlap directories are not allowed" in str(exc), "Overlap guard should explain why vendor/nvdiffrast is rejected")
                else:
                    raise AssertionError("filtered_vendor_paths should reject native overlap directories")

        instance = generator.Trellis2Generator.__new__(generator.Trellis2Generator)
        with patched_attr(generator, "module_spec_origin", lambda _name: str(generator._VENDOR_DIR / "nvdiffrast" / "__init__.py")):
            try:
                instance._require_runtime_dependency("nvdiffrast", "nvdiffrast", allow_vendor=False)
            except RuntimeError as exc:
                assert_true("resolved from vendor/ instead of the extension venv" in str(exc), "Vendor resolution should be rejected for nvdiffrast")
            else:
                raise AssertionError("nvdiffrast should not be allowed to resolve from vendor/")


def test_phase3_manifest_and_docs_contract() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in manifest.get("nodes", [])}
    assert_true(set(nodes) == {"generate", "texture-mesh", "text-to-mesh"}, "Phase 3 must expose generate, texture-mesh, and text-to-mesh nodes")

    image_node = nodes["generate"]
    assert_true(image_node["id"] == "generate", "Phase 2 must preserve the working node id 'generate'")
    assert_true(image_node.get("capability_id") == "image-to-mesh", "Manifest must label the compatibility node as image-to-mesh")
    assert_true(image_node.get("weight_owner_id") == "base-4b", "Manifest must declare the shared weight owner for TRELLIS nodes")
    assert_true(image_node.get("input") == "image" and image_node.get("output") == "mesh", "Generate node contract must remain image -> mesh")

    texture_node = nodes["texture-mesh"]
    assert_true(texture_node.get("capability_id") == "texture-mesh", "Manifest must label the texturing node as texture-mesh")
    assert_true(texture_node.get("weight_owner_id") == "base-4b", "Texture node must reuse the shared TRELLIS weight owner")
    assert_true(texture_node.get("input") == "image" and texture_node.get("output") == "mesh", "Texture node contract must remain image -> mesh")
    assert_true(texture_node.get("download_check") == "texturing_pipeline.json", "Texture node must download-check the texturing pipeline config")

    inputs = texture_node.get("inputs")
    assert_true(isinstance(inputs, list) and len(inputs) == 2, "Texture node must declare exactly two named inputs")
    assert_true(inputs == [
        {"name": "image", "label": "Reference Image", "type": "image", "required": True},
        {"name": "mesh", "label": "Source Mesh", "type": "mesh", "required": True},
    ], "Texture node inputs must use current Modly named-port manifest format")

    text_node = nodes["text-to-mesh"]
    assert_true(text_node.get("capability_id") == "text-to-mesh", "Manifest must label the native text node correctly")
    assert_true(text_node.get("weight_owner_id") == "text-xlarge", "Native text node must use a dedicated text weight owner")
    assert_true(text_node.get("hf_repo") == "microsoft/TRELLIS-text-xlarge", "Native text node must target the official upstream text repo")
    assert_true(text_node.get("input") == "text" and text_node.get("output") == "mesh", "Native text node contract must remain text -> mesh")
    assert_true(text_node.get("download_check") == "pipeline.json", "Native text node must download-check pipeline.json")

    with stubbed_generator_imports():
        generator = load_module("modly_generator_phase2_validation", "generator.py")
        assert_true(image_node.get("params_schema") == generator.IMAGE_TO_MESH_PARAMS_SCHEMA, "Manifest params_schema must stay aligned with generator image-to-mesh schema")
        assert_true(texture_node.get("params_schema") == generator.TEXTURE_MESH_PARAMS_SCHEMA, "Manifest params_schema must stay aligned with generator texture-mesh schema")
        assert_true(text_node.get("params_schema") == generator.TEXT_TO_MESH_PARAMS_SCHEMA, "Manifest params_schema must stay aligned with generator text-to-mesh schema")
        assert_true(generator.CAPABILITIES["generate"].capability_id == "image-to-mesh", "Generator capability map must resolve 'generate' to image-to-mesh")
        assert_true(generator.CAPABILITIES["texture-mesh"].capability_id == "texture-mesh", "Generator capability map must resolve 'texture-mesh' correctly")
        assert_true(generator.CAPABILITIES["text-to-mesh"].capability_id == "text-to-mesh", "Generator capability map must resolve 'text-to-mesh' correctly")
        assert_true(generator.CAPABILITIES["generate"].config_file == "pipeline.json", "Image capability must keep the default pipeline config")
        assert_true(generator.CAPABILITIES["texture-mesh"].config_file == "texturing_pipeline.json", "Texture capability must point at the texturing pipeline config")
        assert_true(generator.CAPABILITIES["texture-mesh"].download_check == "texturing_pipeline.json", "Texture capability download check must match the texturing pipeline config")
        assert_true(generator.CAPABILITIES["text-to-mesh"].family == "trellis-text", "Native text node must resolve to the official trellis family")
        assert_true(generator.Trellis2Generator.params_schema() == generator.IMAGE_TO_MESH_PARAMS_SCHEMA, "Default params_schema must preserve Phase 1 compatibility")
        assert_true(generator.Trellis2Generator.capability_params_schema("texture-mesh") == generator.TEXTURE_MESH_PARAMS_SCHEMA, "Capability-specific params_schema lookup must support texture-mesh")
        assert_true(generator.Trellis2Generator.capability_params_schema("text-to-mesh") == generator.TEXT_TO_MESH_PARAMS_SCHEMA, "Capability-specific params_schema lookup must support text-to-mesh")


def test_capability_specific_pipeline_loading() -> None:
    with stubbed_generator_imports():
        generator = load_module("modly_generator_load_validation", "generator.py")

        loaded: list[tuple[str, str]] = []

        class FakePipeline:
            def __init__(self, capability_id: str):
                self.capability_id = capability_id

            def cuda(self):
                loaded.append((self.capability_id, "cuda"))

        class FakePipelineClass:
            def __init__(self, capability_id: str):
                self.capability_id = capability_id

            def from_pretrained(self, model_dir: str, *, config_file: str):
                loaded.append((model_dir, config_file))
                return FakePipeline(self.capability_id)

        for node_id, expected_config in (("generate", "pipeline.json"), ("texture-mesh", "texturing_pipeline.json"), ("text-to-mesh", generator._TEXT_PIPELINE_CONFIG_FILE)):
            instance = generator.Trellis2Generator.__new__(generator.Trellis2Generator)
            instance._model = None
            instance.model_dir = ROOT / node_id
            instance._auto_download = lambda: (_ for _ in ()).throw(AssertionError("auto-download should not run during config load validation"))
            instance._setup_env = lambda: None
            instance._setup_vendor = lambda: None
            instance._prepare_text_pipeline_config = lambda _model_dir, generator=generator: Path(_model_dir) / generator._TEXT_PIPELINE_CONFIG_FILE
            capability = generator.CAPABILITIES[node_id]
            instance._capability = lambda capability=capability: capability
            instance._resolve_pipeline_class = lambda capability, node_id=node_id: FakePipelineClass(node_id)
            instance.is_downloaded = lambda: True

            instance.load()

            assert_true(loaded[-2] == (str(instance.model_dir), expected_config), f"{node_id} must load the expected pipeline config")
            assert_true(loaded[-1] == (node_id, "cuda"), f"{node_id} must still move the pipeline onto CUDA")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert_true("trellis-2/generate" in readme, "README must document the currently supported node")
    assert_true("trellis-2/texture-mesh" in readme, "README must document the texture-mesh node")
    assert_true("trellis-2/text-to-mesh" in readme, "README must document the text-to-mesh node")
    assert_true("image + mesh -> mesh" in readme, "README must document the texture-mesh workflow shape")
    assert_true("text -> mesh" in readme, "README must document the native text workflow shape")
    assert_true("params.prompt" in readme, "README must document the canonical text prompt transport assumption")
    assert_true("microsoft/TRELLIS-text-xlarge" in readme, "README must document the official text model repo")
    assert_true("24 GB VRAM" in readme, "README must describe practical VRAM expectations")


def test_capability_resolution_prefers_explicit_runtime_node_and_rejects_ambiguous_download_check() -> None:
    with stubbed_generator_imports():
        generator = load_module("modly_generator_capability_resolution_validation", "generator.py")

        env_name = "MODEL_ID"
        with patched_env(env_name, "trellis-2/text-to-mesh"):
            instance = generator.Trellis2Generator.__new__(generator.Trellis2Generator)
            instance.model_dir = Path("/tmp/trellis-2/text-xlarge")
            instance.download_check = "pipeline.json"
            assert_true(
                instance._capability() == generator.CAPABILITIES["text-to-mesh"],
                "Capability resolution must prefer explicit runtime node ids over owner directory names",
            )

        explicit_attr_instance = generator.Trellis2Generator.__new__(generator.Trellis2Generator)
        explicit_attr_instance.node_id = "texture-mesh"
        explicit_attr_instance.model_dir = Path("/tmp/trellis-2/base-4b")
        explicit_attr_instance.download_check = "texturing_pipeline.json"
        assert_true(
            explicit_attr_instance._capability() == generator.CAPABILITIES["texture-mesh"],
            "Capability resolution must honor an explicitly injected node_id attribute when present",
        )

        fallback_instance = generator.Trellis2Generator.__new__(generator.Trellis2Generator)
        fallback_instance.model_dir = Path("/tmp/trellis-2/unknown-owner")
        fallback_instance.download_check = "texturing_pipeline.json"
        assert_true(
            fallback_instance._capability() == generator.CAPABILITIES["texture-mesh"],
            "Unique download_check values may still resolve capabilities as a fallback",
        )

        ambiguous_instance = generator.Trellis2Generator.__new__(generator.Trellis2Generator)
        ambiguous_instance.model_dir = Path("/tmp/trellis-2/text-xlarge")
        ambiguous_instance.download_check = "pipeline.json"
        with patched_env(env_name, None):
            try:
                ambiguous_instance._capability()
            except RuntimeError as exc:
                assert_true("Ambiguous capability resolution" in str(exc), "Ambiguous download_check fallback must fail loudly")
            else:
                raise AssertionError("Ambiguous download_check fallback should not silently choose the wrong capability")


def test_texture_mesh_generator_dispatch_and_validation() -> None:
    with stubbed_generator_imports():
        generator = load_module("modly_generator_texture_dispatch_validation", "generator.py")

        dispatch_instance = generator.Trellis2Generator.__new__(generator.Trellis2Generator)
        dispatch_instance._generate_image_to_mesh = lambda *args, **kwargs: "image-path"
        dispatch_instance._generate_texture_mesh = lambda *args, **kwargs: "texture-path"

        dispatch_instance._capability = lambda: generator.CAPABILITIES["generate"]
        assert_true(
            dispatch_instance.generate(b"img", {}, None, None) == "image-path",
            "Generate node dispatch must keep image-to-mesh behavior stable",
        )

        dispatch_instance._capability = lambda: generator.CAPABILITIES["texture-mesh"]
        assert_true(
            dispatch_instance.generate(b"img", {}, None, None) == "texture-path",
            "Texture node dispatch must route to texture-mesh implementation",
        )

        validation_instance = generator.Trellis2Generator.__new__(generator.Trellis2Generator)
        try:
            validation_instance._resolve_mesh_path({})
        except RuntimeError as exc:
            assert_true("requires a mesh side-input" in str(exc), "Missing mesh_path must raise a clear validation error")
        else:
            raise AssertionError("Missing mesh_path should fail validation")

        try:
            validation_instance._resolve_mesh_path({"mesh_path": "/definitely/missing/example.glb"})
        except RuntimeError as exc:
            assert_true("Mesh side-input was not found" in str(exc), "Invalid mesh_path must raise a clear validation error")
        else:
            raise AssertionError("Invalid mesh_path should fail validation")

        with tempfile.TemporaryDirectory(prefix="trellis2-texture-mesh-") as tmp:
            mesh_path = Path(tmp) / "input.glb"
            mesh_path.write_bytes(b"mesh")

            relative_instance = generator.Trellis2Generator.__new__(generator.Trellis2Generator)
            relative_instance.outputs_dir = Path(tmp) / "workspace" / "outputs"
            relative_instance.outputs_dir.parent.mkdir(parents=True, exist_ok=True)
            relative_mesh_dir = relative_instance.outputs_dir.parent / "meshes"
            relative_mesh_dir.mkdir(parents=True, exist_ok=True)
            relative_mesh_path = relative_mesh_dir / "input.glb"
            relative_mesh_path.write_bytes(b"mesh")
            assert_true(
                relative_instance._resolve_mesh_path({"mesh_path": "meshes/input.glb"}) == relative_mesh_path,
                "Relative mesh_path must resolve against the runtime workspace directory",
            )

            fake_trimesh = types.ModuleType("trimesh")
            loaded_paths: list[tuple[str, str, bool]] = []
            exported_paths: list[tuple[str, str | None]] = []
            model_calls: list[dict[str, object]] = []
            
            class FakeMesh:
                pass

            loaded_mesh = FakeMesh()

            def fake_load(path: str, force: str | None = None, process: bool = True):
                loaded_paths.append((path, str(force), process))
                return loaded_mesh

            class FakeTexturedMesh:
                def export(self, path: str, file_type: str | None = None):
                    exported_paths.append((path, file_type))

            fake_trimesh.load = fake_load
            fake_trimesh.Trimesh = FakeMesh
            fake_trimesh.Scene = type("FakeScene", (), {})
            original_trimesh = sys.modules.get("trimesh")
            sys.modules["trimesh"] = fake_trimesh

            try:
                runtime_instance = generator.Trellis2Generator.__new__(generator.Trellis2Generator)
                runtime_instance._model = types.SimpleNamespace(
                    run=lambda mesh, image, **kwargs: model_calls.append({"mesh": mesh, "image": image, **kwargs}) or FakeTexturedMesh()
                )
                runtime_instance.outputs_dir = Path(tmp) / "outputs"
                runtime_instance._report = lambda *_args, **_kwargs: None
                runtime_instance._check_cancelled = lambda *_args, **_kwargs: None
                runtime_instance._load_image = lambda _bytes: "decoded-image"
                runtime_instance._run_with_smoothed_progress = lambda _progress_cb, **kwargs: kwargs["run"]()

                output_path = runtime_instance._generate_texture_mesh(
                    b"image-bytes",
                    {
                        "mesh_path": str(mesh_path),
                        "pipeline_type": "512",
                        "tex_steps": 9,
                        "texture_size": 2048,
                        "seed": 7,
                    },
                )
            finally:
                if original_trimesh is None:
                    sys.modules.pop("trimesh", None)
                else:
                    sys.modules["trimesh"] = original_trimesh

            assert_true(loaded_paths == [(str(mesh_path), "mesh", False)], "Texture generation must load the provided mesh path as a mesh without trimesh processing")
            assert_true(len(model_calls) == 1, "Texture generation must invoke the TRELLIS texturing pipeline exactly once")
            assert_true(model_calls[0]["mesh"] is loaded_mesh, "Texture generation must pass the loaded mesh into the TRELLIS texturing pipeline")
            assert_true(model_calls[0]["image"] == "decoded-image", "Texture generation must decode and pass the image prompt")
            assert_true(model_calls[0]["resolution"] == 512, "Texture generation must map pipeline_type onto TRELLIS texturing resolution")
            assert_true(model_calls[0]["texture_size"] == 2048, "Texture generation must forward texture_size")
            assert_true(model_calls[0]["seed"] == 7, "Texture generation must forward seed")
            assert_true(model_calls[0]["tex_slat_sampler_params"] == {"steps": 9}, "Texture generation must forward tex_steps to TRELLIS sampler params")
            assert_true(exported_paths == [(str(output_path), "glb")], "Texture generation must export the resulting textured mesh as GLB")


def test_image_mesh_debug_artifact_tracks_topology_stages() -> None:
    with stubbed_generator_imports():
        generator = load_module("modly_generator_image_debug_validation", "generator.py")

        fake_trimesh = types.ModuleType("trimesh")

        class FakeTrimesh:
            def __init__(self, vertices=None, faces=None, process=False, **_kwargs):
                self.vertices = np.asarray(vertices if vertices is not None else np.zeros((0, 3), dtype=np.float32))
                self.faces = np.asarray(faces if faces is not None else np.zeros((0, 3), dtype=np.int32))
                self.process = process
                self.is_watertight = len(self.faces) > 0

            def split(self, only_watertight=False):
                if len(self.faces) == 0:
                    return []
                half = max(1, len(self.faces) // 2)
                return [
                    FakeTrimesh(self.vertices[: max(3, len(self.vertices) // 2)], self.faces[:half], process=False),
                    FakeTrimesh(self.vertices[: max(3, len(self.vertices) // 3)], self.faces[half:], process=False),
                ]

            def export(self, path: str, extension_webp: bool = True, file_type: str | None = None):
                Path(path).write_bytes(b"glb")

        fake_trimesh.Trimesh = FakeTrimesh
        fake_trimesh.Scene = type("FakeScene", (), {})
        fake_trimesh.util = types.SimpleNamespace(concatenate=lambda geometries: geometries[0])
        fake_trimesh.load = lambda _path, force=None, process=False: FakeTrimesh(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
            process=process,
        )

        fake_o_voxel = types.ModuleType("o_voxel")
        fake_o_voxel.postprocess = types.SimpleNamespace(
            to_glb=lambda **_kwargs: FakeTrimesh(
                vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
                faces=np.array([[0, 1, 2]], dtype=np.int32),
                process=False,
            )
        )

        original_trimesh = sys.modules.get("trimesh")
        original_o_voxel = sys.modules.get("o_voxel")
        sys.modules["trimesh"] = fake_trimesh
        sys.modules["o_voxel"] = fake_o_voxel
        try:
            with tempfile.TemporaryDirectory(prefix="trellis2-image-debug-") as tmp:
                runtime_instance = generator.Trellis2Generator.__new__(generator.Trellis2Generator)
                runtime_instance._model = types.SimpleNamespace(
                    run=lambda *_args, **_kwargs: [types.SimpleNamespace(
                        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
                        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
                        attrs="attrs",
                        coords="coords",
                        layout="layout",
                        voxel_size=1.0,
                        simplify=lambda _target: None,
                    )]
                )
                runtime_instance.outputs_dir = Path(tmp) / "outputs"
                runtime_instance._report = lambda *_args, **_kwargs: None
                runtime_instance._check_cancelled = lambda *_args, **_kwargs: None
                runtime_instance._load_image = lambda _bytes: "decoded-image"
                runtime_instance._run_with_smoothed_progress = lambda _progress_cb, **kwargs: kwargs["run"]()

                output_path = runtime_instance._generate_image_to_mesh(
                    b"image-bytes",
                    {
                        "pipeline_type": "512",
                        "sparse_steps": 6,
                        "shape_steps": 7,
                        "tex_steps": 8,
                        "faces": 111,
                        "texture_size": 2048,
                        "seed": 9,
                    },
                )

                debug_payload = json.loads(output_path.with_suffix(".debug.json").read_text(encoding="utf-8"))
        finally:
            if original_trimesh is None:
                sys.modules.pop("trimesh", None)
            else:
                sys.modules["trimesh"] = original_trimesh
            if original_o_voxel is None:
                sys.modules.pop("o_voxel", None)
            else:
                sys.modules["o_voxel"] = original_o_voxel

        assert_true(debug_payload["capability"] == "image-to-mesh", "Image debug artifact must identify the image-to-mesh capability")
        assert_true(debug_payload["output_glb"] == str(output_path), "Image debug artifact must record the output GLB path")
        assert_true(
            [stage["stage"] for stage in debug_payload["stages"]] == [
                "before_simplify",
                "after_simplify",
                "before_final_to_glb",
                "after_postprocess_to_glb",
                "reloaded_exported_glb",
            ],
            "Image debug artifact must capture every topology checkpoint in order",
        )
        assert_true(debug_payload["stages"][0]["component_count"] == 2, "Topology diagnostics must capture connected-component counts")
        assert_true(debug_payload["stages"][-1]["face_count"] == 1, "Reloaded GLB diagnostics must reflect the exported mesh")


def test_text_mesh_generator_dispatch_and_aux_localization() -> None:
    with stubbed_generator_imports():
        generator = load_module("modly_generator_text_dispatch_validation", "generator.py")

        dispatch_instance = generator.Trellis2Generator.__new__(generator.Trellis2Generator)
        dispatch_instance._generate_image_to_mesh = lambda *args, **kwargs: "image-path"
        dispatch_instance._generate_texture_mesh = lambda *args, **kwargs: "texture-path"
        dispatch_instance._generate_text_to_mesh = lambda *args, **kwargs: "text-path"

        dispatch_instance._capability = lambda: generator.CAPABILITIES["text-to-mesh"]
        assert_true(
            dispatch_instance.generate(b"", {"prompt": "chair"}, None, None) == "text-path",
            "Text node dispatch must route to text-to-mesh implementation",
        )

        prompt_instance = generator.Trellis2Generator.__new__(generator.Trellis2Generator)
        assert_true(prompt_instance._normalize_prompt({"prompt": "  chair  "}) == "chair", "Prompt normalization must prefer params.prompt")
        assert_true(prompt_instance._normalize_prompt({"text": "lamp"}) == "lamp", "Prompt normalization may fall back to params.text")

        try:
            prompt_instance._normalize_prompt({})
        except RuntimeError as exc:
            assert_true("requires params.prompt" in str(exc), "Missing prompt must raise a clear text validation error")
        else:
            raise AssertionError("Missing prompt should fail validation")

        with tempfile.TemporaryDirectory(prefix="trellis2-text-localize-") as tmp:
            model_dir = Path(tmp)
            (model_dir / "pipeline.json").write_text(json.dumps({
                "name": "TrellisTextTo3DPipeline",
                "args": {
                    "models": {
                        "sparse_structure_decoder": "JeffreyXiang/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16",
                        "slat_decoder_mesh": "JeffreyXiang/TRELLIS-image-large/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16",
                        "slat_flow_model": "ckpts/slat_flow_txt_dit_XL_64l8p2_fp16",
                    }
                },
            }, indent=2), encoding="utf-8")

            downloads: list[tuple[str, str]] = []
            fake_hf_root = model_dir / "fake-hf"

            def fake_hf_hub_download(repo_id: str, filename: str) -> str:
                downloads.append((repo_id, filename))
                target = fake_hf_root / repo_id.replace("/", "--") / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(f"stub:{repo_id}:{filename}", encoding="utf-8")
                return str(target)

            fake_hf_module = types.ModuleType("huggingface_hub")
            fake_hf_module.hf_hub_download = fake_hf_hub_download
            original_hf_module = sys.modules.get("huggingface_hub")
            sys.modules["huggingface_hub"] = fake_hf_module
            try:
                localized_config = prompt_instance._prepare_text_pipeline_config(model_dir)
            finally:
                if original_hf_module is None:
                    sys.modules.pop("huggingface_hub", None)
                else:
                    sys.modules["huggingface_hub"] = original_hf_module

            localized_payload = json.loads(localized_config.read_text(encoding="utf-8"))
            localized_models = localized_payload["args"]["models"]
            assert_true(localized_config.name == generator._TEXT_PIPELINE_CONFIG_FILE, "Text pipeline localization must emit the dedicated localized config file")
            assert_true(localized_models["sparse_structure_decoder"].startswith(f"{generator._TEXT_AUX_WEIGHTS_DIR}/JeffreyXiang/TRELLIS-image-large/ckpts/"), "Auxiliary decoders must be rewritten under the text owner path")
            assert_true(localized_models["slat_decoder_mesh"].startswith(f"{generator._TEXT_AUX_WEIGHTS_DIR}/JeffreyXiang/TRELLIS-image-large/ckpts/"), "Mesh decoder must also be localized under the text owner path")
            assert_true(localized_models["slat_flow_model"] == "ckpts/slat_flow_txt_dit_XL_64l8p2_fp16", "Local in-owner checkpoints must not be rewritten")
            assert_true(
                downloads == [
                    ("JeffreyXiang/TRELLIS-image-large", "ckpts/ss_dec_conv3d_16l8_fp16.json"),
                    ("JeffreyXiang/TRELLIS-image-large", "ckpts/ss_dec_conv3d_16l8_fp16.safetensors"),
                    ("JeffreyXiang/TRELLIS-image-large", "ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.json"),
                    ("JeffreyXiang/TRELLIS-image-large", "ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors"),
                ],
                "Auxiliary localization must deterministically hydrate each referenced external checkpoint",
            )

        fake_postprocessing_utils = types.ModuleType("trellis.utils.postprocessing_utils")
        glb_exports: list[str] = []
        to_glb_calls: list[dict[str, object]] = []

        class FakeGlb:
            def export(self, path: str):
                glb_exports.append(path)

        fake_postprocessing_utils.to_glb = lambda gaussian, mesh, **kwargs: to_glb_calls.append({"gaussian": gaussian, "mesh": mesh, **kwargs}) or FakeGlb()
        fake_utils_module = types.ModuleType("trellis.utils")
        fake_utils_module.postprocessing_utils = fake_postprocessing_utils
        fake_trellis_module = types.ModuleType("trellis")
        fake_trellis_module.utils = fake_utils_module

        original_trellis = sys.modules.get("trellis")
        original_trellis_utils = sys.modules.get("trellis.utils")
        original_trellis_post = sys.modules.get("trellis.utils.postprocessing_utils")
        sys.modules["trellis"] = fake_trellis_module
        sys.modules["trellis.utils"] = fake_utils_module
        sys.modules["trellis.utils.postprocessing_utils"] = fake_postprocessing_utils
        try:
            with tempfile.TemporaryDirectory(prefix="trellis2-text-runtime-") as tmp:
                runtime_instance = generator.Trellis2Generator.__new__(generator.Trellis2Generator)
                runtime_instance._model = types.SimpleNamespace(
                    run=lambda prompt, **kwargs: {"gaussian": ["gaussian-out"], "mesh": ["mesh-out"], "prompt": prompt, "kwargs": kwargs}
                )
                runtime_instance.outputs_dir = Path(tmp) / "outputs"
                runtime_instance._report = lambda *_args, **_kwargs: None
                runtime_instance._check_cancelled = lambda *_args, **_kwargs: None
                runtime_instance._run_with_smoothed_progress = lambda _progress_cb, **kwargs: kwargs["run"]()

                output_path = runtime_instance._generate_text_to_mesh(
                    {
                        "prompt": "A wooden chair",
                        "sparse_steps": 11,
                        "slat_steps": 13,
                        "sparse_cfg": 6.5,
                        "slat_cfg": 8.0,
                        "texture_size": 2048,
                        "simplify": 0.9,
                        "seed": 123,
                    }
                )
        finally:
            for name, module in (("trellis", original_trellis), ("trellis.utils", original_trellis_utils), ("trellis.utils.postprocessing_utils", original_trellis_post)):
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        assert_true(to_glb_calls[0]["gaussian"] == "gaussian-out", "Text generation must pass the official gaussian output into TRELLIS postprocessing")
        assert_true(to_glb_calls[0]["mesh"] == "mesh-out", "Text generation must pass the official mesh output into TRELLIS postprocessing")
        assert_true(to_glb_calls[0]["simplify"] == 0.9, "Text generation must forward simplify to official postprocessing")
        assert_true(to_glb_calls[0]["texture_size"] == 2048, "Text generation must forward texture_size to official postprocessing")
        assert_true(glb_exports == [str(output_path)], "Text generation must export the official postprocessed GLB")


def test_dinov3_transformers_compatibility_patch() -> None:
    with stubbed_image_feature_extractor_imports() as fake_tensor:
        module = load_module(
            "modly_image_feature_extractor_validation",
            "vendor/trellis2/modules/image_feature_extractor.py",
        )

        fallback_extractor = module.DinoV3FeatureExtractor.__new__(module.DinoV3FeatureExtractor)
        fallback_extractor.model = types.SimpleNamespace(
            embeddings=types.SimpleNamespace(
                patch_embeddings=types.SimpleNamespace(weight=types.SimpleNamespace(dtype="float16"))
            ),
            __call__=lambda _image: types.SimpleNamespace(last_hidden_state="fallback-last-hidden-state"),
        )

        # types.SimpleNamespace does not dispatch __call__, so wrap it in a callable object.
        class ForwardOnlyModel:
            def __init__(self):
                self.embeddings = fallback_extractor.model.embeddings

            def __call__(self, _image):
                return types.SimpleNamespace(last_hidden_state="fallback-last-hidden-state")

        fallback_extractor.model = ForwardOnlyModel()
        assert_true(
            fallback_extractor.extract_features(fake_tensor()) == "fallback-last-hidden-state",
            "DINOv3 fallback must use outputs.last_hidden_state when transformers no longer exposes top-level .layer",
        )

        manual_extractor = module.DinoV3FeatureExtractor.__new__(module.DinoV3FeatureExtractor)

        class FakeLayer:
            def __call__(self, hidden_states, position_embeddings=None):
                del position_embeddings
                return fake_tensor(value=f"{hidden_states.value}-layered")

        manual_extractor.model = types.SimpleNamespace(
            embeddings=types.SimpleNamespace(
                patch_embeddings=types.SimpleNamespace(weight=types.SimpleNamespace(dtype="float16")),
                __call__=lambda image, bool_masked_pos=None: fake_tensor(value="embedded"),
            ),
            rope_embeddings=lambda image: "rope",
            layer=[FakeLayer()],
        )

        class Embeddings:
            patch_embeddings = types.SimpleNamespace(weight=types.SimpleNamespace(dtype="float16"))

            def __call__(self, image, bool_masked_pos=None):
                del image, bool_masked_pos
                return fake_tensor(value="embedded")

        manual_extractor.model.embeddings = Embeddings()
        normalized = manual_extractor.extract_features(fake_tensor())
        assert_true(
            normalized == {"hidden_states": "embedded-layered", "normalized_shape": (3,)},
            "Manual DINOv3 layer walk must stay available for older transformers layouts",
        )


def main() -> None:
    test_setup_plan_and_attention()
    test_optional_and_core_native_install_contracts()
    test_python_runtime_dependency_contract()
    test_native_build_env_steering_for_arm64_source_builds()
    test_arm64_spconv_source_build_env()
    test_patch_installed_cumm_cuda_discovery()
    test_vendor_precedence_guards()
    test_phase3_manifest_and_docs_contract()
    test_capability_specific_pipeline_loading()
    test_texture_mesh_generator_dispatch_and_validation()
    test_text_mesh_generator_dispatch_and_aux_localization()
    test_dinov3_transformers_compatibility_patch()
    print("validate_harden_arm64_native_setup: OK")


if __name__ == "__main__":
    main()
