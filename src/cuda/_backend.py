
import glob
import json
import os
import shutil
from subprocess import DEVNULL, call

from torch.utils.cpp_extension import _get_build_directory, load

PATH = os.path.dirname(os.path.abspath(__file__))

def cuda_toolkit_available():
    """Check if the nvcc is avaiable on the machine."""
    try:
        call(["nvcc"], stdout=DEVNULL, stderr=DEVNULL)
        return True
    except FileNotFoundError:
        return False


def cuda_toolkit_version():
    """Get the cuda toolkit version."""
    cuda_home = os.path.join(os.path.dirname(shutil.which("nvcc")), "..")
    if os.path.exists(os.path.join(cuda_home, "version.txt")):
        with open(os.path.join(cuda_home, "version.txt")) as f:
            cuda_version = f.read().strip().split()[-1]
    elif os.path.exists(os.path.join(cuda_home, "version.json")):
        with open(os.path.join(cuda_home, "version.json")) as f:
            cuda_version = json.load(f)["cuda"]["version"]
    else:
        raise RuntimeError("Cannot find the cuda version.")
    return cuda_version


name = "yanf_cuda"
build_dir = _get_build_directory(name, verbose=False)
extra_include_paths = []
extra_cflags = ["-O3"]
extra_cuda_cflags = ["-O3"]

_C = None
sources = list(glob.glob(os.path.join(PATH, "csrc/*.cu"))) + list(glob.glob(os.path.join(PATH, "csrc/*.cpp")))

# if failed, try with JIT compilation
if cuda_toolkit_available():
    if os.listdir(build_dir) != []:
        # If the build exists, we assume the extension has been built
        # and we can load it.

        print("Loading CUDA (This may take a few minutes the first time)")
        _C = load(
            name=name,
            sources=sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_include_paths=extra_include_paths,
        )
        print("Done!")
    else:
        # Build from scratch. Remove the build directory just to be safe: pytorch jit might stuck
        # if the build directory exists.
        shutil.rmtree(build_dir)
        print("Setting up CUDA (This may take a few minutes the first time)")
        _C = load(
            name=name,
            sources=sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_include_paths=extra_include_paths,
        )
        print("Done!")
else:
    print("Error: No CUDA toolkit found.")
    exit(0)

__all__ = ["_C"]
