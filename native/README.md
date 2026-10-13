# Multiverse native runtime

This optional C++17 extension owns MuJoCo model loading, per-environment
`mjData`, and CPU worker threads. It is not installed by `requirements.txt`.
The base Verse API and Python implementations remain available when the native
extension is absent.

## Build on Windows (PowerShell)

Install Visual Studio 2022 Build Tools with the **Desktop development with
C++** workload, then open a Developer PowerShell for VS 2022:

```powershell
python -m pip install -r requirements_native.txt
$PybindDir = python -m pybind11 --cmakedir
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release `
  -DPython_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" `
  -Dpybind11_DIR=$PybindDir
cmake --build build/native --config Release
python -m pytest -q tests/test_native_physics.py
```

Select the native provider for a supported dynamic verse with
`VerseSpec.params["sim_provider"] = "multiverse_native"`.

## Build on Ubuntu

```bash
python -m pip install -r requirements_native.txt
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build/native --parallel
python -m pytest -q tests/test_native_physics.py
```

MuJoCo's wheel provides the C headers and runtime library. The extension loads
that runtime dynamically, so the build does not need a platform-specific MuJoCo
import library. The optional `sim_workers` parameter controls native CPU lane
workers; by default the engine bounds worker threads by the number of lanes and
available CPU cores.

## Current model coverage

The point-mass and two-link reacher verses use the native solver when selected.
The terrain and warehouse MJCF models are included as scene/physics assets, but
their existing task transition rules remain authoritative while their native
model mappings and renderer integration are brought to parity. `render()`
continues to use each verse's established output path.
