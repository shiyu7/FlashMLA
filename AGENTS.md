# AGENTS.md

## Scope

These instructions apply to the entire repository. `csrc/cutlass` is a Git
submodule maintained upstream; do not edit vendored CUTLASS code unless a task
explicitly requires it.

## Project overview

FlashMLA is a PyTorch CUDA extension for dense and sparse attention kernels.
The main implementation areas are:

- `flash_mla/`: public Python API, validation, and DCP/DCP2 orchestration.
- `csrc/api/`: PyTorch extension bindings.
- `csrc/sm90/` and `csrc/sm100/`: architecture-specific kernels.
- `csrc/extension/sm90/`: dense FP8 DCP and DCP2 extensions.
- `tests/`: correctness tests and PyTorch reference implementations.
- `benchmark/`: performance and crossover benchmarks.
- `docs/`: design notes and implementation deep dives.

Keep Python wrappers, C++ bindings, CUDA declarations, exports in
`flash_mla/__init__.py`, tests, and documentation synchronized when changing a
public API.

## Environment and build

The project requires PyTorch with CUDA support, CUDA 12.8 or newer, and an
SM90 or SM100 GPU. SM100 builds require CUDA 12.9 or newer.

Initialize dependencies and install from the repository root:

```bash
git submodule update --init --recursive
pip install -v .
```

On CUDA 12.8 or when testing only Hopper kernels, disable SM100 compilation:

```bash
FLASH_MLA_DISABLE_SM100=1 pip install -v .
```

The build also recognizes `FLASH_MLA_DISABLE_SM90`,
`FLASH_MLA_DISABLE_FP16`, `FLASH_MLA_DISABLE_FP8`, and `NVCC_THREADS`. Do not
silently change architecture or datatype coverage merely to make a local build
pass; state any build restriction in the handoff.

## Validation

Run the narrowest relevant test first, then expand coverage. Core commands
include:

```bash
pytest -q tests/test_dense_fp8_dcp.py
python tests/test_flash_mla_dense_decoding.py
python tests/test_flash_mla_sparse_decoding.py
python tests/test_flash_mla_sparse_prefill.py
python tests/test_fmha_sm100.py
```

The DCP2 peer-merge test must be launched on two SM90 GPUs:

```bash
torchrun --standalone --nproc-per-node=2 -m pytest -q \
  tests/test_dcp2_peer_merge.py
```

GPU-architecture skips are expected on unsupported hardware. Report the exact
GPU, CUDA version, command, pass/fail/skip result, and any compilation flags.
Do not claim CUDA validation when only import, syntax, or CPU-side checks ran.

Benchmarks are not correctness tests. Run them only when the task concerns
performance, and include warm-up plus before/after measurements for kernel
changes.

## Change guidelines

- Preserve existing public behavior unless the task explicitly changes it.
- Validate tensor device, dtype, shape, stride, and contiguity at the Python or
  C++ boundary before kernel launch.
- Keep asynchronous CUDA code free of accidental host synchronization such as
  `.item()`, implicit device-to-host copies, or unconditional synchronizes on
  hot paths.
- Treat stream ordering, CUDA Graph capture, distributed collectives, peer
  mappings, and buffer lifetimes as correctness constraints.
- Use 64-bit arithmetic for byte offsets and element counts that can exceed
  32-bit ranges. Check launch dimensions and shared-memory requirements.
- Keep architecture-specific code in its existing SM90/SM100 subtree and
  avoid broad refactors while fixing a localized kernel issue.
- Match the surrounding Python and C++/CUDA style. This repository has no
  configured global formatter; avoid unrelated formatting churn.
- Add or update a PyTorch reference comparison for numerical changes. Cover
  boundary shapes, ragged sequence lengths, causal behavior, and relevant
  dtypes.
- Document assumptions that affect cache layout, causal boundaries, rank
  ownership, numerical precision, or supported hardware.

## Working tree discipline

Inspect `git status` before editing. Preserve user changes and do not modify,
discard, stage, or commit unrelated files. Generated build products, shared
objects, caches, benchmark CSV files, and plots are ignored and should not be
added to commits.
