<div align="center">

# Qwen3.5-4B FP8 CUDA Inference Engine

**A text-only, batch-1 inference engine with purpose-built CUDA kernels for the hybrid Qwen3.5-4B architecture on NVIDIA Blackwell GPUs**

**Language: [English](README.md) | [简体中文](README_CN.md)**

</div>

## Status

This repository contains a working research implementation optimized and validated on an NVIDIA GeForce RTX 5060. It runs the language component of Qwen3.5-4B with native E4M3 block-FP8 weights and BF16 for selected unquantized tensors.

The model weights and compiled DLLs are intentionally not stored in the source repository.

## Supported model layout

The engine targets a compatible Qwen3.5-4B checkpoint with:

| Attribute | Value |
|---|---:|
| Transformer layers | 32 |
| Linear-attention layers | 24 |
| Full-attention layers | 8 |
| Hidden size | 2560 |
| MLP intermediate size | 9216 |
| Full-attention heads | 16 Q / 4 KV |
| Full-attention head dimension | 256 |
| Linear-attention heads | 16 key / 32 value |
| Linear-attention head dimension | 128 |
| Vocabulary size | 248320 |
| RoPE | partial rotary, 64 of 256 dimensions |
| Quantization | E4M3, 128 x 128 block scales |

The current implementation is text-only. Vision inputs and MTP are not implemented even if the checkpoint contains those weights.

## Highlights

- Native `float8_e4m3fn` block-FP8 weight loading.
- Custom FP8 GEMV/GEMM, fused MLP, RMSNorm, full-attention, and Gated DeltaNet decode kernels.
- Correct Qwen3.5 RMSNorm convention: `(1 + weight)`.
- Correct 64-dimensional partial RoPE; the remaining 192 dimensions stay unchanged.
- Preallocated decode buffers and GPU-resident greedy token handoff.
- FP32 accumulation where required for numerical stability.
- Chinese QA regression scripts and a 300-step decode benchmark.

## Measured result

Test system: RTX 5060, Windows, batch size 1, greedy decode, native FP8 checkpoint.

- Best repeated decode measurements: approximately **53.9 tokens/s**.
- Peak allocated VRAM: approximately **5.1 GB**.

Performance depends on GPU clocks, driver, CUDA/PyTorch versions, prompt length, and checkpoint layout. This number is not a guarantee for other systems.

## Requirements

- Windows 10/11 x64.
- NVIDIA Blackwell GPU. The default build target is `sm_120` (tested on RTX 5060).
- NVIDIA CUDA Toolkit with `nvcc` available on `PATH`.
- Visual Studio 2022 C++ Build Tools or Community Edition.
- Python 3.10+.
- A CUDA-enabled PyTorch build, plus packages in `requirements.txt`.

Tested Python packages:

```text
torch 2.11.0+cu128
transformers 5.14.1
safetensors 0.8.0
```

## Installation

### 1. Clone and install Python dependencies

```powershell
git clone <your-repository-url>
cd qwen3.5-fp8-cuda
python -m pip install -r requirements.txt
```

Install PyTorch using the command recommended for your CUDA environment if the generic requirement does not select the correct CUDA build.

### 2. Place the model outside Git history

Put a compatible native block-FP8 checkpoint at:

```text
model/qwen3.5-4b-fp8/
```

The directory must contain the model config, tokenizer files, Safetensors index, weight shards, and `weight_scale_inv` tensors with 128 x 128 block scaling. The `model/` directory is ignored by Git.

Optional: place an official BF16 reference checkpoint at:

```text
model/qwen3.5-4b/
```

The BF16 checkpoint is useful for numerical comparison and fallback experiments, but normal native-FP8 execution does not require online conversion from BF16.

### 3. Compile both CUDA libraries

Open an x64 Native Tools Command Prompt for Visual Studio 2022 and run:

```bat
compile.bat
```

The default target is `sm_120`. Override it before compiling if needed:

```bat
set CUDA_ARCH=sm_120
compile.bat
```

This generates:

```text
qwen3_infer/csrc/fp8_kernels.dll
qwen3_infer/csrc/fused_kernels.dll
```

If CUDA is installed in a non-standard location, set `CUDA_PATH` or `QWEN_CUDA_BIN`. The DLL paths can also be overridden with `QWEN_FP8_DLL` and `QWEN_FUSED_DLL`.

## Usage

Single question:

```powershell
python main.py --ask "中国的首都是哪里？"
```

Interactive prompt:

```powershell
python main.py
```

Built-in short benchmark:

```powershell
python main.py --benchmark
```

Stable 300-step greedy-decode benchmark:

```powershell
python _benchmark_native_fp8.py
```

Chinese correctness regression:

```powershell
python qa_debug.py
python qa_test.py
```

## Repository contents

```text
.
├── .gitignore
├── LICENSE
├── README.md
├── README_CN.md
├── requirements.txt
├── compile.bat
├── main.py
├── qa_debug.py
├── qa_test.py
├── _benchmark_native_fp8.py
└── qwen3_infer/
    └── csrc/
        ├── fp8_kernels.cu
        └── fused_kernels.cu
```

Do not commit model weights, local `.workbuddy` data, pending-delete folders, profiling reports, caches, or build products. If precompiled Windows DLLs are distributed, attach them to a versioned GitHub Release and clearly state the CUDA version and GPU architecture used to build them.

## Correctness notes

The production path was checked against explicit FP8 block dequantization and an official BF16 reference. Basic FP8 GEMV/GEMM cosine similarity was approximately `0.999997`; the repaired multi-step greedy path matched the BF16 reference on the tested sequence. These checks are implementation evidence, not a general accuracy benchmark.

## Limitations

- Windows DLL loading and MSVC compilation only.
- Batch size 1 and text generation only.
- Kernels are tuned for the tested Qwen3.5-4B tensor shapes and Blackwell architecture.
- Greedy generation is the primary performance path.
- CUDA libraries must be compiled locally or downloaded from a matching release.

## License

Source code in this repository is released under the Apache License 2.0. Model weights are not included and remain subject to their own model license and usage terms.
