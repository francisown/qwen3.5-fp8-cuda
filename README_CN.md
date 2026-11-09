<div align="center">

# Qwen3.5-4B FP8 CUDA 推理引擎

**面向 NVIDIA Blackwell GPU、使用定制 CUDA 算子的 Qwen3.5-4B 混合架构文本推理引擎**

**语言：[English](README.md) | [简体中文](README_CN.md)**

</div>

## 当前状态

这是一个已可运行的研究型实现，针对 NVIDIA GeForce RTX 5060 做过正确性修复和性能优化。引擎运行 Qwen3.5-4B 的语言模型部分，主要权重使用原生 E4M3 block-FP8，少量未量化张量保持 BF16。

模型权重和编译后的 DLL 不放入源码仓库。

## 支持的模型结构

引擎针对以下 Qwen3.5-4B 结构：

| 属性 | 数值 |
|---|---:|
| Transformer 层数 | 32 |
| Linear Attention 层 | 24 |
| Full Attention 层 | 8 |
| Hidden Size | 2560 |
| MLP Intermediate Size | 9216 |
| Full Attention 头 | 16 Q / 4 KV |
| Full Attention Head Dim | 256 |
| Linear Attention 头 | 16 Key / 32 Value |
| Linear Attention Head Dim | 128 |
| 词表大小 | 248320 |
| RoPE | 只旋转 256 维中的前 64 维 |
| 量化方式 | E4M3，128 x 128 block scale |

当前只实现文本推理。即使模型目录包含视觉和 MTP 权重，本引擎也不会使用这些部分。

## 主要特性

- 直接加载原生 `float8_e4m3fn` block-FP8 权重。
- 自定义 FP8 GEMV/GEMM、融合 MLP、RMSNorm、Full Attention 和 Gated DeltaNet Decode Kernel。
- 正确实现 Qwen3.5 的 RMSNorm：权重使用 `(1 + weight)`。
- 正确实现 64 维 partial RoPE，后 192 维保持不变。
- Decode 缓冲区预分配，Greedy Token 保持在 GPU 内传递。
- 对精度敏感的中间计算使用 FP32 累加。
- 提供中文问答回归和 300 步 Decode 基准脚本。

## 实测结果

测试环境：RTX 5060、Windows、Batch Size 1、Greedy Decode、原生 FP8 模型。

- 多轮最佳 Decode：约 **53.9 token/s**。
- 峰值显存：约 **5.1 GB**。

速度会受 GPU 频率、驱动、CUDA/PyTorch 版本、上下文长度和模型权重布局影响，其他机器不保证达到相同结果。

## 环境要求

- Windows 10/11 x64。
- NVIDIA Blackwell GPU；默认编译目标为 `sm_120`，已在 RTX 5060 测试。
- NVIDIA CUDA Toolkit，`nvcc` 已加入 `PATH`。
- Visual Studio 2022 C++ Build Tools 或 Community。
- Python 3.10+。
- 支持 CUDA 的 PyTorch，以及 `requirements.txt` 中的依赖。

当前验证过的 Python 包版本：

```text
torch 2.11.0+cu128
transformers 5.14.1
safetensors 0.8.0
```

## 安装

### 1. 克隆代码并安装依赖

```powershell
git clone <你的仓库地址>
cd qwen3.5-fp8-cuda
python -m pip install -r requirements.txt
```

如果通用依赖没有安装正确的 CUDA 版 PyTorch，请按 PyTorch 官网针对你的 CUDA 环境安装。

### 2. 放置模型，不要提交进 Git

将兼容的原生 block-FP8 模型放到：

```text
model/qwen3.5-4b-fp8/
```

目录内需要包含模型配置、Tokenizer、Safetensors 索引和分片权重；FP8 权重应带有按 128 x 128 block 排列的 `weight_scale_inv`。整个 `model/` 已由 `.gitignore` 排除。

可选的官方 BF16 对照模型可放到：

```text
model/qwen3.5-4b/
```

BF16 模型用于数值对照和回退实验。正常运行现成原生 FP8 模型时，不需要从 BF16 在线重新转换。

### 3. 编译两个 CUDA DLL

打开 Visual Studio 2022 的 x64 Native Tools Command Prompt，运行：

```bat
compile.bat
```

默认目标为 `sm_120`。需要时可在编译前覆盖：

```bat
set CUDA_ARCH=sm_120
compile.bat
```

成功后生成：

```text
qwen3_infer/csrc/fp8_kernels.dll
qwen3_infer/csrc/fused_kernels.dll
```

如果 CUDA 不在默认位置，可设置 `CUDA_PATH` 或 `QWEN_CUDA_BIN`。也可以使用 `QWEN_FP8_DLL` 和 `QWEN_FUSED_DLL` 指定两个 DLL 的路径。

## 使用方法

单次中文提问：

```powershell
python main.py --ask "中国的首都是哪里？"
```

交互输入：

```powershell
python main.py
```

内置短基准：

```powershell
python main.py --benchmark
```

稳定的 300 步 Greedy Decode 基准：

```powershell
python _benchmark_native_fp8.py
```

中文正确性回归：

```powershell
python qa_debug.py
python qa_test.py
```

## GitHub 应上传的结构

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

不要上传模型、`.workbuddy`、`待删除`、`待删除备份-*`、Nsight 报告、缓存或编译产物。若要提供预编译 DLL，推荐放入带版本号的 GitHub Release，并注明编译使用的 CUDA 版本和 GPU 架构。

## 正确性说明

生产路径已与显式 FP8 block 反量化参考及官方 BF16 路径对照。基础 FP8 GEMV/GEMM cosine 约为 `0.999997`；修复后的测试序列与 BF16 的多步 Greedy Token 一致。这些数据是实现验证记录，不代表完整模型评测结果。

## 限制

- 当前仅支持 Windows DLL 加载和 MSVC 编译。
- 当前仅支持 Batch Size 1 和文本生成。
- CUDA Kernel 针对本项目使用的 Qwen3.5-4B 固定张量形状与 Blackwell 架构优化。
- Greedy Decode 是主要性能路径。
- 使用前必须本地编译 DLL，或下载与 CUDA/架构匹配的 Release 二进制。

## 许可证

本仓库源码按 Apache License 2.0 开源。模型权重不包含在仓库中，其许可证和使用条款以模型发布方为准。
