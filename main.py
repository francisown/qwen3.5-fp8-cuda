"""
main.py — Qwen3.5-4B FP8 推理引擎
支持混合架构: full_attention (8层) + linear_attention / GatedDeltaNet (24层)

- FP8 E4M3 block-wise 量化 (block=128)
- CUDA kernel: FP8 GEMV/Gemm, RMSNorm, fused QKV/GateUp
- Attention: PyTorch (full) + GatedDeltaNet recurrent scan (linear)
"""
import os, sys, time, json, glob, ctypes
import math
import importlib.util

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "qwen3_infer"))

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer

# PyTorch fallbacks for GatedDeltaNet (linear_attention)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    torch_recurrent_gated_delta_rule,
    torch_chunk_gated_delta_rule,
    torch_causal_conv1d_update,
    l2norm,
)

DEV = "cuda"
FP8_DIR = os.path.join(os.path.dirname(__file__), "model", "qwen3.5-4b-fp8")
BF16_DIR = os.path.join(os.path.dirname(__file__), "model", "qwen3.5-4b")
MAX_NEW = 512
BLOCK = 128
TEMP = 0.8
TOP_P = 0.9
REP_PENALTY = 1.1  # repetition penalty (>1.0 penalizes repeated tokens)


def compile_if_available(fn):
    """Compile decode math when Triton is installed; otherwise stay runnable.

    PyTorch's default Inductor CUDA backend requires Triton, which is commonly
    unavailable on native Windows. Deferring this decision at import time avoids
    a fatal error on the first generated token.
    """
    if importlib.util.find_spec("triton") is None:
        return fn
    return torch.compile(fn, mode="reduce-overhead")

def top_p_sample(logits, temperature=0.7, top_p=0.9, rep_penalty=1.0, recent_ids=None):
    """Sample with temperature + top-p (nucleus) + repetition penalty.
    Uses topk + sort on subset to avoid O(vocab log vocab) full sort.
    logits: [vocab] GPU tensor. Returns token id (int)."""
    with torch.no_grad():
        if temperature > 0:
            logits = logits / temperature
        # Apply rep penalty: do it before topk to ensure penalized tokens might still be in topk
        if rep_penalty > 1.0 and recent_ids:
            unique_ids = torch.as_tensor(list(set(recent_ids)), dtype=torch.long, device=logits.device)
            logits = logits.clone()
            repeated = logits[unique_ids]
            logits[unique_ids] = torch.where(
                repeated < 0, repeated * rep_penalty, repeated / rep_penalty
            )
        # Nucleus sampling only needs the high-probability candidates. Selecting
        # logits first avoids materializing a 248K-element softmax result.
        topk_logits, topk_indices = logits.topk(min(2000, logits.numel()))
        sorted_logits, order_in_topk = topk_logits.sort(descending=True)
        order = topk_indices[order_in_topk]
        # Normalize against the full vocabulary so the nucleus cutoff matches
        # the original full-softmax implementation exactly (up to round-off).
        log_partition = torch.logsumexp(logits, dim=-1)
        sorted_probs = torch.exp(sorted_logits - log_partition)
        cumsum = sorted_probs.cumsum(dim=-1)
        if cumsum[-1] < top_p:
            # Extremely flat distributions may need more than 2K candidates.
            # Fall back to the exact full sort rather than truncating mass.
            sorted_logits, order = logits.sort(descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumsum = sorted_probs.cumsum(dim=-1)
        keep = cumsum <= top_p
        if not keep.any():
            keep[0] = True
        sorted_probs[~keep] = 0.0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        sampled_idx = int(torch.multinomial(sorted_probs, 1).item())
        token_id = int(order[sampled_idx].item())
    return token_id

# ── 加载 FP8 kernel DLL ──
# On Windows, make the CUDA runtime discoverable without hard-coding a
# developer-specific installation path. QWEN_CUDA_BIN takes precedence.
if os.name == "nt":
    _cuda_bin_candidates = [
        os.environ.get("QWEN_CUDA_BIN"),
        os.path.join(os.environ.get("CUDA_PATH", ""), "bin"),
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin",
    ]
    for _cuda_bin in _cuda_bin_candidates:
        if _cuda_bin and os.path.isdir(_cuda_bin):
            os.add_dll_directory(_cuda_bin)
            break

_fp8_dll_path = os.environ.get(
    "QWEN_FP8_DLL",
    os.path.join(os.path.dirname(__file__), "qwen3_infer", "csrc", "fp8_kernels.dll"),
)
_fp8_dll = ctypes.CDLL(_fp8_dll_path)
_fp8_dll.fp8_gemm_batch.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p
]

# ── 加载融合 kernel DLL ──
_fused_dll_path = os.environ.get(
    "QWEN_FUSED_DLL",
    os.path.join(os.path.dirname(__file__), "qwen3_infer", "csrc", "fused_kernels.dll"),
)
_fused_dll = ctypes.CDLL(_fused_dll_path)
_fused_dll.launch_gemv_v4_extern.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p
]
_HAS_LINEAR_IN_PROJ4 = hasattr(_fused_dll, "launch_linear_in_proj4")
if _HAS_LINEAR_IN_PROJ4:
    _fused_dll.launch_linear_in_proj4.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_void_p,
    ]
_fused_dll.launch_fused_qkv.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p
]
_fused_dll.launch_fused_gate_up.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p
]
_fused_dll.launch_rmsnorm_ip.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_float,
    ctypes.c_void_p
]
_HAS_RMSNORM_OUT = (
    hasattr(_fused_dll, "launch_rmsnorm_out")
    and os.environ.get("QWEN_CUDA_PREFILL_RMSNORM") == "1"
)
if _HAS_RMSNORM_OUT:
    _fused_dll.launch_rmsnorm_out.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p,
    ]
_fused_dll.launch_fused_attn_v2.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p
]
_fused_dll.launch_fused_attn.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p
]
_HAS_MID_ATTN = hasattr(_fused_dll, "layer_mid_attn")
if _HAS_MID_ATTN:
    _fused_dll.layer_mid_attn.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
    ]
_fused_dll.layer_pre_attn.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,  # hidden, norm_weight
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,  # w_q, s_q, N_q, N_q_blk
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,  # w_k, s_k, N_k, N_k_blk
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,  # w_v, s_v, N_v, N_v_blk
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # q_out, k_out, v_out
    ctypes.c_void_p,  # norm_ws
    ctypes.c_int, ctypes.c_int,  # h_dim, K_blk
    ctypes.c_void_p  # stream
]
_fused_dll.layer_post_attn.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,  # hidden, attn_out
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,  # w_o, s_o, N_o, N_o_blk
    ctypes.c_void_p,  # norm_weight
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,  # w_g, s_g, N_g, N_g_blk
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,  # w_u, s_u, N_u, N_u_blk
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,  # w_d, s_d, N_d, N_d_blk
    ctypes.c_void_p,  # output
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # ws, gate_buf, up_buf, silu_buf
    ctypes.c_int, ctypes.c_int,  # h_dim, attn_dim
    ctypes.c_int, ctypes.c_int, ctypes.c_int,  # K_blk_o, K_blk_gu, K_blk_d
    ctypes.c_void_p  # stream
]
_HAS_MIXED_POST = hasattr(_fused_dll, "layer_post_o_fp8_mlp")
if _HAS_MIXED_POST:
    _fused_dll.layer_post_o_fp8_mlp.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
_HAS_LINEAR_GATE_KERNELS = (
    hasattr(_fused_dll, "launch_prepare_linear_gates")
    and hasattr(_fused_dll, "launch_rmsnorm_gated_decode")
)
if _HAS_LINEAR_GATE_KERNELS:
    _fused_dll.launch_prepare_linear_gates.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
    ]
    _fused_dll.launch_rmsnorm_gated_decode.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p,
    ]
_HAS_FUSED_DELTA = (
    hasattr(_fused_dll, "launch_delta_rule_decode")
    and os.environ.get("QWEN_DISABLE_FUSED_DELTA") != "1"
)
if _HAS_FUSED_DELTA:
    _fused_dll.launch_delta_rule_decode.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]

def fp8_linear(input_bf16, w_bytes, w_scale_inv):
    """FP8 Linear: output = input @ W^T (GEMV v4 for decode, batch GEMM for prefill)"""
    orig_shape = input_bf16.shape
    in_dim = orig_shape[-1]
    out_dim = w_bytes.shape[0]
    x_2d = input_bf16.reshape(-1, in_dim)
    M = x_2d.shape[0]
    k_blocks = w_scale_inv.shape[1]
    # Negative K-block count marks [N, K_blocks] per-row-per-block scales.
    kernel_k_blocks = -k_blocks if w_scale_inv.shape[0] == out_dim and k_blocks > 1 else k_blocks

    y = torch.empty(M, out_dim, dtype=torch.bfloat16, device=DEV)
    if M == 1:
        # Decode: GEMV v4
        _fused_dll.launch_gemv_v4_extern(
            ctypes.c_void_p(x_2d.data_ptr()), ctypes.c_void_p(w_bytes.data_ptr()),
            ctypes.c_void_p(w_scale_inv.data_ptr()), ctypes.c_void_p(y.data_ptr()),
            ctypes.c_int(out_dim), ctypes.c_int(in_dim), ctypes.c_int(kernel_k_blocks),
            ctypes.c_void_p(0)
        )
    else:
        # Prefill: batch GEMM
        _fp8_dll.fp8_gemm_batch(
            ctypes.c_void_p(x_2d.data_ptr()), ctypes.c_void_p(w_bytes.data_ptr()),
            ctypes.c_void_p(w_scale_inv.data_ptr()), ctypes.c_void_p(y.data_ptr()),
            ctypes.c_int(M), ctypes.c_int(out_dim), ctypes.c_int(in_dim),
            ctypes.c_int(kernel_k_blocks), ctypes.c_void_p(0)
        )
    return y.reshape(*orig_shape[:-1], out_dim)


def maybe_fp8_linear(input_bf16, weight):
    """Dispatch: BF16 tensor → F.linear, FP8 tuple (bytes, scale) → fp8_linear"""
    if isinstance(weight, tuple):
        w_bytes, w_scale = weight
        return fp8_linear(input_bf16, w_bytes, w_scale)
    return F.linear(input_bf16, weight)


def fused_qkv(x_bf16, w_q, s_q, w_k, s_k, w_v, s_v):
    """融合 QKV 投影 (仅 M=1 decode)"""
    K = x_bf16.shape[-1]
    N_q, N_k, N_v = w_q.shape[0], w_k.shape[0], w_v.shape[0]
    K_blk = s_q.shape[1]
    q_out = torch.empty(N_q, dtype=torch.bfloat16, device=DEV)
    k_out = torch.empty(N_k, dtype=torch.bfloat16, device=DEV)
    v_out = torch.empty(N_v, dtype=torch.bfloat16, device=DEV)
    _fused_dll.launch_fused_qkv(
        ctypes.c_void_p(x_bf16.data_ptr()),
        ctypes.c_void_p(w_q.data_ptr()), ctypes.c_void_p(w_k.data_ptr()), ctypes.c_void_p(w_v.data_ptr()),
        ctypes.c_void_p(s_q.data_ptr()), ctypes.c_void_p(s_k.data_ptr()), ctypes.c_void_p(s_v.data_ptr()),
        ctypes.c_void_p(q_out.data_ptr()), ctypes.c_void_p(k_out.data_ptr()), ctypes.c_void_p(v_out.data_ptr()),
        ctypes.c_int(K), ctypes.c_int(N_q), ctypes.c_int(N_k), ctypes.c_int(N_v),
        ctypes.c_int(K_blk), ctypes.c_int(s_q.shape[0]), ctypes.c_int(s_k.shape[0]), ctypes.c_int(s_v.shape[0]),
        ctypes.c_void_p(0)
    )
    return q_out, k_out, v_out


def fused_gate_up(x_bf16, w_g, s_g, w_u, s_u):
    """融合 Gate+Up 投影 (仅 M=1 decode)"""
    K = x_bf16.shape[-1]
    N_g, N_u = w_g.shape[0], w_u.shape[0]
    K_blk = -s_g.shape[1] if s_g.shape[0] == N_g and s_g.shape[1] > 1 else s_g.shape[1]
    gate_out = torch.empty(N_g, dtype=torch.bfloat16, device=DEV)
    up_out = torch.empty(N_u, dtype=torch.bfloat16, device=DEV)
    _fused_dll.launch_fused_gate_up(
        ctypes.c_void_p(x_bf16.data_ptr()),
        ctypes.c_void_p(w_g.data_ptr()), ctypes.c_void_p(w_u.data_ptr()),
        ctypes.c_void_p(s_g.data_ptr()), ctypes.c_void_p(s_u.data_ptr()),
        ctypes.c_void_p(gate_out.data_ptr()), ctypes.c_void_p(up_out.data_ptr()),
        ctypes.c_int(K), ctypes.c_int(N_g), ctypes.c_int(N_u),
        ctypes.c_int(K_blk), ctypes.c_int(s_g.shape[0]), ctypes.c_int(s_u.shape[0]),
        ctypes.c_void_p(0)
    )
    return gate_out, up_out


def l2norm(x, dim=-1, eps=1e-6):
    """L2 normalization (aligned with FLA)"""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


class QwenEngineFP8Opt:
    """优化版 FP8 推理引擎 — Qwen3.5 混合架构（full + linear attention）"""

    def __init__(self, model_dir=FP8_DIR, max_seq_len=4096,
                 bf16_mlp_dir=None, bf16_mlp_edge_layers=0,
                 per_row_mlp_dir="auto", per_row_attention_dir=None,
                 fp8_lm_head=False):
        self.max_seq_len = max_seq_len
        # 读取 config（已展平）
        model_cfg = json.load(open(os.path.join(model_dir, "config.json")))
        quant_cfg = model_cfg.get("quantization_config", {})
        self._native_block_fp8 = (
            quant_cfg.get("quant_method") == "fp8"
            and quant_cfg.get("weight_block_size") == [128, 128]
        )
        # Native block-FP8 fused decode remains opt-in until each layer type is
        # numerically validated. Values: "full", "linear", or "all".
        self._native_fp8_fused = os.environ.get(
            "QWEN_NATIVE_FP8_FUSED", ""
        ).strip().lower()
        cfg = model_cfg
        # Handle nested config (online model has text_config wrapper)
        if "text_config" in cfg:
            tc = cfg["text_config"]
            rp = tc.get("rope_parameters", {})
            tc["rope_theta"] = rp.get("rope_theta", 10000000)
            tc["partial_rotary_factor"] = rp.get("partial_rotary_factor", 0.25)
            cfg = tc
        self.n_layers = cfg["num_hidden_layers"]
        self.hidden = cfg["hidden_size"]
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg["head_dim"]
        self.vocab = cfg["vocab_size"]
        self.intermediate = cfg["intermediate_size"]
        self.rope_theta = cfg.get("rope_theta", 10000000)

        self.layer_types = cfg["layer_types"]  # ["full_attention", "linear_attention", ...]
        self.linear_num_v_heads = cfg.get("linear_num_value_heads", 32)
        self.linear_num_k_heads = cfg.get("linear_num_key_heads", 16)
        self.linear_k_head_dim = cfg.get("linear_key_head_dim", 128)
        self.linear_v_head_dim = cfg.get("linear_value_head_dim", 128)
        self.linear_conv_kernel = cfg.get("linear_conv_kernel_dim", 4)
        self.partial_rotary_factor = cfg.get("partial_rotary_factor", 0.25)
        self.head_dim = cfg.get("head_dim", 256)

        # Pre-allocate a weight-relocation buffer BEFORE model loading.
        # On Windows/RTX 4090, cuBLAS selects a slow BF16 matmul kernel when
        # the weight data pointer lies in a specific virtual address range.
        # This buffer (allocated first) ends up at a GOOD range that stays
        # fast for all slices regardless of later model allocations.
        self._has_fix_buf = False
        # Estimate: attention (q/k/v/o + in_proj* + out_proj) + MLP (gate/up/down)
        max_per_layer = (
            self.hidden * self.n_heads * self.head_dim  # q_proj
            + self.hidden * self.n_kv_heads * self.head_dim * 2  # k+v
            + self.hidden * self.n_heads * self.head_dim  # o_proj
            + self.hidden * self.intermediate * 3  # gate/up/down MLP
        )
        self._fix_buf_elms = int(self.n_layers * max_per_layer * 1.2)
        self._fix_buf = None
        if self._fix_buf_elms > 0 and not self._native_block_fp8:
            self._fix_buf = torch.empty(
                self._fix_buf_elms, dtype=torch.bfloat16, device=DEV
            )
            self._has_fix_buf = True

        self._load_weights_fp8(model_dir)
        if per_row_mlp_dir == "auto":
            # A native block-FP8 checkpoint is already production-quantized;
            # never silently replace its MLP with an online BF16 re-quantization.
            per_row_mlp_dir = (
                None if self._native_block_fp8
                else (BF16_DIR if os.path.isdir(BF16_DIR) else None)
            )
        if per_row_mlp_dir:
            self._load_per_row_fp8_mlp(per_row_mlp_dir)
        if per_row_attention_dir == "auto":
            per_row_attention_dir = BF16_DIR if os.path.isdir(BF16_DIR) else None
        if per_row_attention_dir:
            self._load_per_row_fp8_attention(per_row_attention_dir)
        if fp8_lm_head:
            self._enable_fp8_lm_head()
        if bf16_mlp_dir and bf16_mlp_edge_layers:
            self._load_bf16_mlp_overrides(bf16_mlp_dir, bf16_mlp_edge_layers)
        self._fix_weight_addresses()
        self._precompute_rope(max_seq_len)
        self._init_buffers()
        self._reset_cache()

    def _fix_weight_addresses(self):
        """Copy all BF16 weights into the pre-allocated relocation buffer.
        
        The buffer was allocated at the very start of __init__, before any
        model loading, so it occupies a 'good' virtual address range where
        cuBLAS BF16 matmul is fast.  We simply copy each weight into the
        buffer and replace the original dict entry with the buffer slice.
        """
        if not self._has_fix_buf or self._fix_buf is None:
            return
        _FIX_KEYS = {"q_proj","k_proj","v_proj","o_proj",
                     "in_proj_qkv","in_proj_a","in_proj_b","in_proj_z","out_proj",
                     "gate_proj","up_proj","down_proj"}
        entries = []
        for ly in self.layers:
            for k in _FIX_KEYS:
                v = ly.get(k)
                if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                    entries.append((ly, k, v))
        if not entries:
            return
        
        total_elms = sum(e[2].numel() for e in entries)
        if total_elms > self._fix_buf.numel():
            # Safety: should never happen with our generous estimate
            return
        
        # Copy each weight into the pre-allocated buffer
        offset = 0
        for ly, k, v in entries:
            n = v.numel()
            view = self._fix_buf[offset:offset + n].view(v.shape)
            view.copy_(v)
            ly[k] = view
            offset += n
        
        # Also move embed / lm_head into the buffer if they exist.
        # Handle the (common) tied case where embed is lm_head.
        embed_tied = (self.embed is not None and self.lm_head is not None
                      and self.embed.data_ptr() == self.lm_head.data_ptr())
        for attr in ("embed", "lm_head"):
            v = getattr(self, attr, None)
            if not isinstance(v, torch.Tensor) or v.dtype != torch.bfloat16:
                continue
            # Skip lm_head if it's tied to embed (embed was already copied)
            if attr == "lm_head" and embed_tied:
                # Re-point lm_head to the embedded-in-fix-buf view
                self.lm_head = self.embed
                continue
            if v.data_ptr() == self._fix_buf.data_ptr():
                continue  # already in fix_buf
            if offset == 0 or (offset >= self._fix_buf.numel()):
                continue
            n = v.numel()
            if offset + n > self._fix_buf.numel():
                continue
            view = self._fix_buf[offset:offset + n].view(v.shape)
            view.copy_(v)
            setattr(self, attr, view)
            offset += n
        
        # ── Dequantize FP8 MLP weights to BF16 in fix_buf ──
        # The MLP weights are loaded as FP8 tuples (uint8 + scale).  We
        # dequantize them to BF16 here so that cuBLAS F.linear (fast at
        # the fix_buf address range) is used instead of the slow custom
        # FP8 GEMV kernel.
        _MLP_KEYS = ("gate_proj", "up_proj", "down_proj")
        for ly in self.layers:
            for k in _MLP_KEYS:
                v = ly.get(k)
                if not isinstance(v, tuple):
                    continue
                w, s = v  # (uint8 weight, bf16 scale)
                n_blocks = s.shape[1]
                block_size = w.shape[1] // n_blocks
                n_elems = w.numel()
                if offset + n_elems > self._fix_buf.numel():
                    continue  # safety: should not happen with the enlarged buffer
                # Correct dequantize: re-interpret uint8 as float8_e4m3fn,
                # then multiply by per-block scale.  The uint8 values are
                # the bit-pattern encoding of float8_e4m3fn, NOT integers.
                w_fp8 = w.view(torch.float8_e4m3fn)  # (rows, cols) float8
                w_3d = w_fp8.view(w.shape[0], n_blocks, block_size).float()
                s_3d = s.unsqueeze(-1).float()
                self._fix_buf[offset:offset + n_elems].copy_(
                    (w_3d * s_3d).bfloat16().reshape(-1)
                )
                ly[k] = self._fix_buf[offset:offset + n_elems].view(
                    w.shape[0], w.shape[1]
                )
                offset += n_elems

        # Free the original tensors (no longer referenced)
        del entries
        torch.cuda.synchronize()

    # ── 预计算 RoPE ──
    def _precompute_rope(self, max_pos):
        # Full RoPE table (head_dim)
        inv_freq = 1.0 / (self.rope_theta ** (
            torch.arange(0, self.head_dim, 2, device=DEV).float() / self.head_dim
        ))
        self._position_ids = torch.arange(max_pos, dtype=torch.long, device=DEV)
        pos = self._position_ids.float()
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.cos_table = emb.cos().contiguous()
        self.sin_table = emb.sin().contiguous()
        # Partial RoPE table (rotary_dim, for full_attention)
        rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if rotary_dim < self.head_dim:
            inv_freq_p = 1.0 / (self.rope_theta ** (
                torch.arange(0, rotary_dim, 2, device=DEV).float() / rotary_dim
            ))
            freqs_p = torch.outer(pos, inv_freq_p)
            emb_p = torch.cat([freqs_p, freqs_p], dim=-1)
            self._rope_cos = emb_p.cos().contiguous()
            self._rope_sin = emb_p.sin().contiguous()

    def _init_buffers(self):
        """预分配 buffer（避免 decode 时重复分配）"""
        hd = self.hidden
        qm = self.n_heads * self.head_dim  # 4096
        kv = self.n_kv_heads * self.head_dim  # 1024
        im = self.intermediate  # 9728
        q_gated = self.n_heads * self.head_dim * 2  # 8192 gated Q
        mixed_dim = (self.linear_num_k_heads * self.linear_k_head_dim) * 2 \
                    + self.linear_num_v_heads * self.linear_v_head_dim  # 8192 mixed QKV
        self._norm_ws  = torch.empty(hd, dtype=torch.bfloat16, device=DEV)
        self._q_buf    = torch.empty(max(q_gated, mixed_dim), dtype=torch.bfloat16, device=DEV)
        self._k_buf    = torch.empty(kv, dtype=torch.bfloat16, device=DEV)
        self._v_buf    = torch.empty(kv, dtype=torch.bfloat16, device=DEV)
        self._attn_buf = torch.empty(max(qm, self.linear_num_v_heads * self.linear_v_head_dim),
                                     dtype=torch.bfloat16, device=DEV)
        self._o_buf    = torch.empty(hd, dtype=torch.bfloat16, device=DEV)
        self._ws_buf   = torch.empty(max(hd, im), dtype=torch.bfloat16, device=DEV)
        self._gate_buf = torch.empty(im, dtype=torch.bfloat16, device=DEV)
        self._up_buf   = torch.empty(im, dtype=torch.bfloat16, device=DEV)
        self._silu_buf = torch.empty(im, dtype=torch.bfloat16, device=DEV)
        self._res_buf  = torch.empty(hd, dtype=torch.bfloat16, device=DEV)  # residual for decode
        self._logits_buf = torch.empty(self.vocab, dtype=torch.bfloat16, device=DEV)
        self._delta_out_buf = torch.empty(
            self.linear_num_v_heads, self.linear_v_head_dim,
            dtype=torch.bfloat16, device=DEV
        )
        self._linear_g_buf = torch.empty(
            self.linear_num_v_heads, dtype=torch.float32, device=DEV
        )
        self._linear_beta_buf = torch.empty(
            self.linear_num_v_heads, dtype=torch.bfloat16, device=DEV
        )
        self._linear_gated_buf = torch.empty(
            self.linear_num_v_heads * self.linear_v_head_dim,
            dtype=torch.bfloat16, device=DEV
        )

    # ── 加载 FP8 权重（Qwen3.5 混合架构）──
    def _load_weights_fp8(self, md):
        raw = {}
        for fn in sorted(glob.glob(os.path.join(md, "*.safetensors"))):
            with safe_open(fn, "pt") as f:
                for k in f.keys():
                    raw[k] = f.get_tensor(k)

        def load_bf16(k):
            if k not in raw: return None
            return raw[k].to(torch.bfloat16).contiguous().to(DEV)

        def load_fp32(k):
            if k not in raw: return None
            return raw[k].to(torch.float32).contiguous().to(DEV)

        def load_fp8_weight(k):
            """Load weight: returns (fp8_bytes, scale_inv) tuple if FP8,
               or a single BF16 tensor if weight is stored as BF16.
               Consumers use maybe_fp8_linear() which handles both formats."""
            if k not in raw: return None
            t = raw[k]
            scale_key = k.replace(".weight", ".weight_scale_inv")
            if scale_key in raw and t.dtype == torch.float8_e4m3fn:
                w_bytes = t.view(torch.uint8).contiguous().to(DEV)
                w_scale = raw[scale_key].to(torch.bfloat16).contiguous().to(DEV)
                return (w_bytes, w_scale)  # FP8 tuple
            # BF16 weight — keep as BF16 tensor (preserves quality)
            return t.to(torch.bfloat16).contiguous().to(DEV)  # BF16 tensor

        def load_fp8_conv1d(k):
            """conv1d 保持 BF16（非量化）"""
            return load_bf16(k)

        def load_fp8_scalar(k):
            """scalar 参数 (A_log, dt_bias) 保持原 dtype"""
            if k not in raw: return None
            t = raw[k]
            return t.contiguous().to(DEV)

        # ── 全局权重 ──
        # embed 可能在不同前缀
        self.embed = load_bf16("model.language_model.embed_tokens.weight")
        if self.embed is None:
            self.embed = load_bf16("model.embed_tokens.weight")
        if self.embed is None:
            raise KeyError("Missing token embedding weight")

        # lm_head: try BF16 (highest quality) first, fall back to FP8
        self.lm_head = None      # BF16 weight (preferred)
        self.lm_head_fp8 = None  # FP8 weight (fallback)
        self.lm_head_scale = None
        if "lm_head.weight" in raw:
            t = raw["lm_head.weight"]
            if t.dtype == torch.bfloat16:
                self.lm_head = t.contiguous().to(DEV)
                print(f"  lm_head: BF16 {self.lm_head.shape}")
            else:
                loaded_head = load_fp8_weight("lm_head.weight")
                if not isinstance(loaded_head, tuple):
                    raise TypeError("FP8 lm_head is missing block scales")
                self.lm_head_fp8, self.lm_head_scale = loaded_head
                print(f"  lm_head: FP8")
        elif self.embed is not None and self.embed.dtype == torch.bfloat16:
            # Tied embedding: use embed directly as BF16 lm_head (highest quality)
            self.lm_head = self.embed
            print(f"  lm_head: BF16 tied={self.lm_head.shape}")
        else:
            # Fallback: quantize embed to FP8
            if self.embed is not None:
                w_fp8, s_inv = self._quantize_bf16_to_fp8(self.embed)
                self.lm_head_fp8 = w_fp8.view(torch.uint8).contiguous()
                self.lm_head_scale = s_inv
                print(f"  lm_head: FP8 (from embed)")

        self.norm_f = load_fp32("model.language_model.norm.weight")
        if self.norm_f is None:
            self.norm_f = load_fp32("model.norm.weight")

        # ── 逐层加载 ──
        self.layers = []
        for i in range(self.n_layers):
            lt = self.layer_types[i]
            base = f"model.language_model.layers.{i}."
            ly = {}

            # 公共: LayerNorm
            ly["input_layernorm"] = load_fp32(base + "input_layernorm.weight")
            ly["post_attention_layernorm"] = load_fp32(base + "post_attention_layernorm.weight")
            ly["layer_type"] = lt

            if lt == "full_attention":
                ly["q_proj"] = load_fp8_weight(base + "self_attn.q_proj.weight")
                ly["k_proj"] = load_fp8_weight(base + "self_attn.k_proj.weight")
                ly["v_proj"] = load_fp8_weight(base + "self_attn.v_proj.weight")
                ly["o_proj"] = load_fp8_weight(base + "self_attn.o_proj.weight")
                ly["q_norm"] = load_fp32(base + "self_attn.q_norm.weight")
                ly["k_norm"] = load_fp32(base + "self_attn.k_norm.weight")

            elif lt == "linear_attention":
                ly["in_proj_qkv"] = load_fp8_weight(base + "linear_attn.in_proj_qkv.weight")
                ly["in_proj_a"] = load_fp8_weight(base + "linear_attn.in_proj_a.weight")
                ly["in_proj_b"] = load_fp8_weight(base + "linear_attn.in_proj_b.weight")
                ly["in_proj_z"] = load_fp8_weight(base + "linear_attn.in_proj_z.weight")
                ly["out_proj"] = load_fp8_weight(base + "linear_attn.out_proj.weight")
                ly["A_log"] = load_fp8_scalar(base + "linear_attn.A_log")
                ly["dt_bias"] = load_fp8_scalar(base + "linear_attn.dt_bias")
                ly["conv1d_w"] = load_bf16(base + "linear_attn.conv1d.weight")  # [C, 1, 4]
                norm_w = load_fp32(base + "linear_attn.norm.weight")
                if norm_w is None:
                    raise KeyError(base + "linear_attn.norm.weight")
                ly["norm_w"] = norm_w.to(torch.bfloat16)  # [head_v_dim]

            # 公共: MLP (FP8 tuple)
            ly["gate_proj"] = load_fp8_weight(base + "mlp.gate_proj.weight")
            ly["up_proj"] = load_fp8_weight(base + "mlp.up_proj.weight")
            ly["down_proj"] = load_fp8_weight(base + "mlp.down_proj.weight")

            # Concatenate BF16 input projections once. A single GEMM reads the
            # activation once and removes 2-3 CUDA/Python launches per layer.
            if lt == "full_attention":
                projections = (ly["q_proj"], ly["k_proj"], ly["v_proj"])
                if all(isinstance(weight, torch.Tensor) for weight in projections):
                    q_rows = projections[0].shape[0]
                    k_rows = projections[1].shape[0]
                    v_rows = projections[2].shape[0]
                    combined = torch.cat(projections, dim=0).contiguous()
                    ly["qkv_proj_combined"] = combined
                    ly["q_proj"] = combined[:q_rows]
                    ly["k_proj"] = combined[q_rows:q_rows + k_rows]
                    ly["v_proj"] = combined[q_rows + k_rows:q_rows + k_rows + v_rows]
            else:
                projection_names = ("in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z")
                projections = tuple(ly[name] for name in projection_names)
                if all(isinstance(weight, torch.Tensor) for weight in projections):
                    sizes = [weight.shape[0] for weight in projections]
                    combined = torch.cat(projections, dim=0).contiguous()
                    ly["linear_in_proj_combined"] = combined
                    offset = 0
                    for name, size in zip(projection_names, sizes):
                        ly[name] = combined[offset:offset + size]
                        offset += size

            self.layers.append(ly)

        del raw
        torch.cuda.synchronize()
        print(f"  加载完成: {self.n_layers} 层 ({self.layer_types.count('full_attention')} full + {self.layer_types.count('linear_attention')} linear)")
        print(f"  显存: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    def _load_per_row_fp8_mlp(self, model_dir):
        """Quantize BF16 MLP weights with one FP8 scale per output channel."""
        wanted = {}
        for i in range(self.n_layers):
            base = f"model.language_model.layers.{i}.mlp."
            for projection in ("gate_proj", "up_proj", "down_proj"):
                wanted[base + projection + ".weight"] = (i, projection)

        found = set()
        for filename in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
            with safe_open(filename, "pt") as checkpoint:
                for key in checkpoint.keys():
                    target = wanted.get(key)
                    if target is None:
                        continue
                    layer_idx, projection = target
                    weight = checkpoint.get_tensor(key).to(torch.bfloat16).contiguous().to(DEV)
                    self.layers[layer_idx][projection] = self._quantize_per_row_block_fp8(weight)
                    found.add(key)
                    del weight
        missing = wanted.keys() - found
        if missing:
            raise KeyError(f"Missing {len(missing)} BF16 MLP weights in {model_dir}")
        torch.cuda.synchronize()
        print("  MLP quantization: per-output-channel/per-128-input FP8")

    def _enable_fp8_lm_head(self):
        if self.lm_head is None:
            return
        self.lm_head_fp8, self.lm_head_scale = self._quantize_per_row_block_fp8(self.lm_head)
        self.lm_head = None
        torch.cuda.empty_cache()
        print("  lm_head: per-row-block FP8")

    @staticmethod
    def _quantize_per_row_fp8(weight):
        max_abs = weight.abs().amax(dim=1, keepdim=True)
        scale = (max_abs / 448.0).clamp(min=1e-12).to(torch.bfloat16)
        quantized = (weight / scale).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
        return quantized, scale.contiguous()

    @staticmethod
    def _quantize_per_row_block_fp8(weight, block_size=128):
        rows, columns = weight.shape
        if columns % block_size:
            raise ValueError(f"Weight width {columns} is not divisible by {block_size}")
        blocks = weight.view(rows, columns // block_size, block_size)
        max_abs = blocks.abs().amax(dim=2, keepdim=True)
        scale = (max_abs / 448.0).clamp(min=1e-12).to(torch.bfloat16)
        quantized = (blocks / scale).to(torch.float8_e4m3fn)
        return quantized.reshape(rows, columns).view(torch.uint8).contiguous(), scale.squeeze(2).contiguous()

    def _load_per_row_fp8_attention(self, model_dir):
        """Experimental per-channel FP8 attention for the fused decode path."""
        wanted = {}
        for i, layer_type in enumerate(self.layer_types):
            base = f"model.language_model.layers.{i}."
            if layer_type == "full_attention":
                for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    wanted[base + "self_attn." + projection + ".weight"] = (i, projection)
            else:
                for projection in ("in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj"):
                    wanted[base + "linear_attn." + projection + ".weight"] = (i, projection)

        found = set()
        for filename in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
            with safe_open(filename, "pt") as checkpoint:
                for key in checkpoint.keys():
                    target = wanted.get(key)
                    if target is None:
                        continue
                    layer_idx, projection = target
                    weight = checkpoint.get_tensor(key).to(torch.bfloat16).contiguous().to(DEV)
                    self.layers[layer_idx][projection] = self._quantize_per_row_fp8(weight)
                    found.add(key)
                    del weight
        missing = wanted.keys() - found
        if missing:
            raise KeyError(f"Missing {len(missing)} BF16 attention weights in {model_dir}")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("  Attention quantization: per-output-channel FP8 (experimental)")

    def _load_bf16_mlp_overrides(self, model_dir, edge_layers):
        """Restore MLPs in sensitive first/last layers from a BF16 checkpoint."""
        edge_layers = min(int(edge_layers), self.n_layers // 2)
        selected = set(range(edge_layers)) | set(range(self.n_layers - edge_layers, self.n_layers))
        wanted = {}
        for i in selected:
            base = f"model.language_model.layers.{i}.mlp."
            for projection in ("gate_proj", "up_proj", "down_proj"):
                wanted[base + projection + ".weight"] = (i, projection)

        found = set()
        for filename in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
            with safe_open(filename, "pt") as checkpoint:
                for key in checkpoint.keys():
                    target = wanted.get(key)
                    if target is None:
                        continue
                    layer_idx, projection = target
                    self.layers[layer_idx][projection] = (
                        checkpoint.get_tensor(key).to(torch.bfloat16).contiguous().to(DEV)
                    )
                    found.add(key)
        missing = wanted.keys() - found
        if missing:
            raise KeyError(f"Missing {len(missing)} BF16 MLP override weights in {model_dir}")
        torch.cuda.synchronize()
        print(f"  BF16 MLP: {len(selected)} sensitive layers ({sorted(selected)})")

    @staticmethod
    def _quantize_bf16_to_fp8(weight_bf16, block_size=128):
        """将 BF16 权重量化为 FP8 (用于 lm_head)"""
        N, K = weight_bf16.shape
        N_blk = (N + block_size - 1) // block_size
        K_blk = (K + block_size - 1) // block_size
        w_pad = torch.zeros(N_blk * block_size, K_blk * block_size,
                            dtype=weight_bf16.dtype, device=weight_bf16.device)
        w_pad[:N, :K] = weight_bf16
        w_blocks = w_pad.view(N_blk, block_size, K_blk, block_size)
        max_abs = w_blocks.abs().amax(dim=(1, 3))
        scale_inv = (max_abs / 448.0).clamp(min=1e-12).to(torch.bfloat16)
        w_scaled = w_blocks / scale_inv.view(N_blk, 1, K_blk, 1)
        w_fp8 = w_scaled.to(torch.float8_e4m3fn)
        w_fp8 = w_fp8.reshape(N_blk * block_size, K_blk * block_size)
        return w_fp8[:N, :K].contiguous(), scale_inv

    # ── Cache ──
    def _reset_cache(self):
        """重置 KV cache + linear attention states"""
        # Allocate only the state a layer type actually uses, then reuse it.
        # Repeated chat turns no longer churn hundreds of MB through the CUDA
        # allocator. Full-attention cache contents need not be cleared because
        # every readable prefix is overwritten before use.
        if not hasattr(self, "k_cache"):
            cache_shape = (1, self.n_kv_heads, self.max_seq_len, self.head_dim)
            self.k_cache = [
                torch.empty(cache_shape, dtype=torch.bfloat16, device=DEV)
                if lt == "full_attention" else None
                for lt in self.layer_types
            ]
            self.v_cache = [
                torch.empty(cache_shape, dtype=torch.bfloat16, device=DEV)
                if lt == "full_attention" else None
                for lt in self.layer_types
            ]
        # Linear attention states
        key_dim = self.linear_num_k_heads * self.linear_k_head_dim
        value_dim = self.linear_num_v_heads * self.linear_v_head_dim
        conv_dim = key_dim * 2 + value_dim
        # HF cache stores a full kernel-width state. During decode it appends
        # the new token, keeps the last `kernel_size` entries, then convolves.
        ck = self.linear_conv_kernel
        if not hasattr(self, "conv_states"):
            self.conv_states = [
                torch.zeros(1, conv_dim, ck, dtype=torch.bfloat16, device=DEV)
                if lt == "linear_attention" else None
                for lt in self.layer_types
            ]
            self.recurrent_states = [
                torch.zeros(1, self.linear_num_v_heads, self.linear_k_head_dim, self.linear_v_head_dim,
                            dtype=torch.float32, device=DEV)
                if lt == "linear_attention" else None
                for lt in self.layer_types
            ]
        else:
            for i, lt in enumerate(self.layer_types):
                if lt == "linear_attention":
                    conv_state = self.conv_states[i]
                    recurrent_state = self.recurrent_states[i]
                    assert conv_state is not None and recurrent_state is not None
                    conv_state.zero_()
                    recurrent_state.zero_()

    # ── 优化算子 ──
    def rmsnorm(self, x, weight, eps=1e-6):
        """Qwen3.5 RMSNorm with accurate prefill and fast decode paths.

        Full-attention layers amplify tiny prompt-side normalization errors. Use
        the official FP32 formulation whenever the sequence dimension is larger
        than one, while retaining the in-place CUDA kernel for single-token
        decode (including per-head Q/K normalization).
        """
        if weight is None:
            return x
        if x.ndim >= 3 and x.shape[-2] > 1:
            if _HAS_RMSNORM_OUT:
                out = torch.empty_like(x)
                x_2d = x.reshape(-1, x.shape[-1])
                out_2d = out.reshape(-1, out.shape[-1])
                _fused_dll.launch_rmsnorm_out(
                    ctypes.c_void_p(x_2d.data_ptr()),
                    ctypes.c_void_p(weight.data_ptr()),
                    ctypes.c_void_p(out_2d.data_ptr()),
                    ctypes.c_int(x_2d.shape[0]), ctypes.c_int(x_2d.shape[1]),
                    ctypes.c_float(eps), ctypes.c_void_p(0),
                )
                return out
            normed = x.float() * torch.rsqrt(
                x.float().square().mean(-1, keepdim=True) + eps
            )
            return (normed * (1.0 + weight.float())).to(x.dtype)
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        N, D = x_2d.shape
        _fused_dll.launch_rmsnorm_ip(
            ctypes.c_void_p(x_2d.data_ptr()), ctypes.c_void_p(weight.data_ptr()),
            ctypes.c_int(N), ctypes.c_int(D), ctypes.c_float(eps),
            ctypes.c_void_p(0)
        )
        return x

    def apply_rope(self, q, k, position_ids):
        """RoPE — PyTorch"""
        inv_freq = 1.0 / (self.rope_theta ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.int64, device=q.device).float() / self.head_dim
        ))
        inv_freq_expanded = inv_freq[None, :, None].float()
        s = q.shape[2]
        pid_expanded = position_ids[None, None, :].float()
        freqs = (inv_freq_expanded @ pid_expanded).transpose(1, 2).squeeze(0)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(1)
        sin = emb.sin().unsqueeze(0).unsqueeze(1)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def rotate_half(x):
        mid = x.shape[-1] // 2
        return torch.cat((-x[..., mid:], x[..., :mid]), dim=-1)

    def qwen_rmsnorm(self, x, weight, eps=1e-6):
        """Qwen3.5 RMSNorm: (1 + weight) * norm(x) — weight initialized to 0"""
        if weight is None:
            return x
        # Norm in fp32, then apply (1 + weight)
        var = x.float().pow(2).mean(-1, keepdim=True)
        normed = x.float() * torch.rsqrt(var + eps)
        return (normed * (1.0 + weight.float())).to(x.dtype)

    def apply_partial_rope(self, q, k, position_ids):
        """Partial RoPE: only apply to first `rotary_dim` dims"""
        head_dim = q.shape[-1]
        rotary_dim = int(head_dim * self.partial_rotary_factor)
        s = q.shape[2]
        # Lookup cos/sin: [s, rotary_dim] → [1, 1, s, rotary_dim] for proper broadcasting
        cos = self._rope_cos[position_ids].unsqueeze(0).unsqueeze(0)
        sin = self._rope_sin[position_ids].unsqueeze(0).unsqueeze(0)

        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_embed = torch.cat([(q_rot * cos) + (self.rotate_half(q_rot) * sin), q_pass], dim=-1)
        k_embed = torch.cat([(k_rot * cos) + (self.rotate_half(k_rot) * sin), k_pass], dim=-1)
        return q_embed, k_embed

    # ── Full Attention ──
    def full_attention_forward(self, hidden, ly, layer_idx, position_ids, past_len):
        """Qwen3.5 full attention: QKV(gated) → RoPE(partial) → KV cache → attention → gate → O_proj"""
        b, s = hidden.shape[:2]

        # ── QKV projections (FP8 GEMV for FP8 tuples, F.linear for BF16) ──
        def proj_dim(w):
            return w[0].shape[0] if isinstance(w, tuple) else w.shape[0]
        combined_weight = ly.get("qkv_proj_combined")
        if combined_weight is not None:
            projected = F.linear(hidden, combined_weight)
            q_rows = proj_dim(ly["q_proj"])
            k_rows = proj_dim(ly["k_proj"])
            q_all, k, v = torch.split(
                projected, [q_rows, k_rows, proj_dim(ly["v_proj"])], dim=-1
            )
        else:
            q_all = maybe_fp8_linear(hidden, ly["q_proj"])
            k = maybe_fp8_linear(hidden, ly["k_proj"])
            v = maybe_fp8_linear(hidden, ly["v_proj"])

        q_raw, gate = torch.chunk(q_all.view(b, s, -1, self.head_dim * 2), 2, dim=-1)
        q = q_raw.reshape(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        gate = gate.reshape(b, s, -1)
        k = k.view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Q/K Norm (Qwen3.5 style: (1+weight)*norm)
        total_seq = past_len + s
        if s == 1 and _HAS_MID_ATTN:
            # Single-token decode: keep accurate BF16 projections, but fuse Q/K
            # normalization, partial RoPE, KV write and GQA attention.
            q_flat = q.transpose(1, 2).contiguous().reshape(-1)
            k_flat = k.transpose(1, 2).contiguous().reshape(-1)
            v_flat = v.transpose(1, 2).contiguous().reshape(-1)
            _fused_dll.layer_mid_attn(
                ctypes.c_void_p(q_flat.data_ptr()),
                ctypes.c_void_p(k_flat.data_ptr()),
                ctypes.c_void_p(v_flat.data_ptr()),
                ctypes.c_void_p(self._attn_buf.data_ptr()),
                ctypes.c_void_p(ly["q_norm"].data_ptr()) if ly["q_norm"] is not None else ctypes.c_void_p(0),
                ctypes.c_void_p(ly["k_norm"].data_ptr()) if ly["k_norm"] is not None else ctypes.c_void_p(0),
                ctypes.c_void_p(self._rope_cos.data_ptr()),
                ctypes.c_void_p(self._rope_sin.data_ptr()),
                ctypes.c_void_p(self.k_cache[layer_idx].data_ptr()),
                ctypes.c_void_p(self.v_cache[layer_idx].data_ptr()),
                ctypes.c_int(past_len), ctypes.c_int(past_len), ctypes.c_int(total_seq),
                ctypes.c_int(self.max_seq_len), ctypes.c_int(self.head_dim),
                ctypes.c_int(int(self.head_dim * self.partial_rotary_factor)),
                ctypes.c_int(self.n_heads), ctypes.c_int(self.n_kv_heads),
                ctypes.c_void_p(0),
            )
            out = self._attn_buf[:self.n_heads * self.head_dim].view(b, s, -1)
        else:
            if ly["q_norm"] is not None:
                q = self.rmsnorm(q.clone(), ly["q_norm"])
            if ly["k_norm"] is not None:
                k = self.rmsnorm(k.clone(), ly["k_norm"])

            q, k = self.apply_partial_rope(q, k, position_ids)
            self.k_cache[layer_idx][:, :, past_len:past_len + s] = k
            self.v_cache[layer_idx][:, :, past_len:past_len + s] = v

            k_cached = self.k_cache[layer_idx][:, :, :total_seq, :]
            v_cached = self.v_cache[layer_idx][:, :, :total_seq, :]
            g = self.n_heads // self.n_kv_heads
            k_full = k_cached.repeat_interleave(g, dim=1)
            v_full = v_cached.repeat_interleave(g, dim=1)
            scale = self.head_dim ** -0.5
            q_f, k_f, v_f = q.float(), k_full.float(), v_full.float()
            attn = (q_f @ k_f.transpose(-2, -1)) * scale
            if s > 1:
                mask = torch.triu(torch.full((s, total_seq), float("-inf"), device=DEV), diagonal=1 + past_len)
                attn = attn + mask
            attn = F.softmax(attn, dim=-1)
            out = (attn @ v_f).to(torch.bfloat16)
            out = out.transpose(1, 2).contiguous().reshape(b, s, -1)

        # Output gate: attn_out * sigmoid(gate) (gate: [b, s, n_heads*head_dim])
        out = out * torch.sigmoid(gate.float()).to(torch.bfloat16)

        return maybe_fp8_linear(out, ly["o_proj"])

    # ── Linear Attention (Gated Delta Net) ──
    def linear_attention_forward(self, hidden, ly, layer_idx, past_len):
        """Gated Delta Net forward (linear_attention)"""
        b, s = hidden.shape[:2]

        key_dim = self.linear_num_k_heads * self.linear_k_head_dim
        value_dim = self.linear_num_v_heads * self.linear_v_head_dim

        # ── Projections (FP8 GEMV for FP8 tuples, F.linear for BF16) ──
        # Extract dimensions handling both FP8 tuple and BF16 tensor weights
        def proj_dim(w):
            return w[0].shape[0] if isinstance(w, tuple) else w.shape[0]
        combined_weight = ly.get("linear_in_proj_combined")
        if combined_weight is not None:
            projected = F.linear(hidden, combined_weight)
            mixed_qkv, a, b_proj, z = torch.split(
                projected,
                [proj_dim(ly["in_proj_qkv"]), proj_dim(ly["in_proj_a"]),
                 proj_dim(ly["in_proj_b"]), proj_dim(ly["in_proj_z"])],
                dim=-1,
            )
        else:
            mixed_qkv = maybe_fp8_linear(hidden, ly["in_proj_qkv"])
            a = maybe_fp8_linear(hidden, ly["in_proj_a"])
            b_proj = maybe_fp8_linear(hidden, ly["in_proj_b"])
            z = maybe_fp8_linear(hidden, ly["in_proj_z"])

        # ── Conv1d on QKV ──
        mixed_qkv_t = mixed_qkv.transpose(1, 2)  # [b, C, s] for conv
        conv_state = self.conv_states[layer_idx]
        conv_w = ly["conv1d_w"]  # [C, 1, 4]

        if s == 1:
            # Single-token decode: causal_conv1d_update
            mixed_qkv_t = torch_causal_conv1d_update(
                mixed_qkv_t, conv_state,
                conv_w.squeeze(1), None, "silu"
            )
        else:
            # Prefill / chunked decode
            mixed_qkv_t = torch.cat([conv_state, mixed_qkv_t], dim=-1)
            # Update full kernel-width conv state.
            new_conv = F.pad(mixed_qkv_t, (self.linear_conv_kernel - mixed_qkv_t.shape[-1], 0))
            self.conv_states[layer_idx].copy_(new_conv[:, :, -conv_state.shape[-1]:])
            # Causal conv1d
            mixed_qkv_t = F.silu(F.conv1d(mixed_qkv_t, conv_w, padding=0, groups=mixed_qkv.shape[-1]))
            mixed_qkv_t = mixed_qkv_t[:, :, -s:]

        mixed_qkv = mixed_qkv_t.transpose(1, 2)  # [b, s, C]

        # ── Split Q/K/V ──
        query, key, value = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
        query = query.reshape(b, s, -1, self.linear_k_head_dim)
        key = key.reshape(b, s, self.linear_num_k_heads, self.linear_k_head_dim)
        value = value.reshape(b, s, self.linear_num_v_heads, self.linear_v_head_dim)

        # ── Gates ──
        beta = b_proj.sigmoid()
        # g = -exp(A_log) * softplus(a + dt_bias)
        g = -ly["A_log"].float().exp() * F.softplus(a.float() + ly["dt_bias"])

        # Repeat KV heads if needed
        n_repeat = self.linear_num_v_heads // self.linear_num_k_heads
        if n_repeat > 1:
            query = query.repeat_interleave(n_repeat, dim=2)
            key = key.repeat_interleave(n_repeat, dim=2)

        # ── Gated Delta Rule ──
        use_cached = past_len > 0
        initial_state = self.recurrent_states[layer_idx] if use_cached else None

        if s == 1 and use_cached:
            core_out, last_state = self.delta_rule_decode(
                query, key, value, g, beta, initial_state
            )
        else:
            core_out, last_state = torch_chunk_gated_delta_rule(
                query, key, value, g, beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )

        # Update recurrent state
        if last_state is not None:
            self.recurrent_states[layer_idx].copy_(last_state.squeeze(0))

        # ── Output Gate (RMSNormGated) ──
        # norm(x) * silu(z)
        z = z.reshape(-1, self.linear_v_head_dim)
        core_out = core_out.reshape(-1, self.linear_v_head_dim)
        # RMSNorm
        var = core_out.float().pow(2).mean(-1, keepdim=True)
        normed = (core_out.float() * torch.rsqrt(var + 1e-6)).to(core_out.dtype)
        normed = normed * ly["norm_w"]
        gated = normed * F.silu(z)
        gated = gated.reshape(b, s, -1)

        # ── Output Projection ──
        return maybe_fp8_linear(gated, ly["out_proj"])

    # ── Optimized Gated Delta Rule for s=1 (decode) ──
    @staticmethod
    @compile_if_available
    def _delta_rule_kernel(q, k, v, g, beta, initial_state):
        q = l2norm(q, dim=-1, eps=1e-6) * (1.0 / (q.shape[-1] ** 0.5))
        k = l2norm(k, dim=-1, eps=1e-6)
        q = q.squeeze(1).float()
        k = k.squeeze(1).float()
        v = v.squeeze(1).float()
        g = g.squeeze(1).float()
        beta = beta.squeeze(1).float()
        g_exp = g.exp().unsqueeze(-1).unsqueeze(-1)
        s = initial_state.float() * g_exp
        kv_mem = (s * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v - kv_mem) * beta.unsqueeze(-1)
        s = s + k.unsqueeze(-1) * delta.unsqueeze(-2)
        out = (s * q.unsqueeze(-1)).sum(dim=-2)
        return out.unsqueeze(1).to(torch.bfloat16), s

    def delta_rule_decode(self, q, k, v, g, beta, initial_state):
        """Gated Delta Rule for s=1 decode"""
        if _HAS_FUSED_DELTA and self.linear_k_head_dim == self.linear_v_head_dim:
            state = initial_state.reshape(
                self.linear_num_v_heads,
                self.linear_k_head_dim,
                self.linear_v_head_dim,
            )
            _fused_dll.launch_delta_rule_decode(
                q.data_ptr(), k.data_ptr(), v.data_ptr(),
                g.data_ptr(), beta.data_ptr(),
                state.data_ptr(), self._delta_out_buf.data_ptr(),
                self.linear_num_v_heads, self.linear_v_head_dim, 0,
            )
            return self._delta_out_buf.view(
                1, 1, self.linear_num_v_heads, self.linear_v_head_dim
            ), None
        return self._delta_rule_kernel(q, k, v, g, beta, initial_state)

    # ── MLP (共享) ──
    def mlp_forward(self, hidden, ly):
        """SwiGLU MLP: gate_proj → SiLU * up_proj → down_proj"""
        h_2d = hidden.reshape(-1, self.hidden)
        M = h_2d.shape[0]

        if M == 1 and hidden.shape[-2] == 1 and isinstance(ly["gate_proj"], tuple):
            w_g, s_g = ly["gate_proj"]
            w_u, s_u = ly["up_proj"]
            gate_1d, up_1d = fused_gate_up(h_2d[0], w_g, s_g, w_u, s_u)
            gate = gate_1d.view(*hidden.shape[:-1], -1)
            up = up_1d.view(*hidden.shape[:-1], -1)
        else:
            gate = maybe_fp8_linear(hidden, ly["gate_proj"])
            up = maybe_fp8_linear(hidden, ly["up_proj"])
        return maybe_fp8_linear(F.silu(gate) * up, ly["down_proj"])

    # ── BF16 detection (skip optimized decode when BF16 projection weights exist) ──
    _FP8_WEIGHT_KEYS = {"q_proj", "k_proj", "v_proj", "o_proj",
                        "in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj",
                        "gate_proj", "up_proj", "down_proj"}

    def _layer_has_bf16(self, ly):
        """Check if any projection weight in this layer is BF16 (requires generic path)."""
        for k in self._FP8_WEIGHT_KEYS:
            v = ly.get(k)
            if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                return True
        return False

    @staticmethod
    def _has_fp8_mlp(ly):
        return all(isinstance(ly.get(name), tuple) for name in (
            "gate_proj", "up_proj", "down_proj"
        ))

    def _mixed_post_o_fp8_mlp(self, residual, o_out, ly):
        """Residual + accurate decode RMSNorm + FP8 MLP in one DLL call."""
        w_g, s_g = ly["gate_proj"]
        w_u, s_u = ly["up_proj"]
        w_d, s_d = ly["down_proj"]
        k_blk_gu = -s_g.shape[1] if s_g.shape[0] == w_g.shape[0] and s_g.shape[1] > 1 else s_g.shape[1]
        k_blk_d = -s_d.shape[1] if s_d.shape[0] == w_d.shape[0] and s_d.shape[1] > 1 else s_d.shape[1]
        _fused_dll.layer_post_o_fp8_mlp(
            ctypes.c_void_p(residual.data_ptr()),
            ctypes.c_void_p(o_out.data_ptr()),
            ctypes.c_void_p(ly["post_attention_layernorm"].data_ptr()),
            ctypes.c_void_p(w_g.data_ptr()), ctypes.c_void_p(s_g.data_ptr()), ctypes.c_int(w_g.shape[0]),
            ctypes.c_void_p(w_u.data_ptr()), ctypes.c_void_p(s_u.data_ptr()), ctypes.c_int(w_u.shape[0]),
            ctypes.c_void_p(w_d.data_ptr()), ctypes.c_void_p(s_d.data_ptr()), ctypes.c_int(w_d.shape[0]),
            ctypes.c_void_p(self._o_buf.data_ptr()),
            ctypes.c_void_p(self._ws_buf.data_ptr()),
            ctypes.c_void_p(self._gate_buf.data_ptr()),
            ctypes.c_void_p(self._up_buf.data_ptr()),
            ctypes.c_void_p(self._silu_buf.data_ptr()),
            ctypes.c_int(self.hidden), ctypes.c_int(k_blk_gu), ctypes.c_int(k_blk_d),
            ctypes.c_void_p(0),
        )
        return self._o_buf.view(1, 1, self.hidden)

    # ── Decode layer helpers (M=1, uses pre-allocated buffers) ──
    def _decode_layer_full_attn(self, hidden, ly, i, pos, past_len):
        """Full attention decode layer: all operations on pre-allocated buffers, no temporary allocations."""
        hd = self.hidden
        w_q, s_q = ly["q_proj"]
        w_k, s_k = ly["k_proj"]
        w_v, s_v = ly["v_proj"]
        w_o, s_o = ly["o_proj"]

        # ── Pre-attn: save residual + fused RMSNorm + QKV via layer_pre_attn ──
        self._res_buf.copy_(hidden.reshape(-1))
        h_k_blk = hd // 128
        _fused_dll.layer_pre_attn(
            ctypes.c_void_p(hidden.data_ptr()),
            ctypes.c_void_p(ly["input_layernorm"].data_ptr()),
            ctypes.c_void_p(w_q.data_ptr()), ctypes.c_void_p(s_q.data_ptr()),
            ctypes.c_int(w_q.shape[0]), ctypes.c_int(s_q.shape[1]),
            ctypes.c_void_p(w_k.data_ptr()), ctypes.c_void_p(s_k.data_ptr()),
            ctypes.c_int(w_k.shape[0]), ctypes.c_int(s_k.shape[1]),
            ctypes.c_void_p(w_v.data_ptr()), ctypes.c_void_p(s_v.data_ptr()),
            ctypes.c_int(w_v.shape[0]), ctypes.c_int(s_v.shape[1]),
            ctypes.c_void_p(self._q_buf.data_ptr()),
            ctypes.c_void_p(self._k_buf.data_ptr()),
            ctypes.c_void_p(self._v_buf.data_ptr()),
            ctypes.c_void_p(self._ws_buf.data_ptr()),
            ctypes.c_int(hd), ctypes.c_int(h_k_blk),
            ctypes.c_void_p(0)
        )

        # ── Mid: QK Norm → RoPE → KV cache → attention ──
        q_gated = self._q_buf
        q_raw = q_gated[:self.n_heads * self.head_dim]
        gate = q_gated[self.n_heads * self.head_dim:]
        k_1d = self._k_buf

        if ly["q_norm"] is not None:
            self.rmsnorm(q_raw.view(self.n_heads, self.head_dim), ly["q_norm"])
        if ly["k_norm"] is not None:
            self.rmsnorm(k_1d.view(self.n_kv_heads, self.head_dim), ly["k_norm"])

        q = q_raw.view(1, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k_1d.view(1, 1, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self._v_buf.view(1, 1, self.n_kv_heads, self.head_dim).transpose(1, 2)

        rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        cos = self._rope_cos[pos].unsqueeze(0).unsqueeze(0)
        sin = self._rope_sin[pos].unsqueeze(0).unsqueeze(0)
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q = torch.cat([(q_rot * cos) + (self.rotate_half(q_rot) * sin), q_pass], dim=-1)
        k = torch.cat([(k_rot * cos) + (self.rotate_half(k_rot) * sin), k_pass], dim=-1)

        total_seq = past_len + 1
        self.k_cache[i][:, :, past_len] = k[:, :, 0, :]
        self.v_cache[i][:, :, past_len] = v[:, :, 0, :]

        g = self.n_heads // self.n_kv_heads
        k_full = self.k_cache[i][:, :, :total_seq].repeat_interleave(g, dim=1)
        v_full = self.v_cache[i][:, :, :total_seq].repeat_interleave(g, dim=1)
        scale = self.head_dim ** -0.5
        q_f, k_f, v_f = q.float(), k_full.float(), v_full.float()
        attn = (q_f @ k_f.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn_out = (attn @ v_f).to(torch.bfloat16)

        gate_val = torch.sigmoid(gate.float()).to(torch.bfloat16)
        attn_out_1d = (attn_out.reshape(-1) * gate_val.reshape(-1))

        # ── Post-attn: O_proj + residual + norm + GateUpSiLU + Down + residual ──
        h_blk = hd // 128
        gu_n_blk = self.intermediate // 128
        d_n_blk = hd // 128
        _fused_dll.layer_post_attn(
            ctypes.c_void_p(self._res_buf.data_ptr()), ctypes.c_void_p(attn_out_1d.data_ptr()),
            w_o.data_ptr(), s_o.data_ptr(), hd, h_blk,
            ly["post_attention_layernorm"].data_ptr(),
            ly["gate_proj"][0].data_ptr(), ly["gate_proj"][1].data_ptr(), self.intermediate, gu_n_blk,
            ly["up_proj"][0].data_ptr(), ly["up_proj"][1].data_ptr(), self.intermediate, gu_n_blk,
            ly["down_proj"][0].data_ptr(), ly["down_proj"][1].data_ptr(), hd, d_n_blk,
            self._o_buf.data_ptr(),
            self._ws_buf.data_ptr(),
            self._gate_buf.data_ptr(),
            self._up_buf.data_ptr(),
            self._silu_buf.data_ptr(),
            hd, self.n_heads * self.head_dim,
            -s_o.shape[1] if s_o.shape[0] == w_o.shape[0] and s_o.shape[1] > 1 else s_o.shape[1],
            (-ly["gate_proj"][1].shape[1]
             if ly["gate_proj"][1].shape[0] == ly["gate_proj"][0].shape[0]
             and ly["gate_proj"][1].shape[1] > 1 else ly["gate_proj"][1].shape[1]),
            (-ly["down_proj"][1].shape[1]
             if ly["down_proj"][1].shape[0] == ly["down_proj"][0].shape[0]
             and ly["down_proj"][1].shape[1] > 1 else ly["down_proj"][1].shape[1]),
            0
        )
        return self._o_buf.view(1, 1, hd)

    def _decode_layer_linear_attn(self, hidden, ly, i, past_len):
        """Optimized linear_attention decode: individual GEMVs + fused layer_post_attn (pre-allocated buffers)."""
        h_blk = self.hidden // 128
        value_dim = self.linear_num_v_heads * self.linear_v_head_dim
        o_n_blk = self.hidden // 128
        gu_n_blk = self.intermediate // 128
        d_n_blk = self.hidden // 128

        # Save residual + RMSNorm
        self._res_buf.copy_(hidden.reshape(-1))
        self._ws_buf[:self.hidden].copy_(hidden.reshape(-1))
        h_rms = self.rmsnorm(self._ws_buf[:self.hidden].view(1, 1, self.hidden), ly["input_layernorm"])
        h_1d = h_rms.reshape(-1)

        # Projections (using pre-allocated buffers)
        w_qkv, s_qkv = ly["in_proj_qkv"]
        w_a, s_a = ly["in_proj_a"]
        w_b, s_b = ly["in_proj_b"]
        w_z, s_z = ly["in_proj_z"]
        w_o, s_o = ly["out_proj"]

        if _HAS_LINEAR_IN_PROJ4:
            def scale_mode(weight, scale):
                return -scale.shape[1] if scale.shape[0] == weight.shape[0] and scale.shape[1] > 1 else scale.shape[1]
            _fused_dll.launch_linear_in_proj4(
                ctypes.c_void_p(h_1d.data_ptr()),
                ctypes.c_void_p(w_qkv.data_ptr()), ctypes.c_void_p(s_qkv.data_ptr()), ctypes.c_void_p(self._q_buf.data_ptr()), ctypes.c_int(w_qkv.shape[0]), ctypes.c_int(scale_mode(w_qkv, s_qkv)),
                ctypes.c_void_p(w_a.data_ptr()), ctypes.c_void_p(s_a.data_ptr()), ctypes.c_void_p(self._k_buf.data_ptr()), ctypes.c_int(w_a.shape[0]), ctypes.c_int(scale_mode(w_a, s_a)),
                ctypes.c_void_p(w_b.data_ptr()), ctypes.c_void_p(s_b.data_ptr()), ctypes.c_void_p(self._v_buf.data_ptr()), ctypes.c_int(w_b.shape[0]), ctypes.c_int(scale_mode(w_b, s_b)),
                ctypes.c_void_p(w_z.data_ptr()), ctypes.c_void_p(s_z.data_ptr()), ctypes.c_void_p(self._attn_buf.data_ptr()), ctypes.c_int(w_z.shape[0]), ctypes.c_int(scale_mode(w_z, s_z)),
                ctypes.c_int(self.hidden), ctypes.c_void_p(0),
            )
        else:
            _fused_dll.launch_gemv_v4_extern(
                ctypes.c_void_p(h_1d.data_ptr()), ctypes.c_void_p(w_qkv.data_ptr()),
                ctypes.c_void_p(s_qkv.data_ptr()), ctypes.c_void_p(self._q_buf.data_ptr()),
                ctypes.c_int(w_qkv.shape[0]), ctypes.c_int(self.hidden),
                ctypes.c_int(s_qkv.shape[1]), ctypes.c_void_p(0)
            )
            _fused_dll.launch_gemv_v4_extern(
                ctypes.c_void_p(h_1d.data_ptr()), ctypes.c_void_p(w_a.data_ptr()),
                ctypes.c_void_p(s_a.data_ptr()), ctypes.c_void_p(self._k_buf.data_ptr()),
                ctypes.c_int(w_a.shape[0]), ctypes.c_int(self.hidden),
                ctypes.c_int(s_a.shape[1]), ctypes.c_void_p(0)
            )
            _fused_dll.launch_gemv_v4_extern(
                ctypes.c_void_p(h_1d.data_ptr()), ctypes.c_void_p(w_b.data_ptr()),
                ctypes.c_void_p(s_b.data_ptr()), ctypes.c_void_p(self._v_buf.data_ptr()),
                ctypes.c_int(w_b.shape[0]), ctypes.c_int(self.hidden),
                ctypes.c_int(s_b.shape[1]), ctypes.c_void_p(0)
            )
            _fused_dll.launch_gemv_v4_extern(
                ctypes.c_void_p(h_1d.data_ptr()), ctypes.c_void_p(w_z.data_ptr()),
                ctypes.c_void_p(s_z.data_ptr()), ctypes.c_void_p(self._attn_buf.data_ptr()),
                ctypes.c_int(w_z.shape[0]), ctypes.c_int(self.hidden),
                ctypes.c_int(s_z.shape[1]), ctypes.c_void_p(0)
            )

        # Conv1d on QKV
        mixed_qkv = self._q_buf[:w_qkv.shape[0]]  # [key_dim*2 + value_dim]
        a = self._k_buf[:w_a.shape[0]]  # [num_v_heads]
        beta = self._v_buf[:w_b.shape[0]]  # [num_v_heads]
        z = self._attn_buf[:w_z.shape[0]]  # [value_dim]
        key_dim = self.linear_num_k_heads * self.linear_k_head_dim
        mixed_qkv_t = mixed_qkv.unsqueeze(0).unsqueeze(-1)  # [1, C, 1]
        conv_out = torch_causal_conv1d_update(
            mixed_qkv_t, self.conv_states[i],
            ly["conv1d_w"].squeeze(1), None, "silu"
        )
        mixed_qkv = conv_out.squeeze(-1).squeeze(0)  # [C]

        # Split Q/K/V
        query_1d, key_1d, value_1d = torch.split(mixed_qkv, [key_dim, key_dim, value_dim])
        query = query_1d.view(1, 1, -1, self.linear_k_head_dim)
        key = key_1d.view(1, 1, self.linear_num_k_heads, self.linear_k_head_dim)
        value = value_1d.view(1, 1, self.linear_num_v_heads, self.linear_v_head_dim)
        a_2d = a.view(1, 1, -1)
        z_2d = z.view(1, 1, -1)

        if _HAS_LINEAR_GATE_KERNELS:
            _fused_dll.launch_prepare_linear_gates(
                ctypes.c_void_p(a.data_ptr()), ctypes.c_void_p(beta.data_ptr()),
                ctypes.c_void_p(ly["A_log"].data_ptr()), ctypes.c_void_p(ly["dt_bias"].data_ptr()),
                ctypes.c_void_p(self._linear_g_buf.data_ptr()),
                ctypes.c_void_p(self._linear_beta_buf.data_ptr()),
                ctypes.c_int(self.linear_num_v_heads), ctypes.c_void_p(0),
            )
            g = self._linear_g_buf.view(1, 1, -1)
            beta_2d = self._linear_beta_buf.view(1, 1, -1)
        else:
            beta_2d = beta.view(1, 1, -1).sigmoid()
            g = -ly["A_log"].float().exp() * F.softplus(a_2d.float() + ly["dt_bias"])

        # Repeat KV heads
        n_repeat = self.linear_num_v_heads // self.linear_num_k_heads
        if n_repeat > 1:
            query = query.repeat_interleave(n_repeat, dim=2)
            key = key.repeat_interleave(n_repeat, dim=2)

        # Delta rule decode
        core_out, last_state = self.delta_rule_decode(
            query, key, value, g, beta_2d, self.recurrent_states[i]
        )
        if last_state is not None:
            self.recurrent_states[i].copy_(last_state.squeeze(0))

        # Output Gate (RMSNormGated)
        v_hd = self.linear_v_head_dim
        if _HAS_LINEAR_GATE_KERNELS:
            _fused_dll.launch_rmsnorm_gated_decode(
                ctypes.c_void_p(core_out.data_ptr()), ctypes.c_void_p(z.data_ptr()),
                ctypes.c_void_p(ly["norm_w"].data_ptr()),
                ctypes.c_void_p(self._linear_gated_buf.data_ptr()),
                ctypes.c_int(self.linear_num_v_heads), ctypes.c_int(v_hd),
                ctypes.c_float(1e-6), ctypes.c_void_p(0),
            )
            gated = self._linear_gated_buf[:value_dim]
        else:
            z_2d_gate = z_2d.reshape(-1, v_hd)
            core_2d = core_out.reshape(-1, v_hd)
            var = core_2d.float().pow(2).mean(-1, keepdim=True)
            normed = (core_2d.float() * torch.rsqrt(var + 1e-6)).to(core_2d.dtype)
            normed = normed * ly["norm_w"]
            gated = (normed * F.silu(z_2d_gate)).reshape(-1)

        # Post-attn: O_proj + residual + norm + fused GateUpSiLU + Down + residual (1 C call)
        _fused_dll.layer_post_attn(
            ctypes.c_void_p(self._res_buf.data_ptr()), ctypes.c_void_p(gated.data_ptr()),
            w_o.data_ptr(), s_o.data_ptr(), self.hidden, o_n_blk,
            ly["post_attention_layernorm"].data_ptr(),
            ly["gate_proj"][0].data_ptr(), ly["gate_proj"][1].data_ptr(), self.intermediate, gu_n_blk,
            ly["up_proj"][0].data_ptr(), ly["up_proj"][1].data_ptr(), self.intermediate, gu_n_blk,
            ly["down_proj"][0].data_ptr(), ly["down_proj"][1].data_ptr(), self.hidden, d_n_blk,
            self._o_buf.data_ptr(),
            self._ws_buf.data_ptr(),
            self._gate_buf.data_ptr(),
            self._up_buf.data_ptr(),
            self._silu_buf.data_ptr(),
            self.hidden, value_dim,
            -s_o.shape[1] if s_o.shape[0] == w_o.shape[0] and s_o.shape[1] > 1 else s_o.shape[1],
            (-ly["gate_proj"][1].shape[1]
             if ly["gate_proj"][1].shape[0] == ly["gate_proj"][0].shape[0]
             and ly["gate_proj"][1].shape[1] > 1 else ly["gate_proj"][1].shape[1]),
            (-ly["down_proj"][1].shape[1]
             if ly["down_proj"][1].shape[0] == ly["down_proj"][0].shape[0]
             and ly["down_proj"][1].shape[1] > 1 else ly["down_proj"][1].shape[1]),
            0
        )
        return self._o_buf.view(1, 1, self.hidden)

    # ── Decode step (预分配缓冲，消除 clone) ──
    def _decode_step(self, hidden, position_ids, past_len):
        """Generic decode step for mixed BF16/FP8 layers — using pre-allocated buffers"""
        hd = self.hidden
        for i in range(self.n_layers):
            ly = self.layers[i]
            lt = ly["layer_type"]
            native_fused = (
                self._native_block_fp8
                and self._native_fp8_fused in (lt.split("_")[0], "all")
            )
            if (self._native_block_fp8 and not native_fused) or self._layer_has_bf16(ly):
                # Save the pre-attention residual and normalize a workspace copy.
                self._res_buf[:hd].copy_(hidden.reshape(-1))
                self._ws_buf[:hd].copy_(hidden.reshape(-1))
                h = self._ws_buf[:hd].view(1, 1, hd)
                h = self.rmsnorm(h, ly["input_layernorm"])
                if lt == "full_attention":
                    h = self.full_attention_forward(h, ly, i, position_ids, past_len)
                elif lt == "linear_attention":
                    h = self.linear_attention_forward(h, ly, i, past_len)

                if _HAS_MIXED_POST and self._has_fp8_mlp(ly):
                    hidden = self._mixed_post_o_fp8_mlp(
                        self._res_buf[:hd], h.reshape(-1), ly
                    )
                else:
                    hidden = self._res_buf[:hd].view(1, 1, hd) + h
                    self._res_buf[:hd].copy_(hidden.reshape(-1))
                    self._ws_buf[:hd].copy_(hidden.reshape(-1))
                    h = self._ws_buf[:hd].view(1, 1, hd)
                    h = self.rmsnorm(h, ly["post_attention_layernorm"])
                    h = self.mlp_forward(h, ly)
                    hidden = self._res_buf[:hd].view(1, 1, hd) + h
            else:
                if lt == "full_attention":
                    hidden = self._decode_layer_full_attn(hidden, ly, i, position_ids, past_len)
                elif lt == "linear_attention":
                    hidden = self._decode_layer_linear_attn(hidden, ly, i, past_len)
        return hidden

    # ── Prefill (batch mode) ──
    def _prefill_step(self, hidden, position_ids, past_len):
        """Prefill path for all layers (batch operations)"""
        for i in range(self.n_layers):
            ly = self.layers[i]
            lt = ly["layer_type"]
            residual = hidden.clone()
            h = self.rmsnorm(hidden, ly["input_layernorm"])
            if lt == "full_attention":
                h = self.full_attention_forward(h, ly, i, position_ids, past_len)
            elif lt == "linear_attention":
                h = self.linear_attention_forward(h, ly, i, past_len)
            hidden = residual + h
            residual = hidden.clone()
            h = self.rmsnorm(hidden, ly["post_attention_layernorm"])
            h = self.mlp_forward(h, ly)
            hidden = residual + h
        return hidden

    # ── 前向传播 ──
    @torch.inference_mode()
    def forward(self, input_ids, past_len=0, logits_to_keep=1):
        """Run inference.

        ``logits_to_keep`` defaults to one because generation only consumes the
        final prompt position. Pass 0 to return logits for every input token.
        """
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids], dtype=torch.long, device=DEV)
        b, s = input_ids.shape
        if past_len < 0 or past_len + s > self.max_seq_len:
            raise ValueError(
                f"Sequence range [{past_len}, {past_len + s}) exceeds "
                f"max_seq_len={self.max_seq_len}"
            )
        embed = self.embed
        if embed is None:
            raise RuntimeError("Token embeddings are not loaded")
        hidden = F.embedding(input_ids, embed)
        position_ids = self._position_ids[past_len:past_len + s]

        if s == 1:
            hidden = self._decode_step(hidden, position_ids, past_len)
        else:
            hidden = self._prefill_step(hidden, position_ids, past_len)

        if logits_to_keep < 0:
            raise ValueError("logits_to_keep must be non-negative")
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:, :]
        hidden = self.rmsnorm(hidden, self.norm_f)
        if self.lm_head is not None:
            logits = F.linear(hidden, self.lm_head)
        else:
            logits = fp8_linear(hidden, self.lm_head_fp8, self.lm_head_scale)
        return logits.reshape(-1, logits.shape[-1])


# ════════════════════════════════════════════
# Benchmark / Chat (adapted from original)
# ════════════════════════════════════════════
def benchmark(eng, tokenizer, prompt, max_new=128):
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    input_ids = torch.tensor([encoded.input_ids], dtype=torch.long, device=DEV)
    seq_len = input_ids.shape[1]
    print(f"  Prompt: {seq_len} tokens")

    eng._reset_cache()
    logits = eng.forward(input_ids[0].tolist(), past_len=0)
    next_id = logits[-1].argmax().item()
    past_len = seq_len
    inp = torch.empty((1, 1), dtype=torch.long, device=DEV)
    for _ in range(16):
        inp[0, 0] = next_id
        logits = eng.forward(inp, past_len=past_len)
        next_id = logits[-1].argmax().item()
        past_len += 1

    eng._reset_cache()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = eng.forward(input_ids[0].tolist(), past_len=0)
    next_id = logits[-1].argmax().item()
    past_len = seq_len
    generated = 1
    stop_ids = {tokenizer.eos_token_id}
    for token in ("<|endoftext|>", "<|im_end|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            stop_ids.add(token_id)
    inp = torch.empty((1, 1), dtype=torch.long, device=DEV)
    for _ in range(max_new - 1):
        inp[0, 0] = next_id
        logits = eng.forward(inp, past_len=past_len)
        next_id = logits[-1].argmax().item()
        past_len += 1
        generated += 1
        if next_id in stop_ids:
            break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    n_gen = generated
    tok_s = n_gen / elapsed
    print(f"  生成: {n_gen} tokens, {elapsed:.2f}s, {tok_s:.1f} tok/s")
    return tok_s


def main():
    torch.set_grad_enabled(False)

    tokenizer = AutoTokenizer.from_pretrained(FP8_DIR, trust_remote_code=True)
    print("加载 Qwen3.5 FP8 优化引擎...")
    t0 = time.time()
    eng = QwenEngineFP8Opt(FP8_DIR)
    print(f"  耗时: {time.time()-t0:.1f}s\n")

    if "--benchmark" in sys.argv:
        benchmark(eng, tokenizer, "What is the capital of France?", max_new=128)
        return

    if "--ask" in sys.argv:
        idx = sys.argv.index("--ask") + 1
        if idx < len(sys.argv):
            prompt = sys.argv[idx]
        else:
            print("Usage: --ask <question>")
            return
    else:
        prompt = None

    stop_ids = {tokenizer.eos_token_id}
    for token in ("<|endoftext|>", "<|im_end|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            stop_ids.add(token_id)
    messages = []
    if prompt is None:
        print("=== Qwen3.5-4B Chat (FP8 引擎) ===\n")
        prompt = input("You: ")

    if prompt.lower() in ("exit", "quit"):
        return
    messages.append({"role": "user", "content": prompt})
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = torch.tensor([encoded.input_ids], dtype=torch.long, device=DEV)
    seq_len = input_ids.shape[1]
    if seq_len >= 1800:
        messages = messages[-2:]
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_ids = torch.tensor([encoded.input_ids], dtype=torch.long, device=DEV)
        seq_len = input_ids.shape[1]
    eng._reset_cache()

    # Prefill: batch (all prompt tokens at once)
    logits = eng.forward(input_ids[0].tolist(), past_len=0)  # tensor [s, vocab]
    past_len = seq_len
    next_id = logits[-1].argmax().item()
    sys.stdout.write("Bot: ")
    sys.stdout.flush()
    response_ids = [next_id]
    recent_tokens = [next_id]  # for repetition penalty (sliding window)
    t0 = time.time()
    buf = "" if next_id in stop_ids else tokenizer.decode([next_id])
    for step_idx in range(MAX_NEW - 1):
        logits = eng.forward([next_id], past_len=past_len)  # [1, vocab]
        if TEMP > 0:
            next_id = top_p_sample(logits[0], TEMP, TOP_P,
                                   rep_penalty=REP_PENALTY, recent_ids=recent_tokens[-50:])
        else:
            next_id = logits[-1].argmax().item()
        response_ids.append(next_id)
        recent_tokens.append(next_id)
        past_len += 1
        if next_id in stop_ids:
            break
        buf += tokenizer.decode([next_id])
        if len(buf) >= 20:
            sys.stdout.write(buf)
            sys.stdout.flush()
            buf = ""
    if buf:
        sys.stdout.write(buf)
        sys.stdout.flush()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    tok_s = len(response_ids) / elapsed
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    print()
    print(f"  [{len(response_ids)} tokens, {tok_s:.1f} tok/s]\n")
    messages.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()
