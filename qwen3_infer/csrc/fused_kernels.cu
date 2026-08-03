// fused_kernels.cu — 融合算子，减少 ctypes 调用次数
// 1. fused_qkv: 一次 GEMV 算 q, k, v 三个投影
// 2. fused_gate_up: 一次 GEMV 算 gate 和 up 两个投影
// 3. rmsnorm: in-place 版本
// 4. layer_pre_attn: 批量提交 RMSNorm + fused QKV (消除 Python gap)
// 5. layer_post_attn: 批量提交 O_proj + residual + RMSNorm + fused GateUp + SiLU_mul + Down + residual
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <math.h>
#include <float.h>

#define BLOCK_DIM 256

// ═══════════════════════════════════════════════════════════════
// FP8 GEMV v3 — vectorized (uint32 weight load) + native FP8
// Grid: [N], Block: 256
// 每个 block 算一个输出行，每个线程每轮处理 4 个元素
// ═══════════════════════════════════════════════════════════════
__global__ void gemv_v3_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ w_fp8,
    const __nv_bfloat16* __restrict__ w_scale,
    __nv_bfloat16* __restrict__ y,
    int K, int K_blocks
) {
    int row = blockIdx.x;
    int block_n = row / 128;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;

    int scale_row_offset = (K_blocks == 1) ? row : block_n * K_blocks;
    const uint8_t* w_row = w_fp8 + row * (size_t)K;
    float sum = 0.0f;

    // 每轮 4 元素，步长 blockDim.x * 4
    for (int k = tid * 4; k < K; k += blockDim.x * 4) {
        // scale 值（所有 4 个元素共享同个 block 的 scale）
        int block_k = k / 128;
        float s = __bfloat162float(w_scale[scale_row_offset + ((K_blocks == 1) ? 0 : block_k)]);

        // uint32 向量化加载 4 个 FP8 权重字节 + 原生 FP8 转换
        uint32_t packed = *(const uint32_t*)(w_row + k);
        __nv_fp8_e4m3* fp8 = (__nv_fp8_e4m3*)&packed;
        float f0 = (float)fp8[0];
        float f1 = (float)fp8[1];
        float f2 = (float)fp8[2];
        float f3 = (float)fp8[3];

        // 输入
        float x0 = __bfloat162float(x[k]);
        float x1 = __bfloat162float(x[k+1]);
        float x2 = __bfloat162float(x[k+2]);
        float x3 = __bfloat162float(x[k+3]);

        sum += (x0 * f0 + x1 * f1 + x2 * f2 + x3 * f3) * s;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);

    __shared__ float shared_sum[8];
    if (lane_id == 0) shared_sum[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        float final_sum = (tid < (blockDim.x >> 5)) ? shared_sum[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            final_sum += __shfl_xor_sync(0xFFFFFFFF, final_sum, offset);
        if (tid == 0) y[row] = __float2bfloat16(final_sum);
    }
}

static inline void launch_gemv_v3(
    const void* x, const void* w, const void* s, void* y,
    int N, int K, int K_blk, cudaStream_t stream
) {
    gemv_v3_kernel<<<N, BLOCK_DIM, 0, stream>>>(
        (const __nv_bfloat16*)x, (const uint8_t*)w,
        (const __nv_bfloat16*)s, (__nv_bfloat16*)y,
        K, K_blk
    );
}

// ═══════════════════════════════════════════════════════════════
// FP8 GEMV v4 — 128-bit 向量化输入 + 权重加载，每线程 8 元素
// Grid: [N], Block: 256
// ═══════════════════════════════════════════════════════════════
__global__ void gemv_v4_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ w_fp8,
    const __nv_bfloat16* __restrict__ w_scale,
    __nv_bfloat16* __restrict__ y,
    int K, int K_blocks
) {
    bool per_row_block = K_blocks < 0;
    K_blocks = per_row_block ? -K_blocks : K_blocks;
    int row = blockIdx.x;
    int block_n = row / 128;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;

    int scale_row_offset = per_row_block ? row * K_blocks
                         : (K_blocks == 1) ? row : block_n * K_blocks;
    const uint8_t* w_row = w_fp8 + row * (size_t)K;
    float sum = 0.0f;

    // 每轮 8 元素，步长 blockDim.x * 8
    for (int k = tid * 8; k < K; k += blockDim.x * 8) {
        int block_k = k / 128;
        float s = __bfloat162float(w_scale[scale_row_offset + ((K_blocks == 1) ? 0 : block_k)]);

        // 128-bit 加载 8 个 FP8 权重字节
        uint32_t packed0 = *(const uint32_t*)(w_row + k);
        uint32_t packed1 = *(const uint32_t*)(w_row + k + 4);
        __nv_fp8_e4m3* fp8_0 = (__nv_fp8_e4m3*)&packed0;
        __nv_fp8_e4m3* fp8_1 = (__nv_fp8_e4m3*)&packed1;
        float f0 = (float)fp8_0[0], f1 = (float)fp8_0[1], f2 = (float)fp8_0[2], f3 = (float)fp8_0[3];
        float f4 = (float)fp8_1[0], f5 = (float)fp8_1[1], f6 = (float)fp8_1[2], f7 = (float)fp8_1[3];

        // 128-bit 向量化加载 8 个 bf16 输入值
        uint4 x128 = *(const uint4*)(x + k);
        float x0 = __bfloat162float(((__nv_bfloat162*)&x128.x)->x);
        float x1 = __bfloat162float(((__nv_bfloat162*)&x128.x)->y);
        float x2 = __bfloat162float(((__nv_bfloat162*)&x128.y)->x);
        float x3 = __bfloat162float(((__nv_bfloat162*)&x128.y)->y);
        float x4 = __bfloat162float(((__nv_bfloat162*)&x128.z)->x);
        float x5 = __bfloat162float(((__nv_bfloat162*)&x128.z)->y);
        float x6 = __bfloat162float(((__nv_bfloat162*)&x128.w)->x);
        float x7 = __bfloat162float(((__nv_bfloat162*)&x128.w)->y);

        sum += (x0*f0 + x1*f1 + x2*f2 + x3*f3 + x4*f4 + x5*f5 + x6*f6 + x7*f7) * s;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);

    __shared__ float shared_sum[8];
    if (lane_id == 0) shared_sum[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        float final_sum = (tid < (blockDim.x >> 5)) ? shared_sum[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            final_sum += __shfl_xor_sync(0xFFFFFFFF, final_sum, offset);
        if (tid == 0) y[row] = __float2bfloat16(final_sum);
    }
}

static inline void launch_gemv_v4(
    const void* x, const void* w, const void* s, void* y,
    int N, int K, int K_blk, cudaStream_t stream
) {
    gemv_v4_kernel<<<N, BLOCK_DIM, 0, stream>>>(
        (const __nv_bfloat16*)x, (const uint8_t*)w,
        (const __nv_bfloat16*)s, (__nv_bfloat16*)y,
        K, K_blk
    );
}

// Four independent projections sharing one input and one kernel launch.
// Used by Qwen3.5 linear attention for qkv/a/b/z decode projections.
__global__ void linear_in_proj4_v4_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ w0, const __nv_bfloat16* __restrict__ s0, __nv_bfloat16* __restrict__ y0, int n0, int kb0,
    const uint8_t* __restrict__ w1, const __nv_bfloat16* __restrict__ s1, __nv_bfloat16* __restrict__ y1, int n1, int kb1,
    const uint8_t* __restrict__ w2, const __nv_bfloat16* __restrict__ s2, __nv_bfloat16* __restrict__ y2, int n2, int kb2,
    const uint8_t* __restrict__ w3, const __nv_bfloat16* __restrict__ s3, __nv_bfloat16* __restrict__ y3, int n3, int kb3,
    int K
) {
    int global_row = blockIdx.x;
    const uint8_t* w;
    const __nv_bfloat16* scale;
    __nv_bfloat16* y;
    int row;
    int K_blocks;
    if (global_row < n0) {
        w = w0; scale = s0; y = y0; row = global_row; K_blocks = kb0;
    } else if (global_row < n0 + n1) {
        w = w1; scale = s1; y = y1; row = global_row - n0; K_blocks = kb1;
    } else if (global_row < n0 + n1 + n2) {
        w = w2; scale = s2; y = y2; row = global_row - n0 - n1; K_blocks = kb2;
    } else {
        w = w3; scale = s3; y = y3; row = global_row - n0 - n1 - n2; K_blocks = kb3;
    }

    bool per_row_block = K_blocks < 0;
    K_blocks = per_row_block ? -K_blocks : K_blocks;
    int scale_row_offset = per_row_block ? row * K_blocks
                         : (K_blocks == 1) ? row : (row / 128) * K_blocks;
    const uint8_t* w_row = w + row * (size_t)K;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    float sum = 0.0f;

    for (int k = tid * 8; k < K; k += blockDim.x * 8) {
        int block_k = k / 128;
        float sc = __bfloat162float(scale[scale_row_offset + ((K_blocks == 1) ? 0 : block_k)]);
        uint32_t p0 = *(const uint32_t*)(w_row + k);
        uint32_t p1 = *(const uint32_t*)(w_row + k + 4);
        __nv_fp8_e4m3* f0 = (__nv_fp8_e4m3*)&p0;
        __nv_fp8_e4m3* f1 = (__nv_fp8_e4m3*)&p1;
        uint4 xv = *(const uint4*)(x + k);
        float x0 = __bfloat162float(((__nv_bfloat162*)&xv.x)->x);
        float x1 = __bfloat162float(((__nv_bfloat162*)&xv.x)->y);
        float x2 = __bfloat162float(((__nv_bfloat162*)&xv.y)->x);
        float x3 = __bfloat162float(((__nv_bfloat162*)&xv.y)->y);
        float x4 = __bfloat162float(((__nv_bfloat162*)&xv.z)->x);
        float x5 = __bfloat162float(((__nv_bfloat162*)&xv.z)->y);
        float x6 = __bfloat162float(((__nv_bfloat162*)&xv.w)->x);
        float x7 = __bfloat162float(((__nv_bfloat162*)&xv.w)->y);
        sum += (x0*(float)f0[0] + x1*(float)f0[1] + x2*(float)f0[2] + x3*(float)f0[3]
              + x4*(float)f1[0] + x5*(float)f1[1] + x6*(float)f1[2] + x7*(float)f1[3]) * sc;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    __shared__ float shared_sum[8];
    if (lane_id == 0) shared_sum[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
        float final_sum = (tid < (blockDim.x >> 5)) ? shared_sum[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            final_sum += __shfl_xor_sync(0xFFFFFFFF, final_sum, offset);
        if (tid == 0) y[row] = __float2bfloat16(final_sum);
    }
}

// ═══════════════════════════════════════════════════════════════
// Fused QKV kernel (v4: 128-bit 输入加载)
// ═══════════════════════════════════════════════════════════════
__global__ void fused_qkv_v4_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ w_q, const uint8_t* __restrict__ w_k, const uint8_t* __restrict__ w_v,
    const __nv_bfloat16* __restrict__ s_q, const __nv_bfloat16* __restrict__ s_k, const __nv_bfloat16* __restrict__ s_v,
    __nv_bfloat16* __restrict__ q_out, __nv_bfloat16* __restrict__ k_out, __nv_bfloat16* __restrict__ v_out,
    int K, int N_q, int N_k, int N_v,
    int K_blk, int N_q_blk, int N_k_blk, int N_v_blk
) {
    bool per_row_block = K_blk < 0;
    K_blk = per_row_block ? -K_blk : K_blk;
    int row = blockIdx.x, tid = threadIdx.x;
    int warp_id = tid >> 5, lane_id = tid & 31;
    const uint8_t* w; const __nv_bfloat16* s; __nv_bfloat16* out; int local_row;
    if (row < N_q) { w = w_q; s = s_q; out = q_out; local_row = row; }
    else if (row < N_q + N_k) { w = w_k; s = s_k; out = k_out; local_row = row - N_q; }
    else { w = w_v; s = s_v; out = v_out; local_row = row - N_q - N_k; }

    int block_n = local_row / 128;
    int scale_row_offset = per_row_block ? local_row * K_blk
                         : (K_blk == 1) ? local_row : block_n * K_blk;
    const uint8_t* w_row = w + local_row * (size_t)K;
    float sum = 0.0f;
    for (int k = tid * 8; k < K; k += blockDim.x * 8) {
        int block_k = k / 128;
        float scale = __bfloat162float(s[scale_row_offset + ((K_blk == 1) ? 0 : block_k)]);

        uint32_t packed0 = *(const uint32_t*)(w_row + k);
        uint32_t packed1 = *(const uint32_t*)(w_row + k + 4);
        __nv_fp8_e4m3* fp8_0 = (__nv_fp8_e4m3*)&packed0;
        __nv_fp8_e4m3* fp8_1 = (__nv_fp8_e4m3*)&packed1;
        float f0 = (float)fp8_0[0], f1 = (float)fp8_0[1], f2 = (float)fp8_0[2], f3 = (float)fp8_0[3];
        float f4 = (float)fp8_1[0], f5 = (float)fp8_1[1], f6 = (float)fp8_1[2], f7 = (float)fp8_1[3];

        uint4 x128 = *(const uint4*)(x + k);
        float x0 = __bfloat162float(((__nv_bfloat162*)&x128.x)->x);
        float x1 = __bfloat162float(((__nv_bfloat162*)&x128.x)->y);
        float x2 = __bfloat162float(((__nv_bfloat162*)&x128.y)->x);
        float x3 = __bfloat162float(((__nv_bfloat162*)&x128.y)->y);
        float x4 = __bfloat162float(((__nv_bfloat162*)&x128.z)->x);
        float x5 = __bfloat162float(((__nv_bfloat162*)&x128.z)->y);
        float x6 = __bfloat162float(((__nv_bfloat162*)&x128.w)->x);
        float x7 = __bfloat162float(((__nv_bfloat162*)&x128.w)->y);

        sum += (x0*f0 + x1*f1 + x2*f2 + x3*f3 + x4*f4 + x5*f5 + x6*f6 + x7*f7) * scale;
    }
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    __shared__ float shared_sum[8];
    if (lane_id == 0) shared_sum[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
        float final_sum = (tid < (blockDim.x >> 5)) ? shared_sum[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) final_sum += __shfl_xor_sync(0xFFFFFFFF, final_sum, offset);
        if (tid == 0) out[local_row] = __float2bfloat16(final_sum);
    }
}

// Fused Gate+Up kernel v4 (128-bit input loads)
__global__ void fused_gate_up_v4_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ w_gate, const uint8_t* __restrict__ w_up,
    const __nv_bfloat16* __restrict__ s_gate, const __nv_bfloat16* __restrict__ s_up,
    __nv_bfloat16* __restrict__ gate_out, __nv_bfloat16* __restrict__ up_out,
    int K, int N_gate, int N_up, int K_blk, int N_gate_blk, int N_up_blk
) {
    bool per_row_block = K_blk < 0;
    K_blk = per_row_block ? -K_blk : K_blk;
    int row = blockIdx.x, tid = threadIdx.x;
    int warp_id = tid >> 5, lane_id = tid & 31;
    const uint8_t* w; const __nv_bfloat16* s; __nv_bfloat16* out; int local_row;
    if (row < N_gate) { w = w_gate; s = s_gate; out = gate_out; local_row = row; }
    else { w = w_up; s = s_up; out = up_out; local_row = row - N_gate; }

    int block_n = local_row / 128;
    int scale_row_offset = per_row_block ? local_row * K_blk
                         : (K_blk == 1) ? local_row : block_n * K_blk;
    const uint8_t* w_row = w + local_row * (size_t)K;
    float sum = 0.0f;
    for (int k = tid * 8; k < K; k += blockDim.x * 8) {
        int block_k = k / 128;
        float scale = __bfloat162float(s[scale_row_offset + ((K_blk == 1) ? 0 : block_k)]);

        uint32_t packed0 = *(const uint32_t*)(w_row + k);
        uint32_t packed1 = *(const uint32_t*)(w_row + k + 4);
        __nv_fp8_e4m3* fp8_0 = (__nv_fp8_e4m3*)&packed0;
        __nv_fp8_e4m3* fp8_1 = (__nv_fp8_e4m3*)&packed1;
        float f0 = (float)fp8_0[0], f1 = (float)fp8_0[1], f2 = (float)fp8_0[2], f3 = (float)fp8_0[3];
        float f4 = (float)fp8_1[0], f5 = (float)fp8_1[1], f6 = (float)fp8_1[2], f7 = (float)fp8_1[3];

        uint4 x128 = *(const uint4*)(x + k);
        float x0 = __bfloat162float(((__nv_bfloat162*)&x128.x)->x);
        float x1 = __bfloat162float(((__nv_bfloat162*)&x128.x)->y);
        float x2 = __bfloat162float(((__nv_bfloat162*)&x128.y)->x);
        float x3 = __bfloat162float(((__nv_bfloat162*)&x128.y)->y);
        float x4 = __bfloat162float(((__nv_bfloat162*)&x128.z)->x);
        float x5 = __bfloat162float(((__nv_bfloat162*)&x128.z)->y);
        float x6 = __bfloat162float(((__nv_bfloat162*)&x128.w)->x);
        float x7 = __bfloat162float(((__nv_bfloat162*)&x128.w)->y);

        sum += (x0*f0 + x1*f1 + x2*f2 + x3*f3 + x4*f4 + x5*f5 + x6*f6 + x7*f7) * scale;
    }
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    __shared__ float shared_sum[8];
    if (lane_id == 0) shared_sum[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
        float final_sum = (tid < (blockDim.x >> 5)) ? shared_sum[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) final_sum += __shfl_xor_sync(0xFFFFFFFF, final_sum, offset);
        if (tid == 0) out[local_row] = __float2bfloat16(final_sum);
    }
}
__global__ void fused_qkv_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ w_q, const uint8_t* __restrict__ w_k, const uint8_t* __restrict__ w_v,
    const __nv_bfloat16* __restrict__ s_q, const __nv_bfloat16* __restrict__ s_k, const __nv_bfloat16* __restrict__ s_v,
    __nv_bfloat16* __restrict__ q_out, __nv_bfloat16* __restrict__ k_out, __nv_bfloat16* __restrict__ v_out,
    int K, int N_q, int N_k, int N_v,
    int K_blk, int N_q_blk, int N_k_blk, int N_v_blk
) {
    int row = blockIdx.x, tid = threadIdx.x;
    int warp_id = tid >> 5, lane_id = tid & 31;
    const uint8_t* w; const __nv_bfloat16* s; __nv_bfloat16* out; int local_row;
    if (row < N_q) { w = w_q; s = s_q; out = q_out; local_row = row; }
    else if (row < N_q + N_k) { w = w_k; s = s_k; out = k_out; local_row = row - N_q; }
    else { w = w_v; s = s_v; out = v_out; local_row = row - N_q - N_k; }

    int block_n = local_row / 128;
    int scale_row_offset = block_n * K_blk;
    const uint8_t* w_row = w + local_row * (size_t)K;
    float sum = 0.0f;
    for (int k = tid * 4; k < K; k += blockDim.x * 4) {
        int block_k = k / 128;
        float scale = __bfloat162float(s[scale_row_offset + block_k]);

        uint32_t packed = *(const uint32_t*)(w_row + k);
        __nv_fp8_e4m3* fp8 = (__nv_fp8_e4m3*)&packed;
        float f0 = (float)fp8[0];
        float f1 = (float)fp8[1];
        float f2 = (float)fp8[2];
        float f3 = (float)fp8[3];

        float x0 = __bfloat162float(x[k]);
        float x1 = __bfloat162float(x[k+1]);
        float x2 = __bfloat162float(x[k+2]);
        float x3 = __bfloat162float(x[k+3]);

        sum += (x0 * f0 + x1 * f1 + x2 * f2 + x3 * f3) * scale;
    }
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    __shared__ float shared_sum[8];
    if (lane_id == 0) shared_sum[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
        float final_sum = (tid < (blockDim.x >> 5)) ? shared_sum[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) final_sum += __shfl_xor_sync(0xFFFFFFFF, final_sum, offset);
        if (tid == 0) out[local_row] = __float2bfloat16(final_sum);
    }
}

// ═══════════════════════════════════════════════════════════════
// Fused Gate+Up+SiLU kernel (v4: paired, 128-bit loads)
// Grid: [N], Block: BLOCK_DIM  每个 block i 同时算 Gate[i] 和 Up[i]
// 输出: gate_out[i], up_out[i], silu_out[i] = SiLU(gate[i]) * up[i]
// ═══════════════════════════════════════════════════════════════
__global__ void fused_gate_up_silu_v4_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ w_gate, const uint8_t* __restrict__ w_up,
    const __nv_bfloat16* __restrict__ s_gate, const __nv_bfloat16* __restrict__ s_up,
    __nv_bfloat16* __restrict__ gate_out, __nv_bfloat16* __restrict__ up_out,
    __nv_bfloat16* __restrict__ silu_out,
    int K, int N, int K_blk
) {
    bool per_row_block = K_blk < 0;
    K_blk = per_row_block ? -K_blk : K_blk;
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;
    int warp_id = tid >> 5, lane_id = tid & 31;

    int block_n = row / 128;
    int scale_off = per_row_block ? row * K_blk
                  : (K_blk == 1) ? row : block_n * K_blk;
    const uint8_t* w_g_row = w_gate + row * (size_t)K;
    const uint8_t* w_u_row = w_up + row * (size_t)K;
    float sum_g = 0.0f, sum_u = 0.0f;

    for (int k = tid * 8; k < K; k += blockDim.x * 8) {
        int block_k = k / 128;
        int scale_k = (K_blk == 1) ? 0 : block_k;
        float sg = __bfloat162float(s_gate[scale_off + scale_k]);
        float su = __bfloat162float(s_up[scale_off + scale_k]);

        // 128-bit load 8 bf16 inputs
        uint4 x128 = *(const uint4*)(x + k);
        float xf0 = __bfloat162float(((__nv_bfloat162*)&x128.x)->x);
        float xf1 = __bfloat162float(((__nv_bfloat162*)&x128.x)->y);
        float xf2 = __bfloat162float(((__nv_bfloat162*)&x128.y)->x);
        float xf3 = __bfloat162float(((__nv_bfloat162*)&x128.y)->y);
        float xf4 = __bfloat162float(((__nv_bfloat162*)&x128.z)->x);
        float xf5 = __bfloat162float(((__nv_bfloat162*)&x128.z)->y);
        float xf6 = __bfloat162float(((__nv_bfloat162*)&x128.w)->x);
        float xf7 = __bfloat162float(((__nv_bfloat162*)&x128.w)->y);

        // Gate weights (8 FP8 × 2 uint32)
        uint32_t pg0 = *(const uint32_t*)(w_g_row + k);
        uint32_t pg1 = *(const uint32_t*)(w_g_row + k + 4);
        __nv_fp8_e4m3* fpg = (__nv_fp8_e4m3*)&pg0;
        float wg0 = (float)fpg[0], wg1 = (float)fpg[1], wg2 = (float)fpg[2], wg3 = (float)fpg[3];
        fpg = (__nv_fp8_e4m3*)&pg1;
        float wg4 = (float)fpg[0], wg5 = (float)fpg[1], wg6 = (float)fpg[2], wg7 = (float)fpg[3];

        // Up weights (8 FP8 × 2 uint32)
        uint32_t pu0 = *(const uint32_t*)(w_u_row + k);
        uint32_t pu1 = *(const uint32_t*)(w_u_row + k + 4);
        __nv_fp8_e4m3* fpu = (__nv_fp8_e4m3*)&pu0;
        float wu0 = (float)fpu[0], wu1 = (float)fpu[1], wu2 = (float)fpu[2], wu3 = (float)fpu[3];
        fpu = (__nv_fp8_e4m3*)&pu1;
        float wu4 = (float)fpu[0], wu5 = (float)fpu[1], wu6 = (float)fpu[2], wu7 = (float)fpu[3];

        sum_g += (xf0*wg0 + xf1*wg1 + xf2*wg2 + xf3*wg3 + xf4*wg4 + xf5*wg5 + xf6*wg6 + xf7*wg7) * sg;
        sum_u += (xf0*wu0 + xf1*wu1 + xf2*wu2 + xf3*wu3 + xf4*wu4 + xf5*wu5 + xf6*wu6 + xf7*wu7) * su;
    }

    // Warp reduce sum_g and sum_u
    for (int offset = 16; offset > 0; offset >>= 1) sum_g += __shfl_xor_sync(0xFFFFFFFF, sum_g, offset);
    for (int offset = 16; offset > 0; offset >>= 1) sum_u += __shfl_xor_sync(0xFFFFFFFF, sum_u, offset);

    __shared__ float shared_s[16];  // [0..7]=gate, [8..15]=up, each warp's partial sum
    if (lane_id == 0) { shared_s[warp_id] = sum_g; shared_s[8 + warp_id] = sum_u; }
    __syncthreads();

    if (warp_id == 0) {
        float fg = (tid < 8) ? shared_s[tid] : 0.0f;
        float fu = (tid < 8) ? shared_s[8 + tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            fg += __shfl_xor_sync(0xFFFFFFFF, fg, offset);
            fu += __shfl_xor_sync(0xFFFFFFFF, fu, offset);
        }
        if (tid == 0) {
            gate_out[row] = __float2bfloat16(fg);
            up_out[row] = __float2bfloat16(fu);
            // SiLU: g / (1 + exp(-g)) * u  (用 float 精度累积值，更快)
            silu_out[row] = __float2bfloat16((fg / (1.0f + expf(-fg))) * fu);
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// RMSNorm in-place
// ═══════════════════════════════════════════════════════════════
__device__ float warp_reduce_sum_f(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void rmsnorm_out_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int D, float eps
) {
    extern __shared__ float shared[];
    int row = blockIdx.x, tid = threadIdx.x;
    int warp_id = tid >> 5;
    float sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float v = __bfloat162float(x[row * D + i]);
        sum += v * v;
    }
    sum = warp_reduce_sum_f(sum);
    if ((tid & 31) == 0) shared[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
        int num_w = (blockDim.x + 31) / 32;
        sum = (tid < num_w) ? shared[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
        if (tid == 0) shared[0] = sum;
    }
    __syncthreads();
    float rms = rsqrtf(shared[0] / D + eps);
    for (int i = tid; i < D; i += blockDim.x) {
        float xf = __bfloat162float(x[row * D + i]);
        out[row * D + i] = __float2bfloat16(xf * rms * (1.0f + weight[i]));
    }
}

__global__ void rmsnorm_ip_kernel(
    __nv_bfloat16* __restrict__ x, const float* __restrict__ weight,
    int D, float eps
) {
    extern __shared__ float shared[];
    int row = blockIdx.x, tid = threadIdx.x;
    int warp_id = tid >> 5;
    float sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float v = __bfloat162float(x[row * D + i]); sum += v * v;
    }
    sum = warp_reduce_sum_f(sum);
    if ((tid & 31) == 0) shared[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
        int num_w = (blockDim.x + 31) / 32;
        sum = (tid < num_w) ? shared[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
        if (tid == 0) shared[0] = sum;
    }
    __syncthreads();
    float rms = rsqrtf(shared[0] / D + eps);
    for (int i = tid; i < D; i += blockDim.x) {
        float xf = __bfloat162float(x[row * D + i]);
        x[row * D + i] = __float2bfloat16(xf * rms * (1.0f + weight[i]));
    }
}

// ═══════════════════════════════════════════════════════════════
// Elementwise kernels
// ═══════════════════════════════════════════════════════════════
// a + b → out (vector add, in-place or out-of-place)
__global__ void elem_add_kernel(
    const __nv_bfloat16* a, const __nv_bfloat16* b,
    __nv_bfloat16* out, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
}

// SiLU(gate) * up → out  (SiLU(x) = x / (1 + exp(-x)))
__global__ void silu_mul_kernel(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    __nv_bfloat16* out, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float g = __bfloat162float(gate[i]);
        float u = __bfloat162float(up[i]);
        out[i] = __float2bfloat16((g / (1.0f + expf(-g))) * u);
    }
}

// Linear-attention gate preparation and RMSNormGated output for decode.
__global__ void prepare_linear_gates_kernel(
    const __nv_bfloat16* a,
    const __nv_bfloat16* beta_in,
    const float* A_log,
    const float* dt_bias,
    float* g_out,
    __nv_bfloat16* beta_out,
    int heads
) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= heads) return;
    float av = __bfloat162float(a[h]);
    float bv = __bfloat162float(beta_in[h]);
    g_out[h] = -expf(A_log[h]) * log1pf(expf(av + dt_bias[h]));
    beta_out[h] = __float2bfloat16(1.0f / (1.0f + expf(-bv)));
}

__global__ void rmsnorm_gated_decode_kernel(
    const __nv_bfloat16* core,
    const __nv_bfloat16* z,
    const __nv_bfloat16* weight,
    __nv_bfloat16* out,
    int heads,
    int head_dim,
    float eps
) {
    int h = blockIdx.x;
    int d = threadIdx.x;
    if (h >= heads || d >= head_dim) return;
    extern __shared__ float reduce[];
    int offset = h * head_dim + d;
    float x = __bfloat162float(core[offset]);
    reduce[d] = x * x;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (d < stride) reduce[d] += reduce[d + stride];
        __syncthreads();
    }
    float normed = x * rsqrtf(reduce[0] / head_dim + eps)
                 * __bfloat162float(weight[d]);
    float zv = __bfloat162float(z[offset]);
    float silu_z = zv / (1.0f + expf(-zv));
    out[offset] = __float2bfloat16(normed * silu_z);
}

// Fused single-token gated delta rule. One block handles one value head and
// one thread handles one output/value dimension. State is updated in place.
// This replaces the many small eager PyTorch kernels used on native Windows.
__global__ void delta_rule_decode_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ g,
    const __nv_bfloat16* __restrict__ beta,
    float* __restrict__ state,
    __nv_bfloat16* __restrict__ output,
    int head_dim
) {
    int h = blockIdx.x;
    int d = threadIdx.x;
    if (d >= head_dim) return;

    extern __shared__ float shared[];
    float* q_shared = shared;
    float* k_shared = shared + head_dim;
    float* reduce_q = shared + 2 * head_dim;
    float* reduce_k = reduce_q + blockDim.x;

    int vec_off = h * head_dim + d;
    float qv = __bfloat162float(q[vec_off]);
    float kv = __bfloat162float(k[vec_off]);
    reduce_q[d] = qv * qv;
    reduce_k[d] = kv * kv;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (d < stride) {
            reduce_q[d] += reduce_q[d + stride];
            reduce_k[d] += reduce_k[d + stride];
        }
        __syncthreads();
    }

    float q_scale = rsqrtf(reduce_q[0] + 1e-6f) * rsqrtf((float)head_dim);
    float k_scale = rsqrtf(reduce_k[0] + 1e-6f);
    q_shared[d] = qv * q_scale;
    k_shared[d] = kv * k_scale;
    __syncthreads();

    size_t state_base = (size_t)h * head_dim * head_dim;
    float decay = expf(g[h]);
    float prediction = 0.0f;
    #pragma unroll 4
    for (int kd = 0; kd < head_dim; kd++) {
        prediction += state[state_base + (size_t)kd * head_dim + d]
                    * decay * k_shared[kd];
    }
    float delta = (__bfloat162float(v[vec_off]) - prediction)
                * __bfloat162float(beta[h]);

    float out = 0.0f;
    #pragma unroll 4
    for (int kd = 0; kd < head_dim; kd++) {
        size_t idx = state_base + (size_t)kd * head_dim + d;
        float updated = state[idx] * decay + k_shared[kd] * delta;
        state[idx] = updated;
        out += updated * q_shared[kd];
    }
    output[vec_off] = __float2bfloat16(out);
}

// ═══════════════════════════════════════════════════════════════
// Fused GQA attention for decode (single query)
// Grid: [n_heads], Block: 64 (head_dim=128 → 2 elem/thread)
//
// 不写中间张量，直接读 KV cache → online softmax → 写输出
// ═══════════════════════════════════════════════════════════════
__global__ void fused_attn_kernel(
    const __nv_bfloat16* __restrict__ q,         // [n_heads * head_dim]
    const __nv_bfloat16* __restrict__ k_cache,    // [n_kv_heads * max_seq * head_dim]
    const __nv_bfloat16* __restrict__ v_cache,    // [n_kv_heads * max_seq * head_dim]
    __nv_bfloat16* __restrict__ output,            // [n_heads * head_dim]
    int seq_len,
    int kv_cache_stride_head,  // max_seq_len * head_dim
    int head_dim,
    int n_heads,
    int n_kv_heads
) {
    int q_head = blockIdx.x;
    int kv_head = q_head * n_kv_heads / n_heads;  // GQA: Q head → KV head
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    int npt = head_dim / blockDim.x;  // 128/64 = 2

    // 共享内存布局: scores[0..seq_len-1] = 最终点积 + softmax
    //                tmp[0..1] = warp reduction 临时 (位于 scores 之后)
    extern __shared__ float shared[];
    float* scores = shared;

    // 预加载 Q (带 attention scale: 1/sqrt(head_dim))
    float scale = rsqrtf((float)head_dim);
    float qv[2];
    #pragma unroll
    for (int d = 0; d < npt; d++)
        qv[d] = __bfloat162float(q[q_head * head_dim + tid * npt + d]) * scale;

    // Phase 1: QK^T + online softmax (running max)
    float max_val = -FLT_MAX;
    for (int pos = 0; pos < seq_len; pos++) {
        float partial = 0.0f;
        #pragma unroll
        for (int d = 0; d < npt; d++) {
            float kv = __bfloat162float(
                k_cache[kv_head * kv_cache_stride_head + pos * head_dim + tid * npt + d]
            );
            partial += qv[d] * kv;
        }
        // warp reduce (within warp: 32 threads → 1)
        for (int offset = 16; offset > 0; offset >>= 1)
            partial += __shfl_xor_sync(0xFFFFFFFF, partial, offset);
        // 每个 warp 的 lane 0 存结果到独立 temp 位置
        if (lane_id == 0) shared[seq_len + warp_id] = partial;
        __syncthreads();
        // warp 0 合并两个 warp 的结果
        if (warp_id == 0) {
            float dot = shared[seq_len] + shared[seq_len + 1];
            if (tid == 0) {
                scores[pos] = dot;
                max_val = fmaxf(max_val, dot);
            }
        }
        __syncthreads();
    }

    // Phase 2: Stable softmax
    if (tid == 0) {
        float sum = 0.0f;
        for (int pos = 0; pos < seq_len; pos++) {
            scores[pos] = expf(scores[pos] - max_val);
            sum += scores[pos];
        }
        float inv_sum = 1.0f / sum;
        for (int pos = 0; pos < seq_len; pos++)
            scores[pos] *= inv_sum;
    }
    __syncthreads();

    // Phase 3: Weighted sum of V
    float outv[2] = {0.0f, 0.0f};
    for (int pos = 0; pos < seq_len; pos++) {
        float w = scores[pos];
        #pragma unroll
        for (int d = 0; d < npt; d++) {
            float vv = __bfloat162float(
                v_cache[kv_head * kv_cache_stride_head + pos * head_dim + tid * npt + d]
            );
            outv[d] += w * vv;
        }
    }

    // Write output
    #pragma unroll
    for (int d = 0; d < npt; d++)
        output[q_head * head_dim + tid * npt + d] = __float2bfloat16(outv[d]);
}

// ═══════════════════════════════════════════════════════════════
// Fused Attention v2 — 1-pass online softmax, 单 warp (No sync in hot loop)
// Block: [32], Grid: [n_heads]
// 每个线程处理 4 个 head_dim 元素 (128/32=4)
// ═══════════════════════════════════════════════════════════════
template<int NPT>
__global__ void fused_attn_v2_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ output,
    int seq_len,
    int kv_cache_stride_head,
    int head_dim,
    int n_heads,
    int n_kv_heads
) {
    int q_head = blockIdx.x;
    int kv_head = q_head * n_kv_heads / n_heads;
    int tid = threadIdx.x;       // 0..31

    // Preload Q x scale (NPT per thread)
    float scale = rsqrtf((float)head_dim);
    float qv[NPT];
    const __nv_bfloat16* q_base = q + q_head * head_dim;
    #pragma unroll
    for (int d = 0; d < NPT; d++)
        qv[d] = __bfloat162float(q_base[tid * NPT + d]) * scale;

    // Online softmax, 1-pass: compute attention & weighted V-sum simultaneously
    float max_val = -FLT_MAX;
    float sum = 0.0f;
    float out[NPT] = {0.0f};

    const __nv_bfloat16* k_base = k_cache + kv_head * kv_cache_stride_head;
    const __nv_bfloat16* v_base = v_cache + kv_head * kv_cache_stride_head;

    for (int pos = 0; pos < seq_len; pos++) {
        // --- Q·K dot product ---
        const __nv_bfloat16* kp = k_base + pos * head_dim + tid * NPT;
        float dot = 0.0f;
        #pragma unroll
        for (int d = 0; d < NPT; d++) {
            float kv = __bfloat162float(kp[d]);
            dot += qv[d] * kv;
        }
        // warp reduce (32 threads → 1)
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            dot += __shfl_xor_sync(0xFFFFFFFF, dot, offset);

        // --- Online softmax update ---
        float new_max = fmaxf(max_val, dot);
        float exp_old = expf(max_val - new_max);
        float exp_new = expf(dot - new_max);

        // Scale existing V-sum
        #pragma unroll
        for (int d = 0; d < NPT; d++)
            out[d] *= exp_old;

        // Add V[pos] contribution
        const __nv_bfloat16* vp = v_base + pos * head_dim + tid * NPT;
        #pragma unroll
        for (int d = 0; d < NPT; d++) {
            float vv = __bfloat162float(vp[d]);
            out[d] += exp_new * vv;
        }

        sum = sum * exp_old + exp_new;
        max_val = new_max;
        // 单 warp: 不需要 __syncthreads！！！
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    #pragma unroll
    for (int d = 0; d < NPT; d++)
        out[d] *= inv_sum;

    // Write output
    __nv_bfloat16* out_base = output + q_head * head_dim;
    #pragma unroll
    for (int d = 0; d < NPT; d++)
        out_base[tid * NPT + d] = __float2bfloat16(out[d]);
}

// ═══════════════════════════════════════════════════════════════
// Inline decode RoPE (single position)
// ═══════════════════════════════════════════════════════════════
__global__ void rope_decode_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int pos, int n_heads, int n_kv_heads, int head_dim, int rotary_dim
) {
    // Qwen3.5 partial RoPE rotates only the first rotary_dim channels.
    // One thread owns a channel pair so in-place reads/writes cannot race.
    int half = rotary_dim / 2;
    int pair_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_pairs = (n_heads + n_kv_heads) * half;
    if (pair_idx >= total_pairs) return;

    int combined_head = pair_idx / half;
    int col = pair_idx % half;
    bool is_q = combined_head < n_heads;
    int local_head = is_q ? combined_head : combined_head - n_heads;
    __nv_bfloat16* data = is_q ? q : k;
    int base = local_head * head_dim;

    float x0 = __bfloat162float(data[base + col]);
    float x1 = __bfloat162float(data[base + col + half]);
    int table_base = pos * rotary_dim;
    float c0 = cos_table[table_base + col];
    float s0 = sin_table[table_base + col];
    float c1 = cos_table[table_base + col + half];
    float s1 = sin_table[table_base + col + half];

    data[base + col] = __float2bfloat16(x0 * c0 - x1 * s0);
    data[base + col + half] = __float2bfloat16(x1 * c1 + x0 * s1);
}

// ═══════════════════════════════════════════════════════════════
// Batched layer: pre-attention (RMSNorm + fused QKV)
// 一次调用提交所有 GPU kernel，消除 Python 间隙
// ═══════════════════════════════════════════════════════════════
static void launch_rmsnorm_ip(
    void* x, const float* weight, int D, float eps, cudaStream_t stream
) {
    int num_warps = (BLOCK_DIM + 31) / 32;
    rmsnorm_ip_kernel<<<1, BLOCK_DIM, num_warps * sizeof(float), stream>>>(
        (__nv_bfloat16*)x, weight, D, eps);
}

// 单个元素的向量加法
static void launch_elem_add(
    const void* a, const void* b, void* out, int N, cudaStream_t stream
) {
    int grid = (N + 255) / 256;
    elem_add_kernel<<<grid, 256, 0, stream>>>(
        (const __nv_bfloat16*)a, (const __nv_bfloat16*)b,
        (__nv_bfloat16*)out, N);
}

// 单个元素的 SiLU mul
static void launch_silu_mul(
    const void* gate, const void* up, void* out, int N, cudaStream_t stream
) {
    int grid = (N + 255) / 256;
    silu_mul_kernel<<<grid, 256, 0, stream>>>(
        (const __nv_bfloat16*)gate, (const __nv_bfloat16*)up,
        (__nv_bfloat16*)out, N);
}

// ═══════════════════════════════════════════════════════════════
// Extern C API
// ═══════════════════════════════════════════════════════════════
extern "C" {

__declspec(dllexport) void launch_prepare_linear_gates(
    const void* a, const void* beta_in,
    const float* A_log, const float* dt_bias,
    void* g_out, void* beta_out, int heads,
    cudaStream_t stream
) {
    int block = 32;
    int grid = (heads + block - 1) / block;
    prepare_linear_gates_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)a, (const __nv_bfloat16*)beta_in,
        A_log, dt_bias, (float*)g_out, (__nv_bfloat16*)beta_out, heads
    );
}

__declspec(dllexport) void launch_rmsnorm_gated_decode(
    const void* core, const void* z, const void* weight, void* out,
    int heads, int head_dim, float eps, cudaStream_t stream
) {
    rmsnorm_gated_decode_kernel<<<heads, head_dim, head_dim * sizeof(float), stream>>>(
        (const __nv_bfloat16*)core, (const __nv_bfloat16*)z,
        (const __nv_bfloat16*)weight, (__nv_bfloat16*)out,
        heads, head_dim, eps
    );
}

__declspec(dllexport) void launch_delta_rule_decode(
    const void* q, const void* k, const void* v,
    const void* g, const void* beta, void* state, void* output,
    int n_heads, int head_dim, cudaStream_t stream
) {
    size_t shared = (size_t)(2 * head_dim + 2 * head_dim) * sizeof(float);
    delta_rule_decode_kernel<<<n_heads, head_dim, shared, stream>>>(
        (const __nv_bfloat16*)q, (const __nv_bfloat16*)k,
        (const __nv_bfloat16*)v, (const float*)g,
        (const __nv_bfloat16*)beta, (float*)state,
        (__nv_bfloat16*)output, head_dim
    );
}

__declspec(dllexport) void launch_gemv_v3_extern(
    const void* x, const void* w, const void* s, void* y,
    int N, int K, int K_blk, cudaStream_t stream
) {
    gemv_v3_kernel<<<N, BLOCK_DIM, 0, stream>>>(
        (const __nv_bfloat16*)x, (const uint8_t*)w,
        (const __nv_bfloat16*)s, (__nv_bfloat16*)y, K, K_blk
    );
}

__declspec(dllexport) void launch_fused_qkv(
    const void* x, const void* w_q, const void* w_k, const void* w_v,
    const void* s_q, const void* s_k, const void* s_v,
    void* q_out, void* k_out, void* v_out,
    int K, int N_q, int N_k, int N_v,
    int K_blk, int N_q_blk, int N_k_blk, int N_v_blk,
    cudaStream_t stream = nullptr
) {
    dim3 grid(N_q + N_k + N_v);
    fused_qkv_v4_kernel<<<grid, BLOCK_DIM, 0, stream>>>(
        (const __nv_bfloat16*)x,
        (const uint8_t*)w_q, (const uint8_t*)w_k, (const uint8_t*)w_v,
        (const __nv_bfloat16*)s_q, (const __nv_bfloat16*)s_k, (const __nv_bfloat16*)s_v,
        (__nv_bfloat16*)q_out, (__nv_bfloat16*)k_out, (__nv_bfloat16*)v_out,
        K, N_q, N_k, N_v, K_blk, N_q_blk, N_k_blk, N_v_blk
    );
}

__declspec(dllexport) void launch_fused_gate_up(
    const void* x, const void* w_gate, const void* w_up,
    const void* s_gate, const void* s_up,
    void* gate_out, void* up_out,
    int K, int N_gate, int N_up,
    int K_blk, int N_gate_blk, int N_up_blk,
    cudaStream_t stream = nullptr
) {
    dim3 grid(N_gate + N_up);
    fused_gate_up_v4_kernel<<<grid, BLOCK_DIM, 0, stream>>>(
        (const __nv_bfloat16*)x,
        (const uint8_t*)w_gate, (const uint8_t*)w_up,
        (const __nv_bfloat16*)s_gate, (const __nv_bfloat16*)s_up,
        (__nv_bfloat16*)gate_out, (__nv_bfloat16*)up_out,
        K, N_gate, N_up, K_blk, N_gate_blk, N_up_blk
    );
}

__declspec(dllexport) void launch_fused_attn(
    const void* q, const void* k_cache, const void* v_cache, void* output,
    int seq_len, int max_seq_len, int head_dim, int n_heads, int n_kv_heads,
    cudaStream_t stream = nullptr
) {
    dim3 grid(n_heads);
    dim3 block(64);  // head_dim=128 → 2 elem/thr
    size_t shared = (size_t)(seq_len + 2) * sizeof(float);  // scores + warp_tmp
    fused_attn_kernel<<<grid, block, shared, stream>>>(
        (const __nv_bfloat16*)q,
        (const __nv_bfloat16*)k_cache,
        (const __nv_bfloat16*)v_cache,
        (__nv_bfloat16*)output,
        seq_len, max_seq_len * head_dim,
        head_dim, n_heads, n_kv_heads
    );
}

__declspec(dllexport) void launch_rmsnorm_out(
    const void* x, const void* weight, void* out, int N, int D, float eps,
    cudaStream_t stream = nullptr
) {
    int num_warps = (BLOCK_DIM + 31) / 32;
    rmsnorm_out_kernel<<<N, BLOCK_DIM, num_warps * sizeof(float), stream>>>(
        (const __nv_bfloat16*)x, (const float*)weight,
        (__nv_bfloat16*)out, D, eps);
}

__declspec(dllexport) void launch_rmsnorm_ip(
    void* x, const void* weight, int N, int D, float eps,
    cudaStream_t stream = nullptr
) {
    int num_warps = (BLOCK_DIM + 31) / 32;
    rmsnorm_ip_kernel<<<N, BLOCK_DIM, num_warps * sizeof(float), stream>>>(
        (__nv_bfloat16*)x, (const float*)weight, D, eps);
}

// ── Layer pre-attention: hidden → RMSNorm → fused QKV → q, k, v ──
// 将 hidden 复制到 norm_ws, 对 norm_ws 做 in-place RMSNorm, 然后 fused QKV
__declspec(dllexport) void layer_pre_attn(
    const void* hidden,
    const float* norm_weight,
    const void* w_q, const void* s_q, int N_q, int N_q_blk,
    const void* w_k, const void* s_k, int N_k, int N_k_blk,
    const void* w_v, const void* s_v, int N_v, int N_v_blk,
    void* q_out, void* k_out, void* v_out,
    void* norm_ws,          // [h_dim] bf16 workspace
    int h_dim, int K_blk,   // K_blk = h_dim/128
    cudaStream_t stream
) {
    // 1. hidden → norm_ws (device to device copy)
    cudaMemcpyAsync(norm_ws, hidden, h_dim * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, stream);
    // 2. RMSNorm in-place
    launch_rmsnorm_ip(norm_ws, norm_weight, h_dim, 1e-6f, stream);
    // 3. Fused QKV
    dim3 grid(N_q + N_k + N_v);
    fused_qkv_v4_kernel<<<grid, BLOCK_DIM, 0, stream>>>(
        (const __nv_bfloat16*)norm_ws,
        (const uint8_t*)w_q, (const uint8_t*)w_k, (const uint8_t*)w_v,
        (const __nv_bfloat16*)s_q, (const __nv_bfloat16*)s_k, (const __nv_bfloat16*)s_v,
        (__nv_bfloat16*)q_out, (__nv_bfloat16*)k_out, (__nv_bfloat16*)v_out,
        h_dim, N_q, N_k, N_v, K_blk, N_q_blk, N_k_blk, N_v_blk
    );
}

// ── Layer post-attention: O_proj + residual + RMSNorm + fused GateUp + SiLU_mul + Down + residual ──
// 数据流:
//   1. O_proj(attn_out) → ws
//   2. ws(O_out) + hidden → output(hidden2 = hidden + O_out)
//   3. output → ws (copy), RMSNorm in-place on ws
//   4. GateUp(ws) → gate_buf, up_buf
//   5. SiLU(gate_buf) * up_buf → silu_buf
//   6. Down(silu_buf) → ws
//   7. output(hidden2) + ws(down) → output(final)
__declspec(dllexport) void layer_post_attn(
    const void* hidden,         // pre-attention hidden [h_dim] bf16
    const void* attn_out,       // attention output [attn_dim] bf16
    const void* w_o, const void* s_o, int N_o, int N_o_blk,
    const float* norm_weight,
    const void* w_g, const void* s_g, int N_g, int N_g_blk,
    const void* w_u, const void* s_u, int N_u, int N_u_blk,
    const void* w_d, const void* s_d, int N_d, int N_d_blk,
    void* output,               // [h_dim] bf16 (最终输出 + 中间结果暂存)
    void* ws,                   // [max(N_o, N_g)] bf16 workspace
    void* gate_buf,             // [N_g] bf16
    void* up_buf,               // [N_u] bf16
    void* silu_buf,             // [N_g] bf16
    int h_dim, int attn_dim,
    int K_blk_o, int K_blk_gu, int K_blk_d,
    cudaStream_t stream
) {
    // 1. O_proj(attn_out) → ws
    launch_gemv_v4(attn_out, w_o, s_o, ws, N_o, attn_dim, K_blk_o, stream);

    // 2. ws (O_out) + hidden → output (hidden2 = hidden + O_out)
    launch_elem_add(hidden, ws, output, h_dim, stream);

    // 3. Copy output → ws, then RMSNorm in-place on ws
    // actually, RMSNorm in-place needs DIM parameter. ws has size at least h_dim.
    cudaMemcpyAsync(ws, output, h_dim * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, stream);
    launch_rmsnorm_ip(ws, norm_weight, h_dim, 1e-6f, stream);

    // 4. Fused Gate+Up+SiLU from ws → gate_buf, up_buf, silu_buf
    {
        dim3 grid_gu(N_g);  // N_g == N_u, 每 block 同时算 Gate[i] 和 Up[i]
        fused_gate_up_silu_v4_kernel<<<grid_gu, BLOCK_DIM, 0, stream>>>(
            (const __nv_bfloat16*)ws,
            (const uint8_t*)w_g, (const uint8_t*)w_u,
            (const __nv_bfloat16*)s_g, (const __nv_bfloat16*)s_u,
            (__nv_bfloat16*)gate_buf, (__nv_bfloat16*)up_buf,
            (__nv_bfloat16*)silu_buf,
            h_dim, N_g, K_blk_gu
        );
    }

    // 5. (SiLU 已融合进 Gate+Up kernel, silu_buf 可直接用于 Down)
    // 6. Down(silu_buf) → ws
    launch_gemv_v4(silu_buf, w_d, s_d, ws, N_d, N_g, K_blk_d, stream);

    // 7. output (hidden2) + ws (down) → output (final)
    launch_elem_add(output, ws, output, h_dim, stream);
}

// ── Mixed-precision post-attention path ──
// Attention projections remain BF16 for quality. Python computes the attention
// output including BF16 O_proj, then this function performs the residual,
// post-attention RMSNorm and FP8 MLP in one host call.
__declspec(dllexport) void layer_post_o_fp8_mlp(
    const void* hidden,         // pre-attention residual [h_dim] bf16
    const void* o_out,          // BF16 O_proj result [h_dim]
    const float* norm_weight,
    const void* w_g, const void* s_g, int N_g,
    const void* w_u, const void* s_u, int N_u,
    const void* w_d, const void* s_d, int N_d,
    void* output,
    void* ws,
    void* gate_buf,
    void* up_buf,
    void* silu_buf,
    int h_dim,
    int K_blk_gu,
    int K_blk_d,
    cudaStream_t stream
) {
    // hidden2 = hidden + O_proj(attention)
    launch_elem_add(hidden, o_out, output, h_dim, stream);

    // Normalize hidden2 for the MLP input.
    cudaMemcpyAsync(ws, output, h_dim * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, stream);
    launch_rmsnorm_ip(ws, norm_weight, h_dim, 1e-6f, stream);

    // Gate/Up projections and SiLU multiplication share one input read.
    dim3 grid_gu(N_g);
    fused_gate_up_silu_v4_kernel<<<grid_gu, BLOCK_DIM, 0, stream>>>(
        (const __nv_bfloat16*)ws,
        (const uint8_t*)w_g, (const uint8_t*)w_u,
        (const __nv_bfloat16*)s_g, (const __nv_bfloat16*)s_u,
        (__nv_bfloat16*)gate_buf, (__nv_bfloat16*)up_buf,
        (__nv_bfloat16*)silu_buf,
        h_dim, N_g, K_blk_gu
    );

    // Down projection and final residual.
    launch_gemv_v4(silu_buf, w_d, s_d, ws, N_d, N_g, K_blk_d, stream);
    launch_elem_add(output, ws, output, h_dim, stream);
}

// ── KV cache write kernel (single launch for all heads) ──
// 每个 thread 写一个 bf16 元素，所有 KV head 并行
__global__ void kv_cache_write_kernel(
    const __nv_bfloat16* __restrict__ k_buf,
    const __nv_bfloat16* __restrict__ v_buf,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    int pos, int n_kv_heads, int head_dim, int cache_stride
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int total = n_kv_heads * head_dim * 2;
    if (idx >= total) return;

    int is_v = idx >= n_kv_heads * head_dim;
    int local_idx = is_v ? idx - n_kv_heads * head_dim : idx;
    int h = local_idx / head_dim;
    int d = local_idx % head_dim;

    const __nv_bfloat16* src = is_v ? v_buf : k_buf;
    __nv_bfloat16* dst = is_v ? v_cache : k_cache;

    dst[h * cache_stride + pos * head_dim + d] = src[h * head_dim + d];
}

// ── Layer mid-attention: Q/K norm + RoPE + KV cache write + fused attention ──
// 将 RMSNorm(Q) + RMSNorm(K) + RoPE(Q,K) + KV cache write + Fused attention
// 合并为一次 C 调用，消除 Python 间隙
__declspec(dllexport) void layer_mid_attn(
    void* q_buf, void* k_buf,       // [n_heads*head_dim], [n_kv_heads*head_dim] bf16 — in-place RoPE
    const void* v_buf,               // [n_kv_heads*head_dim] bf16
    void* attn_out,                  // [n_heads*head_dim] bf16 output
    const float* q_norm_weight,      // nullptr if no norm
    const float* k_norm_weight,      // nullptr if no norm
    const float* cos_table,          // [max_pos, head_dim]
    const float* sin_table,          // [max_pos, head_dim]
    void* k_cache, void* v_cache,    // [n_kv_heads, max_seq, head_dim] bf16
    int pos, int past_len, int total_seq,
    int max_seq_len, int head_dim, int rotary_dim,
    int n_heads, int n_kv_heads, cudaStream_t stream
) {
    // 1. RMSNorm Q (optional) — q_buf 是 [n_heads * head_dim] = 32 行 × 128 列
    if (q_norm_weight != nullptr) {
        // 直接 launch kernel，batch 处理所有 head
        int num_warps = (BLOCK_DIM + 31) / 32;
        rmsnorm_ip_kernel<<<n_heads, BLOCK_DIM, num_warps * sizeof(float), stream>>>(
            (__nv_bfloat16*)q_buf, q_norm_weight, head_dim, 1e-6f
        );
    }
    // 2. RMSNorm K (optional)
    if (k_norm_weight != nullptr) {
        int num_warps = (BLOCK_DIM + 31) / 32;
        rmsnorm_ip_kernel<<<n_kv_heads, BLOCK_DIM, num_warps * sizeof(float), stream>>>(
            (__nv_bfloat16*)k_buf, k_norm_weight, head_dim, 1e-6f
        );
    }
    // 3. Partial RoPE (single decode position)
    {
        int total_pairs = (n_heads + n_kv_heads) * (rotary_dim / 2);
        int block = 256;
        int grid = (total_pairs + block - 1) / block;
        rope_decode_kernel<<<grid, block, 0, stream>>>(
            (__nv_bfloat16*)q_buf, (__nv_bfloat16*)k_buf,
            cos_table, sin_table, pos, n_heads, n_kv_heads,
            head_dim, rotary_dim
        );
    }
    // 4. KV cache write — single kernel for all KV heads
    {
        int total_heads = n_kv_heads * 2;  // K and V
        int grid_kv = (total_heads * head_dim + 255) / 256;
        kv_cache_write_kernel<<<grid_kv, 256, 0, stream>>>(
            (const __nv_bfloat16*)k_buf, (const __nv_bfloat16*)v_buf,
            (__nv_bfloat16*)k_cache, (__nv_bfloat16*)v_cache,
            past_len, n_kv_heads, head_dim, max_seq_len * head_dim
        );
    }
    // 5. Fused attention (v2: 单 warp, 1-pass online softmax)
    {
        dim3 grid2(n_heads);
        dim3 block2(32);
        if (head_dim == 128) {
            fused_attn_v2_kernel<4><<<grid2, block2, 0, stream>>>(
                (const __nv_bfloat16*)q_buf,
                (const __nv_bfloat16*)k_cache,
                (const __nv_bfloat16*)v_cache,
                (__nv_bfloat16*)attn_out,
                total_seq, max_seq_len * head_dim,
                head_dim, n_heads, n_kv_heads
            );
        } else if (head_dim == 256) {
            fused_attn_v2_kernel<8><<<grid2, block2, 0, stream>>>(
                (const __nv_bfloat16*)q_buf,
                (const __nv_bfloat16*)k_cache,
                (const __nv_bfloat16*)v_cache,
                (__nv_bfloat16*)attn_out,
                total_seq, max_seq_len * head_dim,
                head_dim, n_heads, n_kv_heads
            );
        }
    }
}

// Standalone fused attention v2 (extern C for Python)
__declspec(dllexport) void launch_fused_attn_v2(
    const void* q, const void* k_cache, const void* v_cache, void* output,
    int seq_len, int max_seq_len, int head_dim, int n_heads, int n_kv_heads,
    cudaStream_t stream
) {
    dim3 grid(n_heads);
    dim3 block(32);
    if (head_dim == 128) {
        fused_attn_v2_kernel<4><<<grid, block, 0, stream>>>(
            (const __nv_bfloat16*)q,
            (const __nv_bfloat16*)k_cache,
            (const __nv_bfloat16*)v_cache,
            (__nv_bfloat16*)output,
            seq_len, max_seq_len * head_dim,
            head_dim, n_heads, n_kv_heads
        );
    } else if (head_dim == 256) {
        fused_attn_v2_kernel<8><<<grid, block, 0, stream>>>(
            (const __nv_bfloat16*)q,
            (const __nv_bfloat16*)k_cache,
            (const __nv_bfloat16*)v_cache,
            (__nv_bfloat16*)output,
            seq_len, max_seq_len * head_dim,
            head_dim, n_heads, n_kv_heads
        );
    }
}

__declspec(dllexport) void launch_gemv_v4_extern(
    const void* x, const void* w, const void* s, void* y,
    int N, int K, int K_blk, cudaStream_t stream
) {
    gemv_v4_kernel<<<N, BLOCK_DIM, 0, stream>>>(
        (const __nv_bfloat16*)x, (const uint8_t*)w,
        (const __nv_bfloat16*)s, (__nv_bfloat16*)y, K, K_blk
    );
}

__declspec(dllexport) void launch_linear_in_proj4(
    const void* x,
    const void* w0, const void* s0, void* y0, int n0, int kb0,
    const void* w1, const void* s1, void* y1, int n1, int kb1,
    const void* w2, const void* s2, void* y2, int n2, int kb2,
    const void* w3, const void* s3, void* y3, int n3, int kb3,
    int K, cudaStream_t stream
) {
    linear_in_proj4_v4_kernel<<<n0 + n1 + n2 + n3, BLOCK_DIM, 0, stream>>>(
        (const __nv_bfloat16*)x,
        (const uint8_t*)w0, (const __nv_bfloat16*)s0, (__nv_bfloat16*)y0, n0, kb0,
        (const uint8_t*)w1, (const __nv_bfloat16*)s1, (__nv_bfloat16*)y1, n1, kb1,
        (const uint8_t*)w2, (const __nv_bfloat16*)s2, (__nv_bfloat16*)y2, n2, kb2,
        (const uint8_t*)w3, (const __nv_bfloat16*)s3, (__nv_bfloat16*)y3, n3, kb3,
        K
    );
}

} // extern "C"
