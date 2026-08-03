// fp8_kernels.cu — 手写 FP8 算子
// fp8_gemm_batch: FP8 权重反量化 + 批量 GEMV (用于 prefill, batch>1)
//
// 权重格式: block-wise FP8 E4M3 + bf16 scale_inv
//   权重 [N, K]: 每 128x128 block 一个 scale_inv [N/128, K/128]
//   反量化: bf16_val = float(fp8_val) * scale_inv[block_idx]

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <math.h>

// ── Helper: 从位模式解析 FP8 E4M3 到 float ──
__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t val) {
    return (float)(*reinterpret_cast<__nv_fp8_e4m3*>(&val));
}


// ── 批量 GEMV（支持多个 token）──
// 对每个 token 单独调 GEMV
// Grid: [N, M], Block: 256
__global__ void fp8_gemm_batch_kernel(
    const __nv_bfloat16* __restrict__ x,     // [M, K] bf16
    const uint8_t* __restrict__ w_fp8,       // [N, K] FP8
    const __nv_bfloat16* __restrict__ w_scale, // [N_blk, K_blk] bf16
    __nv_bfloat16* __restrict__ y,           // [M, N] bf16
    int M, int N, int K, int K_blocks
) {
    bool per_row_block = K_blocks < 0;
    K_blocks = per_row_block ? -K_blocks : K_blocks;
    int col = blockIdx.x;      // output column
    int row_tok = blockIdx.y;  // token index
    int block_n = col / 128;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    
    float sum = 0.0f;
    const __nv_bfloat16* x_row = x + row_tok * K;
    
    for (int k = tid; k < K; k += blockDim.x) {
        int block_k = k / 128;
        int scale_idx = per_row_block ? col * K_blocks + block_k
                      : (K_blocks == 1) ? col : block_n * K_blocks + block_k;
        float s = __bfloat162float(w_scale[scale_idx]);
        uint8_t fp8_val = w_fp8[col * (size_t)K + k];
        sum += __bfloat162float(x_row[k]) * fp8_e4m3_to_float(fp8_val) * s;
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
        if (tid == 0) y[row_tok * (size_t)N + col] = __float2bfloat16(final_sum);
    }
}


// ── Python 绑定 API ──
extern "C" {

// fp8_gemm_batch: y[M, N] = x[M, K] @ W[N, K]^T
// 用于 prefill (batch_size > 1)
__declspec(dllexport) void fp8_gemm_batch(
    const void* x_bf16,        // [M, K] bf16
    const void* w_fp8,         // [N, K] uint8 FP8
    const void* w_scale_bf16,  // [N_blk, K_blk] bf16
    void* y_bf16,              // [M, N] bf16
    int M, int N, int K, int K_blocks,
    cudaStream_t stream = nullptr
) {
    dim3 grid(N, M);
    dim3 block(256);
    fp8_gemm_batch_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)x_bf16,
        (const uint8_t*)w_fp8,
        (const __nv_bfloat16*)w_scale_bf16,
        (__nv_bfloat16*)y_bf16,
        M, N, K, K_blocks
    );
}

} // extern "C"
