#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Simple naive exact attention forward pass
// Q, K, V are [batch_size, num_heads, seq_len, head_dim]
// This assumes seq_len and head_dim are small enough for threads.

template <typename scalar_t>
__global__ void attention_forward_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ out,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float scale) 
{
    // Block handles one (batch, head, row)
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int row = blockIdx.z; // Sequence row from Q
    const int tid = threadIdx.x;
    
    if (b >= batch_size || h >= num_heads || row >= seq_len) return;
    
    extern __shared__ float shared_mem[];
    float* s_scores = shared_mem; // size = seq_len
    
    // 1. Q * K^T block
    // Each thread process one column of K (i.e., one element in the row's scores).
    // Loop over seq_len with block threads
    for (int col = tid; col < seq_len; col += blockDim.x) {
        float sum = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            float q_val = q[b * (num_heads * seq_len * head_dim) + h * (seq_len * head_dim) + row * head_dim + d];
            float k_val = k[b * (num_heads * seq_len * head_dim) + h * (seq_len * head_dim) + col * head_dim + d];
            sum += q_val * k_val;
        }
        s_scores[col] = sum * scale;
    }
    
    __syncthreads();
    
    // 2. Softmax
    // Find max (thread 0 only for simplicity on small sequences)
    if (tid == 0) {
        float max_val = -1e20f;
        for (int i = 0; i < seq_len; ++i) {
            max_val = max(max_val, s_scores[i]);
        }
        
        float exp_sum = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            s_scores[i] = exp(s_scores[i] - max_val);
            exp_sum += s_scores[i];
        }
        
        for (int i = 0; i < seq_len; ++i) {
            s_scores[i] /= exp_sum;
        }
    }
    
    __syncthreads();
    
    // 3. Score * V
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float out_val = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            float v_val = v[b * (num_heads * seq_len * head_dim) + h * (seq_len * head_dim) + i * head_dim + d];
            out_val += s_scores[i] * v_val;
        }
        out[b * (num_heads * seq_len * head_dim) + h * (seq_len * head_dim) + row * head_dim + d] = static_cast<scalar_t>(out_val);
    }
}

torch::Tensor attention_cuda_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v) 
{
    const int batch_size = q.size(0);
    const int num_heads = q.size(1);
    const int seq_len = q.size(2);
    const int head_dim = q.size(3);
    
    auto out = torch::empty_like(q);
    
    dim3 threads(256); // 256 threads per row processing
    dim3 blocks(batch_size, num_heads, seq_len);
    
    const float scale = 1.0f / sqrt(static_cast<float>(head_dim));
    
    // Allocate shared memory for the sequence row (float per seq_len)
    int shared_mem_bytes = seq_len * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "attention_forward_cuda", ([&] {
        attention_forward_kernel<scalar_t><<<blocks, threads, shared_mem_bytes>>>(
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            scale
        );
    }));

    return out;
}
