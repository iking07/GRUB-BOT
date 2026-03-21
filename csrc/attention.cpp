#include <torch/extension.h>
#include <vector>

// Forward declaration of the CUDA kernel launch function
torch::Tensor attention_cuda_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v);

// C++ interface function
torch::Tensor custom_attention_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v) {
    
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    return attention_cuda_forward(q, k, v);
}

// Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &custom_attention_forward, "Custom Attention forward (CUDA)");
}
