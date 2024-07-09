
#include <torch/extension.h>
#include <ATen/NumericUtils.h>
#include <ATen/cuda/CUDAContext.h>

#include "include/utils_cuda.cuh"

__global__ void distortion_loss_kernel(
    const int64_t n_elements,
    const int64_t* __restrict__ ray_indices,
    const int64_t* __restrict__ chunk_starts,
    const int64_t* __restrict__ chunk_cnts,
    const float* __restrict__ weights,
    const float* __restrict__ steps,
    float* __restrict__ per_weight_distortion_loss)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements)
        return;

    const int64_t ray_index = ray_indices[idx];

    const int64_t chunk_start = chunk_starts[ray_index];
    const int64_t chunk_cnt = chunk_cnts[ray_index];

    float w_i = weights[idx];
    float s_i = steps[idx];

    float loss_sum = 0.0f;
    for (int j = 0; j < chunk_cnt; j++)
    {
        float w_j = weights[chunk_start + j];
        float s_j = steps[chunk_start + j];

        loss_sum += w_j * abs(s_i - s_j);
    }
    per_weight_distortion_loss[idx] = loss_sum;
}

torch::Tensor distortion_loss(
    torch::Tensor ray_indices,
    torch::Tensor chunk_starts,
    torch::Tensor chunk_cnts,
    torch::Tensor weights,
    torch::Tensor steps)
{
    CHECK_INPUT(ray_indices);
    CHECK_INPUT(chunk_starts);
    CHECK_INPUT(chunk_cnts);
    CHECK_INPUT(weights);
    CHECK_INPUT(steps);
    TORCH_CHECK(ray_indices.ndimension() == 1);
    TORCH_CHECK(chunk_starts.ndimension() == 1);
    TORCH_CHECK(chunk_cnts.ndimension() == 1);
    TORCH_CHECK(weights.ndimension() == 1);
    TORCH_CHECK(steps.ndimension() == 1);

    torch::Tensor per_weight_distortion_loss = torch::empty_like(weights);

    const int BLOCK_SIZE = 128;
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    distortion_loss_kernel<<<ceil_div<int32_t>(weights.size(0), BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(
        weights.size(0),
        ray_indices.data_ptr<int64_t>(),
        chunk_starts.data_ptr<int64_t>(),
        chunk_cnts.data_ptr<int64_t>(),
        weights.data_ptr<float>(),
        steps.data_ptr<float>(),
        per_weight_distortion_loss.data_ptr<float>()
    );

    return per_weight_distortion_loss;
} 