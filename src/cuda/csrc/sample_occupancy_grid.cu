
#include <torch/extension.h>
#include <ATen/NumericUtils.h>
#include <ATen/cuda/CUDAContext.h>

#include "include/utils_cuda.cuh"
#include "include/raymarch_common.cuh"

const int GRID_RESOLUTION = 128;

namespace {
namespace device {    

__global__ void sample_occupancy_grid_first_kernel(
    int32_t n_rays,
    const float3* origins,
    const float3* viewdirs,
    const float* nears,
    const float* fars,
    const bool* hits_aabb,

    const bool *occupancy_grid,
    const float3 *aabb,
    int grid_nlvl,

    float step_size,
    float cone_angle,
    int32_t max_n_samples_per_ray,

    int64_t *rays_n_samples)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rays || !hits_aabb[idx])
        return;

    Ray ray { origins[idx], viewdirs[idx] };

    float3 aabb_from = aabb[0];
    float3 aabb_to = aabb[1];

    int sample_cnt = 0;
    float near = nears[idx];
    float far = fars[idx];

    float t1 = near;

    while (sample_cnt < max_n_samples_per_ray)
    {
        float t0 = t1;
        float dt = calculate_stepsize(t0, cone_angle, step_size, 1e10f);
        t1 = t0 + dt;
        float t_mid = (t0 + t1) * 0.5f;

        // printf("%f - %f: %f\n", near, far, t_mid);

        if (t_mid > far)
            break;

        float3 world_point = ray.at(t_mid);
        if (grid_occupied_at_linear<GRID_RESOLUTION>(world_point, aabb_from, aabb_to, grid_nlvl, occupancy_grid))
            sample_cnt++;
    }

    rays_n_samples[idx] = sample_cnt;
}

__global__ void sample_occupancy_grid_second_kernel(
    int32_t n_rays,
    const float3* origins,
    const float3* viewdirs,
    const float* nears,
    const float* fars,
    const bool* hits_aabb,

    const bool *occupancy_grid,
    const float3 *aabb,
    int grid_nlvl,

    float step_size,
    float cone_angle,

    const int64_t *ray_n_samples,
    const int64_t *ray_offsets,

    float *t_starts,
    float *t_ends,
    int64_t *ray_indices)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rays || !hits_aabb[idx])
        return;

    Ray ray { origins[idx], viewdirs[idx] };

    float3 aabb_from = aabb[0];
    float3 aabb_to = aabb[1];

    int sample_cnt = 0;
    float near = nears[idx];
    float far = fars[idx];

    float t1 = near;

    int n_samples = ray_n_samples[idx];
    int ray_offset = ray_offsets[idx];

    while (sample_cnt < n_samples)
    {
        float t0 = t1;
        float dt = calculate_stepsize(t0, cone_angle, step_size, 1e10f);
        t1 = t0 + dt;
        float t_mid = (t0 + t1) * 0.5f;

        if (t_mid > far)
            break;

        float3 world_point = ray.at(t_mid);
        if (grid_occupied_at_linear<GRID_RESOLUTION>(world_point, aabb_from, aabb_to, grid_nlvl, occupancy_grid))
        {
            int sample_offset = ray_offset + sample_cnt;

            t_starts[sample_offset] = t0;
            t_ends[sample_offset] = t1;
            ray_indices[sample_offset] = idx;

            sample_cnt++;
        }
    }
}

} // namespace device
} // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> sample_occupancy_grid(
    const torch::Tensor origins,
    const torch::Tensor viewdirs,
    const torch::Tensor nears,
    const torch::Tensor fars,
    const torch::Tensor hits_aabb,

    const torch::Tensor occupancy_grid,
    const torch::Tensor aabb,

    float step_size,
    float cone_angle,
    int max_n_samples_per_ray)
{
    CHECK_INPUT(origins);
    CHECK_INPUT(viewdirs);
    CHECK_INPUT(nears);
    CHECK_INPUT(fars);
    CHECK_INPUT(hits_aabb);

    CHECK_INPUT(occupancy_grid);
    CHECK_INPUT(aabb);

    int n_rays = origins.size(0);

    int grid_nlvl = occupancy_grid.size(0);
    int3 resolution = make_int3(occupancy_grid.size(1), occupancy_grid.size(2), occupancy_grid.size(3));
    TORCH_CHECK(resolution.x == GRID_RESOLUTION && resolution.y == GRID_RESOLUTION && resolution.z == GRID_RESOLUTION, "only 128 supported");

    torch::Tensor ray_n_samples = torch::zeros({n_rays}, origins.options().dtype(torch::kLong));

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    device::sample_occupancy_grid_first_kernel<<<dim3(n_rays), 128, 0, stream>>>(
        n_rays,
        (float3*) origins.data_ptr<float>(),
        (float3*) viewdirs.data_ptr<float>(),
        nears.data_ptr<float>(),
        fars.data_ptr<float>(),
        hits_aabb.data_ptr<bool>(),

        occupancy_grid.data_ptr<bool>(),
        (float3*) aabb.data_ptr<float>(),
        grid_nlvl,

        step_size,
        cone_angle,
        max_n_samples_per_ray,
        ray_n_samples.data_ptr<int64_t>()
    );

    torch::Tensor cumsum = torch::cumsum(ray_n_samples, 0, ray_n_samples.scalar_type());
    int64_t n_samples = cumsum[-1].item<int64_t>();

    torch::Tensor ray_offsets = cumsum - ray_n_samples;

    torch::Tensor t_starts = torch::empty({n_samples}, origins.options().dtype(torch::kFloat32));
    torch::Tensor t_ends = torch::empty({n_samples}, origins.options().dtype(torch::kFloat32));
    torch::Tensor ray_indices = torch::empty({n_samples}, origins.options().dtype(torch::kLong));

    const int BLOCK_SIZE = 128;
    device::sample_occupancy_grid_second_kernel<<<ceil_div(n_rays, BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(
        n_rays,
        (float3*) origins.data_ptr<float>(),
        (float3*) viewdirs.data_ptr<float>(),
        nears.data_ptr<float>(),
        fars.data_ptr<float>(),
        hits_aabb.data_ptr<bool>(),

        occupancy_grid.data_ptr<bool>(),
        (float3*) aabb.data_ptr<float>(),
        grid_nlvl,

        step_size,
        cone_angle,

        ray_n_samples.data_ptr<int64_t>(),
        ray_offsets.data_ptr<int64_t>(),

        t_starts.data_ptr<float>(),
        t_ends.data_ptr<float>(),
        ray_indices.data_ptr<int64_t>()
    );

    return {t_starts, t_ends, ray_indices, torch::stack({ray_offsets, ray_n_samples}, -1)};
}