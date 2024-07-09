
#include <torch/extension.h>

// scan
torch::Tensor inclusive_sum(
    torch::Tensor chunk_starts,
    torch::Tensor chunk_cnts,
    torch::Tensor inputs,
    bool normalize,
    bool backward);
torch::Tensor exclusive_sum(
    torch::Tensor chunk_starts,
    torch::Tensor chunk_cnts,
    torch::Tensor inputs,
    bool normalize,
    bool backward);
torch::Tensor inclusive_prod_forward(
    torch::Tensor chunk_starts,
    torch::Tensor chunk_cnts,
    torch::Tensor inputs);
torch::Tensor inclusive_prod_backward(
    torch::Tensor chunk_starts,
    torch::Tensor chunk_cnts,
    torch::Tensor inputs,
    torch::Tensor outputs,
    torch::Tensor grad_outputs);
torch::Tensor exclusive_prod_forward(
    torch::Tensor chunk_starts,
    torch::Tensor chunk_cnts,
    torch::Tensor inputs);
torch::Tensor exclusive_prod_backward(
    torch::Tensor chunk_starts,
    torch::Tensor chunk_cnts,
    torch::Tensor inputs,
    torch::Tensor outputs,
    torch::Tensor grad_outputs);

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
    int max_n_samples_per_ray);

torch::Tensor distortion_loss(
    torch::Tensor ray_indices,
    torch::Tensor chunk_starts,
    torch::Tensor chunk_cnts,
    torch::Tensor weights,
    torch::Tensor steps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sample_occupancy_grid", &sample_occupancy_grid);
    m.def("distortion_loss", &distortion_loss);

    m.def("inclusive_sum", &inclusive_sum);
    m.def("exclusive_sum", &exclusive_sum);
    m.def("inclusive_prod_forward", &inclusive_prod_forward);
    m.def("inclusive_prod_backward", &inclusive_prod_backward);
    m.def("exclusive_prod_forward", &exclusive_prod_forward);
    m.def("exclusive_prod_backward", &exclusive_prod_backward);
}