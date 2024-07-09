import math

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import collate, default_collate_fn_map

from . import cuda as _C
from .model import InterNerf
from .occupancy_grid import OccupancyGrid
from .pack import pack_info
from .scan import exclusive_sum


##### Define with: #####
# https://gradient-scaling.github.io/
class GradientScaler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, ray_dist):
        ctx.save_for_backward(ray_dist)
        return value, ray_dist

    @staticmethod
    def backward(ctx, grad_value, grad_output_ray_dist):
        (ray_dist,) = ctx.saved_tensors
        scaling = torch.square(ray_dist).clamp(0, 1)
        return grad_value * (scaling if grad_value.shape[-1] == scaling.shape[-1] else scaling.unsqueeze(-1)) , grad_output_ray_dist


def to_stepping_space(t, cone_angle, dt_min, near_plane):
    log1p_c = math.log(1.0 + cone_angle)

    # For t < at, we have linear step size dt_min
    step_min = 1 / cone_angle - near_plane / dt_min
    t_step_min = dt_min / cone_angle

    stepping_t = torch.where(t > t_step_min, torch.log(t / t_step_min) / log1p_c + step_min, (t - near_plane) / dt_min)

    return stepping_t


def to_normalized_stepping_space(t, cone_angle, dt_min, near_plane, far_plane):
    return to_stepping_space(t, cone_angle, dt_min, near_plane) / to_stepping_space(torch.tensor(far_plane), cone_angle, dt_min, near_plane)


@torch.no_grad()
def calculate_n_rendered_samples(
    radiance_field: InterNerf,
    occupancy_grid: OccupancyGrid,
    origins: torch.Tensor,
    viewdirs: torch.Tensor,
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    early_stop_eps: float = 1e-4,
    alpha_thre: float = 0.0,
    cone_angle: float = 0.0,
    chunk_size: int = 8192,
) -> torch.Tensor:
    rays_shape = origins.shape
    num_rays = rays_shape[:-1].numel()

    origins = origins.reshape(-1, 3)
    viewdirs = viewdirs.reshape(-1, 3)

    def sigma_fn(t_starts, t_ends, ray_indices):
        tmp_origins = chunk_origins[ray_indices]
        tmp_viewdirs = chunk_viewdirs[ray_indices]
        sample_positions = tmp_origins + tmp_viewdirs * (t_starts + t_ends)[:, None] / 2.0

        return radiance_field.query_density(sample_positions)[0]

    n_rendered_samples = 0
    for i in range(0, num_rays, chunk_size):
        actual_chunk_size = min(i + chunk_size, num_rays) - i
        chunk_origins = origins[i : i + actual_chunk_size]
        chunk_viewdirs = viewdirs[i : i + actual_chunk_size]

        t_starts, _, _ = occupancy_grid.sample(
            chunk_origins,
            chunk_viewdirs,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            early_stop_eps=early_stop_eps,
            alpha_thre=alpha_thre,
            cone_angle=cone_angle,
        )
        n_rendered_samples += t_starts.shape[0]

    return n_rendered_samples


def train_batch(
    radiance_field: InterNerf,
    occupancy_grid: OccupancyGrid,
    origins: torch.Tensor,
    viewdirs: torch.Tensor,
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    early_stop_eps: float = 1e-4,
    alpha_thre: float = 0.0,
    cone_angle: float = 0.0,
    render_bkgd: Optional[torch.Tensor] = None,
    chunk_size: int = 8192,
    use_interpol_training: bool = False,
    interpol_viewdirs: Optional[torch.Tensor] = None,
    interpol_coeffs: Optional[torch.Tensor] = None,
    interpol_scale: float = 1.0,
    additional_losses: Dict[str, torch.Tensor] = None,
    viewdep_train: bool = False,
    random_viewdirs: Optional[torch.Tensor] = None,
    random_viewdir_weights: Optional[torch.Tensor] = None,
    gradient_scaling: bool = False,
    max_n_samples_per_ray: int = 1024,
    normalize_interpolated_latents: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Renders all given rays"""
    rays_shape = origins.shape
    num_rays = rays_shape[:-1].numel()

    origins = origins.reshape(-1, 3)
    viewdirs = viewdirs.reshape(-1, 3)
    render_bkgd = render_bkgd.reshape(-1, 3)

    n_init_viewdirs = random_viewdirs.shape[0] if random_viewdirs is not None else 1
    random_viewdirs = random_viewdirs.reshape(n_init_viewdirs, -1, 3) if random_viewdirs is not None else None
    random_viewdir_weights = random_viewdir_weights.reshape(n_init_viewdirs, -1, 1) if random_viewdir_weights is not None else None


    if use_interpol_training:
        interpol_viewdirs = interpol_viewdirs.reshape([-1] + list(interpol_viewdirs.shape[-2:]))
        interpol_coeffs = interpol_coeffs.reshape([-1] + list(interpol_coeffs.shape[-2:]))

    def sigma_fn(t_starts, t_ends, ray_indices):
        tmp_origins = chunk_origins[ray_indices]
        tmp_viewdirs = chunk_viewdirs[ray_indices]
        sample_positions = tmp_origins + tmp_viewdirs * (t_starts + t_ends)[:, None] / 2.0

        sigmas = radiance_field.query_density(sample_positions)[0]
        return sigmas

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        tmp_origins = chunk_origins[ray_indices]
        tmp_viewdirs = chunk_viewdirs[ray_indices]
        tmp_init_viewdirs = chunk_init_viewdirs[:, ray_indices]
        tmp_init_viewdir_weights = chunk_init_viewdir_weights[:, ray_indices]
        sample_positions = tmp_origins + tmp_viewdirs * (t_starts + t_ends)[:, None] / 2.0

        sigmas, mlp_base_output, sigmas_before_activation = radiance_field._forward_base(sample_positions)
        
        if additional_losses is not None and "density_regularization" in additional_losses:
            additional_losses["density_regularization"].add_(torch.maximum(torch.abs(sigmas_before_activation - 10.0, 0.0).pow(2).mean() * chunk_percentage_total))

        rgbs = torch.empty_like(tmp_init_viewdirs)
        for vi in range(tmp_init_viewdirs.shape[0]):
            latents = radiance_field._forward_first_head(tmp_init_viewdirs[vi], mlp_base_output)
            rgbs[vi] = radiance_field._forward_head(tmp_init_viewdirs[vi], tmp_viewdirs, latents, sample_positions)

            if additional_losses is not None and "latent_regularization" in additional_losses:
                additional_losses["latent_regularization"].add_(torch.relu(torch.abs(latents[..., (1 if radiance_field.first_latent_is_density else 0) :]) - 10.0).pow(2).mean() / (n_init_viewdirs) * chunk_percentage_total)
        rgbs = (rgbs * tmp_init_viewdir_weights).sum(dim=0)

        if gradient_scaling:
            rgbs, t_starts = GradientScaler.apply(rgbs, t_starts)
            sigmas, t_starts = GradientScaler.apply(sigmas, t_starts)

        return rgbs, sigmas, sigmas_before_activation

    def t_to_normalized_stepping(t_starts, t_ends):
        return to_normalized_stepping_space((t_starts + t_ends) / 2.0, cone_angle, render_step_size, near_plane, far_plane)

    chunk_results = [[] for _ in range(n_init_viewdirs)]

    for i in range(0, num_rays, chunk_size):
        actual_chunk_size = min(i + chunk_size, num_rays) - i
        chunk_origins = origins[i : i + actual_chunk_size]
        chunk_viewdirs = viewdirs[i : i + actual_chunk_size]
        chunk_init_viewdirs = random_viewdirs[:, i : i + actual_chunk_size] if random_viewdirs is not None else chunk_viewdirs.unsqueeze(0)
        chunk_init_viewdir_weights = random_viewdir_weights[:, i : i + actual_chunk_size] if random_viewdirs is not None else torch.full_like(chunk_viewdirs[..., :1].unsqueeze(0), 1.0 / n_init_viewdirs)
        chunk_percentage_total = actual_chunk_size / num_rays

        t_starts, t_ends, ray_indices = occupancy_grid.sample(
            chunk_origins,
            chunk_viewdirs,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            early_stop_eps=early_stop_eps,
            alpha_thre=alpha_thre,
            cone_angle=cone_angle,
            max_n_samples_per_ray=max_n_samples_per_ray,
        )

        sample_pos = (t_starts + t_ends)[:, None] / 2.0 * chunk_viewdirs[ray_indices] + chunk_origins[ray_indices]

        if use_interpol_training and t_starts.shape[0] > 0:
            assert interpol_coeffs is not None and interpol_viewdirs is not None
            assert interpol_coeffs.shape[0] == origins.shape[0]

            chunk_interpol_viewdirs = interpol_viewdirs[i : i + actual_chunk_size]
            chunk_interpol_coeffs = interpol_coeffs[i : i + actual_chunk_size]

            dt = (t_ends - t_starts)[:, None, None]
            t_mids = (t_starts + t_ends)[:, None, None] / 2.0

            if chunk_interpol_coeffs.shape[-2] == 8:
                tz = (chunk_interpol_coeffs[ray_indices, :, 2].reshape(-1, 4, 2) * torch.tensor([-1, 1], device=chunk_interpol_coeffs.device)).reshape(-1, 8, 1)
            elif chunk_interpol_coeffs.shape[-2] == 2:
                tz = (chunk_interpol_coeffs[ray_indices, :, 0].reshape(-1, 1, 2) * torch.tensor([-1, 1], device=chunk_interpol_coeffs.device)).reshape(-1, 2, 1)
            elif chunk_interpol_coeffs.shape[-2] == 1:
                tz = torch.zeros_like(chunk_interpol_coeffs[ray_indices, :, 0].reshape(-1, 1, 1)) # No offset (no interpol)

            t_offset = t_mids + (tz * interpol_scale) * dt
            interpol_samples_pos = chunk_origins[ray_indices, None, :] + t_offset * chunk_interpol_viewdirs[ray_indices]

            def interpolate_sigmas(interpol_samples_sigma, interpol_samples_occupied, lerp_weights):
                return torch.sum(lerp_weights * interpol_samples_occupied * interpol_samples_sigma, dim=-1)

            def interpolate_latents(interpol_samples_mlp_base_output, init_viewdirs, weights, weights_sum):
                interpol_samples_latents = radiance_field._forward_first_head(init_viewdirs, interpol_samples_mlp_base_output)
                # assert interpol_samples_latents.isfinite().all()
                interpol_latents = torch.sum(weights.unsqueeze(-1) * interpol_samples_latents, dim=-2)
                if normalize_interpolated_latents:
                    interpol_latents /= torch.clamp(weights_sum, min=1e-9)
                return interpol_latents.to(interpol_samples_latents)

            tmp_viewdirs = chunk_viewdirs[ray_indices]
            tmp_init_viewdirs = chunk_init_viewdirs[:, ray_indices]
            tmp_init_viewdir_weights = chunk_init_viewdir_weights[:, ray_indices]
            broadcasted_init_viewdirs = tmp_init_viewdirs.unsqueeze(-2).broadcast_to([n_init_viewdirs] + list(interpol_samples_pos.shape[:-1]) + [3])

            interpol_samples_sigma, interpol_samples_mlp_base_output, interpol_samples_sigma_before_activation = radiance_field._forward_base(interpol_samples_pos)
            interpol_samples_occupied = occupancy_grid.occupied_at(interpol_samples_pos)

            lerp_weights = torch.prod(1 - chunk_interpol_coeffs, dim=-1)[ray_indices]
            weights = lerp_weights * interpol_samples_occupied
            weights_sum = weights.sum(dim=-1, keepdim=True)

            interpol_sigmas = interpolate_sigmas(interpol_samples_sigma, interpol_samples_occupied, lerp_weights)
            
            if gradient_scaling:
                interpol_sigmas, t_starts = GradientScaler.apply(interpol_sigmas, t_starts)
            
            if additional_losses is not None and "density_regularization" in additional_losses:
                additional_losses["density_regularization"].add_(torch.relu(torch.abs(interpol_samples_sigma_before_activation) - 10.0).pow(2).mean() * chunk_percentage_total)

            for vi in range(tmp_init_viewdirs.shape[0]):
                interpol_latents = interpolate_latents(interpol_samples_mlp_base_output, broadcasted_init_viewdirs[vi], weights, weights_sum)

                if radiance_field.first_latent_is_density:
                    interpol_latents[:, 0] = torch.log(torch.clamp(interpol_sigmas, min=1e-9)).to(interpol_latents)

                # assert interpol_latents.isfinite().all()
                rgbs = radiance_field._forward_head(tmp_init_viewdirs[vi], tmp_viewdirs, interpol_latents, sample_pos)

                if additional_losses is not None and "sample_vs_interpol" in additional_losses:
                    samples_sigmas, samples_latents, _ = radiance_field.query_latents(sample_pos, tmp_init_viewdirs[vi])
                    additional_losses["sample_vs_interpol"].add_(F.mse_loss(interpol_latents.view(-1, radiance_field.n_latents), samples_latents.detach().view(-1, radiance_field.n_latents)) / (n_init_viewdirs) * chunk_percentage_total)

                if additional_losses is not None and "latent_regularization" in additional_losses:
                    additional_losses["latent_regularization"].add_(torch.relu(torch.abs(interpol_latents[..., (1 if radiance_field.first_latent_is_density else 0) :]) - 10.0).pow(2).mean() / (n_init_viewdirs) * chunk_percentage_total)

                if gradient_scaling:
                    rgbs, t_starts = GradientScaler.apply(rgbs, t_starts)
                    
                rgb, opacity, distortion_loss = render_chunk_rgb_sigma(
                    rgbs,
                    interpol_sigmas,
                    t_starts,
                    t_ends,
                    ray_indices,
                    actual_chunk_size,
                    t_to_normalized_stepping,
                    render_bkgd[i : i + actual_chunk_size],
                )
                chunk_results[vi].append([rgb, opacity, t_starts.shape[0]])
        else:
            rgb, opacity, distortion_loss = render_chunk(
                t_starts,
                t_ends,
                ray_indices,
                actual_chunk_size,
                rgb_sigma_fn,
                t_to_normalized_stepping,
                render_bkgd[i : i + actual_chunk_size],
            )

            # assert not (rgb.isnan().any() or rgb.isinf().any())
            # assert not (opacity.isnan().any() or opacity.isinf().any())

            chunk_results[0].append([rgb, opacity, t_starts.shape[0]])
                
        if additional_losses is not None and "distortion_loss" in additional_losses:
            additional_losses["distortion_loss"].add_(distortion_loss / actual_chunk_size * chunk_percentage_total)

    all_results = []
    for vi in range(n_init_viewdirs):
        colors, opacities, n_render_samples = collate(
            chunk_results[vi],
            collate_fn_map={**default_collate_fn_map, torch.Tensor: lambda x, **_: torch.cat(x, 0)},
        )

        all_results.append((
            colors.view((*rays_shape[:-1], -1)),
            opacities.view((*rays_shape[:-1], -1)),
            torch.sum(n_render_samples).item(),
        ))
    
    return all_results
    

@torch.no_grad()
def render_all(
    radiance_field: InterNerf,
    occupancy_grid: OccupancyGrid,
    origins: torch.Tensor,
    viewdirs: torch.Tensor,
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    early_stop_eps: float = 1e-4,
    alpha_thre: float = 0.0,
    cone_angle: float = 0.0,
    render_bkgd: Optional[torch.Tensor] = None,
    chunk_size: int = 8192,
    random_viewdirs: Optional[torch.Tensor] = None,
    max_n_samples_per_ray: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Renders all given rays"""
    rays_shape = origins.shape
    num_rays = rays_shape[:-1].numel()

    origins = origins.reshape(-1, 3)
    viewdirs = viewdirs.reshape(-1, 3)
    render_bkgd = render_bkgd.reshape(-1, 3)

    random_viewdirs = random_viewdirs.reshape(random_viewdirs.shape[0], -1, 3) if random_viewdirs is not None else None

    def sigma_fn(t_starts, t_ends, ray_indices):
        tmp_origins = chunk_origins[ray_indices]
        tmp_viewdirs = chunk_viewdirs[ray_indices]
        sample_positions = tmp_origins + tmp_viewdirs * (t_starts + t_ends)[:, None] / 2.0

        return radiance_field.query_density(sample_positions)[0]

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        tmp_origins = chunk_origins[ray_indices]
        tmp_viewdirs = chunk_viewdirs[ray_indices]
        tmp_init_viewdirs = chunk_init_viewdirs[:, ray_indices]
        sample_positions = tmp_origins + tmp_viewdirs * (t_starts + t_ends)[:, None] / 2.0
        
        return radiance_field(sample_positions, tmp_viewdirs, tmp_init_viewdirs[vi])

    n_init_viewdirs = random_viewdirs.shape[0] if random_viewdirs is not None else 1
    chunk_results = [[] for _ in range(n_init_viewdirs)]
    for i in range(0, num_rays, chunk_size):
        actual_chunk_size = min(i + chunk_size, num_rays) - i
        chunk_origins = origins[i : i + actual_chunk_size]
        chunk_viewdirs = viewdirs[i : i + actual_chunk_size]
        chunk_init_viewdirs = random_viewdirs[:, i : i + actual_chunk_size] if random_viewdirs is not None else chunk_viewdirs.unsqueeze(0)

        t_starts, t_ends, ray_indices = occupancy_grid.sample(
            chunk_origins,
            chunk_viewdirs,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            early_stop_eps=early_stop_eps,
            alpha_thre=alpha_thre,
            cone_angle=cone_angle,
            max_n_samples_per_ray=max_n_samples_per_ray,
        )
        
        for vi in range(n_init_viewdirs):
            rgb, opacity, _ = render_chunk(
                t_starts,
                t_ends,
                ray_indices,
                actual_chunk_size,
                rgb_sigma_fn,
                None,
                render_bkgd[i : i + actual_chunk_size],
            ) 
            chunk_results[vi].append([rgb, opacity, t_starts.shape[0]])

    all_results = []
    for vi in range(n_init_viewdirs):
        colors, opacities, n_render_samples = collate(
            chunk_results[vi],
            collate_fn_map={**default_collate_fn_map, torch.Tensor: lambda x, **_: torch.cat(x, 0)},
        )

        all_results.append((
            colors.view((*rays_shape[:-1], -1)),
            opacities.view((*rays_shape[:-1], -1)),
            torch.sum(n_render_samples).item(),
        ))
    
    return all_results[0] if random_viewdirs is None else all_results


def render_chunk(
    t_starts: torch.Tensor,
    t_ends: torch.Tensor,
    ray_indices: torch.Tensor,
    num_rays: int,
    rgb_sigma_fn: Callable,
    to_stepping_fn: Callable,
    render_bkgd: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if t_starts.shape[0] > 0:
        rgbs, sigmas, _ = rgb_sigma_fn(t_starts, t_ends, ray_indices)
    else:
        rgbs = torch.empty((0, 3), device=t_starts.device)
        sigmas = torch.empty((0,), device=t_starts.device)
    return render_chunk_rgb_sigma(rgbs, sigmas, t_starts, t_ends, ray_indices, num_rays, to_stepping_fn, render_bkgd)


def render_chunk_rgb_sigma(
    rgbs: torch.Tensor,
    sigmas: torch.Tensor,
    t_starts: torch.Tensor,
    t_ends: torch.Tensor,
    ray_indices: torch.Tensor,
    num_rays: int,
    to_stepping_fn: Optional[Callable] = None,
    render_bkgd: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    packed_info = pack_info(ray_indices, num_rays)
    chunk_starts, chunk_cnts = packed_info.unbind(dim=-1)

    weights, transmittances, alphas = render_weight_from_density(t_starts, t_ends, sigmas, packed_info)
    
    if to_stepping_fn:
        s = to_stepping_fn(t_starts, t_ends)

        per_weight_distortion_loss = _C.distortion_loss(
            ray_indices.contiguous(),
            chunk_starts.contiguous(),
            chunk_cnts.contiguous(),
            weights.contiguous(),
            s.contiguous(),
        )
        distortion_loss = (weights * per_weight_distortion_loss).sum()
    else:
        distortion_loss = None
        

    colors = accumulate(weights, values=rgbs, ray_indices=ray_indices, n_rays=num_rays)
    opacities = accumulate(weights, values=None, ray_indices=ray_indices, n_rays=num_rays)

    if render_bkgd is not None:
        colors = colors + render_bkgd * (1.0 - opacities)

    return colors, opacities, distortion_loss


def render_weight_from_density(
    t_starts: torch.Tensor, t_ends: torch.Tensor, sigmas: torch.Tensor, packed_info: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sigmas_dt = sigmas * (t_ends - t_starts)
    alphas = 1.0 - torch.exp(-sigmas_dt)
    transmittances = torch.exp(-exclusive_sum(sigmas_dt, packed_info))

    weights = transmittances * alphas
    return weights, transmittances, alphas


def accumulate(
    weights: torch.Tensor,
    values: Optional[torch.Tensor],
    ray_indices: torch.Tensor,
    n_rays: int,
) -> torch.Tensor:
    if values is None:
        src = weights[..., None]
    else:
        src = weights[..., None] * values

    outputs = torch.zeros((n_rays, src.shape[-1]), device=src.device, dtype=src.dtype)
    outputs.index_add_(0, ray_indices, src)
    return outputs
