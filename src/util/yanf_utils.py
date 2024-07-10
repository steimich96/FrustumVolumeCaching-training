"""
Copyright (C) 2024, Michael Steiner, Graz University of Technology.
This code is licensed under the MIT license.
"""
 
import itertools
import torch

def spherical_to_cartesian(phi: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    sin_theta = torch.sin(theta)
    x = torch.cos(phi) * sin_theta
    y = torch.sin(phi) * sin_theta
    z = torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)

def rotate_z_to_vector(z_directions: torch.Tensor, rotation_vector: torch.Tensor) -> torch.Tensor:
    """Takes a random direction in hemisphere around [0, 0, 1] and rotates it towards the rotation vector"""
    nx, ny, nz = rotation_vector[..., 0], rotation_vector[..., 1], rotation_vector[..., 2]
    z_minus_1 = torch.abs(1.0 + nz) < 1e-9
    nz = torch.where(z_minus_1, 1e-9, nz)
    rotation_matrix = torch.stack(
        [
            1.0 - nx * nx / (1.0 + nz),      -nx * ny / (1.0 + nz), -nx,
                 -nx * ny / (1.0 + nz), 1.0 - ny * ny / (1.0 + nz), -ny,
                                    nx,                         ny,  nz,
        ],
        dim=-1,
    ).reshape(rotation_vector.shape[:-1] + (3, 3)).unsqueeze(-3)

    result = (z_directions.unsqueeze(-2) @ rotation_matrix).squeeze(-2)
    result[z_minus_1] = -z_directions[z_minus_1]
    return result


def random_direction_angle_uniform(
    directions_like: torch.Tensor, 
    max_cone_angle: float = torch.pi / 2.0
) -> torch.Tensor:
    """Creates uniformly random hemisphere directions around [0, 0, 1] up to a maximum cone angle"""
    r1 = torch.rand_like(directions_like[..., 2])
    r2 = torch.rand_like(directions_like[..., 2])
    phi = 2.0 * torch.pi * r1

    z = 1.0 - r2 * (1.0 - torch.cos(torch.tensor(max_cone_angle).to(directions_like)))
    sqrt_z = torch.sqrt(1.0 - z**2)
    x = torch.cos(phi) * sqrt_z
    y = torch.sin(phi) * sqrt_z
    return torch.stack([x, y, z], dim=-1)

def random_direction_angle(
    directions_like: torch.Tensor, 
    max_cone_angle: float = torch.pi / 2.0
) -> torch.Tensor:
    """Creates random hemisphere directions, uniform in (phi, theta) around [0, 0, 1] up to a maximum cone angle"""
    rotation_center_phi = torch.rand_like(directions_like[..., 2]) * 2.0 * torch.pi
    rotation_center_theta = torch.rand_like(directions_like[..., 2]) * max_cone_angle

    return spherical_to_cartesian(rotation_center_phi, rotation_center_theta)

def circular_samples_around_directions(
    directions: torch.Tensor,
    n_samples: int,
    cone_angle: float,
    phi_offsets: torch.Tensor,
) -> torch.Tensor:
    phi_samples_uniform = torch.tensor([1.0 / n_samples * float(i) for i in range(n_samples)]).to(directions)
    
    z_phis = (phi_offsets + phi_samples_uniform) * 2.0 * torch.pi
    z_thetas = torch.full_like(z_phis, cone_angle)
    
    z_sphere_samples = spherical_to_cartesian(z_phis, z_thetas)
    return rotate_z_to_vector(z_sphere_samples, directions)
    
def random_directions_angle_circular(
    directions_like: torch.Tensor, 
    n_directions: int = 4, 
    max_cone_angle: float = torch.pi / 2.0,
) -> torch.Tensor:
    # Generate a random direction with a maximum angle of "max_cone_angle / 2.0"
    rotation_center = random_direction_angle(directions_like, max_cone_angle / 2.0)

    # Generate N circular samples with "theta = max_cone_angle / 2.0" and phi uniformly spaced with random initial offset
    r1_phi_initial_offsets = (1.0 / n_directions) * torch.rand_like(directions_like[..., 2:])
    return circular_samples_around_directions(rotation_center, n_directions, max_cone_angle * 0.5, r1_phi_initial_offsets)
    

def generate_random_interpol_coefficients(num_rays: int, n_random_dims: int, device: str) -> torch.Tensor:
    if (n_random_dims > 0):
        t = torch.rand(size=(num_rays, n_random_dims), device=device) # [num_rays, 3] = [du, dv, dt] OR [num_rays, 1] = [dt]
        m = torch.tensor(list(itertools.product([-1, 1], repeat=n_random_dims)), device=device) # [8, 3] OR [2, 1]
        c = torch.tensor(list(itertools.product([1, 0], repeat=n_random_dims)), device=device) # [8, 3] OR [2, 1]

        # = [[du, dv, dt], [du, dv, 1-dt], [du, 1-dv, dt], [du, 1-dv, 1-dt], [1-du, dv, dt], ..., [1-du, 1-dv, 1-dt]] OR = [[dt], [1-dt]]
        return c[None, :, :] + m[None, :, :] * t[:, None, :] # [num_rays, 8, 3] OR [num_rays, 2, 1]
    else:
        return torch.zeros(size=(num_rays, 1, 1), device=device)