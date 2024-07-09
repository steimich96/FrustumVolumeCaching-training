from typing import Callable, List, Optional, Tuple, Union

import torch

from . import cuda as _C
from .scan import exclusive_sum


def _enlarge_aabb(aabb, factor: float) -> torch.Tensor:
    center = (aabb[:3] + aabb[3:]) / 2
    extent = (aabb[3:] - aabb[:3]) / 2
    return torch.cat([center - extent * factor, center + extent * factor])


def _meshgrid3d(res: torch.Tensor, device: Union[torch.device, str] = "cpu") -> torch.Tensor:
    """Create 3D grid coordinates."""
    assert len(res) == 3
    res = res.tolist()
    return torch.stack(
        torch.meshgrid(
            [
                torch.arange(res[0], dtype=torch.long),
                torch.arange(res[1], dtype=torch.long),
                torch.arange(res[2], dtype=torch.long),
            ],
            indexing="ij",
        ),
        dim=-1,
    ).to(device)


class OccupancyGrid(torch.nn.Module):
    RESOLUTION: int = 128

    def __init__(self, aabb: Union[List[int], torch.Tensor], n_levels: int, warmup: bool = True) -> None:
        super().__init__()

        resolution = torch.tensor([self.RESOLUTION] * 3, dtype=torch.int32)
        aabbs = torch.stack([_enlarge_aabb(aabb, 2**i) for i in range(n_levels)], dim=0)

        self.vals_per_lvl = int(resolution.prod().item())
        self.n_levels = n_levels
        self.warmup = warmup

        self.register_buffer("resolution", resolution)  # [3]
        self.register_buffer("aabbs", aabbs)  # [n_aabbs, 6]
        self.register_buffer("occs_float", torch.zeros(n_levels * self.vals_per_lvl, dtype=torch.float32))
        self.register_buffer("occs_binary", torch.zeros([n_levels] + resolution.tolist(), dtype=torch.bool))

        grid_coords = _meshgrid3d(resolution).reshape(self.vals_per_lvl, 3)
        self.register_buffer("grid_coords", grid_coords, persistent=False)
        self.register_buffer("grid_indices", torch.arange(self.vals_per_lvl), persistent=False)

    def ray_aabb_intersect(
        self,
        origins: torch.Tensor,
        viewdirs: torch.Tensor,
        near_plane: float = -float("inf"),
        far_plane: float = float("inf"),
        miss_value: float = float("inf"),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute the minimum and maximum bounds of the AABBs
        aabb_min = self.aabbs[-1, :3]
        aabb_max = self.aabbs[-1, 3:]

        # Compute the intersection distances between the ray and each of the six AABB planes
        t1 = (aabb_min[None, :] - origins) / viewdirs
        t2 = (aabb_max[None, :] - origins) / viewdirs

        # Compute the maximum tmin and minimum tmax for each AABB
        t_mins = torch.max(torch.min(t1, t2), dim=-1)[0]
        t_maxs = torch.min(torch.max(t1, t2), dim=-1)[0]

        # Compute whether each ray-AABB pair intersects
        hits = (t_maxs > t_mins) & (t_maxs > 0)

        # Clip the tmin and tmax values to the near and far planes
        t_mins = torch.clamp(t_mins, min=near_plane, max=far_plane)
        t_maxs = torch.clamp(t_maxs, min=near_plane, max=far_plane)

        # Set the tmin and tmax values to miss_value if there is no intersection
        t_mins = torch.where(hits, t_mins, miss_value)
        t_maxs = torch.where(hits, t_maxs, miss_value)

        return t_mins, t_maxs, hits

    @torch.no_grad()
    def sample(
        self,
        origins: torch.Tensor,  # [n_rays, 3]
        viewdirs: torch.Tensor,  # [n_rays, 3]
        sigma_fn: Optional[Callable],
        near_plane: float = 0.0,
        far_plane: float = 1e10,
        render_step_size: float = 1e-3,
        early_stop_eps: float = 1e-4,
        alpha_thre: float = 0.0,
        cone_angle: float = 0.0,
        max_n_samples_per_ray: int = 1024,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_mins, t_maxs, hits_aabb = self.ray_aabb_intersect(origins, viewdirs)

        near_planes = torch.where(t_mins > near_plane, t_mins, near_plane)
        far_planes = torch.where(t_maxs < far_plane, t_maxs, far_plane)

        if self.training:
            near_planes += torch.rand_like(near_planes) * render_step_size

        t_starts, t_ends, ray_indices, packed_info = _C.sample_occupancy_grid(
            origins.contiguous(),
            viewdirs.contiguous(),
            near_planes.contiguous(),
            far_planes.contiguous(),
            hits_aabb.contiguous(),
            self.occs_binary.contiguous(),
            self.aabbs.contiguous(),
            render_step_size,
            cone_angle,
            max_n_samples_per_ray,
        )

        sigmas = sigma_fn(t_starts, t_ends, ray_indices)

        sigmas_dt = sigmas * (t_ends - t_starts)
        alphas = 1.0 - torch.exp(-sigmas_dt)
        transmittances = torch.exp(-exclusive_sum(sigmas_dt, packed_info))

        visible = (transmittances >= early_stop_eps) & (alphas >= alpha_thre)

        return t_starts[visible], t_ends[visible], ray_indices[visible]

    @torch.no_grad()
    def occupied_at(self, pos: torch.Tensor):
        """Returns a boolean tensor with the same shape, indicating if the grid is set at this position"""

        original_shape = pos.shape
        pos = pos.view(-1, 3)

        # roi -> [0, 1]^3 (in base aabb)
        pos_unit = (pos - self.aabbs[0, :3]) / (self.aabbs[0, 3:] - self.aabbs[0, :3])

        maxval = torch.max(torch.abs(pos_unit - 0.5), dim=-1)[0]
        _, exponent = torch.frexp(maxval)
        mip = exponent + 1

        valid = mip < self.n_levels
        mip = torch.clamp(mip, 0, self.n_levels - 1)

        # [0, 1]^3 (in base aabb) -> [0, 1)^3 (in mip aabb)
        pos_unit_in_mip = torch.clamp(
            (pos_unit - 0.5) * torch.pow(2, -mip.unsqueeze(-1).float()) + 0.5, 0.0, 1.0 - 1e-5
        )

        idcs3D = torch.floor(pos_unit_in_mip * self.resolution).int() * torch.tensor(
            [self.resolution[1] * self.resolution[2], self.resolution[2], 1], dtype=torch.int, device=pos.device
        )
        idcs = torch.sum(idcs3D, dim=-1) + mip * self.vals_per_lvl

        return (valid & self.occs_binary.flatten()[idcs]).reshape(original_shape[:-1])

    @torch.no_grad()
    def update_every_n_steps(
        self,
        step: int,
        occ_eval_fn: Callable,
        occ_thre: float = 1e-2,
        ema_decay: float = 0.95,
        warmup_steps: int = 256,
        n: int = 16,
    ) -> None:
        """Update the estimator every n steps during training.

        Args:
            step: Current training step.
            occ_eval_fn: A function that takes in sample locations :math:`(N, 3)` and
                returns the occupancy values :math:`(N, 1)` at those locations.
            occ_thre: Threshold used to binarize the occupancy grid. Default: 1e-2.
            ema_decay: The decay rate for EMA updates. Default: 0.95.
            warmup_steps: Sample all cells during the warmup stage. After the warmup
                stage we change the sampling strategy to 1/4 uniformly sampled cells
                together with 1/4 occupied cells. Default: 256.
            n: Update the grid every n steps. Default: 16.
        """
        if not self.training:
            raise RuntimeError("You should only call this function only during training.")

        if step % n == 0 and self.training:
            self._update(
                step=step,
                occ_eval_fn=occ_eval_fn,
                occ_thre=occ_thre,
                ema_decay=ema_decay,
                warmup_steps=warmup_steps,
            )

    # adapted from https://github.com/kwea123/ngp_pl/blob/master/models/networks.py
    @torch.no_grad()
    def mark_invisible_cells(
        self,
        K: torch.Tensor,
        c2w: torch.Tensor,
        width: int,
        height: int,
        near_plane: float = 0.0,
        chunk: int = 32**3,
    ) -> None:
        """Mark the cells that aren't covered by the cameras with density -1.
        Should only be executed once before training starts.

        Args:
            K: Camera intrinsics of shape (N, 3, 3) or (1, 3, 3).
            c2w: Camera to world poses of shape (N, 3, 4) or (N, 4, 4).
            width: Image width in pixels
            height: Image height in pixels
            near_plane: Near plane distance
            chunk: The chunk size to split the cells (to avoid OOM)
        """
        assert K.dim() == 3 and K.shape[1:] == (3, 3)
        assert c2w.dim() == 3 and (c2w.shape[1:] == (3, 4) or c2w.shape[1:] == (4, 4))
        assert K.shape[0] == c2w.shape[0] or K.shape[0] == 1

        N_cams = c2w.shape[0]
        w2c_R = c2w[:, :3, :3].transpose(2, 1)  # (N_cams, 3, 3)
        w2c_T = -w2c_R @ c2w[:, :3, 3:]  # (N_cams, 3, 1)

        lvl_indices = self._get_all_cells()
        for lvl, indices in enumerate(lvl_indices):
            grid_coords = self.grid_coords[indices]

            for i in range(0, len(indices), chunk):
                x = grid_coords[i : i + chunk] / (self.resolution - 1)
                indices_chunk = indices[i : i + chunk]

                # voxel coordinates [0, 1]^3 -> world
                xyzs_w = (self.aabbs[lvl, :3] + x * (self.aabbs[lvl, 3:] - self.aabbs[lvl, :3])).T
                xyzs_c = w2c_R @ xyzs_w + w2c_T  # (N_cams, 3, chunk)
                uvd = K @ xyzs_c  # (N_cams, 3, chunk)
                uv = uvd[:, :2] / uvd[:, 2:]  # (N_cams, 2, chunk)
                in_image = (
                    (uvd[:, 2] >= 0) & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
                )
                covered_by_cam = (uvd[:, 2] >= near_plane) & in_image  # (N_cams, chunk)
                # if the cell is visible by at least one camera

                count = covered_by_cam.sum(0) / N_cams

                too_near_to_cam = (uvd[:, 2] < near_plane) & in_image  # (N, chunk)
                # if the cell is too close (in front) to any camera
                too_near_to_any_cam = too_near_to_cam.any(0)
                # a valid cell should be visible by at least one camera and not too close to any camera
                valid_mask = (count > 0) & (~too_near_to_any_cam)

                cell_ids_base = lvl * self.vals_per_lvl
                self.occs_float[cell_ids_base + indices_chunk] = torch.where(valid_mask, 0.0, -1.0)

    @torch.no_grad()
    def _get_all_cells(self) -> List[torch.Tensor]:
        """Returns all cells of the grid."""
        lvl_indices = []
        for lvl in range(self.n_levels):
            # filter out the cells with -1 density (non-visible to any camera)
            cell_ids = lvl * self.vals_per_lvl + self.grid_indices
            indices = self.grid_indices[self.occs_float[cell_ids] >= 0.0]
            lvl_indices.append(indices)
        return lvl_indices

    @torch.no_grad()
    def _sample_uniform_and_occupied_cells(self, n: int) -> List[torch.Tensor]:
        """Samples both n uniform and occupied cells."""
        lvl_indices = []
        for lvl in range(self.n_levels):
            uniform_indices = torch.randint(self.vals_per_lvl, (n,), device=self.occs_float.device)

            # filter out the cells with -1 density (non-visible to any camera)
            cell_ids = lvl * self.vals_per_lvl + uniform_indices
            uniform_indices = uniform_indices[self.occs_float[cell_ids] >= 0.0]
            occupied_indices = torch.nonzero(self.occs_binary[lvl].flatten())[:, 0]

            if n < len(occupied_indices):
                selector = torch.randint(len(occupied_indices), (n,), device=self.occs_float.device)
                occupied_indices = occupied_indices[selector]

            indices = torch.cat([uniform_indices, occupied_indices], dim=0)
            lvl_indices.append(indices)
        return lvl_indices

    @torch.no_grad()
    def _update(
        self,
        step: int,
        occ_eval_fn: Callable,
        occ_thre: float = 0.01,
        ema_decay: float = 0.95,
        warmup_steps: int = 256,
    ) -> None:
        """Update the occ field in the EMA way."""
        # sample cells
        if self.warmup and step < warmup_steps:
            lvl_indices = self._get_all_cells()
        else:
            N = self.vals_per_lvl // 4
            lvl_indices = self._sample_uniform_and_occupied_cells(N)

        for lvl, indices in enumerate(lvl_indices):
            # infer occupancy: density * step_size
            grid_coords = self.grid_coords[indices]
            x = (grid_coords + torch.rand_like(grid_coords, dtype=torch.float32)) / self.resolution

            # voxel coordinates [0, 1]^3 -> world
            x = self.aabbs[lvl, :3] + x * (self.aabbs[lvl, 3:] - self.aabbs[lvl, :3])
            occ = occ_eval_fn(x).squeeze(-1)

            # ema update
            cell_ids = lvl * self.vals_per_lvl + indices
            self.occs_float[cell_ids] = torch.maximum(self.occs_float[cell_ids] * ema_decay, occ)

        thre = torch.clamp(self.occs_float[self.occs_float >= 0].mean(), max=occ_thre)
        self.occs_binary = (self.occs_float > thre).view(self.occs_binary.shape)
