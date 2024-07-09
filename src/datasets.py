import os
import sys
import itertools

import numpy as np
import imageio
import torch
import torch.nn.functional as F
import tqdm

from src.util.yanf_utils import random_directions_angle_circular, rotate_z_to_vector, generate_random_interpol_coefficients


_PATH = os.path.abspath(__file__)

sys.path.insert(0, os.path.join(os.path.dirname(_PATH), "pycolmap", "pycolmap"))
from scene_manager import SceneManager


def _load_colmap(data_root: str, scene: str, factor: int = 1):
    assert factor in [1, 2, 4, 8]

    data_dir = os.path.join(data_root, scene)
    colmap_dir = os.path.join(data_dir, "sparse/0/")

    manager = SceneManager(colmap_dir)
    manager.load_cameras()
    manager.load_images()

    # Assume shared intrinsics between all cameras.
    cam = manager.cameras[1]
    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    K[:2, :] /= factor

    # Extract extrinsic matrices in world-to-camera format.
    imdata = manager.images
    w2c_mats = []
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    for k in imdata:
        im = imdata[k]
        rot = im.R()
        trans = im.tvec.reshape(3, 1)
        w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
        w2c_mats.append(w2c)
    w2c_mats = np.stack(w2c_mats, axis=0)

    # Convert extrinsics to camera-to-world.
    camtoworlds = np.linalg.inv(w2c_mats)

    # Image names from COLMAP. No need for permuting the poses according to
    # image names anymore.
    image_names = [imdata[k].name for k in imdata]

    # Get distortion parameters.
    type_ = cam.camera_type

    if type_ == 0 or type_ == "SIMPLE_PINHOLE":
        params = None
        camtype = "perspective"

    elif type_ == 1 or type_ == "PINHOLE":
        params = None
        camtype = "perspective"

    if type_ == 2 or type_ == "SIMPLE_RADIAL":
        params = {k: 0.0 for k in ["k1", "k2", "k3", "p1", "p2"]}
        params["k1"] = cam.k1
        camtype = "perspective"

    elif type_ == 3 or type_ == "RADIAL":
        params = {k: 0.0 for k in ["k1", "k2", "k3", "p1", "p2"]}
        params["k1"] = cam.k1
        params["k2"] = cam.k2
        camtype = "perspective"

    elif type_ == 4 or type_ == "OPENCV":
        params = {k: 0.0 for k in ["k1", "k2", "k3", "p1", "p2"]}
        params["k1"] = cam.k1
        params["k2"] = cam.k2
        params["p1"] = cam.p1
        params["p2"] = cam.p2
        camtype = "perspective"

    elif type_ == 5 or type_ == "OPENCV_FISHEYE":
        params = {k: 0.0 for k in ["k1", "k2", "k3", "k4"]}
        params["k1"] = cam.k1
        params["k2"] = cam.k2
        params["k3"] = cam.k3
        params["k4"] = cam.k4
        camtype = "fisheye"

    assert params is None, "Only support pinhole camera model."

    # Previous Nerf results were generated with images sorted by filename,
    # ensure metrics are reported on the same test set.
    inds = np.argsort(image_names)
    image_names = [image_names[i] for i in inds]
    camtoworlds = camtoworlds[inds]

    colmap_image_dir = os.path.join(data_dir, "images")
    image_dir = os.path.join(data_dir, "images" + (f"_{factor}" if factor > 1 else ""))
    for d in [image_dir, colmap_image_dir]:
        if not os.path.exists(d):
            raise ValueError(f"Image folder {d} does not exist.")

    # Downsampled images may have different names vs images used for COLMAP,
    # so we need to map between the two sorted lists of files.
    colmap_files = sorted(os.listdir(colmap_image_dir))
    image_files = sorted(os.listdir(image_dir))
    colmap_to_image = dict(zip(colmap_files, image_files))
    image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

    print("Loading images...")
    images = [imageio.imread(x) for x in tqdm.tqdm(image_paths)]
    images = np.stack(images, axis=0)

    # Select the split.
    all_indices = np.arange(images.shape[0])
    split_indices = {
        "test": all_indices[all_indices % 8 == 0],
        "train": all_indices[all_indices % 8 != 0],
    }
    return images, camtoworlds, K, split_indices, image_paths


def similarity_from_cameras(c2w, strict_scaling):
    """
    reference: nerf-factory
    Get a similarity transform to normalize dataset
    from c2w (OpenCV convention) cameras
    :param c2w: (N, 4)
    :return T (4,4) , scale (float)
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # (1) Rotate the world so that z+ is the up axis
    # we estimate the up axis by averaging the camera up axes
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        # In the unlikely case the original data has y+ up axis,
        # rotate 180-deg about x axis
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    #  R_align = np.eye(3) # DEBUG
    R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    # (2) Recenter the scene using camera center rays
    # find the closest point to the origin for each camera's center ray
    nearest = t + (fwds * -t).sum(-1)[:, None] * fwds

    # median for more robustness
    translate = -np.median(nearest, axis=0)

    #  translate = -np.mean(t, axis=0)  # DEBUG

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    # (3) Rescale the scene using camera distances
    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))

    return transform, scale


class MipNerf360DataLoader:
    INDOOR_SCENES = ["bonsai", "counter", "kitchen", "room"]
    OUTDOOR_SCENES = ["garden", "bicycle", "stump", "flowers", "treehill"]
    SCENES = INDOOR_SCENES + OUTDOOR_SCENES
    OPENGL_CAMERA = False

    def __init__(self, scene: str, data_root: str, downscale_factor: int = 1):
        assert scene in self.SCENES, "%s" % scene

        self.images, self.c2w, self.K, self.split_indices, self.image_paths = _load_colmap(data_root, scene, downscale_factor)

        # normalize the scene
        T, sscale = similarity_from_cameras(self.c2w, strict_scaling=False)
        self.c2w = np.einsum("nij, ki -> nkj", self.c2w, T)
        self.c2w[:, :3, 3] *= sscale


class MipNerf360Dataset(torch.utils.data.Dataset):
    SPLITS = ["train", "test"]

    def __init__(
        self, 
        data_loader: MipNerf360DataLoader, 
        split: str, 
        add_interpolated_samples: bool = False, 
        add_z_interpolated_samples: bool = False, 
        interpol_scale: float = 1.0,
        random_viewdirs: bool = False,
        random_viewdirs_max_cone_angle_deg: float = 22.5,
        random_viewdirs_weighted: bool = True,
        n_random_viewdirs: int = 4,
        device: str = "cpu"
    ):
        super().__init__()
        assert split in self.SPLITS, "%s" % split
        
        assert not (add_z_interpolated_samples and add_interpolated_samples), "either interpolate only z or all (both not allowed)"

        self.training = split == "train"
        self.add_interpolated_samples = add_interpolated_samples
        self.add_z_interpolated_samples = add_z_interpolated_samples
        self.interpol_scale = interpol_scale
        
        self.random_viewdirs = random_viewdirs
        self.random_viewdirs_max_cone_angle_rad = (random_viewdirs_max_cone_angle_deg / 180.0) * torch.pi
        self.random_viewdirs_weighted = random_viewdirs_weighted
        self.n_random_viewdirs = n_random_viewdirs

        indices = data_loader.split_indices[split]
        self.image_paths = [data_loader.image_paths[i] for i in indices]
        self.images = torch.from_numpy(data_loader.images[indices]).to(torch.uint8).to(device)
        self.c2w = torch.from_numpy(data_loader.c2w[indices]).to(torch.float32).to(device)
        self.K = torch.tensor(data_loader.K).to(torch.float32).to(device)
        self.height, self.width = self.images.shape[1:3]
        self.OPENGL_CAMERA = data_loader.OPENGL_CAMERA

        print(f"Loaded {self.images.shape[0]} images for split '{split}'")

    def __len__(self):
        return len(self.images)

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        pixels, origins, viewdirs = data["rgb"], data["origins"], data["viewdirs"]

        if self.training:
            color_bkgd = torch.rand_like(pixels)
        else:
            color_bkgd = torch.ones_like(pixels)

        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "origins": origins,  # [n_rays, 3] or [h, w, 3]
            "viewdirs": viewdirs,  # [n_rays, 3] or [h, w, 3]
            "color_bkgd": color_bkgd,  # [n_rays, 3] or [h, w, 3]
            **{k: v for k, v in data.items() if k not in ["rgb", "origins", "viewdirs"]},
        }

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""

        if self.training:
            num_rays = self.num_rays

            image_id = torch.randint(0, len(self.images), size=(num_rays,), device=self.images.device)
            x = torch.randint(0, self.width, size=(num_rays,), device=self.images.device)
            y = torch.randint(0, self.height, size=(num_rays,), device=self.images.device)
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.width, device=self.images.device),
                torch.arange(self.height, device=self.images.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        # generate rays
        rgb = self.images[image_id, y, x] / 255.0  # (num_rays, 3)
        c2w = self.c2w[image_id]  # (num_rays, 3, 4)
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - self.K[0, 2] + 0.5) / self.K[0, 0],
                    (y - self.K[1, 2] + 0.5) / self.K[1, 1] * (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        # [num_rays, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(directions, dim=-1, keepdims=True)

        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))
            rgb = torch.reshape(rgb, (num_rays, 3))
        else:
            origins = torch.reshape(origins, (self.height, self.width, 3))
            viewdirs = torch.reshape(viewdirs, (self.height, self.width, 3))
            rgb = torch.reshape(rgb, (self.height, self.width, 3))

        output = {
            "rgb": rgb,  # [h, w, 3] or [num_rays, 3]
            "origins": origins,  # [h, w, 3] or [num_rays, 3]
            "viewdirs": viewdirs,  # [h, w, 3] or [num_rays, 3]
        }

        if self.training and (self.add_interpolated_samples or self.add_z_interpolated_samples or self.random_viewdirs):
            n_random_dims = 1 if self.add_z_interpolated_samples else (3 if self.add_interpolated_samples else 0)
            coefficients = generate_random_interpol_coefficients(num_rays, n_random_dims, self.images.device) # [num_rays, 8, 3] OR [num_rays, 2, 1] OR [num_rays, 1, 1]
            
            if (n_random_dims > 0):
                x_offsets = torch.zeros_like(coefficients[:, :, 0]) if self.add_z_interpolated_samples else (coefficients[:, :, 0] - 0.5) * self.interpol_scale
                y_offsets = torch.zeros_like(coefficients[:, :, 0]) if self.add_z_interpolated_samples else (coefficients[:, :, 1] - 0.5) * self.interpol_scale

                interpol_camera_dirs = F.pad(
                    torch.stack(
                        [
                            (x.unsqueeze(-1) - self.K[0, 2] + 0.5 + x_offsets) / self.K[0, 0],
                            (y.unsqueeze(-1) - self.K[1, 2] + 0.5 + y_offsets) / self.K[1, 1] * (-1.0 if self.OPENGL_CAMERA else 1.0),
                        ],
                        dim=-1,
                    ),
                    (0, 1),
                    value=(-1.0 if self.OPENGL_CAMERA else 1.0),
                )  # [num_rays, 8, 3] OR [num_rays, 2, 3]
                
                directions = (interpol_camera_dirs[:, :, None, :] * c2w[:, None, :3, :3]).sum(dim=-1)
                interpol_viewdirs = directions / torch.linalg.norm(directions, dim=-1, keepdims=True)
                output["interpol_viewdirs"] = interpol_viewdirs # [num_rays, 8, 3] OR [num_rays, 2, 3]
                output["interpol_coeffs"] = coefficients # [num_rays, 8, 3] OR [num_rays, 2, 1]
            else:
                output["interpol_viewdirs"] = viewdirs.clone()[:, None, :] # [num_rays, 1, 3]
                output["interpol_coeffs"] = coefficients # [num_rays, 1, 1]
            
        if self.training and self.random_viewdirs:
            random_z_rotations = random_directions_angle_circular(viewdirs, n_directions=self.n_random_viewdirs, max_cone_angle=self.random_viewdirs_max_cone_angle_rad)
            random_viewdir_rotations = rotate_z_to_vector(random_z_rotations, viewdirs)
            
            if (self.random_viewdirs_weighted):
                scaling_factor = 0.5
                scaling_fun = lambda x : x**2
                
                diff_angle = torch.arccos(torch.clamp((viewdirs.unsqueeze(-2) * random_viewdir_rotations).sum(dim=-1, keepdim=True), max=1.0))
                weights_inv_lin = diff_angle / self.random_viewdirs_max_cone_angle_rad
                
                weights_unnormalized = 1.0 - scaling_factor * scaling_fun(weights_inv_lin) # [0, 1] -> [1, 0.5]
                weights = weights_unnormalized / weights_unnormalized.sum(-2, keepdims=True)
            else:
                weights = torch.full_like(random_viewdir_rotations[...,:1], 1.0 / self.n_random_viewdirs) # uniformly weighted
            
            output["random_viewdirs"] = random_viewdir_rotations.transpose(0, 1) # [n_random_viewdirs, n_rays, 3]
            output["random_viewdir_weights"] = weights.transpose(0, 1) # [n_random_viewdirs, n_rays, 1]
            
        return output

    def get_transforms_dict(self):
        return {
            "width": self.width,
            "height": self.height,
            "intrinsics": self.K.cpu().tolist(),
            "is_open_gl": self.OPENGL_CAMERA,
            "frames": [
                {"transform_matrix": c2w, "image_path": image_path}
                for c2w, image_path in zip(self.c2w.cpu().tolist(), self.image_paths)
            ],
        }
