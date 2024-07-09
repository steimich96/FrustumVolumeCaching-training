import argparse
import json
import math
import os
import pathlib
import random
import time
import tqdm

import imageio
from lpips import LPIPS
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.cm as cm

from src.datasets import MipNerf360DataLoader, MipNerf360Dataset
from src.model import InterNerf
from src.occupancy_grid import OccupancyGrid
from src.renderer import render_all

from src.util.yanf_utils import circular_samples_around_directions
from src.util.loss_utils import ssim
from src.util.flip_loss import LDRFLIPLoss


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default=str(pathlib.Path.cwd() / "../data/360_v2"), help="the root dir of the dataset")
parser.add_argument("--scene", type=str, default="bicycle", choices=MipNerf360DataLoader.SCENES, help="which scene to use")

parser.add_argument("--interpol_train", action="store_true", help="Use interpolation during training")
parser.add_argument("--zinterpol_train", action="store_true", help="Use interpolation of only z dimension during training")
parser.add_argument("--interpol_loss_factor", type=float, default=0.0, help="Loss of interpolated sample vs normal sample during training")
parser.add_argument("--interpol_scale_factor", type=float, default=1.0, help="Size of the interpolation area (1.0 means same pixel and stepsize scale)")
parser.add_argument("--disable_interpol_latent_normalization", action="store_true", help="Disables latent normalization durint interpolation")

parser.add_argument("--viewdep_train", action="store_true", help="Use limited view-independent first mlp head part during training")
parser.add_argument("--n_viewdep_samples", type=int, default=4, help="")
parser.add_argument("--weight_viewdep_samples", action="store_true", help="")
parser.add_argument("--viewdep_max_cone_angle", type=float, default=22.5, help="in degrees")
parser.add_argument("--latent_regularization_factor", type=float, default=1e-4, help="")

parser.add_argument("--separate_density_encoding", action="store_true", help="")
parser.add_argument("--separate_density_network", action="store_true", help="")
parser.add_argument("--density_network_n_neurons", type=int, default=32, help="")
parser.add_argument("--density_network_n_layers", type=int, default=1, help="")
parser.add_argument("--density_network_tcnn", action="store_true", help="")

parser.add_argument("--unbounded", action="store_true", help="Use unbounded space for the radiance field (occupancy grid still bounded)")
parser.add_argument("--max_steps", type=int, default=80000, help="Loss of interpolated sample vs normal sample during training")
parser.add_argument("--gradient_scaling", action="store_true", help="Use gradient scaling from Floater-No-More")
parser.add_argument("--pre_calculate_num_rays", action="store_true", help="Leads to reacalculation of number of actual rays after updating occupancy grid")
parser.add_argument("--distortion_loss_factor", type=float, default=0.0, help="")

parser.add_argument("--log2_hashmap_size", type=int, default=21, help="")
parser.add_argument("--gridenc_n_levels", type=int, default=8, help="")
parser.add_argument("--gridenc_n_features", type=int, default=4, help="")
parser.add_argument("--gridenc_max_resolution", type=int, default=4096, help="")

parser.add_argument("--no_mlp_base", action="store_true", help="")
parser.add_argument("--mlp_base_n_layers", type=int, default=1, help="")
parser.add_argument("--mlp_base_n_neurons", type=int, default=128, help="")
parser.add_argument("--n_mlp_base_outputs", type=int, default=16, help="")
parser.add_argument("--sh_small_degree", type=int, default=3, help="")
parser.add_argument("--sh_large_degree", type=int, default=4, help="")

parser.add_argument("--mlp_first_head_n_neurons", type=int, default=128, help="")
parser.add_argument("--mlp_first_head_n_layers", type=int, default=2, help="")
parser.add_argument("--n_latents", type=int, default=16, help="")
parser.add_argument("--not_density_in_latents", action="store_true", help="")
parser.add_argument("--density_regularization_factor", type=float, default=0.0, help="")

parser.add_argument("--use_freq_encoding", action="store_true", help="")
parser.add_argument("--freq_encoding_degree", type=int, default=4, help="")

parser.add_argument("--mlp_head_n_neurons", type=int, default=128, help="")
parser.add_argument("--mlp_head_n_layers", type=int, default=2, help="")

parser.add_argument("--experiment", type=str, default="base", help="experiment name")
parser.add_argument("--save_model", action="store_true")
parser.add_argument("--save_dir", type=str, default=None)
parser.add_argument("--export_model", action="store_true")
parser.add_argument("--export_dir", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default=None)

parser.add_argument("--eval_viewdep", action="store_true", help="")
parser.add_argument("--eval_viewdep_max_cone_angle", type=float, default=25.0, help="in degrees")
parser.add_argument("--eval_viewdep_n_samples", type=int, default=4, help="")
args = parser.parse_args()

device = "cuda:0"
set_random_seed(42)

print(args)

factor = 4 if args.scene in MipNerf360DataLoader.OUTDOOR_SCENES else 2
occ_grid_n_lvls = (5 if args.scene in MipNerf360DataLoader.OUTDOOR_SCENES else 4) + args.unbounded # Increase aabb by 1 level if unbounded

data_loader = MipNerf360DataLoader(args.scene, args.data_root, factor)
test_dataset = MipNerf360Dataset(data_loader, "test", add_interpolated_samples=False, random_viewdirs=False, device=device)


render_step_size = math.sqrt(3) / 1024
cone_angle = 1 / 256.0

near_plane = 0.2
far_plane = 1.0e10

alpha_thre = 1e-5
early_stop_eps = 1e-4

log2_hashmap_size = args.log2_hashmap_size
mlp_base_n_neurons = args.mlp_base_n_neurons
n_mlp_base_outputs = args.n_mlp_base_outputs

mlp_first_head_n_neurons = args.mlp_first_head_n_neurons
mlp_first_head_n_layers = args.mlp_first_head_n_layers
n_latents = args.n_latents

mlp_head_n_neurons = args.mlp_head_n_neurons
mlp_head_n_layers = args.mlp_head_n_layers

aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
occupancy_grid = OccupancyGrid(aabb, occ_grid_n_lvls, warmup=args.checkpoint is None).to(device)
radiance_field = InterNerf(
    occupancy_grid.aabbs[1] if args.unbounded else occupancy_grid.aabbs[-1], # Contract with second AABB as base if unbounded
    
    log2_hashmap_size=args.log2_hashmap_size,
    n_levels=args.gridenc_n_levels,
    max_resolution=args.gridenc_max_resolution,
    n_features=args.gridenc_n_features,
    
    use_mlp_base=not args.no_mlp_base,
    mlp_base_n_layers=args.mlp_base_n_layers,
    mlp_base_n_neurons=args.mlp_base_n_neurons,
    n_mlp_base_outputs=args.n_mlp_base_outputs,
    
    mlp_first_head_n_neurons=args.mlp_first_head_n_neurons,
    mlp_first_head_n_layers=args.mlp_first_head_n_layers,
    n_latents = args.n_latents,
    
    mlp_head_n_neurons=args.mlp_head_n_neurons,
    mlp_head_n_layers=args.mlp_head_n_layers,
    
    unbounded=args.unbounded,
    split_mlp_head=args.viewdep_train,
    first_latent_is_density=not args.not_density_in_latents,
    sh_small_degree=args.sh_small_degree,
    sh_large_degree=args.sh_large_degree,
    
    use_freq_encoding=args.use_freq_encoding,
    freq_encoding_degree=args.freq_encoding_degree,
    
    separate_density_network = args.separate_density_network,
    separate_density_encoding = args.separate_density_encoding,
    density_network_n_neurons = args.density_network_n_neurons,
    density_network_n_layers = args.density_network_n_layers,
    density_network_tcnn = args.density_network_tcnn
).to(device)

if (args.checkpoint is not None):
    checkpoint = torch.load(args.checkpoint)
    radiance_field.load_state_dict(checkpoint["radiance_field_state_dict"])
    occupancy_grid.load_state_dict(checkpoint["occupancy_grid_state_dict"])

tic = time.time()
model_name = f"{args.experiment}"

if args.export_model:
    export_config_dict = radiance_field.get_config()     
    export_config_dict["scene"] = {
        "near": near_plane,
        "far": far_plane,
        "stepsize": render_step_size,
        "alpha_thre": alpha_thre,
        "cone_angle": cone_angle,
        "aabb": occupancy_grid.aabbs[-1].cpu().tolist(),
        "grid_nlvl": occupancy_grid.n_levels,
        "grid_resolution": occupancy_grid.RESOLUTION,
        "is_open_gl": test_dataset.OPENGL_CAMERA,
        "contraction": { "aabb": radiance_field.aabb.cpu().tolist() } if args.unbounded else None
    }

    export_dir = (pathlib.Path.cwd() if args.export_dir is None else pathlib.Path(args.export_dir)) / model_name
    export_dir.mkdir(exist_ok=True)

    json_object = json.dumps(export_config_dict, indent=4)
    with open(f"{export_dir}/config.json", "w") as f:
        f.write(json_object)

    test_frames_dict = test_dataset.get_transforms_dict()
    json_object = json.dumps(test_frames_dict, indent=4)
    with open(f"{export_dir}/test_transforms.json", "w") as f:
        f.write(json_object)
        
    params_dict = radiance_field.get_params_dict()
    params_dict["occupancy_grid"] = occupancy_grid.occs_binary
    
    for key, params in params_dict.items():
        params.cpu().numpy().tofile(f"{export_dir}/{key}.dat")

if args.save_model:
    save_dir = (pathlib.Path.cwd() if args.save_dir is None else pathlib.Path(args.save_dir)) / model_name
    save_dir.mkdir(exist_ok=True)

    model_save_path = str(save_dir / f"model.ckpt")
    torch.save(
        {
            "radiance_field_state_dict": radiance_field.state_dict(),
            "occupancy_grid_state_dict": occupancy_grid.state_dict(),
        },
        model_save_path,
    )


magma_cm = cm.get_cmap('magma')
psnr_fn = lambda x, y: (-10.0 * torch.log(F.mse_loss(x, y)) / np.log(10.0))
ssim_fn = lambda x, y: ssim(x, y).mean()

lpips_net = LPIPS(net="vgg").to(device)
lpips_fn = lambda x, y: lpips_net(x, y, normalize=False).mean()
flip = LDRFLIPLoss()
flip_fn = lambda x, y: flip(x, y).mean()

# evaluation
radiance_field.eval()
occupancy_grid.eval()

MAX_N_SAMPLES_PER_RAY = 1024
CHUNK_SIZE = 2**14

error_fns = {"PSNR": psnr_fn, "SSIM": ssim_fn, "FLIP": flip_fn, "LPIPS": lpips_fn}

n_samples_ppx = []
errors = {k: [] for k in error_fns.keys()}

EVAL_DEGREES = [5, 10, 25, 45, 90]
errors_thetas_vs_original = [{k: [] for k in error_fns.keys()} for _ in EVAL_DEGREES]
errors_thetas_vs_render = [{k: [] for k in error_fns.keys()} for _ in EVAL_DEGREES]

with torch.no_grad():
    for i in tqdm.tqdm(range(len(test_dataset))):
        data = test_dataset[i]
        render_bkgd = data["color_bkgd"]
        origins = data["origins"]
        viewdirs = data["viewdirs"]
        pixels = data["pixels"]
        
        # rendering
        rgb, acc, n_render_samples = render_all(
            radiance_field,
            occupancy_grid,
            origins,
            viewdirs,
            near_plane,
            far_plane,
            render_step_size,
            early_stop_eps,
            alpha_thre,
            cone_angle,
            render_bkgd,
            chunk_size=CHUNK_SIZE,
            max_n_samples_per_ray=MAX_N_SAMPLES_PER_RAY
        )
        
        pixels_fmt = pixels[None, ...].permute(0, 3, 1, 2)
        rgb_fmt = rgb[None, ...].permute(0, 3, 1, 2)
        for error_key in error_fns.keys():
            errors[error_key].append(error_fns[error_key](rgb_fmt, pixels_fmt).item())
        
        n_samples_ppx.append(n_render_samples / rgb.shape[:-1].numel())
        
        output_folder = f"output/{args.scene}/{args.experiment}/eval"
        os.makedirs(output_folder, exist_ok=True)
        
        if args.viewdep_train and args.eval_viewdep:
            for theta_i, theta in enumerate(EVAL_DEGREES):
                if theta > args.eval_viewdep_max_cone_angle:
                    break
                
                theta_rad = torch.tensor(theta, dtype=torch.float) / 180.0 * torch.pi
                init_viewdirs = circular_samples_around_directions(viewdirs, args.eval_viewdep_n_samples, theta_rad, torch.zeros_like(viewdirs[..., 2:]))
                init_viewdirs = init_viewdirs.movedim(2, 0) # [n_samples, height, width, 3]
                
                viewdep_results = render_all(
                    radiance_field,
                    occupancy_grid,
                    origins,
                    viewdirs,
                    near_plane,
                    far_plane,
                    render_step_size,
                    early_stop_eps,
                    alpha_thre,
                    cone_angle,
                    render_bkgd,
                    random_viewdirs=init_viewdirs,
                    chunk_size=CHUNK_SIZE,
                    max_n_samples_per_ray=MAX_N_SAMPLES_PER_RAY
                )
                
                # theta_output_folder = f"{output_folder}/{theta}"
                # os.makedirs(theta_output_folder, exist_ok=True)
                
                theta_img_0 = viewdep_results[0][0]
                # imageio.imwrite(
                #     f"{output_folder}/{i}_{theta}_rgb_test.png",
                #     (theta_img_0.cpu().numpy() * 255).astype(np.uint8),
                # )
                # imageio.imwrite(
                #     f"{output_folder}/{i}_{theta}_mse_error_test.png",
                #     (magma_cm(np.clip(torch.mean((theta_img_0 - pixels)**2, axis=-1).cpu().numpy(), 0.0, 1.0)) * 255).astype(np.uint8),
                # )
                
                # for error_key in error_fns.keys():
                #     print("vs orig", theta, error_key, error_fns[error_key](theta_img_0[None, ...].permute(0, 3, 1, 2), pixels_fmt).item())
                # for error_key in error_fns.keys():
                #     print("vs render", theta, error_key, error_fns[error_key](theta_img_0[None, ...].permute(0, 3, 1, 2), rgb_fmt).item())
                
                def viewdep_mean_error(error_fn, ref_img):
                    return np.mean([error_fn(viewdep_results[sample_idx][0][None, ...].permute(0, 3, 1, 2), ref_img).item() for sample_idx in range(args.eval_viewdep_n_samples)])
                
                for error_key in error_fns.keys():
                    errors_thetas_vs_original[theta_i][error_key].append(viewdep_mean_error(error_fns[error_key], pixels_fmt))
                    errors_thetas_vs_render[theta_i][error_key].append(viewdep_mean_error(error_fns[error_key], rgb_fmt))


        imageio.imwrite(
            f"{output_folder}/{i}_rgb_test.png",
            (rgb.cpu().numpy() * 255).astype(np.uint8),
        )
        # imageio.imwrite(
        #     f"{output_folder}/{i}_mse_error_test.png",
        #     (magma_cm(np.clip(torch.mean((rgb - pixels)**2, axis=-1).cpu().numpy(), 0.0, 1.0)) * 255).astype(np.uint8)
        # )
        # for error_key in error_fns.keys():
        #     print(0, error_key, error_fns[error_key](rgb[None, ...].permute(0, 3, 1, 2), pixels_fmt).item())
        # imageio.imwrite(
        #     f"{output_folder}/{i}_original_test.png",
        #     (pixels.cpu().numpy() * 255).astype(np.uint8),
        # )

if args.viewdep_train and args.eval_viewdep:
    for theta_i, theta in enumerate(EVAL_DEGREES):
        if theta > args.eval_viewdep_max_cone_angle:
            break
        viewdep_out_str = f"Eval Degree {theta}"
        viewdep_out_str += " | vs. original - "
        viewdep_out_str += ", ".join([f"{error_key}: {np.mean(errors_thetas_vs_original[theta_i][error_key]):.3f}" for error_key in error_fns.keys()])
        viewdep_out_str += " | vs. render - "
        viewdep_out_str += ", ".join([f"{error_key}: {np.mean(errors_thetas_vs_render[theta_i][error_key]):.3f}" for error_key in error_fns.keys()])
        print(viewdep_out_str)

ovr_out_str = "Eval | "
ovr_out_str += " | ".join([f"{error_key}: {np.mean(errors[error_key]):.3f}" for error_key in error_fns.keys()])
ovr_out_str += f" | samples ppx: {np.mean(n_samples_ppx):.2f} | {np.min(n_samples_ppx):.2f} | {np.max(n_samples_ppx):.2f}"
print(ovr_out_str)
