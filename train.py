import argparse
import json
import math
import os
import pathlib
import random
import time
import tqdm

import imageio
import numpy as np
import torch
import torch.nn.functional as F

from src.datasets import MipNerf360DataLoader, MipNerf360Dataset
from src.model import InterNerf
from src.occupancy_grid import OccupancyGrid
from src.renderer import calculate_n_rendered_samples, train_batch, render_all


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
args = parser.parse_args()

print(args)

device = "cuda:0"
set_random_seed(42)

factor = 4 if args.scene in MipNerf360DataLoader.OUTDOOR_SCENES else 2
occ_grid_n_lvls = (5 if args.scene in MipNerf360DataLoader.OUTDOOR_SCENES else 4) + args.unbounded # Increase aabb by 1 level if unbounded

data_loader = MipNerf360DataLoader(args.scene, args.data_root, factor)
train_dataset = MipNerf360Dataset(
    data_loader, "train", device=device,
    add_interpolated_samples=args.interpol_train, 
    add_z_interpolated_samples=args.zinterpol_train,
    interpol_scale=args.interpol_scale_factor, 
    random_viewdirs=args.viewdep_train,
    random_viewdirs_max_cone_angle_deg=args.viewdep_max_cone_angle, 
    random_viewdirs_weighted=args.weight_viewdep_samples,
    n_random_viewdirs=args.n_viewdep_samples
)
test_dataset = MipNerf360Dataset(data_loader, "test", device=device, add_interpolated_samples=False, random_viewdirs=False, )

BATCH_SIZE_DOWNSCALE_FACTOR = 1 # 4 means 4x smaller batches but 4x the number of iterations

update_occupancy_grid_every_n = 16 * BATCH_SIZE_DOWNSCALE_FACTOR
occupancy_grid_warmup_steps = 16 * update_occupancy_grid_every_n

max_steps = args.max_steps * BATCH_SIZE_DOWNSCALE_FACTOR
init_n_rays = 512 // BATCH_SIZE_DOWNSCALE_FACTOR
target_sample_batch_size = (1 << 18) // BATCH_SIZE_DOWNSCALE_FACTOR
train_dataset.update_num_rays(init_n_rays)

aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
occupancy_grid = OccupancyGrid(aabb, occ_grid_n_lvls, warmup=args.checkpoint is None).to(device)
radiance_field = InterNerf(
    occupancy_grid.aabbs[1] if args.unbounded else occupancy_grid.aabbs[-1], # Contract with second AABB as base if unbounded
    log2_hashmap_size=args.log2_hashmap_size,
    
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
    
    n_levels=args.gridenc_n_levels,
    max_resolution=args.gridenc_max_resolution,
    n_features=args.gridenc_n_features,
    
    separate_density_network = args.separate_density_network,
    separate_density_encoding = args.separate_density_encoding,
    density_network_n_neurons = args.density_network_n_neurons,
    density_network_n_layers = args.density_network_n_layers,
    density_network_tcnn = args.density_network_tcnn
).to(device)

occupancy_grid.mark_invisible_cells(train_dataset.K.unsqueeze(0), train_dataset.c2w, train_dataset.width, train_dataset.height)

render_step_size = math.sqrt(3) / 1024
cone_angle = 1 / 256.0

# Assuming that all cameras are inside max AABB
near_plane = 0.2
far_plane = torch.linalg.norm(occupancy_grid.aabbs[-1].view(2, 3)[0] - occupancy_grid.aabbs[-1].view(2, 3)[1]).cpu().item() # max aabb diagonal length

alpha_thre = 1e-5
early_stop_eps = 1e-4

if (args.checkpoint is not None):
    checkpoint = torch.load(args.checkpoint)
    radiance_field.load_state_dict(checkpoint["radiance_field_state_dict"])
    occupancy_grid.load_state_dict(checkpoint["occupancy_grid_state_dict"])


# hp = radiance_field.pos_encoding.encoding_config
# resolution_per_level = [math.ceil(hp["base_resolution"] * 2**(math.log2(hp["per_level_scale"])*i) - 1) + 1 for i in range(hp["n_levels"])]
# entries_per_level = [min(2**hp["log2_hashmap_size"], math.ceil((res**3) / 8.0) * 8) for res in resolution_per_level]
# params_per_level = [e * hp["n_features_per_level"] for e in entries_per_level]
# offsets = [0] + list(accumulate(params_per_level))
# weights = torch.tensor([params_per_level[-1] / n_params for n_params in params_per_level], dtype=radiance_field.pos_encoding.dtype).to(device)
# idcs = torch.empty(offsets[-1], dtype=torch.int64).to(device)
# for i in range(hp["n_levels"]):
#     idcs[offsets[i]:offsets[i+1]] = i

#[min(2**hp["log2_hashmap_size"], (math.ceil(hp["base_resolution"] * 2**(math.log2(hp["per_level_scale"])*i) - 1) + 1)**3) for i in range(hp["n_levels"])]


grad_scaler = torch.cuda.amp.GradScaler(2**10)
# params = [
#         {'params': radiance_field.pos_encoding.parameters(), 'weight_decay': 1e-6},
#         {'params': radiance_field.mlp_head.parameters(), 'weight_decay':1e-6}
#     ]

# if args.viewdep_train:
#     params.append({'params': radiance_field.mlp_first_head.parameters(), 'weight_decay':1e-6})
# if not args.no_mlp_base:
#     params.append({'params': radiance_field.mlp_base.parameters(), 'weight_decay':1e-6})
# if args.separate_density_network:
#     params.append({'params': radiance_field.density_network.parameters(), 'weight_decay':1e-6})
    
    
optimizer = torch.optim.Adam(radiance_field.parameters(), lr=1e-2, eps=1e-15, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.ChainedScheduler(
    [
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=max_steps // 10),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-4)
    ]
)

train_print_every_n_steps = 1000
eval_every_n_steps = 20000

interpol_train = args.interpol_train or args.zinterpol_train or args.viewdep_train

tic = time.time()
for step in range(max_steps + 1):
    radiance_field.train()
    occupancy_grid.train()

    i = torch.randint(0, len(train_dataset), (1,)).item()
    data = train_dataset[i]

    ref_pixels = data["pixels"]
    origins = data["origins"]
    viewdirs = data["viewdirs"]
    render_bkgd = data["color_bkgd"]

    interpol_viewdirs = data["interpol_viewdirs"] if interpol_train else None
    interpol_coeffs = data["interpol_coeffs"] if interpol_train else None
    random_viewdirs = data["random_viewdirs"] if args.viewdep_train else None
    random_viewdir_weights = data["random_viewdir_weights"] if args.viewdep_train else None
        
    def occ_eval_fn(x):
        density = radiance_field.query_density(x)[0]
        return density * render_step_size

    # update occupancy grid
    occupancy_grid.update_every_n_steps(
        step=step,
        n=update_occupancy_grid_every_n,
        occ_eval_fn=occ_eval_fn,
        occ_thre=1e-2,
        warmup_steps=occupancy_grid_warmup_steps
    )
    
    num_rays = len(ref_pixels)
    if (step % update_occupancy_grid_every_n) == 0 and args.pre_calculate_num_rays:
        # This can prevent OOM exceptions when already pretty on the limit
        # Sometimes after updating the occupancy grid, the actual number of rays is far off from the estimated number, that
        #  was based on the previous iteration. This code pre-calculates the exact number of rays and truncate if
        #  the sample count exceeds the target sample batch size
        
        n_render_samples = calculate_n_rendered_samples(
            radiance_field,
            occupancy_grid,
            origins,
            viewdirs,
            near_plane,
            far_plane,
            render_step_size,
            early_stop_eps,
            alpha_thre,
            cone_angle
        )
        
        target_num_rays = int(num_rays * (target_sample_batch_size / float(max(n_render_samples, 1))))
        num_rays = num_rays if target_num_rays > num_rays else target_num_rays
        
        ref_pixels = ref_pixels[:num_rays]
        origins = origins[:num_rays]
        viewdirs = viewdirs[:num_rays]
        render_bkgd = render_bkgd[:num_rays]

        interpol_viewdirs = interpol_viewdirs[:num_rays] if interpol_viewdirs is not None else None
        interpol_coeffs = interpol_coeffs[:num_rays] if interpol_coeffs is not None else None
        random_viewdirs = random_viewdirs[:,:num_rays] if random_viewdirs is not None else None
        random_viewdir_weights = random_viewdir_weights[:,:num_rays] if random_viewdir_weights is not None else None
    

    additional_losses = {}
    if (args.viewdep_train and args.latent_regularization_factor > 0.0):
        additional_losses["latent_regularization"] = torch.tensor(0.0).to(origins.device)
    if (args.density_regularization_factor > 0.0):
        additional_losses["density_regularization"] = torch.tensor(0.0).to(origins.device)
    if (interpol_train and args.interpol_loss_factor > 0.0):
        additional_losses["sample_vs_interpol"] = torch.tensor(0.0).to(origins.device)
    if (args.distortion_loss_factor > 0.0):
        additional_losses["distortion_loss"] = torch.tensor(0.0).to(origins.device)
    
    results = train_batch(
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
        use_interpol_training=interpol_train,
        interpol_viewdirs=interpol_viewdirs,
        interpol_coeffs=interpol_coeffs,
        interpol_scale=args.interpol_scale_factor,
        additional_losses=additional_losses,
        viewdep_train=args.viewdep_train,
        random_viewdirs=random_viewdirs,
        random_viewdir_weights=random_viewdir_weights,
        gradient_scaling=args.gradient_scaling,
        normalize_interpolated_latents=not args.disable_interpol_latent_normalization
    )
    
    loss = torch.tensor(0.0).to(origins.device)
    n_render_samples = results[0][2]
    
    for vi, result in enumerate(results):
        colors, _, _ = result
        loss += ((colors - ref_pixels)**2 * (random_viewdir_weights[vi] if random_viewdir_weights is not None else 1.0)).mean() * 0.5

    num_rays = len(ref_pixels)
    num_rays = min(int(num_rays * target_sample_batch_size / float(max(n_render_samples, 1))), int(target_sample_batch_size / 4.0)) # min 4 samples per ray (slow otherwise)
    train_dataset.update_num_rays(num_rays)
    # print(step, num_rays, n_render_samples)

    # tmp = args.distortion_loss_factor * additional_losses["distortion_loss"].item()
    # print(f"\r{loss.item() * 1000:3.4f} {tmp * 1000:3.4f}", end="")
    if ("latent_regularization" in additional_losses):
        loss += args.latent_regularization_factor * additional_losses["latent_regularization"]
    if ("density_regularization" in additional_losses):
        loss += args.density_regularization_factor * additional_losses["density_regularization"]
    if ("sample_vs_interpol" in additional_losses):
        loss += args.interpol_loss_factor * additional_losses["sample_vs_interpol"]
    if ("distortion_loss" in additional_losses):
        loss += args.distortion_loss_factor * additional_losses["distortion_loss"]

    optimizer.zero_grad()
    grad_scaler.scale(loss).backward()
    
    # assert radiance_field.pos_encoding.params.grad.isfinite().all()
    # assert not radiance_field.use_mlp_base or radiance_field.mlp_base.params.grad.isfinite().all()
    # assert radiance_field.mlp_head.params.grad.isfinite().all()
    # assert all([p.grad.isfinite().all() for p in radiance_field.density_network.parameters()])
    # assert not args.viewdep_train or radiance_field.mlp_first_head.params.grad.isfinite().all()

    optimizer.step()
    scheduler.step()
    
    # assert radiance_field.pos_encoding.params.isfinite().all()
    # assert not radiance_field.use_mlp_base or radiance_field.mlp_base.params.isfinite().all()
    # assert radiance_field.mlp_head.params.isfinite().all()
    # assert all([p.isfinite().all() for p in radiance_field.density_network.parameters()])
    # assert not args.viewdep_train or radiance_field.mlp_first_head.params.isfinite().all()

    if step % train_print_every_n_steps == 0:
        elapsed_time = time.time() - tic
        loss = F.mse_loss(colors, ref_pixels) # Only taking last init_viewdir but good enough approximate mse loss
        psnr = -10.0 * torch.log(loss) / np.log(10.0)
        print(
            f"elapsed_time={elapsed_time:.2f}s | step={step} | "
            f"loss={loss:.5f} | psnr={psnr:.2f} | "
            f"n_rendering_samples={n_render_samples:d} | num_rays={len(ref_pixels):d}",
            flush = True
        )

    if step > 0 and step % max_steps == 0:
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

    if step > 0 and (step % eval_every_n_steps) == 0:
        # evaluation
        radiance_field.eval()
        occupancy_grid.eval()

        psnrs = []
        n_samples_ppx = []
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
                )
                mse = F.mse_loss(rgb, pixels)
                psnr = -10.0 * torch.log(mse) / np.log(10.0)
                psnrs.append(psnr.item())
                n_samples_ppx.append(n_render_samples / rgb.shape[:-1].numel())

                # output_folder = f"output/{args.scene}/{args.experiment}/{step}"
                # os.makedirs(output_folder, exist_ok=True)
                # imageio.imwrite(
                #     f"{output_folder}/{i}_rgb_test.png",
                #     (rgb.cpu().numpy() * 255).astype(np.uint8),
                # )
                # imageio.imwrite(
                #     f"{output_folder}/{i}_original_test.png",
                #     (pixels.cpu().numpy() * 255).astype(np.uint8),
                # )

        psnr_avg = sum(psnrs) / len(psnrs)
        print(
            f"Eval - step: {step} | psnr_avg: {np.mean(psnrs):.3f}, samples ppx: {np.mean(n_samples_ppx):.2f} | {np.min(n_samples_ppx):.2f} | {np.max(n_samples_ppx):.2f}",
            flush=True
        )
