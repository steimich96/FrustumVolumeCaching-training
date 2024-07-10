"""
Copyright (C) 2024, Michael Steiner, Graz University of Technology.
This code is licensed under the MIT license.
"""

from typing import Callable, List, Union

import numpy as np
import tinycudann as tcnn
import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(torch.clamp(x, max=10))

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=10))
    
trunc_exp = _TruncExp.apply

def contract_to_unisphere(
    x: torch.Tensor,
    aabb: torch.Tensor,
    ord: Union[float, int] = 2,
):
    """MipNerf-360 style contraction: x if norm(x) < 1 else warp(x)"""
    aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
    x = (x - aabb_min) / (aabb_max - aabb_min)
    x = x * 2 - 1  # aabb is at [-1, 1]

    norm = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
    mask = norm.squeeze(-1) > 1

    x[mask] = (2 - 1 / norm[mask]) * (x[mask] / norm[mask]) # [-inf, inf] -> [-2, 2]
    x = x / 4 + 0.5  # [-2, 2] -> [0, 1]
    return x

def frequency_encoding(opt, input, L):  # [B,...,N]
    shape = input.shape
    freq = (2 ** torch.arange(L, dtype=torch.float32, device=opt.device) * np.pi)  # [L]
    spectrum = input[..., None] * freq  # [B,...,N,L]
    sin, cos = spectrum.sin(), spectrum.cos()  # [B,...,N,L]
    input_enc = torch.stack([sin, cos], dim=-2)  # [B,...,N,2,L]
    input_enc = input_enc.view(*shape[:-1], -1)  # [B,...,2NL]
    return input_enc

def contract_to_aabb(
    x: torch.Tensor,
    aabb: torch.Tensor,
    ord: Union[float, int] = 2,
):
    aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
    x = (x - aabb_min) / (aabb_max - aabb_min)
    return x

def normalize_viewdir(viewdir: torch.Tensor) -> torch.Tensor:
    return (viewdir + 1.0) / 2.0

class InterNerf(torch.nn.Module):
    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        unbounded: bool = False,
        
        log2_hashmap_size: int = 19,
        max_resolution = 4096,
        n_levels = 8,
        n_features = 4,
        
        use_mlp_base: bool = True,
        mlp_base_n_neurons: int = 64,
        mlp_base_n_layers: int = 1,
        n_mlp_base_outputs: int = 16,  
            
        split_mlp_head: bool = False,
        first_latent_is_density: bool = True,
        sh_small_degree: int = 3,
        mlp_first_head_n_neurons: int = 64,
        mlp_first_head_n_layers: int = 2,
        n_latents: int = 16,
        
        separate_density_network: bool = False,
        separate_density_encoding: bool = False,
        density_network_n_neurons: int = 32,
        density_network_n_layers: int = 1,
        density_network_tcnn: bool = False,
        
        use_freq_encoding: bool = False,
        freq_encoding_degree: int = 4,
        
        sh_large_degree: int = 4,
        mlp_head_n_neurons: int = 64,
        mlp_head_n_layers: int = 2
    ) -> None:
        super().__init__()
        
        assert split_mlp_head or n_latents == n_mlp_base_outputs, "If not split, then MLP Base output = Latents"
        assert first_latent_is_density or split_mlp_head, "If first latent is not density -> this requires a split MLP"

        self.aabb = aabb
        self.base_resolution = 16
        self.max_resolution = max_resolution
        self.n_levels = n_levels
        self.n_features = n_features
        self.log2_hashmap_size = log2_hashmap_size
        self.per_level_scale = np.exp(
            (np.log(self.max_resolution) - np.log(self.base_resolution)) / (self.n_levels - 1)
        ).tolist()


        self.n_latents = n_latents
        self.mlp_base_n_neurons = mlp_base_n_neurons
        self.mlp_head_n_neurons = mlp_head_n_neurons
        
        self.use_mlp_base = use_mlp_base
        self.n_mlp_base_outputs = n_mlp_base_outputs
        
        self.mlp_first_head_n_neurons = mlp_first_head_n_neurons
        self.mlp_first_head_n_layers = mlp_first_head_n_layers
        self.split_mlp_head = split_mlp_head
        self.first_latent_is_density = first_latent_is_density

        self.unbounded = unbounded
        self.scene_contraction = contract_to_unisphere if unbounded else contract_to_aabb
        
        self.pos_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": self.n_levels,
                "n_features_per_level": self.n_features,
                "log2_hashmap_size": self.log2_hashmap_size,
                "base_resolution": self.base_resolution,
                "per_level_scale": self.per_level_scale,
            },
        )

        self.separate_density_network = separate_density_network
        self.separate_density_encoding = separate_density_encoding
        self.density_network_n_neurons = density_network_n_neurons
        self.density_network_n_layers = density_network_n_layers
        self.density_network_tcnn = density_network_tcnn
        
        if separate_density_network:
            if separate_density_encoding:
                self.density_n_levels = 16
                self.density_encoding = tcnn.Encoding(
                    n_input_dims=3,
                    encoding_config={
                        "otype": "HashGrid",
                        "n_levels": self.density_n_levels,
                        "n_features_per_level": 1,
                        "log2_hashmap_size": self.log2_hashmap_size,
                        "base_resolution": self.base_resolution,
                        "per_level_scale": np.exp((np.log(self.max_resolution) - np.log(self.base_resolution)) / (self.density_n_levels - 1)).tolist()
                    },
                )
                
            self.density_network_input_width = self.density_encoding.n_output_dims if self.separate_density_encoding else self.pos_encoding.n_output_dims
            
            if density_network_tcnn:
                self.density_network = tcnn.Network(
                    n_input_dims=self.density_network_input_width,
                    n_output_dims=1,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": self.density_network_n_neurons,
                        "n_hidden_layers": self.density_network_n_layers,
                    },
                )
            else:
                density_network_layers = [torch.nn.Linear(in_features=self.density_network_input_width, bias=False, out_features=1 if self.density_network_n_layers == 0 else self.density_network_n_neurons)]
                
                for i in range(self.density_network_n_layers):
                    density_network_layers.append(torch.nn.ReLU())
                    n_output_features = 1 if i == (self.density_network_n_layers - 1) else self.density_network_n_neurons
                    density_network_layers.append(torch.nn.Linear(in_features=self.density_network_n_neurons, bias=False, out_features = n_output_features))
                
                self.density_network = torch.nn.Sequential(*tuple(density_network_layers))
            
        self.mlp_base = None
        if self.use_mlp_base:
            self.mlp_base = tcnn.Network(
                n_input_dims=self.pos_encoding.n_output_dims,
                n_output_dims=self.n_mlp_base_outputs,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": mlp_base_n_neurons,
                    "n_hidden_layers": mlp_base_n_layers,
                },
            )
        self.base_output_width = self.n_mlp_base_outputs if self.use_mlp_base else self.pos_encoding.n_output_dims

        self.sh_large_degree = sh_large_degree
        self.sh_large_coeff_offset = 1 if self.sh_large_degree in (3, 5) else 0 # {3: 9, 4: 16, 5: 25}
        self.viewdir_encoding_large = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "Composite",
                "nested": [
                    {
                        "n_dims_to_encode": 3,
                        "otype": "SphericalHarmonics",
                        "degree": self.sh_large_degree,
                    }
                ],
            },
        )
        
        n_mlp_head_inputs = (self.viewdir_encoding_large.n_output_dims - self.sh_large_coeff_offset) + self.n_latents
        
        self.use_freq_encoding = use_freq_encoding
        self.freq_encoding_degree = freq_encoding_degree
        self.freq_encoding_n_outputs = 6 * freq_encoding_degree
        
        if (self.use_freq_encoding):
            n_mlp_head_inputs += self.freq_encoding_n_outputs
        
        self.sh_small_degree = sh_small_degree
        self.sh_small_coeff_offset = 1 if self.sh_small_degree in (3, 5) else 0 # {3: 9, 4: 16, 5: 25}
        if (self.split_mlp_head):            
            self.viewdir_encoding_small = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "Composite",
                    "nested": [
                        {
                            "n_dims_to_encode": 3,
                            "otype": "SphericalHarmonics",
                            "degree": self.sh_small_degree,
                        }
                    ],
                },
            )
        
            self.mlp_first_head = tcnn.Network(
                n_input_dims=(self.viewdir_encoding_small.n_output_dims - self.sh_small_coeff_offset) + self.base_output_width,
                n_output_dims=self.n_latents,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.mlp_first_head_n_neurons,
                    "n_hidden_layers": mlp_first_head_n_layers,
                },
            )
            n_mlp_head_inputs += (self.viewdir_encoding_small.n_output_dims - self.sh_small_coeff_offset)# append original viewdir diff

        self.mlp_head = tcnn.Network(
            n_input_dims=n_mlp_head_inputs,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": mlp_head_n_neurons,
                "n_hidden_layers": mlp_head_n_layers,
            },
        )
        
    def get_params_dict(self):
        params = {}
        
        params["pos_enc"] = self.pos_encoding.params.to(self.pos_encoding.dtype).detach()
        params["mlp_head"] = self.mlp_head.params.to(self.mlp_head.dtype).detach()
        
        if self.separate_density_network:
            params["density_network"] = self.density_network.params.to(self.density_network.dtype).detach()
            if self.separate_density_encoding:
                params["density_enc"] = self.density_encoding.params.to(self.density_encoding.dtype).detach()
        
        if self.use_mlp_base:
            params["mlp_base"] = self.mlp_base.params.to(self.mlp_base.dtype).detach()
            
        if (self.split_mlp_head):
            params["mlp_first_head"] = self.mlp_first_head.params.to(self.mlp_first_head.dtype).detach()
        
        return params
        
    
    def get_config(self):
        export_config_dict = {
            "mlp_head": {
                "n_input": self.mlp_head.n_input_dims,
                "n_output": self.mlp_head.n_output_dims,
                "encoding_offset": self.sh_large_coeff_offset,
                "encoding": self.viewdir_encoding_large.encoding_config,
                "network": self.mlp_head.network_config,
                "freq_encoding_degree": self.freq_encoding_degree if self.use_freq_encoding else 0
            }
        }
        
        if self.separate_density_network:
            export_config_dict["density_network"] = {
                "n_input": self.density_network.n_input_dims,
                "n_output": self.density_network.n_output_dims,
                "network": self.density_network.network_config,
            }
            if self.separate_density_encoding:
                export_config_dict["density_network"]["encoding"] = self.density_encoding.encoding_config
        
        if self.use_mlp_base:
            export_config_dict["mlp_base"] = {
                "n_input": self.mlp_base.n_input_dims,
                "n_output": self.mlp_base.n_output_dims,
                "encoding": self.pos_encoding.encoding_config,
                "network": self.mlp_base.network_config,
            }
        
        if self.split_mlp_head:
            export_config_dict["mlp_first_head"] = {
                "n_input": self.mlp_first_head.n_input_dims,
                "n_output": self.mlp_first_head.n_output_dims,
                "encoding_offset": self.sh_small_coeff_offset,
                "first_latent_is_density": self.first_latent_is_density,
                "encoding": self.viewdir_encoding_small.encoding_config,
                "network": self.mlp_first_head.network_config,
            }
        
        return export_config_dict
        
    def _forward_base(self, pos, only_density: bool = False):
        x = self.scene_contraction(pos, self.aabb, ord=float("inf")) if self.scene_contraction is not None else pos
        x_enc = self.pos_encoding(x.view(-1, 3))
        
        if self.separate_density_network:
            x_density_enc = self.density_encoding(x.view(-1, 3)) if self.separate_density_encoding else x_enc
            density_before_activation = self.density_network(x_density_enc if self.density_network_tcnn else x_density_enc.to(x))
            density = trunc_exp(density_before_activation.to(x))

        if self.use_mlp_base and not (self.separate_density_network and only_density):
            mlp_base_output = self.mlp_base(x_enc)
            if not self.separate_density_network:
                density_before_activation = mlp_base_output[..., 0]
                density = trunc_exp(density_before_activation.to(x))
        
        base_output = (mlp_base_output if self.use_mlp_base else x_enc)
        return density.view(x.shape[:-1]), base_output.view(list(x.shape[:-1]) + [base_output.shape[-1]]), density_before_activation.view(x.shape[:-1])
    
    def _forward_first_head(self, init_viewdir, mlp_base_output):
        if not self.split_mlp_head:
            return mlp_base_output
        
        enc_init_viewdir = self.viewdir_encoding_small(normalize_viewdir(init_viewdir).reshape(-1, 3))[...,self.sh_small_coeff_offset:]
        mlp_first_head_input = torch.concat([mlp_base_output.reshape(-1, self.base_output_width), enc_init_viewdir], dim=-1)
        latents = self.mlp_first_head(mlp_first_head_input).reshape(list(mlp_base_output.shape[:-1]) + [self.n_latents])
        
        return torch.concat([mlp_base_output[..., :1], latents[..., :-1]], dim=-1) if self.first_latent_is_density else latents
        
    def _forward_head(self, init_viewdir, viewdir, latents, pos = None):
        enc_viewdir = self.viewdir_encoding_large(normalize_viewdir(viewdir).reshape(-1, 3))
        mlp_head_input = torch.concat([latents.reshape(-1, self.n_latents), enc_viewdir[..., self.sh_large_coeff_offset:]], dim=-1)
        
        if self.split_mlp_head:
            enc_viewdir_small = enc_viewdir[..., self.sh_small_coeff_offset:self.viewdir_encoding_small.n_output_dims]
            enc_init_viewdir = self.viewdir_encoding_small(normalize_viewdir(init_viewdir).reshape(-1, 3))[...,self.sh_small_coeff_offset:]
            enc_viewdir_diff = enc_viewdir_small - enc_init_viewdir
            mlp_head_input = torch.concat([mlp_head_input, enc_viewdir_diff], dim=-1)
            
        if self.use_freq_encoding:
            assert pos is not None
            pos_contracted = self.scene_contraction(pos, self.aabb, ord=float("inf")) if self.scene_contraction is not None else pos
            freq_enc_pos = frequency_encoding(mlp_head_input, pos_contracted, self.freq_encoding_degree)
            mlp_head_input = torch.concat([mlp_head_input, freq_enc_pos], dim=-1)
        
        mlp_head_output = self.mlp_head(mlp_head_input).reshape(list(latents.shape[:-1]) + [3])

        rgb = torch.sigmoid(mlp_head_output)
        return rgb
    
    def query_latents(self, pos: torch.Tensor, init_viewdir: torch.Tensor):
        sigmas, mlp_base_output, sigmas_before_activation =  self._forward_base(pos)
        latents = self._forward_first_head(init_viewdir, mlp_base_output)
        return sigmas, latents, sigmas_before_activation

    def forward(self, pos: torch.Tensor, viewdir: torch.Tensor, init_viewdir: torch.Tensor):
        density, latents, density_before_activation = self.query_latents(pos, init_viewdir)
        rgb = self._forward_head(init_viewdir, viewdir, latents, pos)
        return rgb, density, density_before_activation
    
    def query_density(self, pos: torch.Tensor):
        density, _, density_before_activation =  self._forward_base(pos, only_density=True)
        return density, density_before_activation
