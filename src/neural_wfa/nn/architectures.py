import torch
import torch.nn as nn
import numpy as np
from typing import Union, List
from neural_wfa.nn.encoding import HashEmbedder2D

class MLP(nn.Module):
    """
    DenseNet-style MLP with Fourier Features and Residual Blocks.
    
    Used for representing Neural Fields (Coordinate-based networks).
    """
    def __init__(
        self,
        dim_in: int = 2,
        dim_out: int = 1,
        num_resnet_blocks: int = 3,
        num_layers_per_block: int = 2,
        dim_hidden: int = 50,
        activation: nn.Module = nn.GELU(),
        fourier_features: bool = False,
        m_freqs: int = 100,
        sigma: Union[float, List[float]] = 10.0,
        tune_beta: bool = False,
    ):
        super().__init__()
        self.num_resnet_blocks = num_resnet_blocks
        self.num_layers_per_block = num_layers_per_block
        self.fourier_features = fourier_features
        self.activation = activation
        self.tune_beta = tune_beta
        self.sigma = sigma
        
        # --- Beta (Activation Scaling) ---
        if tune_beta:
            self.beta0 = nn.Parameter(torch.ones(1, 1))
            self.beta = nn.Parameter(
                torch.ones(self.num_resnet_blocks, self.num_layers_per_block)
            )
        else:
            self.register_buffer('beta0', torch.ones(1, 1))
            self.register_buffer('beta', torch.ones(self.num_resnet_blocks, self.num_layers_per_block))

        # --- Fourier Features ---
        input_dim = dim_in
        if fourier_features:
            input_dim = 2 * m_freqs + dim_in
            n_param = len(sigma) if isinstance(sigma, list) else dim_in
            B_init = np.random.normal(0.0, sigma, (m_freqs, n_param)).T
            self.register_buffer('B', torch.from_numpy(B_init).float())

        # --- Network Layers ---
        self.first = nn.Linear(input_dim, dim_hidden)
        
        self.resblocks = nn.ModuleList()
        for _ in range(num_resnet_blocks):
            block = nn.ModuleList()
            for _ in range(num_layers_per_block):
                block.append(nn.Linear(dim_hidden, dim_hidden))
            self.resblocks.append(block)
            
        self.last = nn.Linear(dim_hidden, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = x.clone()
        if self.fourier_features:
            proj = torch.matmul(x, self.B)
            cosx = torch.cos(proj)
            sinx = torch.sin(proj)
            x = torch.cat((cosx, sinx, xx), dim=1)
            x = self.activation(self.beta0 * self.first(x))
        else:
            x = self.activation(self.beta0 * self.first(x))

        for i in range(self.num_resnet_blocks):
            z = self.activation(self.beta[i][0] * self.resblocks[i][0](x))
            for j in range(1, self.num_layers_per_block):
                z = self.activation(self.beta[i][j] * self.resblocks[i][j](z))
            x = z + x
        return self.last(x)

class TemporalMLP(nn.Module):
    """
    Temporal MLP variant using FiLM (Feature-wise Linear Modulation) for time information.
    """
    def __init__(
        self,
        dim_in=3,
        dim_out=1,
        num_resnet_blocks=3,
        num_layers_per_block=2,
        dim_hidden=50,
        activation=nn.GELU(),
        fourier_features=False,
        m_freqs=100,
        sigma=10.0,
    ):
        super().__init__()
        self.num_resnet_blocks = num_resnet_blocks
        self.num_layers_per_block = num_layers_per_block
        self.activation = activation
        self.dim_hidden = dim_hidden
        
        # FiLM network processes time (dim 0 of input)
        self.film = MLP(
            dim_in=1,
            dim_out=num_resnet_blocks,
            num_resnet_blocks=num_resnet_blocks,
            num_layers_per_block=num_layers_per_block,
            dim_hidden=dim_hidden,
            activation=activation,
            fourier_features=True,
            m_freqs=m_freqs,
            sigma=sigma if isinstance(sigma, float) else sigma[0]
        )

        sigma_spatial = sigma[1:] if isinstance(sigma, list) else sigma
        self.register_buffer('beta0', torch.ones(1, 1))

        spatial_dim_in = 2
        input_dim = spatial_dim_in
        if fourier_features:
            input_dim = 2 * m_freqs + spatial_dim_in
            n_param = len(sigma_spatial) if isinstance(sigma_spatial, list) else spatial_dim_in
            B_init = np.random.normal(0.0, sigma_spatial, (m_freqs, n_param)).T
            self.register_buffer('B', torch.from_numpy(B_init).float())

        self.first = nn.Linear(input_dim, dim_hidden)
        self.resblocks = nn.ModuleList([
            nn.ModuleList([nn.Linear(dim_hidden, dim_hidden) for _ in range(num_layers_per_block)])
            for _ in range(num_resnet_blocks)
        ])
        self.last = nn.Linear(dim_hidden, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx_temporal = x[:, 0:1]
        xx_spatial = x[:, 1:]

        # Gamma modulation parameters from FiLM
        gamma = self.film(xx_temporal).reshape(x.shape[0], self.num_resnet_blocks, 1)

        if hasattr(self, 'B'):
            proj = torch.matmul(xx_spatial, self.B)
            x = torch.cat((torch.cos(proj), torch.sin(proj), xx_spatial), dim=1)
        else:
            x = xx_spatial

        x = self.activation(self.beta0 * self.first(x))

        for i in range(self.num_resnet_blocks):
            z = self.activation(gamma[:, i] + self.resblocks[i][0](x))
            for j in range(1, self.num_layers_per_block):
                z = self.activation(gamma[:, i] + self.resblocks[i][j](z))
            x = z + x
        return self.last(x)

class AdditiveTemporalMLP(nn.Module):
    """
    Temporal MLP variant where output is f(spatial) + g(temporal, spatial)
    """
    def __init__(self, **kwargs):
        super().__init__()
        # Standard MLP for full spatio-temporal input
        self.mlp_full = MLP(**kwargs)
        
        # MLP for spatial-only part
        spatial_kwargs = kwargs.copy()
        spatial_kwargs['dim_in'] = 2
        if isinstance(kwargs.get('sigma'), list):
            spatial_kwargs['sigma'] = kwargs['sigma'][1:]
            
        self.mlp_spatial = MLP(**spatial_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx_spatial = x[:, 1:]
        xx_temporal = x[:, 0:1]
        return xx_temporal * self.mlp_full(x) + self.mlp_spatial(xx_spatial)

class HashMLP(nn.Module):
    """
    MLP that uses Multi-Resolution Hash Encoding for inputs.
    
    This architecture is usually much smaller (shallower/narrower) than standard MLPs
    because the Hash Grid handles the high-frequency learning.
    
    Supports 2D and 3D inputs via unified HashEmbedding factory.
    """
    def __init__(
        self,
        dim_in: int = 2,  # 2D or 3D supported
        dim_out: int = 1,
        dim_hidden: int = 64,
        num_layers: int = 2,
        activation: nn.Module = nn.GELU(),
        # Hash Grid Params (spatial)
        num_levels: int = 16,
        base_resolution: int = 16,
        log2_hashmap_size: int = 19,
        features_per_level: int = 2,
        max_resolution: int = 2048,
        version: int = 0,  # Version selector (0-6)
        # 3D-specific params
        mode: str = 'auto',  # 'auto', 'full', or 'hybrid' for 3D
        temporal_base_resolution: int = None,  # For full 3D
        temporal_max_resolution: int = None,   # For full 3D
        temporal_m_freqs: int = 64,            # For hybrid 3D
        temporal_sigma: float = 10.0,          # For hybrid 3D
        # Optional shared encoder
        encoder: nn.Module = None
    ):
        super().__init__()
        self.dim_in = dim_in
        
        # 1. Encoding
        if encoder is not None:
            self.encoder = encoder
        else:
            from neural_wfa.nn.encoding import HashEmbedding
            
            # Build kwargs based on mode and dim_in
            common_kwargs = dict(
                num_levels=num_levels,
                base_resolution=base_resolution,
                features_per_level=features_per_level,
                log2_hashmap_size=log2_hashmap_size,
                max_resolution=max_resolution,
                version=version,
            )
            
            if dim_in == 3:
                if mode == 'hybrid':
                    # Hybrid mode: spatial params prefixed
                    common_kwargs = dict(
                        spatial_num_levels=num_levels,
                        spatial_base_resolution=base_resolution,
                        spatial_features_per_level=features_per_level,
                        spatial_log2_hashmap_size=log2_hashmap_size,
                        spatial_max_resolution=max_resolution,
                        spatial_version=version,
                        temporal_m_freqs=temporal_m_freqs,
                        temporal_sigma=temporal_sigma,
                    )
                else:
                    # Full 3D mode
                    common_kwargs['temporal_base_resolution'] = temporal_base_resolution
                    common_kwargs['temporal_max_resolution'] = temporal_max_resolution
            
            self.encoder = HashEmbedding(dim_in=dim_in, mode=mode, **common_kwargs)
        
        # Get output dimension from encoder
        if hasattr(self.encoder, 'output_dim'):
            input_dim = self.encoder.output_dim
        elif hasattr(self.encoder, 'total_levels'):
            input_dim = self.encoder.total_levels * features_per_level
        else:
            input_dim = num_levels * features_per_level
        
        # 2. MLP Head (Small)
        layers = []
        layers.append(nn.Linear(input_dim, dim_hidden))
        layers.append(activation)
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(dim_hidden, dim_hidden))
            layers.append(activation)
            
        layers.append(nn.Linear(dim_hidden, dim_out))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.net(features)

