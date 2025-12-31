import torch
import torch.nn as nn
import numpy as np
from typing import Union, List

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
        # Defines learnable or fixed scaling factors for activations.
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
            
            # Initialize frequencies B
            # sigma can be a float or a list of floats (per dimension)
            n_param = len(sigma) if isinstance(sigma, list) else dim_in
            
            # Random normal frequencies
            # Shape (dim_in, m_freqs) to map (N, dim_in) -> (N, m_freqs)
            # Original code: np.random.normal(0.0, sigma, (m_freqs, n_param)).T -> (n_param, m_freqs)
            # Then matmul(x, B). x is (N, dim_in). B should be (dim_in, m_freqs).
            # So (n_param, m_freqs) is correct if n_param == dim_in.
            
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
        # Clone input to preserve concatenation
        xx = x.clone()

        # Fourier Mapping
        if self.fourier_features:
            # x: (N, dim_in), B: (dim_in, m_freqs)
            # proj: (N, m_freqs)
            proj = torch.matmul(x, self.B)
            cosx = torch.cos(proj)
            sinx = torch.sin(proj)
            x = torch.cat((cosx, sinx, xx), dim=1)
            
            # First layer with beta0 scaling
            x = self.activation(self.beta0 * self.first(x))
        else:
            x = self.activation(self.beta0 * self.first(x))

        # Residual Blocks
        for i in range(self.num_resnet_blocks):
            # First layer of block adds to residual but has residual connection?
            # Original: z = activation(beta... * layer(x))
            # Inner loop: z = activation(beta... * layer(z))
            # x = z + x
            
            # Block 0
            z = self.activation(self.beta[i][0] * self.resblocks[i][0](x))

            # Subsequent layers in block
            for j in range(1, self.num_layers_per_block):
                z = self.activation(self.beta[i][j] * self.resblocks[i][j](z))
            
            # Residual connection
            x = z + x

        out = self.last(x)
        return out
