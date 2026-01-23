import torch
import torch.nn as nn
import numpy as np

class HashEmbedder2D(nn.Module):
    """
    Multi-Resolution Hash Encoding for 2D coordinates.
    
    Based on 'Instant Neural Graphics Primitives with a Multiresolution Hash Encoding' 
    (Müller et al., 2022).
    
    This implementation uses pure PyTorch operations for portability, 
    vectorized for efficiency.
    """
    def __init__(
        self,
        num_levels: int = 16,
        base_resolution: int = 16,
        features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        max_resolution: int = 2048,
        bounding_box: tuple = ((-1.0, -1.0), (1.0, 1.0)), # (min_vals, max_vals)
        version: int = 0  # Version selector (0-6)
    ):
        super().__init__()
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hashmap_size = 2**log2_hashmap_size
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.bounding_box = bounding_box
        self.version = version
        
        # Calculate resolution for each level
        # growth_factor = exp( (ln(max) - ln(base)) / (num_levels - 1) )
        b = np.exp((np.log(max_resolution) - np.log(base_resolution)) / (num_levels - 1))
        self.resolutions = [int(np.floor(base_resolution * b**i)) for i in range(num_levels)]
        
        # Hash tables parameters
        # Version 1+: Use dense grids for coarse levels instead of hash collisions
        self.embeddings = nn.ModuleList()
        self.is_dense = []  # Track which levels use dense indexing
        
        for res in self.resolutions:
            n_grid_points = res**2
            
            # Version 1+: Use dense grid if it fits in hash table
            if version >= 1 and n_grid_points <= self.hashmap_size:
                # Dense grid (no hash collisions)
                table_size = n_grid_points
                is_dense = True
            else:
                # Sparse hash grid
                table_size = min(n_grid_points, self.hashmap_size)
                is_dense = False
                
            embedding = nn.Embedding(table_size, features_per_level)
            nn.init.uniform_(embedding.weight, -1e-4, 1e-4) 
            
            self.embeddings.append(embedding)
            self.is_dense.append(is_dense)
            
        # Primes for hashing (from the paper)
        self.primes = [1, 2654435761, 805459861]
        
        # Versions 2+: Learnable smoothing or progressive training
        if version >= 2:
            hidden_dim = features_per_level * 4
            if version in [2, 4, 6]:
                # Shared smoother (excellent in V2, V6)
                self.smoother = nn.Sequential(
                    nn.Linear(features_per_level, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, features_per_level)
                )
                self.smooth_weight = nn.Parameter(torch.tensor(0.1))
            else:
                # Version 3, 5: Level-specific smoothers (Higher capacity for adaptation)
                self.smoothers = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(features_per_level, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, features_per_level)
                    ) for _ in range(num_levels)
                ])
                self.smooth_weights = nn.Parameter(torch.ones(num_levels) * 0.1)
        
        # Version 4+: Progressive training support
        # Supports float active_levels for smooth alpha-blending activation
        self.active_levels = float(num_levels)
        
        # Version 6: Multi-Plane Hybrid (dense plane + hash grid)
        self.num_levels = num_levels
        if version >= 6:
            # Strictly respect max_resolution as requested.
            # This ensures the hybrid plane doesn't exceed the user's frequency budget.
            res = max_resolution
            embedding = nn.Embedding(res**2, features_per_level)
            nn.init.uniform_(embedding.weight, -1e-4, 1e-4)
            self.embeddings.append(embedding)
            self.is_dense.append(True)
            self.resolutions.append(res)
            
        self.total_levels = len(self.embeddings)
        
        # Version 4+: Progressive training support
        # Supports float active_levels for smooth alpha-blending activation
        self.active_levels = float(self.total_levels)
    
    def set_active_levels(self, n: float):
        """Set number of active levels for progressive training (V4+).
        
        Args:
            n: Number of levels to activate (from coarsest). Supports float for smooth blending.
        """
        self.active_levels = min(max(1.0, float(n)), float(self.total_levels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, 2) coordinates.
        Returns:
            (Batch, total_levels * features_per_level) concatenated features.
        """
        # Normalize x to [0, 1]
        min_v = torch.tensor(self.bounding_box[0], device=x.device)
        max_v = torch.tensor(self.bounding_box[1], device=x.device)
        x_norm = torch.clamp((x - min_v) / (max_v - min_v), 0.0, 1.0)
        
        features = []
        
        # Progressive parameters
        full_levels = int(self.active_levels)
        fractional = self.active_levels - full_levels
        n_to_compute = int(np.ceil(self.active_levels))
        
        for i in range(self.total_levels):
            if i < n_to_compute:
                res = self.resolutions[i]
                x_scaled = x_norm * res
                
                # Bilinear Grid Lookup
                x0 = torch.floor(x_scaled).long()
                x1 = torch.clamp(x0 + 1, 0, res - 1)
                x0 = torch.clamp(x0, 0, res - 1)
                
                w = x_scaled - torch.floor(x_scaled)
                wx, wy = w[..., 0:1], w[..., 1:2]
                
                def get_idx(coords):
                    if self.is_dense[i]:
                        return coords[:, 1] * res + coords[:, 0]
                    h = (coords[:, 0] * self.primes[1]) ^ (coords[:, 1] * self.primes[2])
                    return h % self.embeddings[i].num_embeddings

                c00 = get_idx(torch.stack([x0[:, 0], x0[:, 1]], dim=-1))
                c10 = get_idx(torch.stack([x1[:, 0], x0[:, 1]], dim=-1))
                c01 = get_idx(torch.stack([x0[:, 0], x1[:, 1]], dim=-1))
                c11 = get_idx(torch.stack([x1[:, 0], x1[:, 1]], dim=-1))
                
                val = torch.lerp(
                    torch.lerp(self.embeddings[i](c00), self.embeddings[i](c10), wx),
                    torch.lerp(self.embeddings[i](c01), self.embeddings[i](c11), wx),
                    wy
                )
                
                # Progressive alpha blending
                if i == full_levels and fractional > 1e-5:
                    val = val * fractional
                
                # Apply smoothing (Versions 2+)
                if self.version >= 2:
                    # Note: V6 uses the shared smoother for its grid levels.
                    # The Hybrid Plane (level 16) bypasses the smoother for raw global fidelity.
                    if i < self.num_levels:
                        if hasattr(self, 'smoother'):
                            smooth_delta = self.smoother(val)
                            sw = self.smooth_weight
                        else:
                            smooth_delta = self.smoothers[i](val)
                            sw = self.smooth_weights[i]
                        val = val + torch.sigmoid(sw) * smooth_delta
            else:
                # Level is inactive (Progressive)
                val = torch.zeros(x.shape[0], self.features_per_level, device=x.device)
            
            features.append(val)
            
        return torch.cat(features, dim=-1)
