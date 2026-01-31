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


class HashEmbedder3D(nn.Module):
    """
    Multi-Resolution Hash Encoding for 3D coordinates (t, y, x).
    
    Extends HashEmbedder2D with trilinear interpolation (8 corners).
    Supports different resolution ranges for temporal vs spatial dimensions.
    """
    def __init__(
        self,
        num_levels: int = 16,
        base_resolution: int = 16,
        features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        max_resolution: int = 2048,
        # Separate temporal parameters (if None, use spatial params)
        temporal_base_resolution: int = None,
        temporal_max_resolution: int = None,
        bounding_box: tuple = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),  # (min, max) for (t, y, x)
        version: int = 0
    ):
        super().__init__()
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hashmap_size = 2**log2_hashmap_size
        self.bounding_box = bounding_box
        self.version = version
        
        # Spatial resolutions
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        
        # Temporal resolutions (default to spatial if not specified)
        self.temporal_base_resolution = temporal_base_resolution or base_resolution
        self.temporal_max_resolution = temporal_max_resolution or max_resolution
        
        # Calculate resolutions for each level
        b_spatial = np.exp((np.log(max_resolution) - np.log(base_resolution)) / max(num_levels - 1, 1))
        b_temporal = np.exp((np.log(self.temporal_max_resolution) - np.log(self.temporal_base_resolution)) / max(num_levels - 1, 1))
        
        self.resolutions_spatial = [int(np.floor(base_resolution * b_spatial**i)) for i in range(num_levels)]
        self.resolutions_temporal = [int(np.floor(self.temporal_base_resolution * b_temporal**i)) for i in range(num_levels)]
        
        # Hash tables
        self.embeddings = nn.ModuleList()
        self.is_dense = []
        
        for i in range(num_levels):
            res_t = self.resolutions_temporal[i]
            res_s = self.resolutions_spatial[i]
            n_grid_points = res_t * res_s * res_s  # t * y * x
            
            # Dense if fits in hash table
            if version >= 1 and n_grid_points <= self.hashmap_size:
                table_size = n_grid_points
                is_dense = True
            else:
                table_size = min(n_grid_points, self.hashmap_size)
                is_dense = False
            
            embedding = nn.Embedding(table_size, features_per_level)
            nn.init.uniform_(embedding.weight, -1e-4, 1e-4)
            self.embeddings.append(embedding)
            self.is_dense.append(is_dense)
        
        # Primes for 3D hashing
        self.primes = [1, 2654435761, 805459861, 3674653429]
        
        # Progressive training support
        self.active_levels = float(num_levels)
    
    def set_active_levels(self, n: float):
        self.active_levels = min(max(1.0, float(n)), float(self.num_levels))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, 3) coordinates: [t, y, x]
        Returns:
            (Batch, num_levels * features_per_level) features
        """
        # Normalize to [0, 1]
        min_v = torch.tensor(self.bounding_box[0], device=x.device)
        max_v = torch.tensor(self.bounding_box[1], device=x.device)
        x_norm = torch.clamp((x - min_v) / (max_v - min_v), 0.0, 1.0)
        
        features = []
        
        full_levels = int(self.active_levels)
        fractional = self.active_levels - full_levels
        n_to_compute = int(np.ceil(self.active_levels))
        
        for i in range(self.num_levels):
            if i < n_to_compute:
                res_t = self.resolutions_temporal[i]
                res_s = self.resolutions_spatial[i]
                
                # Scale coordinates
                t_scaled = x_norm[:, 0:1] * res_t
                yx_scaled = x_norm[:, 1:3] * res_s
                x_scaled = torch.cat([t_scaled, yx_scaled], dim=-1)
                
                # Trilinear interpolation (8 corners)
                x0 = torch.floor(x_scaled).long()
                x1 = x0 + 1
                
                # Clamp per dimension
                max_vals = torch.tensor([res_t-1, res_s-1, res_s-1], device=x.device)
                x0 = torch.clamp(x0, min=torch.zeros(3, device=x.device, dtype=torch.long), max=max_vals)
                x1 = torch.clamp(x1, min=torch.zeros(3, device=x.device, dtype=torch.long), max=max_vals)
                
                # Interpolation weights
                w = x_scaled - torch.floor(x_scaled)
                wt, wy, wx = w[:, 0:1], w[:, 1:2], w[:, 2:3]
                
                def get_idx(coords):
                    t_idx, y_idx, x_idx = coords[:, 0], coords[:, 1], coords[:, 2]
                    if self.is_dense[i]:
                        return (t_idx * res_s * res_s + y_idx * res_s + x_idx).long()
                    # XOR hash with 3 primes
                    h = (t_idx.long() * self.primes[1]) ^ (y_idx.long() * self.primes[2]) ^ (x_idx.long() * self.primes[3])
                    return h % self.embeddings[i].num_embeddings
                
                # 8 corners (c_tyx where t,y,x are 0 or 1)
                c000 = get_idx(torch.stack([x0[:, 0], x0[:, 1], x0[:, 2]], dim=-1))
                c001 = get_idx(torch.stack([x0[:, 0], x0[:, 1], x1[:, 2]], dim=-1))
                c010 = get_idx(torch.stack([x0[:, 0], x1[:, 1], x0[:, 2]], dim=-1))
                c011 = get_idx(torch.stack([x0[:, 0], x1[:, 1], x1[:, 2]], dim=-1))
                c100 = get_idx(torch.stack([x1[:, 0], x0[:, 1], x0[:, 2]], dim=-1))
                c101 = get_idx(torch.stack([x1[:, 0], x0[:, 1], x1[:, 2]], dim=-1))
                c110 = get_idx(torch.stack([x1[:, 0], x1[:, 1], x0[:, 2]], dim=-1))
                c111 = get_idx(torch.stack([x1[:, 0], x1[:, 1], x1[:, 2]], dim=-1))
                
                # Trilinear interpolation
                val = (
                    (1 - wt) * (1 - wy) * (1 - wx) * self.embeddings[i](c000) +
                    (1 - wt) * (1 - wy) * wx * self.embeddings[i](c001) +
                    (1 - wt) * wy * (1 - wx) * self.embeddings[i](c010) +
                    (1 - wt) * wy * wx * self.embeddings[i](c011) +
                    wt * (1 - wy) * (1 - wx) * self.embeddings[i](c100) +
                    wt * (1 - wy) * wx * self.embeddings[i](c101) +
                    wt * wy * (1 - wx) * self.embeddings[i](c110) +
                    wt * wy * wx * self.embeddings[i](c111)
                )
                
                # Progressive blending
                if i == full_levels and fractional > 1e-5:
                    val = val * fractional
            else:
                val = torch.zeros(x.shape[0], self.features_per_level, device=x.device)
            
            features.append(val)
        
        return torch.cat(features, dim=-1)


class HybridHashEmbedder3D(nn.Module):
    """
    Hybrid 3D encoding: 2D spatial hash (y, x) + 1D temporal Fourier (t).
    
    Allows separate parameter control for spatial and temporal dimensions.
    """
    def __init__(
        self,
        # Spatial Hash parameters
        spatial_num_levels: int = 16,
        spatial_base_resolution: int = 16,
        spatial_max_resolution: int = 2048,
        spatial_features_per_level: int = 2,
        spatial_log2_hashmap_size: int = 19,
        spatial_version: int = 1,
        # Temporal Fourier parameters
        temporal_m_freqs: int = 64,
        temporal_sigma: float = 10.0,
        # Bounding box (only spatial part used for hash)
        bounding_box: tuple = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
    ):
        super().__init__()
        
        # Spatial encoder (2D Hash)
        spatial_bbox = ((bounding_box[0][1], bounding_box[0][2]), 
                        (bounding_box[1][1], bounding_box[1][2]))
        self.spatial_encoder = HashEmbedder2D(
            num_levels=spatial_num_levels,
            base_resolution=spatial_base_resolution,
            max_resolution=spatial_max_resolution,
            features_per_level=spatial_features_per_level,
            log2_hashmap_size=spatial_log2_hashmap_size,
            bounding_box=spatial_bbox,
            version=spatial_version
        )
        
        # Temporal encoder (Fourier features)
        self.temporal_m_freqs = temporal_m_freqs
        self.temporal_sigma = temporal_sigma
        self.register_buffer('B_temporal', torch.randn(1, temporal_m_freqs) * temporal_sigma)
        
        # Output dimensions
        self.spatial_dim = spatial_num_levels * spatial_features_per_level
        if spatial_version >= 6:
            self.spatial_dim += spatial_features_per_level  # Extra hybrid plane level
        self.temporal_dim = 2 * temporal_m_freqs  # cos + sin
        self.output_dim = self.spatial_dim + self.temporal_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, 3) coordinates: [t, y, x]
        Returns:
            (Batch, spatial_dim + temporal_dim) concatenated features
        """
        t = x[:, 0:1]  # (Batch, 1)
        yx = x[:, 1:3]  # (Batch, 2)
        
        # Spatial features (Hash encoding)
        spatial_features = self.spatial_encoder(yx)
        
        # Temporal features (Fourier encoding)
        t_proj = t @ self.B_temporal  # (Batch, m_freqs)
        temporal_features = torch.cat([torch.cos(t_proj), torch.sin(t_proj)], dim=-1)
        
        return torch.cat([temporal_features, spatial_features], dim=-1)


class HashEmbedding(nn.Module):
    """
    Unified factory for Hash Encoding that auto-dispatches based on input dimensions.
    
    Args:
        dim_in: Input dimensions (2 or 3)
        mode: 'auto', 'full', or 'hybrid'
            - 'auto': HashEmbedder2D for 2D, HashEmbedder3D for 3D
            - 'full': Force full 3D (trilinear)
            - 'hybrid': Force hybrid (2D hash + 1D Fourier)
        **kwargs: Passed to the underlying encoder
    """
    def __init__(self, dim_in: int, mode: str = 'auto', **kwargs):
        super().__init__()
        self.dim_in = dim_in
        self.mode = mode
        
        if dim_in == 2:
            self.encoder = HashEmbedder2D(**self._filter_kwargs(kwargs, HashEmbedder2D))
        elif dim_in == 3:
            if mode in ['auto', 'full']:
                self.encoder = HashEmbedder3D(**self._filter_kwargs(kwargs, HashEmbedder3D))
            else:  # hybrid
                self.encoder = HybridHashEmbedder3D(**self._filter_kwargs(kwargs, HybridHashEmbedder3D))
        else:
            raise ValueError(f"HashEmbedding only supports dim_in=2 or 3, got {dim_in}")
        
        # Output dimension for compatibility
        if hasattr(self.encoder, 'output_dim'):
            self.output_dim = self.encoder.output_dim
        else:
            # Compute from embeddings
            self.output_dim = len(self.encoder.embeddings) * self.encoder.features_per_level
    
    def _filter_kwargs(self, kwargs, cls):
        """Filter kwargs to only include those accepted by cls.__init__"""
        import inspect
        sig = inspect.signature(cls.__init__)
        valid_keys = set(sig.parameters.keys()) - {'self'}
        return {k: v for k, v in kwargs.items() if k in valid_keys}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def set_active_levels(self, n: float):
        if hasattr(self.encoder, 'set_active_levels'):
            self.encoder.set_active_levels(n)

