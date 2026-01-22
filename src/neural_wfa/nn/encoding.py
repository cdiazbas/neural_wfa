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
        
        # Version 2: Learnable smoothing with residual connection
        # Use larger hidden dim and residual to avoid degrading bilinear features
        if version >= 2:
            hidden_dim = features_per_level * 4  # Expand for capacity
            self.smoother = nn.Sequential(
                nn.Linear(features_per_level, hidden_dim),
                nn.LayerNorm(hidden_dim),  # Stabilize training
                nn.GELU(),
                nn.Linear(hidden_dim, features_per_level)
            )
            # Weight for residual (learnable)
            self.smooth_weight = nn.Parameter(torch.tensor(0.1))  # Start with small contribution
        
        # Version 3: Adaptive multi-scale fusion
        if version >= 3:
            self.level_predictor = nn.Sequential(
                nn.Linear(2, 16),  # Input: normalized coordinates
                nn.GELU(),
                nn.Linear(16, num_levels),
                nn.Softmax(dim=-1)
            )
        
        # Version 4+: Progressive training support
        # Start with all levels active (can be reduced dynamically)
        self.active_levels = num_levels
        
        # Version 6: Multi-Plane Hybrid (dense plane + hash grid)
        if version >= 6:
            # Dense 2D feature plane for smooth global structure
            self.plane_resolution = 128  # Fixed resolution
            self.feature_plane = nn.Parameter(
                torch.randn(1, features_per_level, self.plane_resolution, self.plane_resolution) * 0.01
            )
    
    def set_active_levels(self, n: int):
        """Set number of active levels for progressive training (V4+).
        
        Args:
            n: Number of levels to activate (from coarsest)
        """
        self.active_levels = min(max(1, n), self.num_levels)

    def _bicubic_interp(self, x_norm: torch.Tensor, res: int, level_idx: int, 
                        x0: torch.Tensor, x1: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Catmull-Rom bicubic interpolation for smooth gradients.
        
        Args:
            x_norm: Normalized coordinates (N, 2)
            res: Current level resolution
            level_idx: Index of current level
            x0, x1: Floor and ceil coordinates
            w: Fractional weights
        Returns:
            Interpolated features (N, F)
        """
        wx, wy = w[..., 0:1], w[..., 1:2]
        
        # Helper to get index with clamping
        def get_index_clamped(coords):
            # Clamp coordinates to valid range
            coords_clamped = torch.stack([
                torch.clamp(coords[:, 0], 0, res - 1),
                torch.clamp(coords[:, 1], 0, res - 1)
            ], dim=-1)
            
            if self.is_dense[level_idx]:
                idx = coords_clamped[:, 1] * res + coords_clamped[:, 0]
            else:
                h = (coords_clamped[:, 0] * self.primes[1]) ^ (coords_clamped[:, 1] * self.primes[2])
                idx = h % self.embeddings[level_idx].num_embeddings
            return idx
        
        # Fetch 4x4 grid (Catmull-Rom needs 4 points per dimension)
        embeddings = []
        for dy in range(-1, 3):  # -1, 0, 1, 2
            for dx in range(-1, 3):
                coords = torch.stack([x0[:, 0] + dx, x0[:, 1] + dy], dim=-1)
                idx = get_index_clamped(coords)
                embeddings.append(self.embeddings[level_idx](idx))
        
        # Catmull-Rom weights
        def catmull_rom(t):
            # Returns 4 weights for positions -1, 0, 1, 2
            return torch.stack([
                -0.5 * t**3 + t**2 - 0.5 * t,
                1.5 * t**3 - 2.5 * t**2 + 1.0,
                -1.5 * t**3 + 2.0 * t**2 + 0.5 * t,
                0.5 * t**3 - 0.5 * t**2
            ], dim=-1)
        
        weights_x = catmull_rom(wx)  # (N, 1, 4)
        weights_y = catmull_rom(wy)  # (N, 1, 4)
        
        # Apply weights
        result = torch.zeros_like(embeddings[0])
        for iy in range(4):
            for ix in range(4):
                grid_idx = iy * 4 + ix
                weight = weights_x[..., ix:ix+1] * weights_y[..., iy:iy+1]
                result = result + weight * embeddings[grid_idx]
        
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, 2) coordinates. Expected range matches bounding_box.
        Returns:
            (Batch, num_levels * features_per_level) embedded features.
        """
        # Normalize x to [0, 1] based on bounding_box
        min_v = torch.tensor(self.bounding_box[0], device=x.device)
        max_v = torch.tensor(self.bounding_box[1], device=x.device)
        x_norm = (x - min_v) / (max_v - min_v)
        # Clip to ensure bounds
        x_norm = torch.clamp(x_norm, 0.0, 1.0) # Actually allow slightly outside if needed? Clamping is safer.
        
        features = []
        
        # V4+: Only process active levels for progressive training
        for i in range(self.active_levels):
            res = self.resolutions[i]
            # Scale to grid resolution
            x_scaled = x_norm * res
            
            # Get integer and fractional parts
            x0 = torch.floor(x_scaled).long()
            x1_unclamped = x0 + 1
            
            # Clamp to grid bounds
            x0 = torch.clamp(x0, min=0, max=res - 1)
            x1 = torch.clamp(x1_unclamped, min=0, max=res - 1)
            
            # Fractional part (weights for interpolation)
            # When at boundary, w should be 0 (use only x0/y0)
            w = x_scaled - torch.floor(x_scaled)  # (N, 2)
            wx, wy = w[..., 0:1], w[..., 1:2]
            
            # 4 Corner vertices of the cell
            # vertices: (0,0), (1,0), (0,1), (1,1)
            # Shapes: x0 is (N, 2)
            
            # Helper to compute indices
            def get_index(coords):
                """Compute embedding indices for coordinates."""
                # coords: (N, 2) long tensor
                if self.is_dense[i]:
                    # Dense grid: direct linear indexing
                    # idx = y * res + x
                    idx = coords[:, 1] * res + coords[:, 0]
                else:
                    # Sparse hash grid: XOR hashing
                    h = (coords[:, 0] * self.primes[1]) ^ (coords[:, 1] * self.primes[2])
                    idx = h % self.embeddings[i].num_embeddings
                return idx

            # Collect vertex indices
            c00 = get_index(torch.stack([x0[:, 0], x0[:, 1]], dim=-1))
            c10 = get_index(torch.stack([x1[:, 0], x0[:, 1]], dim=-1))
            c01 = get_index(torch.stack([x0[:, 0], x1[:, 1]], dim=-1))
            c11 = get_index(torch.stack([x1[:, 0], x1[:, 1]], dim=-1))
            
            # Look up embeddings
            e00 = self.embeddings[i](c00)
            e10 = self.embeddings[i](c10)
            e01 = self.embeddings[i](c01)
            e11 = self.embeddings[i](c11)
            
            # Bilinear interpolation (V0, V1, V2, V3 all use bilinear base)
            fx0 = torch.lerp(e00, e10, wx)
            fx1 = torch.lerp(e01, e11, wx)
            val = torch.lerp(fx0, fx1, wy)
            
            # Version 2+: Apply learnable smoothing with residual connection
            if self.version >= 2 and self.version < 3:
                # Residual connection: original + alpha * smooth(original)
                smooth_delta = self.smoother(val)
                val = val + torch.sigmoid(self.smooth_weight) * smooth_delta
            
            features.append(val)
        
        # Version 6: Sample from dense feature plane
        if self.version >= 6:
            # x_norm is already in [0, 1], grid_sample expects [-1, 1]
            grid_coords = x_norm * 2 - 1  # (N, 2) -> [-1, 1]
            # grid_sample expects (N, 1, 1, 2) for 2D sampling
            grid = grid_coords.view(-1, 1, 1, 2)
            
            # Sample from plane (bilinear interpolation)
            plane_features = torch.nn.functional.grid_sample(
                self.feature_plane.expand(x.shape[0], -1, -1, -1),
                grid,
                align_corners=True,
                mode='bilinear'
            )  # (N, F, 1, 1)
            
            # Reshape to (N, F)
            plane_features = plane_features.squeeze(-1).squeeze(-1).permute(1, 0).contiguous().t()
            features.append(plane_features)
            
            
        # Version 3: Adaptive multi-scale fusion (optimized)
        if self.version >= 3 and self.version < 6:
            # Stack all level features for batch processing
            # features is list of (N, F), stack to (N, active_levels, F)
            stacked_features = torch.stack(features, dim=1)  # (N, L, F)
            
            # Reshape for batch smoother application: (N*L, F)
            N, L, F = stacked_features.shape
            features_flat = stacked_features.reshape(N * L, F)
            
            # Apply smoother once to all features (batched)
            smoothed_flat = self.smoother(features_flat)  # (N*L, F)
            smoothed = smoothed_flat.reshape(N, L, F)  # (N, L, F)
            
            # Predict level weights based on coordinates
            # NOTE: level_predictor was initialized with num_levels outputs,
            # but we only use the first active_levels weights
            all_weights = self.level_predictor(x_norm)  # (N, num_levels)
            level_weights = all_weights[:, :self.active_levels]  # (N, active_levels)
            # Renormalize to sum to 1
            level_weights = level_weights / (level_weights.sum(dim=-1, keepdim=True) + 1e-8)
            
            # Weighted sum: broadcast and sum over level dimension
            # level_weights: (N, L) -> (N, L, 1) for broadcasting
            weighted = smoothed * level_weights.unsqueeze(-1)  # (N, L, F)
            output = weighted.sum(dim=1)  # (N, F)
            
            return output
        else:
            # Version 0, 1, 2: Concatenate all level features
            return torch.cat(features, dim=-1)  # (N, L*F)

