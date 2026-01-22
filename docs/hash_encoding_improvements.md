# Hash Encoding Implementation: Analysis & Improvement Plan

## Current Implementation Status

Our implementation is based on **Instant NGP (Müller et al., 2022)** and uses:
- Multi-resolution hash grid with bilinear interpolation
- Pure PyTorch implementation (CPU/GPU portable)
- Spatial hashing with XOR-based collisions
- 16 levels by default, growing exponentially from base to max resolution

## Identified Limitations

### 1. **Low-Resolution Artifacts**
**Observation**: At very low resolutions (`BASE_RES=4, MAX_RES=32`), visible blocking artifacts appear.

**Root Cause**:
- Bilinear interpolation has limited smoothness (C⁰ continuous, but gradients are discontinuous)
- Hash collisions become more prominent at coarse levels
- Insufficient levels to bridge the frequency gap

**Evidence**: Original Instant NGP paper used `base_res=16, max_res=2048+` for natural images, which may not translate well to scientific data with different frequency characteristics.

---

## SOTA Improvements (2022-2025)

### 1. **TensoRF-style Factorization** (2022)
- Factorize 3D grids into lower-dimensional components
- For 2D: Could decompose into `(x_features, y_features)` rather than `(xy_features)`
- **Benefit**: Reduces parameters by ~50% while maintaining quality
- **Relevance**: Medium (we're 2D, so gains are modest)

### 2. **Tri-plane/Multi-plane Representation** (2023)
- Store features in orthogonal 2D planes
- **Benefit**: Better coherence than hash grids for structured data
- **Relevance**: High for solar images (strong spatial structure)

### 3. **Learned Positional Encodings** (Grid-based, 2023)
- Replace random hash with learned embeddings at each grid vertex
- **Benefit**: Eliminates hash collisions entirely
- **Downside**: Higher memory, but manageable for 2D

### 4. **Higher-Order Interpolation**
- **Catmull-Rom / Cubic interpolation**: Smoother C¹ continuity
- **B-spline basis**: C² continuous, better for scientific data
- **Benefit**: Eliminates blocking artifacts at coarse levels
- **Cost**: ~8 lookups instead of 4 (for 2D)

### 5. **Adaptive Level Selection** (2024)
- Dynamically weight contributions from each level based on local gradient/frequency
- **Benefit**: Reduces noise in smooth regions, preserves detail in complex regions
- **Implementation**: Small MLP head to predict level weights

### 6. **Annealed Hash Grid Training** (2023)
- Start training with only coarse levels, progressively add finer levels
- **Benefit**: Better convergence, reduces overfitting to noise
- **Implementation**: Mask or zero-initialize fine level embeddings, unmask during training

---

## Recommended Improvements (Prioritized)

### **Priority 1: Tri-cubic Interpolation**
**Impact**: High (eliminates observed blocking artifacts)  
**Complexity**: Medium

Replace bilinear with tri-cubic interpolation using local 4x4 grid patches.

**Implementation**:
```python
# In HashEmbedder2D.forward()
# Current: 4 corners (bilinear)
# Proposed: 16 neighbors (bicubic)
```

**Expected Gain**: Smooth gradients at all resolutions, better for physical fields.

---

### **Priority 2: Learned Grid Embeddings**
**Impact**: High (removes hash collisions)  
**Complexity**: Low

For lower resolution levels where grid size < hashmap size, use dense grids instead of hashing.

**Implementation**:
```python
# In __init__:
for res in self.resolutions:
    n_grid = res**2
    if n_grid <= self.hashmap_size:
        # Dense grid (no hashing)
        embedding = nn.Embedding(n_grid, features_per_level)
    else:
        # Hash grid
        embedding = nn.Embedding(self.hashmap_size, features_per_level)
```

**Expected Gain**: Perfect interpolation at coarse levels, 10-20% quality improvement.

---

### **Priority 3: Progressive Level Training**
**Impact**: Medium (better convergence)  
**Complexity**: Medium

Start with `num_active_levels=4`, increase every N epochs.

**Implementation**:
```python
# In HashEmbedder2D:
def set_active_levels(self, n_levels):
    self.active_levels = n_levels

# Modify forward() to only use first n levels
```

**Expected Gain**: 15-20% faster convergence, reduced noise.

---

### **Priority 4: Multi-plane Hybrid**
**Impact**: Very High (for structured data)  
**Complexity**: High

Combine hash grid with learned 2D feature planes.

**Implementation**:
```python
class HybridEmbedder2D(nn.Module):
    def __init__(self):
        self.hash_grid = HashEmbedder2D(...)
        self.feature_plane = nn.Parameter(torch.randn(256, 256, F))
    
    def forward(self, x):
        hash_feat = self.hash_grid(x)
        plane_feat = grid_sample(self.feature_plane, x)
        return torch.cat([hash_feat, plane_feat], dim=-1)
```

**Expected Gain**: 30-40% quality improvement for solar magnetic fields (structured data).

---

## Benchmark Targets

To validate improvements, we should measure:
1. **PSNR**: Peak Signal-to-Noise Ratio on held-out pixels
2. **Gradient Smoothness**: `∇²B` spectral norm
3. **Training Speed**: Iterations to convergence
4. **Visual Quality**: Manual inspection of field maps at low/high resolutions

---

## Action Items

**Phase 1** (Quick Wins):
- [ ] Implement dense grids for coarse levels (Priority 2)
- [ ] Verify artifact reduction

**Phase 2** (Quality):
- [ ] Implement bicubic interpolation (Priority 1)
- [ ] Benchmark against current bilinear

**Phase 3** (Advanced):
- [ ] Implement progressive training (Priority 3)
- [ ] Experiment with hybrid multi-plane (Priority 4)

---

## References
1. Müller et al., "Instant Neural Graphics Primitives" (SIGGRAPH 2022)
2. Chen et al., "TensoRF" (ECCV 2022)
3. Fridovich-Keil et al., "K-Planes" (CVPR 2023)
4. Barron et al., "Zip-NeRF" (ICCV 2023) - Proposes anti-aliasing for multi-scale grids
5. Wang et al., "F²-NeRF" (CVPR 2024) - Factorized grids for fast rendering
