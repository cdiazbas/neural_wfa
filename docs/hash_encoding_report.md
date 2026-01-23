# Hash Encoding Technical Report: Implementation & Benchmarks

This report consolidates the theoretical improvement plan, finalized implementation details, and benchmark results for the Neural WFA Hash Encoding suite (Versions 0-6).

## 1. Experimental Summary (Plage SSR Data)

Benchmarks were conducted on 200x178 solar data using 400 iterations (200/phase), Batch Size 200k, and a strict **MAX_RESOLUTION = 32** constraint for all versions.

| Version | Description | Final Loss | vs V0 Loss | Time (s) | vs V0 Time | Status |
|---------|-------------|------------|------------|----------|------------|--------|
| **V0** | Baseline Hash | 4.37e-04| 0% | 17.5s | 0% | Baseline |
| **V1** | **Dense Grids** | **4.29e-04**| **-1.9%** | **15.2s** | **-13%** | **Winner (Strict)** |
| **V2** | Shared Smooth | 4.33e-04| -1.1% | 21.4s | +22% | Smooth Grad |
| **V3** | Spec. Smooth | 4.31e-04| -1.4% | 24.1s | +38% | Adaptive Grid |
| **V4** | Progressive (S)| 4.32e-04| -1.2% | 21.6s | +23% | Coarse-to-Fine |
| **V5** | Progressive (P)| 4.35e-04| -0.4% | 24.3s | +39% | Hybrid Alpha |
| **V6** | Hybrid Plane | 4.31e-04| -1.4% | 23.0s | +31% | **Speed Optimized** |

> [!NOTE]
> **Optimized V6**: We achieved a 2.7x speedup in V6 (62s → 23s) by replacing the generic `grid_sample` with specialized dense-grid indexing. 

---

## 2. Implementation & Architectural Overhaul

### The Information Bottleneck Fix (V3/V5)
The main failure in previous iterations was an information bottleneck that reduced 16 levels of features into just 2 dimensions. 
- **Solution**: We restored **Multi-Scale Concatenation** across all versions.
- **Result**: Versions 3, 4, and 5 now leverage the full 32-feature vector, resolving the previous visual patterns and high loss.

### Dense Grid Optimization (V1)
Scientific data often lacks the high-frequency variation of natural images in the coarse-to-mid range. By using direct dense indexing for low resolutions, we removed hash collisions where the hash map was under-utilized. This provides a "Free Lunch" (better quality + faster speed).

### Hybrid Plane Frequency Matching (V6)
The superior quality previously seen in V6 (3.7e-04) was due to a fixed 128x128 plane. To ensure scientific fairness, we have now **restricted the plane to strictly follow the keyword-defined `max_resolution`**. When restricted to the same budget as the other models, V1 (Dense Grids) proves to be the most efficient representation.

---

## 3. Future SOTA Considerations

| Category | Improvement | Impact | Status |
|----------|-------------|--------|--------|
| **Interpolation** | Bicubic (Catmull-Rom) | High | Researched (V5) |
| **Smoothness** | B-Spline Grids | High | Theoretical |
| **Memory** | TensoRF Factorization | Medium| Not required for 2D |

## 4. Final Recommendations

1. **Production Choice**: **Version 1 (Dense Grids)**. It provides the best loss-per-second ratio under strict resolution constraints.
2. **Quality Choice (Unrestricted)**: Use **Version 6** with an unrestricted plane resolution (e.g., 128) if global smoothness is prioritized over strict frequency budgets.
3. **Training Stability**: Use **Version 4** (Progressive) for difficult inversions where coarse features must be locked in before adding fine-scale noise.
