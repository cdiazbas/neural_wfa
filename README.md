# Spectropolarimetric inversions using neural fields under the weak-field approximation

This repository contains the code used to generate the results of the paper "Exploring spectropolarimetric inversions using neural fields: solar chromospheric magnetic field under the weak-field approximation" ([arXiv:2409.05156](https://arxiv.org/abs/2409.05156)).




![example](docs/sketch_wINRWFA_large.png?raw=true "")
**Figure 1** — Sketch of the neural field inversion of the magnetic field vector under the weak-field approximation. The neural field is a continuous representation of the magnetic field vector over the spatial and temporal domain.

![example](docs/transverse_comparison.png?raw=true "")
**Figure 2** — Comparison of the transverse component of the magnetic field vector inferred from a simulated observation at Ca II 8542A using the pixel-wise weak-field approximation (WFA) inversion (middle panel) and the neural field inversion (right panel).

## Abstract
Full-Stokes polarimetric datasets, originating from slit-spectrograph or narrow-band filtergrams, are routinely acquired nowadays. The data rate is increasing with the advent of bi-dimensional spectropolarimeters and observing techniques that allow long-time sequences of high-quality observations. There is a clear need to go beyond the traditional pixel-by-pixel strategy in spectropolarimetric inversions by exploiting the spatiotemporal coherence of the inferred physical quantities that contain valuable information about the conditions of the solar atmosphere. We explore the potential of neural networks as a continuous representation of the physical quantities over time and space (also known as neural fields), for spectropolarimetric inversions. We have implemented and tested a neural field to perform one of the simplest forms of spectropolarimetric inversions, the inference of the magnetic field vector under the weak-field approximation (WFA). By using a neural field to describe the magnetic field vector, we can regularize the solution in the spatial and temporal domain by assuming that the physical quantities are continuous functions of the coordinates. This technique can be trivially generalized to account for more complex inversion methods. We have tested the performance of the neural field to describe the magnetic field of a realistic 3D magnetohydrodynamic (MHD) simulation. We have also tested the neural field as a magnetic field inference tool (approach also known as physics-informed neural networks) using the WFA as our radiative transfer model. We investigated the results in synthetic and real observations of the Ca II 8542 A line. We also explored the impact of other explicit regularizations, such as using the information of an extrapolated magnetic field, or the orientation of the chromospheric fibrils. Compared to the traditional pixel-by-pixel inversion, the neural field approach improves the fidelity of the reconstruction of the magnetic field vector, especially the transverse component. This implicit regularization is a way of increasing the effective signal-to-noise of the observations. Although it is slower than the pixel-wise WFA estimation, this approach shows a promising potential for depth-stratified inversions, by reducing the number of free parameters and inducing spatio-temporal constraints in the solution.

## Installation

To install the package in editable mode:
```bash
pip install -e .
```

## Project Structure

The library has been reorganized into a modular `src`-layout to ensure clean imports and clear separation of concerns:

- **`core/`**: Canonical data structures and the WFA physics engine. Contains `MagneticField` (state container), `Observation` (data container), and `WFAProblem` (physics & loss).
- **`physics/`**: Low-level physics routines including atomic line information, numerical derivatives, and potential field extrapolation.
- **`nn/`**: Neural Network architectures including MLPs and Fourier Feature mappings for neural field representations.
- **`optimization/`**: High-performance solvers for both pixel-wise (explicit) and neural field inversions.
- **`regularization/`**: Unified spatial and temporal regularization schemes (L1, L2, TV, and legacy kernels).
- **`analysis/`**: Tools for post-inversion analysis, including calibrated uncertainty estimation and quality metrics (PSNR, BPP).
- **`utils/`**: Scientific I/O (FITS) and visualization helpers optimized for spectropolarimetric data.

## Core API Highlights

### `MagneticField`
The class holding the magnetic field parameters. It handles internal normalization for optimization stability while providing physical property accessors for $B_{los}$, $B_{trans}$, and Azimuth ($\phi$). It automatically handles the 180-degree ambiguity for visualization.

### `Observation`
Standardizes spectropolarimetric data handling, providing automatic coordinate generation for neural fields and seamless batching support.

### `WFAProblem`
The bridge between physics and optimization. It manages precomputed derivatives and provides robust loss functions (Huber, Cauchy) with multi-stokes weighting.

## Performance Improvements

The refactored implementation provides significant speedups over the legacy codebase (implemented at the time of the paper submission) by leveraging vectorized logic and optimized GPU memory management:

| Solver | Legacy Speed | Refactored Speed | Speedup | Key Optimization |
| :--- | :--- | :--- | :--- | :--- |
| **Explicit (Pixel)** | ~15 it/s | **~120 it/s** | **~8x** | Vectorized "hot-paths" |
| **Neural Field** | ~7.4 it/s | **~52 it/s** | **~7x** | GPU Buffer registration |

## Advanced Features

- **Hash Encoding**: Supports multi-resolution hash encodings (Dense Grids, Progressive Encodings, and Hybrid Planes) for faster convergence and higher-fidelity reconstructions.
- **Calibrated Uncertainty**: Unified uncertainty estimation matching analytical Taylor expansions with high numerical stability.
- **Flexible Regularization**: Support for potential field extrapolation and fibril-based orientation constraints.

## Usage

### Neural Field Inversion
See `examples_ipynb/neural.ipynb` for a complete example.

```python
from neural_wfa import Observation, WFAProblem
from neural_wfa.physics import LineInfo
from neural_wfa.nn import MLP
from neural_wfa.optimization import NeuralSolver

obs = Observation(img, xl, device="auto")
problem = WFAProblem(obs, LineInfo(8542))
solver = NeuralSolver(problem, model_blos, model_bqu, coords)
solver.train(n_epochs=200)
```

### Explicit (Pixel-wise) Inversion
See `examples_ipynb/explicit.ipynb` for a complete example.

```python
from neural_wfa import Observation, WFAProblem
from neural_wfa.physics import LineInfo
from neural_wfa.optimization import PixelSolver

obs = Observation(img, xl, device="auto")
problem = WFAProblem(obs, LineInfo(8542))
solver = PixelSolver(problem)
solver.solve(n_iterations=200)
```

### Example scripts and notebooks
The `examples_py/` folder contains runnable Python scripts. The `examples_ipynb/` folder contains the equivalent Jupyter notebooks:

| Example | Description |
| :--- | :--- |
| `explicit.py` / `explicit.ipynb` | Pixel-wise (explicit) WFA inversion |
| `explicit_time.py` / `explicit_time.ipynb` | Explicit inversion with temporal data |
| `neural.py` / `neural.ipynb` | Neural field inversion |
| `neural_time.py` / `neural_time.ipynb` | Neural field with temporal sequences |
| `neural_hash.py` / `neural_hash.ipynb` | Neural field with hash encoding |
| `neural_hash_time.py` / `neural_hash_time.ipynb` | Hash encoding + temporal sequences |
