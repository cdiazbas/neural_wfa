#!/usr/bin/env python
# coding: utf-8

# # Neural WFA Inversion (Time Series - Hash Encoding)
# 
# This script demonstrates the usage of the refactored `neural_wfa` package for
# inverting solar spectropolarimetric data using **Hash Encoding** Neural Fields
# on a **simulated time series**.
#
# Tests both `mode='full'` (3D trilinear hash) and `mode='hybrid'` (2D hash + 1D Fourier).

# In[ ]:


import torch
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import os, sys

# Ensure src is in path if running locally
sys.path.append("src")
sys.path.append("../src")

from neural_wfa import Observation, WFAProblem, MagneticField
from neural_wfa.physics import LineInfo
from neural_wfa.nn import HashMLP
from neural_wfa.optimization import NeuralSolver
from neural_wfa.utils.viz import set_params
from neural_wfa.utils.viz import plot_wfa_results, plot_temporal_evolution, torch2numpy

set_params()


# ## 1. Load Data & Simulate Time Series

# In[ ]:


datadir = "example_py/plage_sst/"
if not os.path.exists(datadir):
    datadir = "plage_sst/"

# Original Single Snapshot (Ny, Nx, 4, Nw)
img_raw = np.ascontiguousarray(
    fits.open(datadir + "CRISP_5173_plage_dat.fits", "readonly")[0].data,
    dtype="float32",
)
xl = np.ascontiguousarray(
    fits.open(datadir + "CRISP_5173_plage_wav.fits", "readonly")[0].data,
    dtype="float32",
)

print("Original Data shape:", img_raw.shape)
ny_orig, nx_orig, ns, nw = img_raw.shape

# --- Simulate Time Series ---
Nt = 4
amplify_factor = 1.5

print(f"Generating {Nt} frames with Q/U/V amplification factor {amplify_factor}...")

img_series_list = []

for t in range(Nt):
    frame = img_raw.copy()
    scale = amplify_factor ** t
    frame[:, :, 1, :] *= scale  # Q
    frame[:, :, 2, :] *= scale  # U
    frame[:, :, 3, :] *= scale  # V
    img_series_list.append(frame)

# Stack to (Nt, Ny, Nx, Ns, Nw)
img_series = np.stack(img_series_list, axis=0)
print("Time Series Data shape:", img_series.shape)
nt, ny, nx, ns, nw = img_series.shape


# ## 2. Setup Problem

# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Observation (5D input)
obs = Observation(img_series, xl, active_wav_idx=[5, 6, 7], device=str(device))
print(f"Observation loaded. Grid Shape: {obs.grid_shape}")

# Line Parameters
lin = LineInfo(5173)

# WFA Physics Engine
problem = WFAProblem(obs, lin, device=device)


# ## 3. Initialize Hash Encoding Neural Fields (3D)
#
# === MODE SELECTOR ===
# Set MODE to 'full' for full 3D trilinear hash encoding
# Set MODE to 'hybrid' for 2D spatial hash + 1D temporal Fourier
MODE = 'hybrid'  # Try 'hybrid' for alternative

# In[ ]:


# Coordinate Grid (normalized -1 to 1 for all dimensions)
t_norm = np.linspace(-1, 1, nt)
y_norm = np.linspace(-1, 1, ny)
x_norm = np.linspace(-1, 1, nx)

# Create 3D meshgrid (t, y, x) -> shape (Nt, Ny, Nx, 3)
TT, YY, XX = np.meshgrid(t_norm, y_norm, x_norm, indexing='ij')
coords = np.stack([TT, YY, XX], axis=-1).reshape(-1, 3)  # Flatten to (Nt*Ny*Nx, 3)
coords = torch.from_numpy(coords.astype(np.float32)).to(device)

print(f"Coordinate Grid Shape: {coords.shape} (should be (Nt*Ny*Nx, 3))")
print(f"Using Hash Encoding mode: '{MODE}'")

# === HASH ENCODING PARAMETERS ===
# Spatial parameters (for both modes)
SPATIAL_BASE_RES = 2**0
SPATIAL_MAX_RES = 2**5
NUM_LEVELS = 16
VERSION = 1

# Temporal parameters
if MODE == 'full':
    # Full 3D: separate temporal resolutions
    TEMPORAL_BASE_RES = 2  # Lower temporal resolution (fewer time samples)
    TEMPORAL_MAX_RES = 8
else:
    # Hybrid: Fourier parameters for time
    TEMPORAL_M_FREQS = 64
    TEMPORAL_SIGMA = 10.0

# Model for Blos
if MODE == 'full':
    model_blos = HashMLP(
        dim_in=3,
        dim_out=1,
        dim_hidden=64,
        num_layers=2,
        num_levels=NUM_LEVELS,
        base_resolution=SPATIAL_BASE_RES,
        max_resolution=SPATIAL_MAX_RES,
        version=VERSION,
        mode='full',
        temporal_base_resolution=TEMPORAL_BASE_RES,
        temporal_max_resolution=TEMPORAL_MAX_RES,
    ).to(device)
else:
    model_blos = HashMLP(
        dim_in=3,
        dim_out=1,
        dim_hidden=64,
        num_layers=2,
        num_levels=NUM_LEVELS,
        base_resolution=SPATIAL_BASE_RES,
        max_resolution=SPATIAL_MAX_RES,
        version=VERSION,
        mode='hybrid',
        temporal_m_freqs=TEMPORAL_M_FREQS,
        temporal_sigma=TEMPORAL_SIGMA,
    ).to(device)

# Model for BQU
if MODE == 'full':
    model_bqu = HashMLP(
        dim_in=3,
        dim_out=2,
        dim_hidden=64,
        num_layers=2,
        num_levels=NUM_LEVELS,
        base_resolution=SPATIAL_BASE_RES,
        max_resolution=SPATIAL_MAX_RES,
        version=VERSION,
        mode='full',
        temporal_base_resolution=TEMPORAL_BASE_RES,
        temporal_max_resolution=TEMPORAL_MAX_RES,
    ).to(device)
else:
    model_bqu = HashMLP(
        dim_in=3,
        dim_out=2,
        dim_hidden=64,
        num_layers=2,
        num_levels=NUM_LEVELS,
        base_resolution=SPATIAL_BASE_RES,
        max_resolution=SPATIAL_MAX_RES,
        version=VERSION,
        mode='hybrid',
        temporal_m_freqs=TEMPORAL_M_FREQS,
        temporal_sigma=TEMPORAL_SIGMA,
    ).to(device)

print(f"Models created with mode='{MODE}'")


# ## 4. Train using Neural Solver

# In[ ]:


solver = NeuralSolver(
    problem=problem,
    model_blos=model_blos,
    model_bqu=model_bqu,
    coordinates=coords,
    lr=5e-3,  # Higher LR for Hash Encoding
    batch_size=200000,
    device=device
)
solver.set_normalization(w_blos=1.0, w_bqu=1000.0)

print("Training Phase 1: Blos Only...")
solver.train(n_epochs=200, optimize_blos=True, optimize_bqu=False)
loss_blos = np.array(solver.loss_history)
lr_blos = np.array(solver.lr_history)
solver.loss_history = []
solver.lr_history = []

print("Training Phase 2: BQU Only...")
solver.train(n_epochs=200, optimize_blos=False, optimize_bqu=True)
loss_bqu = np.array(solver.loss_history)
lr_bqu = np.array(solver.lr_history)


from neural_wfa.utils.viz import plot_loss

# Phase 1
plot_loss({'loss': loss_blos, 'lr': lr_blos})
plt.savefig(f"ref_hash_time_{MODE}_loss_blos.png", dpi=300)
plt.close()

# Phase 2
plot_loss({'loss': loss_bqu, 'lr': lr_bqu})
plt.savefig(f"ref_hash_time_{MODE}_loss_bqu.png", dpi=300)
plt.close()


# ## 5. Visualize Results & Analysis

# In[ ]:


final_field = solver.get_full_field()

blos_map = torch2numpy(final_field.blos_map)
btrans_map = torch2numpy(final_field.btrans_map)
azi_map = torch2numpy(final_field.phi_map)

print(f"Output Shape: blos_map = {blos_map.shape} (Expected: ({nt}, {ny}, {nx}))")

# Plot LAST frame
if len(blos_map.shape) == 3:
    plot_wfa_results(blos_map[-1], btrans_map[-1], azi_map[-1], 
                     save_name=f"ref_hash_time_{MODE}_results_last_frame.png")
else:
    blos_map = blos_map.reshape(nt, ny, nx)
    btrans_map = btrans_map.reshape(nt, ny, nx)
    azi_map = azi_map.reshape(nt, ny, nx)
    plot_wfa_results(blos_map[-1], btrans_map[-1], azi_map[-1], 
                     save_name=f"ref_hash_time_{MODE}_results_last_frame.png")

# Total Loss
loss_val = problem.compute_loss(final_field).item()
print(f"Total Loss: {loss_val:.4e}")

# Baseline WFA
print("Computing Baseline WFA (for comparison)...")
from neural_wfa.optimization import PixelSolver
solver_wfa = PixelSolver(problem, device=device)
solver_wfa.initialize_parameters(method='weak_field')
wfa_field = solver_wfa.get_field()

wfa_blos = torch2numpy(wfa_field.blos_map)
wfa_btrans = torch2numpy(wfa_field.btrans_map)
wfa_azi = torch2numpy(wfa_field.phi_map)

if len(wfa_blos.shape) != 3:
    wfa_blos = wfa_blos.reshape(nt, ny, nx)
    wfa_btrans = wfa_btrans.reshape(nt, ny, nx)
    wfa_azi = wfa_azi.reshape(nt, ny, nx)

plot_wfa_results(wfa_blos[-1], wfa_btrans[-1], wfa_azi[-1], 
                 save_name=f"ref_hash_time_{MODE}_wfa_baseline.png")


# ## 6. Temporal Evolution at a Single Pixel

# In[ ]:


py, px = 100, 100

# Use smart get_full_field for efficient single-pixel query
print(f"Querying pixel (y={py}, x={px}) across all time frames...")

pixel_field = solver.get_full_field(y=py, x=px)
blos_neural_ts = torch2numpy(pixel_field.blos_map)
btrans_neural_ts = torch2numpy(pixel_field.btrans_map)
azi_neural_ts = torch2numpy(pixel_field.phi_map)

print(f"  Neural Field pixel shape: {blos_neural_ts.shape} (expected: ({nt},))")

# WFA time series at this pixel
blos_wfa_ts = wfa_blos[:, py, px]
btrans_wfa_ts = wfa_btrans[:, py, px]
azi_wfa_ts = wfa_azi[:, py, px]

time_frames = np.arange(nt)

# Plot comparison
plot_temporal_evolution(
    time_frames,
    blos_wfa_ts, btrans_wfa_ts, azi_wfa_ts,
    blos_neural_ts, btrans_neural_ts, azi_neural_ts,
    label_1="WFA (Analytical)",
    label_2=f"Hash Encoding ({MODE})",
    pixel_coords=(py, px),
    save_name=f"ref_hash_time_{MODE}_pixel_evolution.png"
)

print(f"Saved temporal evolution plot to ref_hash_time_{MODE}_pixel_evolution.png")

# Demonstrate single frame query
print(f"\nQuerying single frame (t=2) for full spatial map...")
frame_field = solver.get_full_field(t=2)
print(f"  Single frame shape: blos={torch2numpy(frame_field.blos_map).shape} (expected: ({ny}, {nx}))")

print(f"Done. Saved results to ref_hash_time_{MODE}_*.png")
