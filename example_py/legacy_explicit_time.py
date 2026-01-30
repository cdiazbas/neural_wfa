#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os, sys
from tqdm import tqdm, trange
from einops import rearrange

import astropy.io.fits as fits

sys.path.append("../")


# Legacy imports
from legacy import utils
from legacy import neural_fields, bfield
from legacy import explicit
from legacy.plot_params import set_params
set_params()

# ## 1. Load Data & Simulate Time Series

# In[2]:


datadir = "example_py/plage_sst/"
if not os.path.exists(datadir):
    datadir = "plage_sst/"

# Original Single Snapshot
# Shape: (ny, nx, nStokes, nWav) -> NOTE: Legacy seems to expect (ny, nx, ns, nw) based on bfield.py
img_raw = np.ascontiguousarray(
    fits.open(datadir + "CRISP_5173_plage_dat.fits", "readonly")[0].data,
    dtype="float32",
)
xl = np.ascontiguousarray(
    fits.open(datadir + "CRISP_5173_plage_wav.fits", "readonly")[0].data,
    dtype="float32",
)

# Legacy might expect different shape?
# Explicit Refactored used (Ny, Nx, 4, Nw).
# Legacy explicit.py line 47: ny, nx, nStokes, nWav = img.shape
# bfield.py line 206: ny, nx, ns, nw = self.data.shape
# So shape is compatible.

print("Original Data shape:", img_raw.shape)

# --- Simulate Time Series ---
Nt = 4
amplify_factor = 1.5

print(f"Generating {Nt} frames with Q/U/V amplification factor {amplify_factor}...")

img_series_list = []

for t in range(Nt):
    frame = img_raw.copy()
    
    # Scale factor for this timestep: 1.5^t
    scale = amplify_factor ** t
    
    # Apply to Q, U, V (indices 1, 2, 3)
    frame[:, :, 1, :] *= scale
    frame[:, :, 2, :] *= scale
    frame[:, :, 3, :] *= scale
    
    img_series_list.append(frame)

# Stack to (Nt, Ny, Nx, Ns, Nw)
img_series = np.stack(img_series_list, axis=0)
print("Time Series Data shape:", img_series.shape)


# ## 2. Model & Optimization

# In[3]:

# Model
niter = 50 # Reduced for benchmark
mask_index = [5, 6, 7]  

# WFA_model3D handles (Nt, Ny, Nx, Ns, Nw) -> flattens internally
mymodel = bfield.WFA_model3D(img_series, xl, mask=mask_index, spectral_line=5173, verbose=True)

# Regularization Config
reguV = 1e-4
reguQU = 1e-2

# Temporal Regularization
reguT_Blos = 1e-2
reguT_BQU  = 1e-2


# Initial Guess
# prepare_initial_guess calls mymodel.initial_guess()
# initial_guess in bfield.py reshapes output?
# Let's rely on prepare_initial_guess
out = explicit.prepare_initial_guess(mymodel)
print("Initial Guess shape:", out.shape) # Should be (Nt*Ny*Nx, params) or similar

optimizer = torch.optim.Adam([out], lr=1e-2)

print("Starting Inversion...")
outplot, out = explicit.optimization(
    optimizer=optimizer,
    niterations=niter, 
    parameters=out,
    model=mymodel, 
    reguV=reguV, 
    reguQU=reguQU, 
    reguT_Blos=reguT_Blos,
    reguT_BQU=reguT_BQU,
    weights=[10,10,10], 
    normgrad=True
)

# ## 3. Analyze Results

# In[4]:

# outplot shape: (Nt, Ny, Nx, 3) because explicit.py optimization routine reshapes it back
# line 202 in explicit.py (user edit): outplot = rearrange(outplot, '(nt ny nx) p -> nt ny nx p', ...)

print("Final Output Shape:", outplot.shape)

# Plots (Last Frame)
# outplot is [Blos, B_trans, B_azi] ?
# bfield.forward returns Q, U, V.
# optimization returns outplot_final which constructs [Blos*Vnorm, Btrans, Phi]

Blos = outplot[-1,:,:,0]
Bhor = outplot[-1,:,:,1]
Bazi = outplot[-1,:,:,2]

plt.close("all");
ny, nx = Blos.shape
extent = np.float64((0,nx,0,ny))
f, ax = plt.subplots(nrows=1, ncols=3, figsize=(13.5, 6.75))

im0 = ax[0].imshow(Blos, vmax=800, vmin=-800, cmap='RdGy',      interpolation='nearest', extent=extent)
im1 = ax[1].imshow(Bhor, vmin=0, vmax=800,   cmap='gist_gray', interpolation='nearest', extent=extent)
im2 = ax[2].imshow(Bazi, vmax=np.pi, vmin=0,    cmap='twilight',      interpolation='nearest', extent=extent)

names = [r'B$_\parallel$', r'B$_\bot$', r'$\Phi_B$']
f.colorbar(im0, ax=ax[0], orientation='horizontal', label=names[0]+' [G]', pad=0.17)
f.colorbar(im1, ax=ax[1], orientation='horizontal', label=names[1]+' [G]', pad=0.17)
f.colorbar(im2, ax=ax[2], orientation='horizontal', label=names[2]+' [rad]', pad=0.17)

for ii in range(3):
    ax[ii].set_title(names[ii] + f" (Frame {Nt-1})")
    ax[ii].set_xlabel('x [pixels]')

plt.savefig("legacy_time_result_last_frame.png")
print("Saved result to legacy_time_result_last_frame.png")
