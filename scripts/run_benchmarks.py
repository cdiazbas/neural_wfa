#!/usr/bin/env python
import torch
import numpy as np
import os, sys
import time
from pathlib import Path

# Ensure src is in path
sys.path.append("src")

from neural_wfa import Observation, WFAProblem, MagneticField
from neural_wfa.physics import LineInfo
from neural_wfa.nn import MLP, HashMLP
from neural_wfa.optimization import NeuralSolver, PixelSolver
from neural_wfa.utils.viz import torch2numpy

def run_benchmark():
    datadir = "example_py/plage_sst/"
    if not os.path.exists(datadir):
        datadir = "plage_sst/"

    def fits_open(path):
        import astropy.io.fits as fits
        return fits.open(path, "readonly")[0].data

    img = fits_open(datadir + "CRISP_5173_plage_dat.fits").astype("float32")
    xl = fits_open(datadir + "CRISP_5173_plage_wav.fits").astype("float32")


    ny, nx, ns, nw = img.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs = Observation(img, xl, active_wav_idx=[5, 6, 7], device=str(device))
    lin = LineInfo(5173)
    problem = WFAProblem(obs, lin, active_wav_idx=obs.active_wav_idx, device=device)

    y = np.linspace(-1, 1, ny)
    x = np.linspace(-1, 1, nx)
    YY, XX = np.meshgrid(y, x, indexing='ij')
    coords = np.stack([YY, XX], axis=-1).reshape(-1, 2)
    coords = torch.from_numpy(coords.astype(np.float32)).to(device)

    versions = [0, 1, 2, 3, 4, 6]
    results = {}

    for v in versions:
        print(f"\n>>> Benchmarking Version {v}...")
        
        model_blos = HashMLP(
            dim_in=2, dim_out=1, dim_hidden=64, num_layers=2,
            version=v, base_resolution=1, max_resolution=32, num_levels=16
        ).to(device)
        model_bqu = HashMLP(
            dim_in=2, dim_out=2, dim_hidden=64, num_layers=2,
            version=v, base_resolution=1, max_resolution=32, num_levels=16
        ).to(device)

        solver = NeuralSolver(
            problem=problem,
            model_blos=model_blos,
            model_bqu=model_bqu,
            coordinates=coords,
            lr=5e-3,
            batch_size=200000,
            device=device
        )
        solver.set_normalization(w_blos=1.0, w_bqu=1000.0)

        prog_schedule = None
        if v == 4:
            prog_schedule = {0: 4, 50: 8, 100: 12, 150: 16}

        start_time = time.time()
        
        # Phase 1
        solver.train(n_epochs=200, optimize_blos=True, optimize_bqu=False, 
                     progressive_schedule=prog_schedule, verbose=False)
        # Phase 2
        solver.train(n_epochs=200, optimize_blos=False, optimize_bqu=True, 
                     progressive_schedule=prog_schedule, verbose=False)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        field = solver.get_full_field()
        loss = problem.compute_loss(field).item()
        
        results[v] = {'loss': loss, 'time': elapsed}
        print(f"Version {v}: Loss = {loss:.4e}, Time = {elapsed:.2f}s")

    # Save to file
    with open("docs/hash_encoding_benchmarks_raw.txt", "w") as f:
        f.write("Version | Loss | Time (s)\n")
        f.write("--------|------|----------\n")
        for v in versions:
            f.write(f"{v} | {results[v]['loss']:.4e} | {results[v]['time']:.2f}\n")

    return results

if __name__ == "__main__":
    run_benchmark()
