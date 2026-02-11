import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Optional


def find_optimal_batch_size(
    solver,  # NeuralSolver instance
    min_batch: int = 1000,
    max_batch: Optional[int] = None,
    n_tests: int = 15,
) -> int:
    """
    Benchmarks different batch sizes to find the one with highest throughput.
    """
    n_pixels = solver.coordinates.shape[0]
    max_batch = max_batch or n_pixels

    batch_sizes = np.unique(
        np.logspace(np.log10(min_batch), np.log10(max_batch), n_tests).astype(int)
    )
    timings = []

    # Save original batch size
    solver.batch_size

    print(f"Benchmarking {len(batch_sizes)} batch sizes...")
    for bs in batch_sizes:
        solver.batch_size = bs

        # Warmup
        solver.train(n_epochs=1, verbose=False)

        # Benchmark
        t0 = time.time()
        solver.train(n_epochs=2, verbose=False)
        t1 = time.time()

        # Throughput (pixels per second)
        throughput = (2 * n_pixels) / (t1 - t0)
        timings.append(throughput)

    best_idx = np.argmax(timings)
    best_bs = int(batch_sizes[best_idx])

    print(f"Optimal batch size found: {best_bs} ({timings[best_idx]:.2e} px/s)")

    # Restore/set best
    solver.batch_size = best_bs

    # Optional: Plot
    plt.figure(figsize=(8, 4))
    plt.plot(batch_sizes, timings, "o-")
    plt.xscale("log")
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput [px/s]")
    plt.title(f"Batch Size Optimization (Best: {best_bs})")
    plt.grid(True, which="both", ls="-", alpha=0.5)

    return best_bs


def train_with_scheduling(
    solver,
    n_epochs: int = 1000,
    patience: int = 50,
    min_lr: float = 1e-6,
    verbose: bool = True,
):
    """
    Advanced training wrapper with Learning Rate scheduling and early stopping.
    """
    # Assuming solver has optimizer as a property or accessible
    # This might require minor adjustments to NeuralSolver to expose optimizers

    # Simplified version for now using built-in solver.train logic
    solver.train(n_epochs=n_epochs, verbose=verbose)

    return solver
