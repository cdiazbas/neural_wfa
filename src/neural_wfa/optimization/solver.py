import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Dict, Tuple, Optional

from neural_wfa.core.problem import WFAProblem
from neural_wfa.core.magnetic_field import MagneticField


class NeuralSolver:
    """
    Solver for Neural WFA Inversion.

    Optimizes Neural Fields (MLPs) to minimize the WFA loss.
    """

    def __init__(
        self,
        problem: WFAProblem,
        model_blos: nn.Module,
        model_bqu: nn.Module,
        coordinates: torch.Tensor,
        lr: float = 1e-3,
        batch_size: int = 4096,
        device: torch.device = None,
    ):
        self.problem = problem
        self.model_blos = model_blos
        self.model_bqu = model_bqu
        self.coordinates = coordinates
        self.batch_size = batch_size
        self.device = device if device else problem.device

        # Move models to device
        self.model_blos.to(self.device)
        self.model_bqu.to(self.device)
        self.coordinates = self.coordinates.to(self.device)

        # Optimizers (Separate or Joint? Usually separate allow flexible rates/freezing)
        self.optimizer_blos = optim.Adam(self.model_blos.parameters(), lr=lr)
        self.optimizer_bqu = optim.Adam(self.model_bqu.parameters(), lr=lr)

        self.loss_history = []
        self.lr_history = []

        # Grid dimensions for selective evaluation
        grid_shape = self.problem.obs.grid_shape
        if len(grid_shape) == 3:  # (Nt, Ny, Nx)
            self.nt, self.ny, self.nx = grid_shape
        else:  # (Ny, Nx)
            self.nt = 1
            self.ny, self.nx = grid_shape

        # Normalization
        self.w_blos_norm = 1.0
        self.w_bqu_norm = 1000.0

    def set_normalization(self, w_blos: float, w_bqu: float):
        self.w_blos_norm = w_blos
        self.w_bqu_norm = w_bqu

    def train(
        self,
        n_epochs: int = 100,
        optimize_blos: bool = True,
        optimize_bqu: bool = True,
        scheduler_patience: int = 50,
        normalize_gradients: bool = True,
        regu_potential: Optional[Tuple[torch.Tensor, torch.Tensor, float]] = None,
        regu_azimuth: Optional[Tuple[torch.Tensor, float]] = None,
        n_active_levels: Optional[int] = None,
        progressive_schedule: Optional[Dict[int, int]] = None,
        verbose: bool = True,
    ):
        """
        Runs the optimization loop.

        Args:
            n_epochs: Total number of epochs.
            optimize_blos: Whether to update Blos model.
            optimize_bqu: Whether to update BQU model.
            scheduler_patience: Patience for ReduceLROnPlateau.
            normalize_gradients: If True, scales gradients by their mean absolute value.
            regu_potential: (Bq_ref, Bu_ref, weight) for potential field regularization.
            regu_azimuth: (azimuth_ref, weight) for azimuth regularization.
            n_active_levels: If provided, sets number of active levels in encoder.
            progressive_schedule: Dictionary mapping epoch -> n_active_levels.
            verbose: If True, prints progress.
        """
        n_pixels = self.coordinates.shape[0]
        batch_size = min(self.batch_size, n_pixels)
        n_batches = (n_pixels + batch_size - 1) // batch_size

        # Schedulers
        sched_blos = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_blos, "min", patience=scheduler_patience
        )
        sched_bqu = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_bqu, "min", patience=scheduler_patience
        )

        # Prep Azimuth reg if provided
        if regu_azimuth is not None:
            ref_azi, _ = regu_azimuth
            sin2phi_ref = torch.sin(2 * ref_azi).to(self.device).flatten()
            cos2phi_ref = torch.cos(2 * ref_azi).to(self.device).flatten()

        iterator = range(n_epochs)
        if verbose:
            iterator = tqdm(iterator, desc="Epochs")

        for epoch in iterator:
            # Progressive Level Activation (V4)
            if progressive_schedule is not None and epoch in progressive_schedule:
                levels = progressive_schedule[epoch]
                if hasattr(self.model_blos, "encoder") and hasattr(
                    self.model_blos.encoder, "set_active_levels"
                ):
                    self.model_blos.encoder.set_active_levels(levels)
                if hasattr(self.model_bqu, "encoder") and hasattr(
                    self.model_bqu.encoder, "set_active_levels"
                ):
                    self.model_bqu.encoder.set_active_levels(levels)
            elif n_active_levels is not None:
                if hasattr(self.model_blos, "encoder") and hasattr(
                    self.model_blos.encoder, "set_active_levels"
                ):
                    self.model_blos.encoder.set_active_levels(n_active_levels)
                if hasattr(self.model_bqu, "encoder") and hasattr(
                    self.model_bqu.encoder, "set_active_levels"
                ):
                    self.model_bqu.encoder.set_active_levels(n_active_levels)

            # Shuffle indices
            perm = torch.randperm(n_pixels, device=self.device)

            epoch_loss = 0.0

            for i in range(n_batches):
                idx = perm[i * batch_size : (i + 1) * batch_size]
                coords_batch = self.coordinates[idx]

                # Zero grads
                if optimize_blos:
                    self.optimizer_blos.zero_grad()
                if optimize_bqu:
                    self.optimizer_bqu.zero_grad()

                # Forward Pass Models
                blos_raw = self.model_blos(coords_batch)
                bqu_raw = self.model_bqu(coords_batch)

                # Construct MagneticField
                field = MagneticField(
                    blos_raw, bqu_raw, w_blos=self.w_blos_norm, w_bqu=self.w_bqu_norm
                )

                # Calculate Loss
                loss = self.problem.compute_loss(
                    field,
                    active_blos=optimize_blos,
                    active_bqu=optimize_bqu,
                    indices=idx,
                )

                # Additional Regularizations
                if regu_potential is not None and optimize_bqu:
                    Bq_ref, Bu_ref, weight = regu_potential
                    loss += weight * torch.mean(
                        (bqu_raw[:, 0] - Bq_ref.to(self.device).flatten()[idx]) ** 2
                    )
                    loss += weight * torch.mean(
                        (bqu_raw[:, 1] - Bu_ref.to(self.device).flatten()[idx]) ** 2
                    )

                if regu_azimuth is not None and optimize_bqu:
                    _, weight = regu_azimuth
                    # Bt for azimuth calculation (normalized)
                    bt_norm = torch.sqrt(bqu_raw[:, 0] ** 2 + bqu_raw[:, 1] ** 2 + 1e-9)
                    sin2phi = bqu_raw[:, 1] / bt_norm
                    cos2phi = bqu_raw[:, 0] / bt_norm
                    loss += weight * torch.mean((sin2phi - sin2phi_ref[idx]) ** 2)
                    loss += weight * torch.mean((cos2phi - cos2phi_ref[idx]) ** 2)

                loss.backward()

                # Gradient Normalization
                if normalize_gradients:
                    if optimize_blos:
                        for p in self.model_blos.parameters():
                            if p.grad is not None:
                                p.grad /= torch.mean(torch.abs(p.grad)) + 1e-9
                    if optimize_bqu:
                        for p in self.model_bqu.parameters():
                            if p.grad is not None:
                                p.grad /= torch.mean(torch.abs(p.grad)) + 1e-9

                if optimize_blos:
                    self.optimizer_blos.step()
                if optimize_bqu:
                    self.optimizer_bqu.step()

                epoch_loss += loss.item()

            self.loss_history.append(epoch_loss / n_batches)

            # Step Schedulers
            if optimize_blos:
                sched_blos.step(epoch_loss / n_batches)
            if optimize_bqu:
                sched_bqu.step(epoch_loss / n_batches)

            # Track LR (take first group of active optimizer)
            current_lr = 0.0
            if optimize_blos:
                current_lr = self.optimizer_blos.param_groups[0]["lr"]
            elif optimize_bqu:
                current_lr = self.optimizer_bqu.param_groups[0]["lr"]
            self.lr_history.append(current_lr)

            if verbose:
                iterator.set_postfix(loss=epoch_loss / n_batches, lr=current_lr)

    def get_full_field(self, t=None, y=None, x=None) -> MagneticField:
        """
        Evaluates models on selected coordinates and returns MagneticField.

        Args:
            t: Time index (int), list of indices, or None for all times.
            y: Y pixel index (int), list of indices, or None for all y.
            x: X pixel index (int), list of indices, or None for all x.

        Returns:
            MagneticField with shape determined by selection:
            - No args: (Nt, Ny, Nx)
            - t=5: (Ny, Nx)
            - y=100, x=100: (Nt,)
            - t=5, y=100, x=100: scalar

        Examples:
            field = solver.get_full_field()             # Full grid
            field = solver.get_full_field(y=100, x=100) # Pixel time series
            field = solver.get_full_field(t=5)          # Single frame
        """
        import numpy as np

        # Determine ranges for each dimension
        t_range = (
            [t]
            if isinstance(t, int)
            else (t if t is not None else list(range(self.nt)))
        )
        y_range = (
            [y]
            if isinstance(y, int)
            else (y if y is not None else list(range(self.ny)))
        )
        x_range = (
            [x]
            if isinstance(x, int)
            else (x if x is not None else list(range(self.nx)))
        )

        # If querying full grid, use pre-computed coordinates for speed
        if t is None and y is None and x is None:
            with torch.no_grad():
                blos_full = self.model_blos(self.coordinates)
                bqu_full = self.model_bqu(self.coordinates)

                return MagneticField(
                    blos_full,
                    bqu_full,
                    w_blos=self.w_blos_norm,
                    w_bqu=self.w_bqu_norm,
                    grid_shape=self.problem.obs.grid_shape,
                )

        # Build selected coordinates (normalized -1 to 1)
        t_norm = np.linspace(-1, 1, self.nt) if self.nt > 1 else np.array([0.0])
        y_norm = np.linspace(-1, 1, self.ny)
        x_norm = np.linspace(-1, 1, self.nx)

        # Create meshgrid for selected indices
        t_sel = t_norm[t_range]
        y_sel = y_norm[y_range]
        x_sel = x_norm[x_range]

        if self.nt > 1:
            TT, YY, XX = np.meshgrid(t_sel, y_sel, x_sel, indexing="ij")
            coords = np.stack([TT, YY, XX], axis=-1).reshape(-1, 3)
        else:
            YY, XX = np.meshgrid(y_sel, x_sel, indexing="ij")
            coords = np.stack([YY, XX], axis=-1).reshape(-1, 2)

        coords_tensor = torch.from_numpy(coords.astype(np.float32)).to(self.device)

        # Evaluate models
        with torch.no_grad():
            blos = self.model_blos(coords_tensor)
            bqu = self.model_bqu(coords_tensor)

        # Determine output grid shape
        out_shape = (len(t_range), len(y_range), len(x_range))
        # Squeeze singleton dimensions
        out_shape = tuple(s for s in out_shape if s > 1) or (1,)

        return MagneticField(
            blos,
            bqu,
            w_blos=self.w_blos_norm,
            w_bqu=self.w_bqu_norm,
            grid_shape=out_shape,
        )
