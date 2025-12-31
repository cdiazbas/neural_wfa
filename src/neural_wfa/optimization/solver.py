import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Optional

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
        device: torch.device = None
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
        
        # Wait, check baseline_neural.py:
        # blos = nfmodel_blos(coords) * wfamodel.Vnorm
        # wfamodel.Vnorm is 1 by default (checked in verify_phase2).
        # But `prepare_initial_guess` in explicit uses Vnorm=1000? NO.
        # `baseline_explicit` uses Vnorm=1000.
        # `baseline_neural` uses `wfamodel` initialized standardly.
        # Standard WFA_model3D init: self.Vnorm = 1.
        # So in Neural Baseline, Blos is effectively raw output * 1.
        # BUT the network `mlp` is initialized with sigma=40.0 (wBlos).
        # This sigma scales the Fourier frequencies, changing the "bandwidth".
        # It doesn't scale the output amplitude directly, though `beta0` is 1.
        # The output magnitude is whatever the network learns.
        
        # But `MagneticField` expects `w_blos` to denormalize if we consider network output as "normalized".
        # If network output is treated as "Raw Blos", then w_blos=1.
        # If network output is "Normalized", w_blos=40?
        # In legacy: `Blos = params[:, 0] * self.Vnorm`.
        # `nfmodel` outputs `params`.
        # So `w_blos` corresponds to `Vnorm`.
        
        # I will assume Vnorm=1, QUnorm=1000 for now, as per Legacy class defaults.
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
        verbose: bool = True
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
            verbose: If True, prints progress.
        """
        n_pixels = self.coordinates.shape[0]
        batch_size = min(self.batch_size, n_pixels)
        n_batches = (n_pixels + batch_size - 1) // batch_size
        
        # Schedulers
        sched_blos = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_blos, 'min', patience=scheduler_patience)
        sched_bqu = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_bqu, 'min', patience=scheduler_patience)
        
        # Prep Azimuth reg if provided
        if regu_azimuth is not None:
            ref_azi, _ = regu_azimuth
            sin2phi_ref = torch.sin(2 * ref_azi).to(self.device).flatten()
            cos2phi_ref = torch.cos(2 * ref_azi).to(self.device).flatten()
            
        iterator = range(n_epochs)
        if verbose:
            iterator = tqdm(iterator, desc="Epochs")
            
        for epoch in iterator:
            # Shuffle indices
            perm = torch.randperm(n_pixels, device=self.device)
            
            epoch_loss = 0.0
            
            for i in range(n_batches):
                idx = perm[i * batch_size : (i + 1) * batch_size]
                coords_batch = self.coordinates[idx]
                
                # Zero grads
                if optimize_blos: self.optimizer_blos.zero_grad()
                if optimize_bqu: self.optimizer_bqu.zero_grad()
                
                # Forward Pass Models
                # Get raw outputs
                # blos_raw: (batch, 1) usually. 
                # bqu_raw: (batch, 2)
                blos_raw = self.model_blos(coords_batch)
                bqu_raw = self.model_bqu(coords_batch)
                
                # Construct MagneticField
                field = MagneticField(
                    blos_raw, 
                    bqu_raw, 
                    w_blos=self.w_blos_norm, 
                    w_bqu=self.w_bqu_norm
                )
                
                # We need to construct a subset Observation for loss?
                # WFAProblem uses full self.obs.
                # WFAProblem.compute_loss computes DiffQ = abs(Obs.Q - Model.Q).
                # Obs.Q is full image. Model.Q is batch.
                # We need WFAProblem to accept indices or handle batching!
                # I missed adding batching support to WFAProblem.
                # WFAProblem.compute_loss calculates residuals. 
                # If field is batch, model outputs batch.
                # Obs must be sliced to batch.
                
                # FIX: WFAProblem.compute_loss needs 'indices' argument.
                # Or WFAProblem should be lightweight and we pass sliced Obs data?
                # WFAProblem holds Obs.
                # Let's add 'indices' to compute_loss.
                
                # Calculate Loss
                loss = self.problem.compute_loss(
                    field, 
                    mask_blos=optimize_blos, 
                    mask_bqu=optimize_bqu,
                    indices=idx 
                )
                
                # Additional Regularizations
                if regu_potential is not None and optimize_bqu:
                    Bq_ref, Bu_ref, weight = regu_potential
                    loss += weight * torch.mean((bqu_raw[:, 0] - Bq_ref.to(self.device).flatten()[idx])**2)
                    loss += weight * torch.mean((bqu_raw[:, 1] - Bu_ref.to(self.device).flatten()[idx])**2)
                    
                if regu_azimuth is not None and optimize_bqu:
                    _, weight = regu_azimuth
                    # Bt for azimuth calculation (normalized)
                    bt_norm = torch.sqrt(bqu_raw[:, 0]**2 + bqu_raw[:, 1]**2 + 1e-9)
                    sin2phi = bqu_raw[:, 1] / bt_norm
                    cos2phi = bqu_raw[:, 0] / bt_norm
                    loss += weight * torch.mean((sin2phi - sin2phi_ref[idx])**2)
                    loss += weight * torch.mean((cos2phi - cos2phi_ref[idx])**2)

                loss.backward()
                
                # Gradient Normalization
                if normalize_gradients:
                    if optimize_blos:
                        for p in self.model_blos.parameters():
                            if p.grad is not None: p.grad /= (torch.mean(torch.abs(p.grad)) + 1e-9)
                    if optimize_bqu:
                        for p in self.model_bqu.parameters():
                            if p.grad is not None: p.grad /= (torch.mean(torch.abs(p.grad)) + 1e-9)

                if optimize_blos: self.optimizer_blos.step()
                if optimize_bqu: self.optimizer_bqu.step()
                
                epoch_loss += loss.item()
                
            self.loss_history.append(epoch_loss / n_batches)
            
            # Step Schedulers
            if optimize_blos: sched_blos.step(epoch_loss / n_batches)
            if optimize_bqu: sched_bqu.step(epoch_loss / n_batches)
            
            if verbose:
                iterator.set_postfix(loss=epoch_loss / n_batches, lr=self.optimizer_blos.param_groups[0]['lr'])

    def get_full_field(self) -> MagneticField:
        """Evaluates models on full coordinates and returns MagneticField."""
        with torch.no_grad():
            # Process in chunks if too large? 
            # For 200x200 it fits in memory.
            blos_full = self.model_blos(self.coordinates)
            bqu_full = self.model_bqu(self.coordinates)
            
            return MagneticField(
                blos_full, 
                bqu_full, 
                w_blos=self.w_blos_norm, 
                w_bqu=self.w_bqu_norm
            )
