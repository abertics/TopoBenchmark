"""Loss module for human motion prediction in the topobenchmark package."""

from typing import Any

import torch
from torchmetrics import Metric


class MeanPerJointPositionError(Metric):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Define frames for different states
        self.train_frames = [2, 4, 8, 10]  # Fewer frames for training
        self.test_frames = [2, 4, 8, 10, 12, 14, 18, 22, 25]  # All frames for testing
        
        # Initialize states with maximum possible size
        self.add_state(
            "sum_errors",
            default=torch.zeros(len(self.test_frames)),  # Use max length
            dist_reduce_fx="sum",
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, state_str: str) -> None:
        # Choose frames based on state
        frames_to_evaluate = self.train_frames if state_str == "Train" else self.test_frames

        # Reshape inputs from (bs*f*j*c, 1) to (bs, f, j, c)
        batch_size = preds.shape[0] // (50 * 22 * 3)  # 50 frames, 22 joints, 3 channels
        preds = preds.view(batch_size, 50, 22, 3)
        target = target.view(batch_size, 50, 22, 3)

        # Calculate MPJPE for each frame count
        for i, k in enumerate(frames_to_evaluate):
            # Extract first k frames
            preds_k = preds[:, :k, :, :]  # [batch_size, k, 22, 3]
            target_k = target[:, :k, :, :]  # [batch_size, k, 22, 3]

            # Convert to millimetres and get difference
            error_mm = 1000 * (preds_k - target_k)

            # Calculate euclidean distance across xyz coordinates
            euc_dist = torch.norm(error_mm, dim=3)  # [batch_size, k, 22]

            # Average error across joints and frames
            mpjpe = torch.mean(euc_dist, dim=(1, 2))  # [batch_size]

            # Sum across batch (to be divided by total later)
            self.sum_errors[i] += torch.sum(mpjpe)

        self.total += batch_size

    def compute(self) -> dict:
        """Compute the final metrics.

        Returns
        -------
        dict
            Dictionary of MPJPE values for different frame counts in millimeters.
        """
        mpjpe_values = self.sum_errors / self.total
        
        # Find which frames were used by looking at non-zero values
        # Test frames array is always used for initialization, so we need to check which values were actually updated
        active_indices = torch.nonzero(self.sum_errors).squeeze()
        active_frames = (self.train_frames if len(active_indices) == len(self.train_frames) 
                        else self.test_frames)
        return {
            f"mpjpe_{k:02d}_frames": value.item()
            for k, value in zip(active_frames, mpjpe_values[:len(active_frames)])
        }