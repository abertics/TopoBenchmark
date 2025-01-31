"""Evaluation callback class for human motion prediction."""

import numpy as np
import torch
import os
from lightning import Callback


class MotionEvaluationCallback(Callback):
    """Custom Callback function to evaluate human motion predictions autoregressively.
    """

    def __init__(self):
        super().__init__()
        self.frames_to_evaluate = [2, 4, 8, 10, 14, 18, 22, 25]

    # def on_train_epoch_start(self, trainer, pl_module): # for debugging
    def on_train_epoch_end(self, trainer, pl_module):
        """Calculate mpjpe at the end of each epoch."""
        motion_outputs = []
        num_samples = 0

        # TODO: want it to run through all data in test set
        batch = next(iter(trainer.train_dataloader))
        num_samples += batch.batch_size
        batch.x_0 = batch.x.to(pl_module.device)

        step = 10 # config.motion.h36m_target_length_train
        num_step = 1 if step == 25 else 25 // step + 1

        motion_input = batch.x_0.clone() # (844800, 1) 256*50*22*3

        pl_module.eval()
        for _ in range(num_step):
            batch_ = batch.clone()
            batch_.x_0 = motion_input
            batch_.x = motion_input


            # Get predictions
            with torch.no_grad():
                outputs = pl_module(batch_)
            
            model_output = outputs["x_0"].reshape(
                    256, 50, 22, 3
                )  # [B, T, J, 3]
            output = model_output[:, :10, :, :]
            motion_outputs.append(output)

            motion_input = motion_input.reshape(256, 50, 22, 3)
                
            motion_input = torch.cat([motion_input[:, step:], output], axis=1)
            motion_input = motion_input.reshape(256*50*22*3, 1)

        motion_pred = torch.cat(motion_outputs, axis=1)[:,:25]
        motion_target = outputs["labels"].reshape(
                256, 50, 22, 3
            )
        b,n,c,_ = motion_target.shape


        mpjpes = []
        # Calculate MPJPE for each frame count
        for _, k in enumerate(self.frames_to_evaluate):
            # Extract first k frames
            preds_k = motion_pred[:, :k, :, :].cuda()  # [batch_size, k, 22, 3]
            target_k = motion_target[:, :k, :, :].cuda()  # [batch_size, k, 22, 3]

            # Convert to millimetres and get difference
            error_mm = 1000 * (preds_k - target_k)

            # Calculate euclidean distance across xyz coordinates
            euc_dist = torch.norm(error_mm, dim=3)  # [batch_size, k, 22]

            # Average error across joints and frames
            mpjpe = torch.mean(euc_dist, dim=(1, 2))  # [batch_size]

            # Sum across batch (to be divided by total later)
            mpjpes.append(torch.sum(mpjpe) / num_samples)

        # Write mpjpe values to file bc i am sketch
        os.makedirs("zzevals", exist_ok=True)
        with open(f"zzevals/mpjpe_epoch-{trainer.current_epoch}.txt", "w") as f:
            for mpjpe, frame in zip(mpjpes, self.frames_to_evaluate, strict=False):
                f.write(f"Frame {frame}:\t{round(mpjpe.item(), 4)}\n")

        pl_module.train()
