"""Loss module for the topobenchmarkx package."""

import torch
import torch_geometric

from topobenchmarkx.loss.base import AbstractLoss
from topobenchmarkx.loss.custom_losses import DatasetLoss


class TBXLoss(AbstractLoss):
    r"""Defines the default model loss for the given task.

    Parameters
    ----------
    dataset_loss : dict
        Dictionary containing the dataset loss information.
    model_loss : AbstractLoss, optional
        Custom model loss to be used.
    """

    def __init__(self, dataset_loss, model_loss=None):
        super().__init__()
        self.losses = []
        # Dataset loss
        self.losses.append(DatasetLoss(dataset_loss))
        # Model loss
        if model_loss is not None:
            self.losses.append(model_loss)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(task={self.task}, loss_type={self.loss_type})"

    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Forward pass of the loss function.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing the model output with the loss.
        """
        losses = [loss(model_out, batch) for loss in self.losses]

        model_out["loss"] = torch.stack(losses).sum()

        return model_out
