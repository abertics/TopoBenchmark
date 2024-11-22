"""Unit tests for GraphMLP."""

import torch
import torch_geometric
from topobenchmarkx.nn.backbones.graph import GraphMLP
from topobenchmarkx.nn.wrappers.graph import GraphMLPWrapper
from topobenchmarkx.loss.model import GraphMLPLoss

def testGraphMLP(random_graph_input):
    """ Unit test for GraphMLP.
    
    Parameters
    ----------
    random_graph_input : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        A tuple of input tensors for testing EDGNN.
    """
    x, x_1, x_2, edges_1, edges_2 = random_graph_input
    batch = torch_geometric.data.Data(x_0=x, y=x, edge_index=edges_1, batch_0=torch.zeros(x.shape[0], dtype=torch.long))
    model = GraphMLP(x.shape[1], x.shape[1])
    wrapper = GraphMLPWrapper(model, **{"out_channels": x.shape[1], "num_cell_dimensions": 1})
    loss_fn = GraphMLPLoss()
    
    model_out = wrapper(batch)
    assert model_out["x_0"].shape == x.shape
    assert list(model_out["x_dis"].shape) == [8,8]
    
    loss = loss_fn(model_out, batch)
    assert loss.item() >= 0
    
    
