import torch_geometric
from torch_geometric.nn import radius_graph


class InfereRadiusConnectivity(torch_geometric.transforms.BaseTransform):
    r"""A transform that generates the radius connectivity of the input point
    cloud.

    Args:
        kwargs (optional): Parameters for the base transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "infere_radius_connectivity"
        self.parameters = kwargs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r}, parameters={self.parameters!r})"

    def forward(self, data: torch_geometric.data.Data):
        r"""Apply the transform to the input data.

        Args:
            data (torch_geometric.data.Data): The input data.
        Returns:
            torch_geometric.data.Data: The transformed data.
        """
        data.edge_index = radius_graph(data.x, **self.parameters["args"])
        return data
