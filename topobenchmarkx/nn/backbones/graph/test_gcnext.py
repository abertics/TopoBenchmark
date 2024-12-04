"""Unit tests for gcnext."""

import pytest
import torch

from topobenchmarkx.nn.backbones.graph.gcnext import (
    SkeletalConvolution,
    H36MSkeleton,
)

import numpy as np


def test_skeletal_convolution_equivalence():
    """Test that SkeletalConvolution forward pass matches reference implementation."""
    # Setup test parameters
    batch_size = 4
    num_joints = 22
    channels = 3
    seq_len = 50
    vertices = num_joints * channels

    # Create random input tensor
    x = torch.randn(batch_size, vertices, seq_len)

    # Current implementation output
    model = SkeletalConvolution()
    output_current = model(x)

    ## COPIED FROM GCNEXT ##
    skl = H36MSkeleton().skl_mask.clone().detach()
    bi_skl = torch.zeros(22, 22, requires_grad=False)
    bi_skl[skl != 0] = 1.0
    skl_mask = bi_skl

    # Reference implementation
    b, v, t = x.shape
    x1 = x.reshape(b, v // 3, 3, t)
    x1 = torch.einsum("vj,bjct->bvct", model.weights.mul(skl_mask), x1)
    output_reference = x1.reshape(b, v, t)

    # Check outputs match
    assert torch.allclose(
        output_current, output_reference, rtol=1e-5, atol=1e-5
    ), "Current and reference implementations produce different outputs"
