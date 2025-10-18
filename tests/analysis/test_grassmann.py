from __future__ import annotations

import pytest
import torch

from mutual_concept_loss.analysis import (
    compute_principal_components,
    grassmann_distance,
    pairwise_grassmann_distances,
)


def test_compute_principal_components_returns_orthonormal_basis() -> None:
    torch.manual_seed(2)
    tensor = torch.randn(10, 6)
    basis = compute_principal_components(tensor, k=4)
    assert basis.shape == (6, 4)
    gram = basis.T @ basis
    assert torch.allclose(gram, torch.eye(4), atol=1e-4)


def test_grassmann_distance_zero_for_identical_subspaces() -> None:
    torch.manual_seed(3)
    tensor = torch.randn(12, 5)
    basis = compute_principal_components(tensor, k=3)
    distance = grassmann_distance(basis, basis)
    assert distance == pytest.approx(0.0, abs=1e-6)


def test_pairwise_grassmann_distances_shape() -> None:
    torch.manual_seed(4)
    tensors = [torch.randn(8, 5), torch.randn(8, 5)]
    names, matrix = pairwise_grassmann_distances({"a": tensors[0], "b": tensors[1]}, k=3)
    assert names == ["a", "b"]
    assert matrix.shape == (2, 2)
    diag = matrix.diagonal()
    assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-6)
