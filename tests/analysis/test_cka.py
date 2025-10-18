from __future__ import annotations

import pytest
import torch

from mutual_concept_loss.analysis import linear_cka, pairwise_linear_cka


def test_linear_cka_identical_features_is_one() -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 4)
    y = 2.0 * x
    value = linear_cka(x, y)
    assert value == pytest.approx(1.0, rel=1e-4, abs=1e-4)


def test_pairwise_linear_cka_returns_symmetric_matrix() -> None:
    torch.manual_seed(1)
    a = torch.randn(6, 3)
    b = torch.randn(6, 3)
    names, matrix = pairwise_linear_cka({"a": a, "b": b})
    assert names == ["a", "b"]
    assert matrix.shape == (2, 2)
    assert torch.allclose(matrix, matrix.T)


def test_linear_cka_requires_matching_samples() -> None:
    x = torch.randn(4, 2)
    y = torch.randn(5, 2)
    with pytest.raises(ValueError):
        linear_cka(x, y)
