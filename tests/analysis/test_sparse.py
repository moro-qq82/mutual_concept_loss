from __future__ import annotations

import torch

from mutual_concept_loss.analysis import summarize_sparse_codes


def test_summarize_sparse_codes_shapes() -> None:
    codes = torch.tensor([[0.5, 0.0, 1.0], [0.0, 0.2, 0.0], [0.1, 0.0, 0.3]])
    primitives = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    summary = summarize_sparse_codes(codes, primitives, activation_threshold=0.05)
    assert summary.mean_activation.shape == (3,)
    assert summary.active_fraction.shape == (3,)
    assert summary.code_primitive_correlation.shape == (3, 2)
