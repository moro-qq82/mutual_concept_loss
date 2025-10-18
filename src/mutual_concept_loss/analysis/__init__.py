"""Utilities for representation analysis in Phase 5."""

from .collection import RepresentationCollection, collect_representations, group_by_label
from .cka import linear_cka, pairwise_linear_cka
from .grassmann import compute_principal_components, grassmann_distance, pairwise_grassmann_distances
from .sparse import SparseSummary, summarize_sparse_codes

__all__ = [
    "RepresentationCollection",
    "collect_representations",
    "group_by_label",
    "linear_cka",
    "pairwise_linear_cka",
    "compute_principal_components",
    "grassmann_distance",
    "pairwise_grassmann_distances",
    "SparseSummary",
    "summarize_sparse_codes",
]
