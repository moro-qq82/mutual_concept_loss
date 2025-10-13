"""Custom samplers tailored for the synthetic grid dataset."""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, Iterator, List, Sequence, Tuple

import torch
from torch.utils.data import Sampler


class GroupedBatchSampler(Sampler[List[int]]):
    """Batch sampler that groups indices by identical multi-hot primitive labels."""

    def __init__(
        self,
        dataset: Sequence,
        batch_size: int,
        *,
        drop_last: bool = False,
        generator: torch.Generator | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self._dataset = dataset
        self._batch_size = batch_size
        self._drop_last = drop_last
        self._generator = generator

    def __iter__(self) -> Iterator[List[int]]:
        if len(self._dataset) == 0:
            return iter(())
        order = self._iteration_order(len(self._dataset))
        buckets: DefaultDict[Tuple[int, ...], List[int]] = defaultdict(list)
        for idx in order:
            sample = self._dataset[idx]
            label = tuple(int(v) for v in sample["primitives"].tolist())
            bucket = buckets[label]
            bucket.append(idx)
            if len(bucket) == self._batch_size:
                yield list(bucket)
                buckets[label] = []
        if not self._drop_last:
            for bucket in buckets.values():
                if bucket:
                    yield list(bucket)

    def __len__(self) -> int:
        if len(self._dataset) == 0:
            return 0
        counts: Dict[Tuple[int, ...], int] = defaultdict(int)
        for idx in range(len(self._dataset)):
            sample = self._dataset[idx]
            label = tuple(int(v) for v in sample["primitives"].tolist())
            counts[label] += 1
        total_batches = 0
        for count in counts.values():
            total_batches += count // self._batch_size
            if not self._drop_last and count % self._batch_size:
                total_batches += 1
        return total_batches

    def _iteration_order(self, length: int) -> Iterable[int]:
        if self._generator is None:
            return range(length)
        permutation = torch.randperm(length, generator=self._generator)
        return permutation.tolist()

