"""Synthetic dataset generator built on top of primitive operations."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .primitives import DEFAULT_PRIMITIVES, PrimitiveContext, PrimitiveDefinition

GridTensor = torch.Tensor


@dataclass(frozen=True)
class GeneratorConfig:
    """Configuration for generating synthetic grid tasks."""

    grid_size: int = 8
    num_colors: int = 4
    min_composition_length: int = 1
    max_composition_length: int = 2
    min_base_shapes: int = 1
    max_base_shapes: int = 3
    background_color: int = 0

    def validate(self) -> None:
        if self.grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if self.num_colors <= 1:
            raise ValueError("num_colors must be greater than 1")
        if self.min_composition_length <= 0:
            raise ValueError("min_composition_length must be positive")
        if self.max_composition_length < self.min_composition_length:
            raise ValueError("max_composition_length must be >= min_composition_length")
        if self.min_base_shapes <= 0:
            raise ValueError("min_base_shapes must be positive")
        if self.max_base_shapes < self.min_base_shapes:
            raise ValueError("max_base_shapes must be >= min_base_shapes")
        if not 0 <= self.background_color < self.num_colors:
            raise ValueError("background_color must be within the available colors")


class SyntheticTaskGenerator:
    """Generates synthetic grid transformation tasks."""

    def __init__(
        self,
        config: GeneratorConfig | None = None,
        primitives: Sequence[PrimitiveDefinition] | None = None,
    ) -> None:
        self.config = config or GeneratorConfig()
        self.config.validate()
        self.primitives = tuple(primitives or DEFAULT_PRIMITIVES)
        if not self.primitives:
            raise ValueError("At least one primitive must be provided")
        self.context = PrimitiveContext(
            grid_size=self.config.grid_size,
            num_colors=self.config.num_colors,
        )

    def generate(
        self, rng: torch.Generator
    ) -> Tuple[GridTensor, GridTensor, torch.Tensor, Tuple[int, ...]]:
        input_grid = self._create_base_grid(rng)
        target_grid = input_grid.clone()
        multi_hot = torch.zeros(len(self.primitives), dtype=torch.float32)
        sequence_indices: list[int] = []
        length = int(
            torch.randint(
                self.config.min_composition_length,
                self.config.max_composition_length + 1,
                (1,),
                generator=rng,
            )
        )
        for _ in range(length):
            primitive_index = int(torch.randint(0, len(self.primitives), (1,), generator=rng))
            primitive = self.primitives[primitive_index]
            instance = primitive.sample(target_grid, rng, self.context)
            target_grid = instance.apply(target_grid)
            multi_hot[primitive_index] = 1.0
            sequence_indices.append(primitive_index)
        return input_grid, target_grid, multi_hot, tuple(sequence_indices)

    def _create_base_grid(self, rng: torch.Generator) -> GridTensor:
        size = self.config.grid_size
        colors = self.config.num_colors
        background = torch.zeros((size, size, colors), dtype=torch.float32)
        background[..., self.config.background_color] = 1.0
        grid = background.clone()
        num_shapes = int(
            torch.randint(
                self.config.min_base_shapes,
                self.config.max_base_shapes + 1,
                (1,),
                generator=rng,
            )
        )
        for _ in range(num_shapes):
            color = int(torch.randint(0, colors, (1,), generator=rng))
            y0 = int(torch.randint(0, size, (1,), generator=rng))
            y1 = int(torch.randint(y0 + 1, size + 1, (1,), generator=rng))
            x0 = int(torch.randint(0, size, (1,), generator=rng))
            x1 = int(torch.randint(x0 + 1, size + 1, (1,), generator=rng))
            grid[y0:y1, x0:x1, :] = 0
            grid[y0:y1, x0:x1, color] = 1
        return grid


class SyntheticTaskDataset(Dataset[Dict[str, torch.Tensor]]):
    """Dataset that exposes synthetic grid transformation samples."""

    def __init__(
        self,
        num_samples: int,
        config: GeneratorConfig | None = None,
        *,
        primitives: Sequence[PrimitiveDefinition] | None = None,
        seed: int = 0,
        buffer_size: int = 0,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")
        self.num_samples = num_samples
        self.generator = SyntheticTaskGenerator(config=config, primitives=primitives)
        self.seed = seed
        self.buffer_size = max(0, buffer_size)
        self._cache: OrderedDict[int, Dict[str, torch.Tensor]] = OrderedDict()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if index < 0 or index >= self.num_samples:
            raise IndexError("index out of range")
        cached = self._cache.get(index)
        if cached is not None:
            return self._clone_sample(cached)
        rng = torch.Generator()
        rng.manual_seed(self.seed + index)
        input_grid, target_grid, multi_hot, sequence_indices = self.generator.generate(rng)
        sample: Dict[str, torch.Tensor] = {
            "input": input_grid,
            "target": target_grid,
            "primitives": multi_hot,
            "primitive_indices": torch.tensor(sequence_indices, dtype=torch.int64),
        }
        sample["sequence_length"] = torch.tensor(len(sequence_indices), dtype=torch.int64)
        self._remember(index, sample)
        return self._clone_sample(sample)

    def _remember(self, index: int, sample: Dict[str, torch.Tensor]) -> None:
        if self.buffer_size == 0:
            return
        while len(self._cache) >= self.buffer_size:
            self._cache.popitem(last=False)
        self._cache[index] = {key: value.clone() for key, value in sample.items()}

    @staticmethod
    def _clone_sample(sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: value.clone() for key, value in sample.items()}

