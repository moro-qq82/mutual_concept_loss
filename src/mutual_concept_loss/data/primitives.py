"""Primitive operations for constructing synthetic grid tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple, TypeVar

import torch

GridTensor = torch.Tensor


@dataclass(frozen=True)
class PrimitiveContext:
    """Context describing the grid configuration."""

    grid_size: int
    num_colors: int


ParamT = TypeVar("ParamT")


@dataclass(frozen=True)
class PrimitiveDefinition:
    """Definition of a grid primitive."""

    name: str
    sampler: Callable[[GridTensor, torch.Generator, PrimitiveContext], ParamT]
    apply_fn: Callable[[GridTensor, ParamT], GridTensor]

    def sample(self, grid: GridTensor, rng: torch.Generator, context: PrimitiveContext) -> "PrimitiveInstance":
        params = self.sampler(grid, rng, context)
        return PrimitiveInstance(definition=self, params=params)

    def apply(self, grid: GridTensor, params: ParamT) -> GridTensor:
        return self.apply_fn(grid, params)


@dataclass(frozen=True)
class PrimitiveInstance:
    """Concrete primitive with frozen parameters."""

    definition: PrimitiveDefinition
    params: ParamT

    def apply(self, grid: GridTensor) -> GridTensor:
        return self.definition.apply(grid, self.params)

    @property
    def name(self) -> str:
        return self.definition.name


@dataclass(frozen=True)
class RotationParams:
    k: int


@dataclass(frozen=True)
class ColorSwapParams:
    source: int
    target: int


@dataclass(frozen=True)
class BorderParams:
    color: int


@dataclass(frozen=True)
class ShiftParams:
    dx: int
    dy: int


@dataclass(frozen=True)
class FillComponentParams:
    source_color: int
    fill_color: int


@dataclass(frozen=True)
class FlipParams:
    horizontal: bool


def _sample_rotation(_: GridTensor, rng: torch.Generator, __: PrimitiveContext) -> RotationParams:
    k = int(torch.randint(1, 4, (1,), generator=rng))
    return RotationParams(k=k)


def _apply_rotation(grid: GridTensor, params: RotationParams) -> GridTensor:
    return torch.rot90(grid, k=params.k, dims=(0, 1))


def _sample_flip(_: GridTensor, rng: torch.Generator, __: PrimitiveContext) -> FlipParams:
    horizontal = bool(torch.randint(0, 2, (1,), generator=rng))
    return FlipParams(horizontal=horizontal)


def _apply_flip(grid: GridTensor, params: FlipParams) -> GridTensor:
    if params.horizontal:
        return grid.flip(dims=(1,))
    return grid.flip(dims=(0,))


def _sample_color_swap(_: GridTensor, rng: torch.Generator, context: PrimitiveContext) -> ColorSwapParams:
    source = int(torch.randint(0, context.num_colors, (1,), generator=rng))
    candidates = [c for c in range(context.num_colors) if c != source]
    target_index = int(torch.randint(0, len(candidates), (1,), generator=rng))
    target = int(candidates[target_index])
    return ColorSwapParams(source=source, target=target)


def _apply_color_swap(grid: GridTensor, params: ColorSwapParams) -> GridTensor:
    swapped = grid.clone()
    source_channel = swapped[..., params.source].clone()
    target_channel = swapped[..., params.target].clone()
    swapped[..., params.source] = target_channel
    swapped[..., params.target] = source_channel
    return swapped


def _sample_border(_: GridTensor, rng: torch.Generator, context: PrimitiveContext) -> BorderParams:
    color = int(torch.randint(0, context.num_colors, (1,), generator=rng))
    return BorderParams(color=color)


def _apply_border(grid: GridTensor, params: BorderParams) -> GridTensor:
    painted = grid.clone()
    painted[0, :, :] = 0
    painted[-1, :, :] = 0
    painted[:, 0, :] = 0
    painted[:, -1, :] = 0
    painted[0, :, params.color] = 1
    painted[-1, :, params.color] = 1
    painted[:, 0, params.color] = 1
    painted[:, -1, params.color] = 1
    return painted


def _sample_shift(_: GridTensor, rng: torch.Generator, __: PrimitiveContext) -> ShiftParams:
    dx = int(torch.randint(-2, 3, (1,), generator=rng))
    dy = int(torch.randint(-2, 3, (1,), generator=rng))
    if dx == 0 and dy == 0:
        dx = 1
    return ShiftParams(dx=dx, dy=dy)


def _apply_shift(grid: GridTensor, params: ShiftParams) -> GridTensor:
    shifted = torch.zeros_like(grid)
    h, w, _ = grid.shape
    src_x0 = max(0, -params.dx)
    src_y0 = max(0, -params.dy)
    src_x1 = min(w, w - params.dx) if params.dx >= 0 else w
    src_y1 = min(h, h - params.dy) if params.dy >= 0 else h
    dst_x0 = max(0, params.dx)
    dst_y0 = max(0, params.dy)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    shifted[dst_y0:dst_y1, dst_x0:dst_x1, :] = grid[src_y0:src_y1, src_x0:src_x1, :]
    return shifted


def _sample_fill_component(grid: GridTensor, rng: torch.Generator, context: PrimitiveContext) -> FillComponentParams:
    present = grid.argmax(dim=-1)
    unique_colors = torch.unique(present)
    source_idx = int(torch.randint(0, len(unique_colors), (1,), generator=rng))
    source_color = int(unique_colors[source_idx])
    candidates = [c for c in range(context.num_colors) if c != source_color]
    fill_idx = int(torch.randint(0, len(candidates), (1,), generator=rng))
    fill_color = int(candidates[fill_idx])
    return FillComponentParams(source_color=source_color, fill_color=fill_color)


def _apply_fill_component(grid: GridTensor, params: FillComponentParams) -> GridTensor:
    mask = grid[..., params.source_color] > 0.5
    if not mask.any():
        return grid.clone()
    component = _largest_component(mask)
    if component is None:
        return grid.clone()
    painted = grid.clone()
    y_idx, x_idx = component
    painted[y_idx, x_idx, :] = 0
    painted[y_idx, x_idx, params.fill_color] = 1
    return painted


def _largest_component(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor] | None:
    h, w = mask.shape
    visited = torch.zeros((h, w), dtype=torch.bool)
    best_component: List[Tuple[int, int]] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            component = _collect_component(mask, visited, x, y)
            if len(component) > len(best_component):
                best_component = component
    if not best_component:
        return None
    y_coords, x_coords = zip(*best_component)
    return torch.tensor(y_coords, dtype=torch.long), torch.tensor(x_coords, dtype=torch.long)


def _collect_component(mask: torch.Tensor, visited: torch.Tensor, start_x: int, start_y: int) -> List[Tuple[int, int]]:
    stack = [(start_x, start_y)]
    component: List[Tuple[int, int]] = []
    while stack:
        x, y = stack.pop()
        if visited[y, x] or not mask[y, x]:
            continue
        visited[y, x] = True
        component.append((y, x))
        neighbors = ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
        for nx, ny in neighbors:
            if 0 <= nx < mask.shape[1] and 0 <= ny < mask.shape[0]:
                if not visited[ny, nx] and mask[ny, nx]:
                    stack.append((nx, ny))
    return component


DEFAULT_PRIMITIVES: Sequence[PrimitiveDefinition] = (
    PrimitiveDefinition(name="ROT90", sampler=_sample_rotation, apply_fn=_apply_rotation),
    PrimitiveDefinition(name="FLIP", sampler=_sample_flip, apply_fn=_apply_flip),
    PrimitiveDefinition(name="COLOR_SWAP", sampler=_sample_color_swap, apply_fn=_apply_color_swap),
    PrimitiveDefinition(name="DRAW_BORDER", sampler=_sample_border, apply_fn=_apply_border),
    PrimitiveDefinition(name="SHIFT", sampler=_sample_shift, apply_fn=_apply_shift),
    PrimitiveDefinition(name="FILL_COMPONENT", sampler=_sample_fill_component, apply_fn=_apply_fill_component),
)

