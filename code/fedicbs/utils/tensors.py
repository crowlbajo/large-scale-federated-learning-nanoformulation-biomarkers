from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor


def move_to_device(value: object, device: torch.device) -> object:
    if isinstance(value, Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        values = {
            name: move_to_device(getattr(value, name), device)
            for name in value.__dataclass_fields__
        }
        return type(value)(**values)
    return value


def detach_state(state: dict[str, Tensor]) -> OrderedDict[str, Tensor]:
    return OrderedDict(
        (name, value.detach().cpu().clone())
        for name, value in state.items()
    )


def finite_tensor(tensor: Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"non-finite values detected in {name}")


def numpy_to_tensor(
    values: NDArray[np.generic],
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> Tensor:
    array = np.asarray(values)
    tensor = torch.from_numpy(array).to(dtype=dtype)
    return tensor.to(device) if device is not None else tensor


def tensor_to_numpy(tensor: Tensor) -> NDArray[np.generic]:
    return tensor.detach().cpu().numpy()


def global_norm(tensors: list[Tensor]) -> Tensor:
    if not tensors:
        return torch.zeros(())
    device = tensors[0].device
    squared = torch.zeros((), dtype=torch.float64, device=device)
    for tensor in tensors:
        squared += torch.square(tensor.to(torch.float64)).sum()
    return torch.sqrt(squared)


def parameter_vector(state: dict[str, Tensor]) -> Tensor:
    values = [
        tensor.detach().to(torch.float64).reshape(-1)
        for tensor in state.values()
        if tensor.is_floating_point()
    ]
    if not values:
        return torch.empty(0, dtype=torch.float64)
    return torch.cat(values)


def state_checksum(state: dict[str, Tensor]) -> float:
    vector = parameter_vector(state)
    if vector.numel() == 0:
        return 0.0
    indices = torch.arange(1, vector.numel() + 1, dtype=torch.float64)
    return float(torch.dot(vector.cpu(), indices).item())


def split_batches(tensor: Tensor, batch_size: int) -> tuple[Tensor, ...]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return tuple(tensor[index : index + batch_size] for index in range(0, len(tensor), batch_size))
