"""Shared abstractions and helpers for neural architectures used in NNCME."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap

from nncme.utils import scaled_dot_product_attention


class BaseModel(nn.Module, ABC):
    """Minimal base class providing gradient utilities for autoregressive nets."""

    @abstractmethod
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Return network outputs for the supplied tensors."""

    @staticmethod
    def _loss_fn(log_probs: torch.Tensor) -> torch.Tensor:
        """Compute the mean loss across the batch."""

        return log_probs.mean(0)

    def _get_params(self) -> Dict[str, torch.Tensor]:
        """Return a detached copy of the learnable parameter tensors."""

        return {k: v.detach() for k, v in self.named_parameters()}

    def _compute_loss(self, params: Dict[str, torch.Tensor], sample: torch.Tensor) -> torch.Tensor:
        """Evaluate the loss for a given parameter dictionary and sample batch."""

        batch = sample.unsqueeze(0)
        log_prob = functional_call(self, (params,), (batch,))
        return self._loss_fn(log_prob)

    def per_sample_grad(self, samples: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return gradients for each sample in ``samples``."""

        compute_grad = grad(self._compute_loss)
        compute_sample_grad = vmap(compute_grad, in_dims=(None, 0))
        return compute_sample_grad(self._get_params(), samples)

    @torch.no_grad()
    def update_params(self, updates_flatten: torch.Tensor, lr: float) -> None:
        """Apply flattened updates to the model parameters."""

        idx = 0
        for _, weight in self.named_parameters():
            numel = weight.numel()
            update = updates_flatten[idx : idx + numel].view(weight.shape)
            weight.data -= lr * update
            idx += numel


__all__ = ["BaseModel", "scaled_dot_product_attention"]
