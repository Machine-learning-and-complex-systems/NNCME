"""Simple cascade reaction network with a single upstream species."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


class cascade1:
    """Linear cascade with one input species driving downstream reactions."""

    def __init__(self, *_, **kwargs) -> None:
        """  init   operation.
        """


        self.L = kwargs["L"]
        self.M = kwargs["M"]
        self.bits = kwargs["bits"]
        self.device = kwargs["device"]
        self.MConstrain = kwargs["MConstrain"]
        self.Para = kwargs["Para"]

    def Propensity(
        self,
        Win: torch.Tensor,
        Wout: torch.Tensor,
        rates: torch.Tensor,
        *_,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the mass-action propensities for every reaction."""

        propensity_in = torch.prod(Win, dim=1) * rates
        propensity_out = torch.prod(Wout, dim=1) * rates
        return propensity_in, propensity_out

    def rates(
        self,
    ) -> Tuple[str, np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        """Return initial conditions and stoichiometric descriptors."""

        beta, k, gamma = 10.0, 5.0, 1.0
        initial_counts = np.zeros((1, self.L)) - 1
        rates = torch.zeros(2 * self.L, device=self.device)
        rates[0] = beta
        for idx in range(self.L):
            rates[2 * idx + 1] = gamma
        for idx in range(1, self.L):
            rates[2 * idx] = k

        reaction_left = torch.zeros((self.L, 2 * self.L), device=self.device)
        for idx in range(self.L):
            reaction_left[idx, 2 * idx + 1] = 1.0
        for idx in range(1, self.L):
            reaction_left[idx - 1, 2 * idx] = 1.0

        reaction_right = torch.zeros((self.L, 2 * self.L), device=self.device)
        reaction_right[0, 0] = 1.0
        for idx in range(1, self.L):
            reaction_right[idx, 2 * idx] = 1.0

        constraints = np.zeros(1, dtype=int)
        conservation = np.ones(1, dtype=int)

        return (
            "delta",
            initial_counts,
            rates,
            reaction_left,
            reaction_right,
            constraints,
            conservation,
        )
