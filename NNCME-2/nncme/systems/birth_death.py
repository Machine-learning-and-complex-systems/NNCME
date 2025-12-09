"""Birth-death reaction system definition."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


class BirthDeath:
    """Canonical birth-death process with linear reaction propensities."""

    def __init__(self, *_, **kwargs) -> None:
        """  init   operation.
        """


        self.L = kwargs["L"]
        self.M = kwargs["M"]
        self.bits = kwargs["bits"]
        self.device = kwargs["device"]
        self.MConstrain = kwargs["MConstrain"]
        self.Para = kwargs["Para"]
        self.IniDistri = kwargs["IniDistri"]
        self.binary = kwargs["binary"]
        self.order = kwargs["order"]

    def Propensity(
        self,
        Win: torch.Tensor,
        Wout: torch.Tensor,
        rates: torch.Tensor,
        *_,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute incoming and outgoing propensities for each reaction."""

        propensity_in = torch.prod(Win, dim=1) * rates
        propensity_out = torch.prod(Wout, dim=1) * rates
        return propensity_in, propensity_out

    def rates(
        self,
    ) -> Tuple[str, np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        """Return the initial distribution and stoichiometric matrices."""

        initial_distribution = "poisson"
        initial_counts = np.array([1], dtype=float)

        rates = torch.zeros(2, device=self.device)
        rates[0] = 0.1
        rates[1] = 0.01

        reaction_left = torch.as_tensor([(0, 1)], device=self.device)
        reaction_right = torch.as_tensor([(1, 0)], device=self.device)
        constraints = np.zeros(1, dtype=int)
        conservation = np.ones(1, dtype=int)

        return (
            initial_distribution,
            initial_counts,
            rates,
            reaction_left,
            reaction_right,
            constraints,
            conservation,
        )
