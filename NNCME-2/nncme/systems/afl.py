"""Activator feed-forward loop (AFL) reaction system definition."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


class AFL:
    """Activator feed-forward loop benchmark system."""

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

    def MaskAFL(self, candidate_states: torch.Tensor, flux: torch.Tensor) -> torch.Tensor:
        """Maskafl operation.
        """


        mask = torch.ones_like(flux)
        occupant = candidate_states[:, 0, :]
        complement = 1 - occupant

        mask[occupant[:, 1] != 1, 1] = 0
        mask[occupant[:, 2] != 1, 2] = 0
        mask[complement[:, 0] != 1, 0] = 0
        mask[complement[:, 3] != 1, 3] = 0
        return mask

    def Propensity(
        self,
        Win: torch.Tensor,
        Wout: torch.Tensor,
        rates: torch.Tensor,
        incoming_states: torch.Tensor,
        outgoing_states: torch.Tensor,
        *_,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute in/out propensities for all reactions.

        Args:
            Win: Tensor of shape ``(batch, species, reactions)`` with reactant
                concentrations for incoming transitions.
            Wout: Tensor with the same shape for outgoing transitions.
            rates: Reaction-rate tensor ``(reactions,)``.
            incoming_states: Reactant states for incoming transitions.
            outgoing_states: Reactant states for outgoing transitions.
            *_: Unused boundary-condition indicators kept for API compatibility.

        Returns:
            Tuple ``(propensity_in, propensity_out)`` with tensors of shape
            ``(batch, reactions)`` describing transition intensities.
        """

        win_product = torch.prod(Win, dim=1)
        mask = self.MaskAFL(incoming_states, win_product)
        propensity_in = win_product * mask * rates

        wout_product = torch.prod(Wout, dim=1)
        mask = self.MaskAFL(outgoing_states, wout_product)
        propensity_out = wout_product * mask * rates

        return propensity_in, propensity_out

    def rates(
        self,
    ) -> Tuple[str, np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        """Return the initial distribution and the stoichiometric description.

        Returns:
            Tuple containing the name of the initial distribution, the initial
            molecule counts, reaction rates, left/right stoichiometric matrices,
            molecule constraints, and the conservation vector.
        """

        initial_counts = np.array([0, 0], dtype=float).reshape(1, self.L)
        rates = torch.zeros(5, device=self.device)
        constraints = np.array([2, self.M], dtype=int)
        conservation = np.ones(1, dtype=int)

        if self.Para == 1:
            sigma_u, sigma_b, rho_u, rho_b = 0.94, 0.01, 8.40, 28.1
        elif self.Para == 2:
            sigma_u, sigma_b, rho_u, rho_b = 0.69, 0.07, 7.2, 40.6
        else:
            sigma_u, sigma_b, rho_u, rho_b = 0.44, 0.08, 0.94, 53.1

        rates[0] = sigma_u
        rates[1] = sigma_b
        rates[2] = rho_u
        rates[3] = rho_b
        rates[4] = 1.0

        reaction_left = torch.as_tensor(
            [(0, 1, 1, 0, 0), (0, 1, 0, 0, 1)],
            device=self.device,
        )
        reaction_right = torch.as_tensor(
            [(1, 0, 1, 0, 0), (1, 0, 1, 1, 0)],
            device=self.device,
        )

        return (
            "delta",
            initial_counts,
            rates,
            reaction_left,
            reaction_right,
            constraints,
            conservation,
        )
