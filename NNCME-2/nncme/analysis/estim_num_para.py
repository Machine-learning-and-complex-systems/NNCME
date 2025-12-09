"""Parameter counting utilities for NADE and transformer architectures."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt


def estimate_transformer_params(M: int, L: int, d_model: int, d_ff: int, n_layers: int) -> int:
    """Estimate the number of trainable parameters in the transformer baseline."""

    embedding_params = M * d_model + L * d_model
    output_layer_params = d_model * M + M
    block_params = 4 * d_model**2 + 2 * d_model * d_ff + d_ff + 6 * d_model
    transformer_params = n_layers * block_params
    return embedding_params + output_layer_params + transformer_params


def count_nade_params(L: int, M: int, hidden_width: int) -> int:
    """Return the parameter count for a NADE network."""

    return hidden_width * (L + 1 + M * L) + L * M


def plot_parameter_scaling(
    L_values: Iterable[int],
    M_values: Iterable[int],
    nade_width: int = 16,
    transformer_ff: int = 16,
    transformer_layers: int = 2,
) -> None:
    """Visualise how model sizes grow as we vary species and vocabulary sizes."""

    nade_params = [count_nade_params(L, 85, nade_width) for L in L_values]
    trans_params_8 = [estimate_transformer_params(85, L, 8, transformer_ff, transformer_layers) for L in L_values]
    trans_params_16 = [estimate_transformer_params(85, L, 16, transformer_ff, transformer_layers) for L in L_values]
    trans_params_dynamic = [
        estimate_transformer_params(85, L, 2 * L, transformer_ff, transformer_layers) for L in L_values
    ]

    plt.figure(figsize=(7, 6))
    plt.plot(L_values, nade_params, marker="o", label="NADE (width=16)")
    plt.plot(L_values, trans_params_8, marker="s", label="Transformer (d_model=8)")
    plt.plot(L_values, trans_params_16, marker="^", label="Transformer (d_model=16)")
    plt.plot(L_values, trans_params_dynamic, marker="d", label="Transformer (d_model=2L)")
    plt.xlabel("Species (L)")
    plt.ylabel("Number of Parameters")
    plt.title("Parameter Comparison vs Species Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    nade_params_M = [count_nade_params(4, M, nade_width) for M in M_values]
    trans_params_fixed_M = [estimate_transformer_params(M, 4, 8, transformer_ff, transformer_layers) for M in M_values]

    plt.figure(figsize=(7, 6))
    plt.plot(M_values, nade_params_M, marker="o", label="NADE (H=16)")
    plt.plot(M_values, trans_params_fixed_M, marker="s", label="Transformer (d_model=8)")
    plt.xlabel("Vocabulary Size (M)")
    plt.ylabel("Number of Parameters")
    plt.title("Parameter Comparison vs Vocabulary Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    example_params = dict(M=85, L=2, d_model=8, d_ff=16, n_layers=2)
    print(f"Transformer parameters: {estimate_transformer_params(**example_params)}")
    print(f"NADE parameters: {count_nade_params(L=2, M=85, hidden_width=16)}")

    plot_parameter_scaling(L_values=[2, 4, 6, 8, 10], M_values=[20, 40, 60, 80, 100, 120, 140])
