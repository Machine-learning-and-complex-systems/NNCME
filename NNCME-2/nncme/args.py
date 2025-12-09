"""Centralised argument parsing for the NNCME training scripts."""

from __future__ import annotations

import argparse
from typing import List


SAVING_DEFAULT: List[float] = [
    0.0,
    1e2,
    5e2,
    2e3,
    1e4,
    2e4,
    5e4,
    1e5,
    1.5e5,
    2e5,
    2.5e5,
    3e5,
    3.5e5,
    4e5,
    5e5,
    6e5,
    7e5,
    8e5,
    9e5,
    1e6,
]

TRAINING_LOSS_DEFAULT: List[float] = [
    0.0,
    1.0,
    2.0,
    1e2,
    1e3,
    2e3,
    4e3,
    8e3,
    1e4,
    1e5,
    2e5,
    3e5,
    4e5,
    5e5,
]


def build_parser() -> argparse.ArgumentParser:
    """Create an argument parser shared by entry scripts."""

    parser = argparse.ArgumentParser(
        description="Neural network solver for the chemical master equation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    system = parser.add_argument_group("system configuration")
    system.add_argument("--Model", type=str, default="No", help="Name of the chemical system to simulate.")
    system.add_argument(
        "--IniDistri",
        type=str,
        default="delta",
        choices=["delta", "poisson", "uniform", "zipf", "MM"],
        help="Initial distribution family used to seed the population.",
    )
    system.add_argument("--Para", type=float, default=1.0, help="Model-specific scaling parameter.")
    system.add_argument("--Tstep", type=int, default=1, help="Number of master-equation time steps to evaluate.")
    system.add_argument("--delta_t", type=float, default=0.05, help="Physical time step between consecutive updates.")
    system.add_argument(
        "--AdaptiveTFold",
        type=float,
        default=100.0,
        help="Factor used when enlarging the time step adaptively.",
    )
    system.add_argument(
        "--boundary",
        type=str,
        default="periodic",
        choices=["open", "periodic"],
        help="Boundary condition applied to the lattice model.",
    )
    system.add_argument("--L", type=int, default=3, help="Number of chemical species.")
    system.add_argument("--Sites", type=int, default=1, help="Number of spatial sites for extended systems.")
    system.add_argument("--M", type=int, default=100, help="Upper bound for molecule counts per species.")
    system.add_argument("--order", type=int, default=1, help="Ordering used to enumerate species.")
    system.add_argument("--beta", type=float, default=1.0, help="Inverse temperature scaling factor.")
    system.add_argument(
        "--conservation",
        type=int,
        default=1,
        help="Integer representing conserved quantity constraints (1 disables the constraint).",
    )
    system.add_argument(
        "--L_label",
        nargs="+",
        type=str,
        default=["X1", "X2", "X3", "X4"],
        help="Labels used when plotting marginal distributions.",
    )
    system.add_argument(
        "--L_plot",
        nargs="+",
        type=int,
        default=[0],
        help="Indices of species that should be visualised.",
    )
    system.add_argument(
        "--initialD",
        nargs="+",
        type=float,
        help="Optional explicit initial molecule counts overriding distribution defaults.",
    )
    system.add_argument(
        "--MConstrain",
        nargs="+",
        type=int,
        help="Optional per-species occupancy constraints.",
    )
    system.add_argument(
        "--absorbed",
        action="store_true",
        help="Enable simulation of absorbing boundary conditions.",
    )
    system.add_argument(
        "--modify",
        action="store_true",
        help="Allow modifications to the default absorbing state definition.",
    )
    system.add_argument(
        "--absorb_state",
        nargs="+",
        type=int,
        help="Absorbing state vector e.g. '--absorb_state 50 50'.",
    )

    sampling = parser.add_argument_group("sampling and variance reduction")
    sampling.add_argument(
        "--sampling",
        type=str,
        default="default",
        choices=["default", "manual", "binomial", "random", "scaling", "diffusive", "alpha"],
        help="Enhanced sampling strategy.",
    )
    sampling.add_argument("--ESNumber", type=int, default=10, help="Number of samples for enhanced sampling.")
    sampling.add_argument("--kernel", type=int, default=1, help="Neighbourhood size for diffusive sampling.")
    sampling.add_argument("--alpha", type=float, default=1.0, help="Scaling exponent for power-law sampling.")
    sampling.add_argument(
        "--reweighted",
        dest="reweighted",
        action="store_true",
        help="Enable importance reweighting during sampling.",
    )
    sampling.add_argument(
        "--noreweight",
        dest="reweighted",
        action="store_false",
        help="Disable importance reweighting.",
    )
    parser.set_defaults(reweighted=True)

    network = parser.add_argument_group("network architecture")
    network.add_argument(
        "--method",
        type=str,
        default="SGD",
        choices=["SGD", "TDVP", "NatGrad", "NatGrad_kl2"],
        help="Optimisation formulation used during training.",
    )
    network.add_argument(
        "--net",
        type=str,
        default="rnn",
        choices=["rnn", "transformer", "NADE", "LSTM"],
        help="Neural architecture producing the probability amplitudes.",
    )
    network.add_argument("--net_depth", type=int, default=3, help="Depth of fully connected segments.")
    network.add_argument("--net_width", type=int, default=64, help="Width of hidden layers for MLP-backed nets.")
    network.add_argument("--d_model", type=int, default=64, help="Transformer latent dimension.")
    network.add_argument("--d_ff", type=int, default=128, help="Transformer feed-forward dimension.")
    network.add_argument("--n_layers", type=int, default=2, help="Number of recurrent/transformer layers.")
    network.add_argument("--n_heads", type=int, default=2, help="Number of attention heads when using transformers.")
    network.add_argument("--bits", type=int, default=1, help="Bit width when representing states in binary form.")
    network.add_argument("--bias", action="store_true", help="Enable bias parameters in the network.")
    network.add_argument("--AdaptiveT", action="store_true", help="Enable adaptive time stepping in training.")
    network.add_argument("--adaptive_lr", action="store_true", help="Enable adaptive learning-rate heuristics.")
    network.add_argument("--res_block", action="store_true", help="Insert residual blocks into the network.")
    network.add_argument("--binary", action="store_true", help="Represent states with binary encodings.")
    network.add_argument("--reverse", action="store_true", help="Augment the flow with reverse conditional models.")

    optimisation = parser.add_argument_group("optimisation")
    optimisation.add_argument("--optimizer", type=str, default="adam", help="Optimizer used for parameter updates.")
    optimisation.add_argument("--batch_size", type=int, default=2000, help="Number of samples per optimisation step.")
    optimisation.add_argument("--save_sample_num", type=int, default=20000, help="Number of samples to persist.")
    optimisation.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimiser.")
    optimisation.add_argument(
        "--delta_factor",
        type=float,
        default=1e-3,
        help="Smoothing parameter for initial delta distributions.",
    )
    optimisation.add_argument("--max_step", type=int, default=1000, help="Maximum number of optimisation steps.")
    optimisation.add_argument(
        "--max_stepAll",
        type=int,
        default=5000,
        help="Epoch budget for the first simulated time step when resuming.",
    )
    optimisation.add_argument(
        "--max_stepLater",
        type=int,
        default=100,
        help="Epoch budget for later time steps after the first one.",
    )
    optimisation.add_argument("--epoch0", type=int, default=10000, help="Epoch budget for the first time step.")
    optimisation.add_argument("--epoch", type=int, default=50, help="Epoch budget for subsequent time steps.")
    optimisation.add_argument(
        "--lr_schedule",
        action="store_true",
        help="Enable learning-rate scheduling following 'lr_schedule_type'.",
    )
    optimisation.add_argument(
        "--lr_schedule_type",
        type=int,
        default=1,
        help="Identifier for the learning-rate scheduler strategy.",
    )
    optimisation.add_argument(
        "--beta_anneal",
        type=float,
        default=0.0,
        help="Annealing speed from zero towards the target inverse temperature.",
    )
    optimisation.add_argument(
        "--clip_grad",
        type=float,
        default=0.0,
        help="Gradient clipping norm; set to zero to disable clipping.",
    )
    optimisation.add_argument(
        "--Percent",
        type=float,
        default=0.1,
        help="Fraction of epochs used for metric averaging and learning-rate decay.",
    )
    optimisation.add_argument(
        "--lossType",
        type=str,
        default="kl",
        choices=[
            "l2",
            "kl",
            "he",
            "ss",
            "ForwardKL1",
            "ForwardKL2",
            "ForwardKL3",
            "ReverseKL1",
            "ReverseKL2",
            "ReverseKL3",
            "DiKL",
        ],
        help="Loss formulation used during optimisation.",
    )
    optimisation.add_argument("--epsilon", type=float, default=1e-30, help="Numerical stabiliser for log-probabilities.")
    optimisation.add_argument(
        "--loadVAN",
        action="store_true",
        help="Resume optimisation from a pre-trained variational ansatz (VAN).",
    )
    optimisation.add_argument("--loadTime", type=int, default=1000, help="Time step from which to resume a VAN run.")

    logging_group = parser.add_argument_group("output and logging")
    logging_group.add_argument("--no_stdout", action="store_true", help="Disable printing to stdout.")
    logging_group.add_argument("--clear_checkpoint", action="store_true", help="Clear checkpoints before starting.")
    logging_group.add_argument("--save_step", type=int, default=100, help="Interval (in steps) for checkpoint saves.")
    logging_group.add_argument("--visual_step", type=int, default=100, help="Interval for generating visualisations.")
    logging_group.add_argument(
        "--print_step",
        type=int,
        default=10,
        help="Interval for printing training statistics.",
    )
    logging_group.add_argument(
        "--plotstep",
        type=int,
        default=100,
        help="Interval between successive plots of distribution snapshots.",
    )
    logging_group.add_argument(
        "--saving_data_time_step",
        nargs="+",
        type=float,
        default=SAVING_DEFAULT,
        help="Time steps at which samples and statistics are persisted.",
    )
    logging_group.add_argument(
        "--training_loss_print_step",
        nargs="+",
        type=float,
        default=TRAINING_LOSS_DEFAULT,
        help="Epoch indices where the training loss is explicitly logged.",
    )
    logging_group.add_argument("--save_sample", action="store_true", help="Persist generated samples at save steps.")
    logging_group.add_argument("--print_sample", type=int, default=1, help="Number of samples to include in logs.")
    logging_group.add_argument("--print_grad", action="store_true", help="Log gradient statistics at visual steps.")
    logging_group.add_argument("--out_infix", type=str, default="", help="Custom infix inserted into output names.")
    logging_group.add_argument("--out_dir", type=str, default="out", help="Directory prefix for experiment outputs.")
    logging_group.add_argument("--num_prints", type=int, default=100, help="Number of checkpoints to log.")
    logging_group.add_argument("--num_plots", type=int, default=10, help="Number of plots to render during training.")
    logging_group.add_argument(
        "--num_absorbs",
        type=int,
        default=50,
        help="Number of absorption-probability snapshots to record.",
    )

    runtime = parser.add_argument_group("runtime and reproducibility")
    runtime.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Floating-point precision used for arrays and tensors.",
    )
    runtime.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility (0 picks a random seed).")
    runtime.add_argument("--cuda", type=int, default=0, help="CUDA device identifier; set to -1 to use the CPU.")

    return parser


args = build_parser().parse_args()
