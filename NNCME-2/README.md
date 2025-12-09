# NNCME-2

This repository provides the **reference implementation of NNCME-2**, accompanying the paper

> **Tracking Large Chemical Reaction Networks and Rare Events by Neural Networks**

NNCME-2 is a neural-network framework for solving the **Chemical Master Equation (CME)** in high-dimensional reaction networks and spatially extended reaction-diffusion systems. It combines **variational autoregressive networks (VANs)** with **second-order optimization** and **enhanced sampling**, enabling efficient modeling of large biochemical networks and accurate characterization of **rare events**.

## Overview

- Implements variational autoregressive networks (VANs) to approximate the CME with built-in support for NADE, RNN, LSTM, and transformer backbones.
- Includes stochastic gradient, natural-gradient, and TDVP optimizers in `nncme/training/main.py`, together with adaptive second-order solvers.
- Adds enhanced sampling strategies (diffusive/alpha/mixtures) that can be toggled via CLI flags in `nncme/args.py`.
- Ships experiment scripts (`MasterEq.py`, `*.sh`) plus the worked notebook `Example_Schlogl_2.ipynb` for rapid prototyping and visualization.

## Environment and Installation

The code targets **Python 3.10** and benefits from a CUDA-capable GPU, although CPU-only execution is supported. We recommend Conda for dependency management; adjust the commands as needed for virtualenv or system Python.

### GPU setup (recommended)

```bash
conda create --name nncme python=3.10
conda activate nncme
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install scipy matplotlib scikit-learn==1.6.1 tqdm==4.67.1 spyder
```

### CPU-only setup

```bash
conda create --name nncme python=3.10
conda activate nncme
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 cpuonly -c pytorch
conda install scipy matplotlib scikit-learn tqdm spyder
```

## Workflow at a Glance

1. **Select or define a system** in `nncme/systems/` (e.g., `Schlogl`, `MAPK`, `ToggleSwitch`). `MasterEq.py` picks up the class associated with `args.Model`.
2. **Choose network/optimizer/sampling settings** via CLI arguments (see `nncme/args.py` for defaults) or by editing `MasterEq.py`.
3. **Train and evaluate** by running `python MasterEq.py ...` or the provided `.sh` scripts on clusters.
4. **Inspect outputs** written to `out/<system_name>/`, including saved samples, loss traces, and optional plots (e.g., `Fig_Schlogl.png`).

## Designing Your Reaction System

To introduce a custom reaction network:

1. **Create a system class** under `nncme/systems/` mirroring the existing templates. Implement at least:
   - `rates(...)`: return the initial distribution (`IniDistri`/`initialD`), reaction rates `r`, stoichiometric matrices `ReactionMatLeftTotal` and `ReactionMatRightTotal`, optional constraints (`MConstrain`, `conservation`), and any diffusion operators.
   - `Propensity(...)`: compute reaction propensities given sampled states (default implementations follow mass-action kinetics; adapt if needed).
2. **Register the model** by adding it to `SYSTEM_REGISTRY` in `MasterEq.py` and ensure `args.Model` defaults suit your experiment.
3. **Tune hyperparameters** such as `args.L`, `args.M`, `args.batch_size`, `args.delta_t`, and the optimizer/sampling flags. All CLI switches documented in `nncme/args.py` work both from the shell and within Spyder.
4. **Reference NNCME-1 resources** &mdash; the `NNCME-1/README.md`, `A Simple Template.ipynb`, and `Detailed Version of Gene Expression.ipynb` provide worked walk-throughs on how to translate stoichiometry, propensity functions, and initial distributions into the required Python interface. The conventions carry over directly to NNCME-2.

## Running the Solver

### Direct Python execution

```bash
cd NNCME-2
python MasterEq.py \
  --Model Schlogl \
  --net NADE \
  --method NatGrad \
  --sampling diffusive \
  --delta_t 0.01 \
  --Tstep 200 \
  --batch_size 2000
```

- All arguments defined in `nncme/args.py` are available from the command line (or through Spyder on Windows).
- Use `--reweighted/--noreweight`, `--sampling alpha`, `--kernel`, etc., to toggle enhanced sampling. 

### Shell scripts and batch jobs

- `schlogl.sh`, `Schlogl_2d.sh`, `MAPK.sh`, and `toggle_switch.sh` showcase bash inputs for SLURM. Adapt the parameter blocks at the top, then run `sbatch schlogl.sh`.
- Environment variables such as `CUDA_VISIBLE_DEVICES` can be exported inside the scripts for multi-GPU runs.

## Optimization Methods (summary)

`nncme/training/main.py` exposes three complementary solvers, all configurable via `--method`:

- **SGD / Adam**: default first-order updates with optional gradient clipping (`args.clip_grad`) and scheduling.
- **Natural Gradient (NatGrad / NatGrad_kl2)**: solves Fisher systems with the Cholesky utilities in `nncme/utils.py`, improving convergence on stiff distributions.
- **Time-Dependent Variational Principle (TDVP)**: linearizes CME dynamics and solves the resulting normal equations via `(OᵀO + λI)^{-1}OᵀR`.

All methods share the same `MasterEq.py` entry point and can be combined with reweighting or adaptive time stepping (`--AdaptiveT`, `--lr_schedule`).

## Enhanced Sampling Methods (summary)

Enhanced sampling is configured via the `--sampling` flag and implemented inside `nncme/training/main.py`:

- `default` / vanilla: draw samples directly from the VAN distribution.
- `diffusive`: smooth distributions with a local kernel (`--kernel`) to better explore neighboring states.
- `alpha`: temper the distribution with exponent `--alpha` to emphasize rare configurations.
- `random`, `binomial`, and mixture modes add auxiliary noise or uniform components to reduce estimator variance.

Each strategy works with any VAN architecture (NADE, transformer, RNN, LSTM) and supports importance reweighting (`--reweighted`).

## Example to Follow

- **`Example_Schlogl_2.ipynb`** demonstrates the full workflow: configuring `args`, running `MasterEq.py`, loading `out/Schlogl/` results, and plotting statistics (`Fig_Schlogl.png`). Start here if you prefer a notebook-driven exploration or need a template for custom visualization.

## Outputs and Visualization

- Training checkpoints, sampled `.npy` data are stored in `out/<model_name>/`.

## License

Copyright (c) Lab of Machine Learning and Stochastic Dynamics. All rights reserved.

