# NNCME-2
This repository provides the **reference implementation of NNCME‑2**, accompanying the paper

> **Tracking Large Chemical Reaction Networks and Rare Events by Neural Networks**  

NNCME‑2 is a neural‑network–based framework for solving the **Chemical Master Equation (CME)** in high‑dimensional reaction networks and spatially extended reaction–diffusion systems. It combines **variational autoregressive networks (VANs)** with **second‑order optimization** and **enhanced sampling**, enabling efficient modeling of large biochemical networks and accurate characterization of **rare events**.


- [NNCME-2](#nncme-2)
  - [0. Requirements](#0-requirements)
    - [GPU installation (recommended)](#gpu-installation-recommended)
    - [CPU‑only installation](#cpuonly-installation)
  - [1. CME Formulation](#1-cme-formulation)
  - [2. Variational Autoregressive Networks (VAN)](#2-variational-autoregressive-networks-van)
    - [Supported architectures](#supported-architectures)
  - [3. Optimization Methods](#3-optimization-methods)
  - [3. Enhanced Sampling Methods](#3-enhanced-sampling-methods)
  - [4. Running Experiments](#4-running-experiments)
  - [5. License](#5-license)


## 0. Requirements

The code base is written in **Python 3.10** and supports both CPU and GPU execution.

### GPU installation (recommended)
```bash
conda create --name nncme python=3.10
conda activate nncme
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install scipy matplotlib scikit-learn==1.6.1 tqdm==4.67.1 spyder
```

### CPU‑only installation
```bash
conda create --name nncme python=3.10
conda activate nncme
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 cpuonly -c pytorch
conda install scipy matplotlib scikit-learn tqdm spyder
```

## 1. CME Formulation

For a reaction network with state $n = (n_1,\ldots,n_L)$ the probability distribution $P_t(n)$ evolves according to the Chemical Master Equation

$$
\partial_t P_t(n) = \sum_{k=1}^K \big[ a_k(n-s_k) P_t(n-s_k) - a_k(n) P_t(n) \big],
$$

where $a_k(n)$ is the propensity function and $s_k$ is the stoichiometric change vector. The exponential growth of the state space $M^L$ makes direct solutions intractable, motivating the neural‑network approach adopted here.

The training objective minimises the reverse Kullback–Leibler divergence between the VAN parameterized distribution $\hat{P}_{t+\delta t}^{\theta_{t+\delta t}}$ and the transition probability $\mathbb{T}\hat{P}_{t}^{\theta_{t}}$, where $\mathbb{T}$ the transition probability.

$$
D_{\mathrm{KL}}(\hat{P}_{t+\delta t}^{\theta_{t+\delta t}} \parallel \mathbb{T}\hat{P}_{t}^{\theta_{t}})
 \;=\; \sum_x \hat{P}_{t+\delta t}^{\theta_{t+\delta t}}(x) \log \frac{\hat{P}_{t+\delta t}^{\theta_{t+\delta t}}(x)}{\mathbb{T}\hat{P}_{t}^{\theta_{t}}(x)}.
$$

In practice we optimise the VAN parameterized distribution implemented in `nncme/training/main.py:270`, where the code computes the log-density ratio `loss = log_prob - LogTP_t.detach()` .

## 2. Variational Autoregressive Networks (VAN)

NNCME‑2 represents the joint distribution using an autoregressive factorization

$$
\hat{P}_{\theta}(n) = \prod_{i=1}^{L} \hat{P}_{\theta}\bigl(n_i \mid n_{<i}\bigr),
$$

which is automatically normalized and allows efficient sampling.

### Supported architectures

- **NADE (default)**  
  Lightweight, feed‑forward autoregressive model with shared parameters. Offers an excellent balance between efficiency and expressivity.

- **Transformer**  
  Attention‑based autoregressive model with higher representational power, suitable for highly structured or strongly correlated systems.

---

## 3. Optimization Methods
Three optimization methods are available inside `nncme/training/main.py`:

* **Stochastic Gradient Descent (SGD)** performs the familiar update $\theta_{k+1} = \theta_k - \eta \nabla_\theta \mathcal{L}$ using the gradient produced by backpropagation (`nncme/training/main.py:270-288`), correspoonding default optimizer is `Adam`.
* **Natural Gradient (NG)** replaces the Euclidean metric with the Fisher information matrix $S$, yielding the update $\theta_{k+1} = \theta_k - \eta S^{-1} \nabla_\theta \mathcal{L}$.  The code builds the Fisher system and solves it through the Cholesky-based routines at `nncme/training/main.py:374-384`, powered by the linear solvers in `nncme/utils.py:279-329`.
* **Time-Dependent Variational Principle (TDVP)** linearises the CME dynamics to update network parameters via $\theta_{k+1} = \theta_k - \eta (O^\top O + \lambda I)^{-1} O^\top R$, where $O$ collects per-sample Jacobians and $R$ captures residuals.  See the TDVP branch in `nncme/training/main.py:382-386`.

For all three optimization methods, the project also supports importance reweighting via the control variates defined in `nncme/training/main.py:263-288`, providing for enhanced sampling.

## 3. Enhanced Sampling Methods
Enhanced sampling is implemented within the network. Let $\hat{P_{\theta}}$ denote the VAN parameterized distribution. Samples $s$ are drawn from below distribution for training in order to improve the ability of sample representation.
* **Vanilla sampling**
$
s \sim \hat{P_{\theta}}.
$

* **Mixture sampling**
$
s \sim \hat{P_{\theta}} + P_u,
$

where $P_u$ is a uniform distribution.
* **Diffusive sampling** 
$
s \sim K * \hat{P_{\theta}},
$
where $K$ is a uniform kernel, and $*$ denotes convolution over the distribution.
* **α sampling** 
$
s \sim \hat{P_{\theta}}^{\alpha},
$
where $\alpha$ is a hyper-parameter to control the degree of over-spreading of the original distribution.

Each method is exposed through `nade` and `transformer`, making it straightforward to switch enhanced sampling on or off via the global configuration in `args.py` or by editing experiment scripts.

## 4. Running Experiments
You can launch a simulation in two ways:

1. **Inline Python entry point** – edit the hyper-parameters in `MasterEq.py` (notably `args.Model`, optimisation settings, and sampling flags) and run:
   ```bash
   python MasterEq.py
   ```
   The script selects the appropriate chemical system from `nncme/systems/`, outputs (logs, checkpoints, plots) are saved to `out/<system_name>/`.

2. **Batch scripts** – for cluster execution, adjust the SLURM-ready shell scripts.  Edit the parameter block near the top of a script, then submit through your scheduler, or run locally with `sbatch schlogl.sh` if SLURM environment variables are available.


## 5. License

© Lab of Machine Learning and Stochastic Dynamics. All rights reserved.
