# NNCME Program Suite

This directory hosts two generations of the Neural Network Chemical Master Equation (NNCME) code base. Use this overview to understand their focus, progress, and how to deploy or cite each version.

## Program Highlights

### NNCME-1
- Original implementation accompanying *Neural-network solutions to stochastic reaction networks*.
- Uses variational autoregressive neural networks to learn the joint probability distribution and marginal statistics for stochastic reaction networks.
- Provides template notebooks (`A Simple Template.ipynb`, `Detailed Version of Gene Expression.ipynb`) and scripts (`MasterEq.py`, `ParameterSearch.py`) for creating new reaction models or scanning hyperparameters.

### NNCME-2
- Reference implementation for *Tracking Large Chemical Reaction Networks and Rare Events by Neural Networks*.
- Retains the VAN-based CME solver while adding second-order optimization (natural gradient, TDVP), enhanced sampling strategies, and better tooling for rare-event characterization and spatially extended systems.
- Includes both inline (`MasterEq.py`) and batch/SLURM workflows plus modular system definitions under `nncme/systems/`.

## Progress in NNCME-2
- **Scalability:** Optimized for high-dimensional reaction networks, including reaction-diffusion setups.
- **Optimization:** Adds natural gradient and TDVP solvers on top of Adam/SGD for faster convergence and better stability.
- **Sampling:** Offers mixture, diffusive, and $\alpha$-tempered sampling modes to resolve rare events without exhaustive brute-force trajectories.
- **Execution paths:** Provides GPU-first Conda environments and batch scripts suitable for HPC deployments.

## Environment & Dependencies

### NNCME-1
- Python >= 3.6; PyTorch >= 1.0.
- Developed and tested on GPU-enabled systems but can run on CPUs for smaller models.
- Optional tools: Spyder IDE for Windows users, shell scripts under `Shell/` for cluster runs.

### NNCME-2
- Python 3.10 environment.
- **GPU setup (recommended):**
  ```bash
  conda create --name nncme python=3.10
  conda activate nncme
  conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=11.8 -c pytorch -c nvidia
  conda install scipy matplotlib scikit-learn==1.6.1 tqdm==4.67.1 spyder
  ```
- **CPU-only setup:**
  ```bash
  conda create --name nncme python=3.10
  conda activate nncme
  conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 cpuonly -c pytorch
  conda install scipy matplotlib scikit-learn tqdm spyder
  ```

## Citations

### NNCME-1
```bibtex
@article{nncme1_2023,
  title   = {Neural-network solutions to stochastic reaction networks},
  author  = {Tang, Ying and Weng, Jiayu and Zhang, Pan},
  journal = {Nat. Mach. Intell.},
  volume  = {5},
  number  = {4},
  pages   = {376--385},
  year    = {2023},
  month   = mar,
  issn    = {2522-5839},
  doi     = {10.1038/s42256-023-00632-6},
  urldate = {2025-11-18}
}
```

### NNCME-2
```bibtex
@article{nncme2,
  title   = {Tracking Large Chemical Reaction Networks and Rare Events by Neural Networks},
  author  = {Weng, Jiayu and Zhu, Xinyi and Liu, Jing and L{\"u}, Linyuan and Zhang, Pan and Tang, Ying},
  journal = {...},
  year    = {2025}
}
```

Please cite the relevant version(s) when publishing results derived from these codes.

# Contact

If you have any questions or need help to implement your model, feel free to contact us.

Contact: Ying Tang, jamestang23@gmail.com; Jiayu Weng, yukiweng0602@gmail.com; Xinyi Zhu, ziky168xixi@gmail.com;
