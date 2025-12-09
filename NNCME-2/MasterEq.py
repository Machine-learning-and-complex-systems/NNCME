"""Experiment entry point for running NNCME models."""

from __future__ import annotations

import time
from typing import Dict, Type

import numpy as np

from nncme.args import args
from nncme.systems.afl import AFL
from nncme.systems.birth_death import BirthDeath
from nncme.systems.cascade1 import cascade1
from nncme.systems.cascade1_inverse import cascade1_inverse
from nncme.systems.cascade2 import cascade2
from nncme.systems.cascade3 import cascade3
from nncme.systems.early_life import EarlyLife
from nncme.systems.epidemic import Epidemic
from nncme.systems.gene_expression import GeneExp
from nncme.systems.mapk import MAPK
from nncme.systems.repressilator import repressilator
from nncme.systems.schlogl import Schlogl
from nncme.systems.toggle_switch import ToggleSwitch
from nncme.training.main import TestNatGrad


SYSTEM_REGISTRY: Dict[str, Type] = {
    "AFL": AFL,
    "BirthDeath": BirthDeath,
    "ToggleSwitch": ToggleSwitch,
    "EarlyLife": EarlyLife,
    "Epidemic": Epidemic,
    "Schlogl": Schlogl,
    "cascade1": cascade1,
    "cascade1_inverse": cascade1_inverse,
    "cascade2": cascade2,
    "cascade3": cascade3,
    "GeneExp": GeneExp,
    "MAPK": MAPK,
    "repressilator": repressilator,
}

# Default configuration overrides for quick experiments
args.dtype = "float32"
args.cuda = 0
args.Model = "BirthDeath"
args.L = 1
args.L_plot = np.arange(args.L)
args.M = 30
args.batch_size = 1000
args.save_sample_num = 2000
args.Tstep = 101
args.delta_t = 1.0
args.Para = 1
args.lossType = "kl"
args.net = "transformer"
args.method = "NatGrad"
args.lr = 0.5
args.epoch0 = 20
args.epoch = 5
args.sampling = "diffusive"
args.alpha = 0.3
args.kernel = 10
args.reweighted = True
args.ESNumber = 100
args.net_depth = 1
args.net_width = 8
args.d_model = 8
args.d_ff = 16
args.n_layers = 2
args.n_heads = 2
args.absorbed = False
args.absorb_state = [5, 15]
args.delta_factor = 0.01
args.binary = False
args.AdaptiveT = False
args.AdaptiveTFold = 5
args.num_prints = 50
args.num_plots = 10
args.num_absorbs = 20
args.print_step = max(1, int(args.Tstep / 5))
args.plotstep = args.print_step
args.saving_data_time_step = [
    0,
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
args.training_loss_print_step = [0, 50, 200, 500, 800, 1000]
args.bias = True
args.bits = 1
if args.binary:
    args.bits = int(np.ceil(np.log2(args.M)))
args.Percent = 0.2
args.clip_grad = 1
args.epsilon = 1e-30
args.lr_schedule = False


def build_model() -> object:
    """Instantiate the configured chemical system."""

    if args.Model not in SYSTEM_REGISTRY:
        raise KeyError(f"Unknown model '{args.Model}'. Available: {sorted(SYSTEM_REGISTRY)}")
    return SYSTEM_REGISTRY[args.Model](**vars(args))


if __name__ == "__main__":
    model = build_model()
    start = time.time()
    Loss, SampleSum, delta_TSum, Lossmean, Lossstd, argsSave = TestNatGrad(model)
    minutes = (time.time() - start) / 60.0
    print(f"The system {args.Model} completed in {minutes:.2f} minutes.")
