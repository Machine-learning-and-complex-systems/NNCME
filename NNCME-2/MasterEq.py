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
from nncme.systems.schlogl import Schlogl
from nncme.systems.Schlogl_2d import Schlogl_2d
from nncme.systems.toggle_switch import ToggleSwitch
from nncme.training.main import TestNatGrad


SYSTEM_REGISTRY: Dict[str, Type] = {
    "AFL": AFL,
    "BirthDeath": BirthDeath,
    "ToggleSwitch": ToggleSwitch,
    "EarlyLife": EarlyLife,
    "Epidemic": Epidemic,
    "Schlogl": Schlogl,
    "Schlogl_2d": Schlogl_2d,
    "cascade1": cascade1,
    "cascade1_inverse": cascade1_inverse,
    "cascade2": cascade2,
    "cascade3": cascade3,
    "GeneExp": GeneExp,
    "MAPK": MAPK,
}

# Default configuration overrides for quick experiments
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
