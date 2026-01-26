"""Chemical system definitions used by NNCME."""

from .afl import AFL
from .birth_death import BirthDeath
from .cascade1 import cascade1
from .cascade1_inverse import cascade1_inverse
from .cascade2 import cascade2
from .cascade3 import cascade3
from .early_life import EarlyLife
from .epidemic import Epidemic
from .gene_expression import GeneExp
from .mapk import MAPK
from .schlogl import Schlogl
from .toggle_switch import ToggleSwitch

__all__ = [
    "AFL",
    "BirthDeath",
    "cascade1",
    "cascade1_inverse",
    "cascade2",
    "cascade3",
    "EarlyLife",
    "Epidemic",
    "GeneExp",
    "MAPK",
    "Schlogl",
    "ToggleSwitch",
]
