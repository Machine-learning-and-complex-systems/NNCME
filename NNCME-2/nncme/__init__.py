"""Top-level package for the NNCME project.

This package provides tools to simulate and learn chemical master equation
systems with neural-network-based solvers. Subpackages include chemical system
definitions, neural network architectures, training routines, stochastic
simulation algorithms, and analysis utilities."""

from .args import args

__all__ = ["args"]
