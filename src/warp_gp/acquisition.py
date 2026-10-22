"""Acquisition scores over a candidate pool.

eig       posterior variance (equivalent to EIG with fixed noise)
eq        posterior variance times the geometric equalizer
warp      posterior variance under the reconstruction warp
eq_warp   warp times the geometric equalizer (proposed method)
eq_kumar  posterior variance under a trained Kumaraswamy warp times the geometric equalizer
"""
import torch

from .equalizer import geometric_equalizer
from .gp import posterior_variance
from .kumaraswamy import fit_kumaraswamy
from .warp import build_reconstruction_warp

METHODS = ("eig", "eq", "warp", "eq_warp", "eq_kumar")


def score(method, X, y, C, noise, ell, g=None):
    """Return the score of each candidate. g(C, n) computes the geometric equalizer; by default
    it is estimated directly."""
    D, n = X.shape[1], len(X)
    g = g or (lambda C, n: geometric_equalizer(C, n, D, ell, noise))
    if method in ("warp", "eq_warp"):
        warp = build_reconstruction_warp(X, y, noise, ell, D)
    elif method == "eq_kumar":
        warp = fit_kumaraswamy(X, y, noise, ell)
    elif method in ("eig", "eq"):
        warp = None
    else:
        raise ValueError(method)
    with torch.no_grad():
        v = posterior_variance(X, C, noise, ell, warp)
        return v * g(C, n) if method.startswith("eq") else v
