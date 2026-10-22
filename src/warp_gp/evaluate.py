"""Nearest-neighbour reconstruction error of a design on [0,1]^D."""
import numpy as np
import torch

from .functions import standardized
from .gp import DT


def true_field(func_name, n_test=16384, seed=12321):
    """Return a Sobol test set, the standardized function values on it, and the standardized function."""
    f, D = standardized(func_name)
    fn = lambda Z: f(torch.as_tensor(Z, dtype=DT)).numpy()
    T = torch.quasirandom.SobolEngine(D, scramble=True, seed=seed).draw(n_test).to(DT)
    return T, fn(T), fn


def onenn_trajectory(X0, picks, T, fT, fn):
    """Mean squared error of the 1-NN reconstruction on T after each acquired point."""
    X = torch.as_tensor(np.vstack([X0, picks]), dtype=DT)
    vals, n0 = fn(X), len(X0)
    dmin, imin = torch.cdist(T, X[:n0]).min(1)
    dmin, vnn = dmin.numpy(), vals[imin.numpy()]
    out = []
    for k in range(n0, len(X)):
        dk = torch.cdist(T, X[k:k + 1]).squeeze(1).numpy()
        closer = dk < dmin
        dmin, vnn = np.where(closer, dk, dmin), np.where(closer, vals[k], vnn)
        out.append(((vnn - fT) ** 2).mean())
    return np.array(out)
