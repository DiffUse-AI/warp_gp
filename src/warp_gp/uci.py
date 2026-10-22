"""Real-data benchmarks: posterior means of GPs fitted to the UCI Combined Cycle Power Plant
(id 294) and Airfoil Self-Noise (id 291) datasets."""
import math
import os

import numpy as np
import torch

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")


def _surrogate(name):
    z = np.load(os.path.join(DATA, f"uci_{name}_surrogate.npz"))

    def f(U):
        U = U.reshape(1, -1) if U.ndim == 1 else U
        Xtr, alpha, ls = (torch.as_tensor(z[k], dtype=U.dtype) for k in ("Xtr", "alpha", "ls"))
        a = math.sqrt(5.0) * torch.cdist(U / ls, Xtr / ls)
        mu = float(z["s2"]) * (((1 + a + a * a / 3) * torch.exp(-a)) @ alpha)
        return mu * float(z["ysd"]) + float(z["ymean"])
    return f


UCI = {"power_plant": 4, "airfoil": 5}
