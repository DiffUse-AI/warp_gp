"""Benchmark functions on [0,1]^D. Each maps inputs of shape [N, D] to outputs of shape [N]."""
import math

import torch

from .uci import UCI, _surrogate


def _to_domain(U, lo, hi):
    lo = torch.as_tensor(lo, dtype=U.dtype)
    hi = torch.as_tensor(hi, dtype=U.dtype)
    return lo + U * (hi - lo)


def _2d(U):
    return U.reshape(1, -1) if U.ndim == 1 else U


def peaks(U, a=3.0, b=10.0, c=1.0 / 3.0):
    X = _to_domain(_2d(U), [-3.0, -3.0], [3.0, 3.0])
    x, y = X[..., 0], X[..., 1]
    return (a * (1.0 - x).pow(2) * torch.exp(-(x.pow(2)) - (y + 1.0).pow(2))
            - b * (x / 5.0 - x.pow(3) - y.pow(5)) * torch.exp(-x.pow(2) - y.pow(2))
            - c * torch.exp(-(x + 1.0).pow(2) - y.pow(2)))


def grlee08(U):
    X = _to_domain(_2d(U), [-2.0, -2.0], [6.0, 6.0])
    x1, x2 = X[..., 0], X[..., 1]
    return x1 * torch.exp(-(x1 ** 2 + x2 ** 2))


def zhou98(U):
    X = _2d(U)
    d = X.shape[1]
    log_norm = -0.5 * d * math.log(2.0 * math.pi)
    phi1 = torch.exp((10.0 * (X - 1.0 / 3.0)).pow(2).sum(dim=1).mul_(-0.5).add_(log_norm))
    phi2 = torch.exp((10.0 * (X - 2.0 / 3.0)).pow(2).sum(dim=1).mul_(-0.5).add_(log_norm))
    return ((10.0 ** d) / 2.0 * (phi1 + phi2)).reshape(-1)


def beale(U):
    X = _to_domain(_2d(U), [-4.5, -4.5], [4.5, 4.5])
    x, y = X[..., 0], X[..., 1]
    return ((1.5 - x + x * y).pow(2) + (2.25 - x + x * y ** 2).pow(2)
            + (2.625 - x + x * y ** 3).pow(2))


def _levy(U, D):
    X = _to_domain(_2d(U), [-10.0] * D, [10.0] * D)
    w = 1.0 + (X - 1.0) / 4.0
    term1 = torch.sin(math.pi * w[:, 0]) ** 2
    wi = w[:, :-1]
    mid = torch.sum((wi - 1.0) ** 2 * (1.0 + 10.0 * torch.sin(math.pi * wi + 1.0) ** 2), dim=1)
    last = (w[:, -1] - 1.0) ** 2 * (1.0 + torch.sin(2.0 * math.pi * w[:, -1]) ** 2)
    return term1 + mid + last


def levy2d(U):
    return _levy(U, 2)


def levy_3d(U):
    return _levy(U, 3)


def levy_4d(U):
    return _levy(U, 4)


def levy_5d(U):
    return _levy(U, 5)


def _smooth_box(U, D, k=20, box_start=-0.5):
    X = _to_domain(_2d(U), [-3.0] * D, [3.0] * D)
    out = torch.ones(X.shape[0], dtype=X.dtype)
    for d in range(D):
        out = out * (1 / (1 + torch.exp(-k * (X[:, d] - box_start))))
    return out


def smooth_box_2d(U):
    return _smooth_box(U, 2)


def smooth_box_3d(U):
    return _smooth_box(U, 3)


def smooth_box_4d(U):
    return _smooth_box(U, 4)


_H_ALPHA = [1.0, 1.2, 3.0, 3.2]


def _hartmann(U, A, P):
    X = _2d(U)
    A = torch.tensor(A, dtype=X.dtype)
    P = torch.tensor(P, dtype=X.dtype)
    alpha = torch.tensor(_H_ALPHA, dtype=X.dtype)
    return torch.sum(alpha * torch.exp(-torch.sum(A * (X.unsqueeze(1) - P) ** 2, dim=2)), dim=1)


def hartmann_3d(U):
    A = [[3.0, 10.0, 30.0], [0.1, 10.0, 35.0], [3.0, 10.0, 30.0], [0.1, 10.0, 35.0]]
    P = [[0.3689, 0.1170, 0.2673], [0.4699, 0.4387, 0.7470],
         [0.1091, 0.8732, 0.5547], [0.0381, 0.5743, 0.8828]]
    return -_hartmann(U, A, P)


_A6 = [[10.0, 3.0, 17.0, 3.5, 1.7, 8.0], [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
       [3.0, 3.5, 1.7, 10.0, 17.0, 8.0], [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]]
_P6 = [[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
       [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
       [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
       [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]]


def hartmann_4d(U):
    A = [r[:4] for r in _A6]
    P = [r[:4] for r in _P6]
    return (1.1 - _hartmann(U, A, P)) / 0.839


def hartmann_6d(U):
    return -_hartmann(U, _A6, _P6)


def styblinski_tang_3d(U):
    X = _to_domain(_2d(U), [-5.0] * 3, [5.0] * 3)
    return 0.5 * torch.sum(X ** 4 - 16.0 * X ** 2 + 5.0 * X, dim=1)


def friedman_5d(U):
    X = _2d(U)
    x1, x2, x3, x4, x5 = X.unbind(1)
    return (10.0 * torch.sin(math.pi * x1 * x2) + 20.0 * (x3 - 0.5) ** 2
            + 10.0 * x4 + 5.0 * x5)


SYNTHETIC = {
    "peaks": (peaks, 2), "smooth_box_2d": (smooth_box_2d, 2), "grlee08": (grlee08, 2),
    "levy2d": (levy2d, 2), "zhou98": (zhou98, 2), "beale": (beale, 2),
    "smooth_box_3d": (smooth_box_3d, 3), "hartmann_3d": (hartmann_3d, 3),
    "levy_3d": (levy_3d, 3), "styblinski_tang_3d": (styblinski_tang_3d, 3),
    "smooth_box_4d": (smooth_box_4d, 4), "hartmann_4d": (hartmann_4d, 4),
    "levy_4d": (levy_4d, 4),
    "friedman_5d": (friedman_5d, 5), "levy_5d": (levy_5d, 5),
    "hartmann_6d": (hartmann_6d, 6),
}


def get_function(name):
    """Return the benchmark function and its dimension."""
    if name in UCI:
        return _surrogate(name), UCI[name]
    return SYNTHETIC[name]


def standardized(name):
    """Return the benchmark scaled to zero mean and unit standard deviation, and its dimension."""
    f, D = get_function(name)
    fb = f(torch.quasirandom.SobolEngine(D, scramble=True, seed=999).draw(8192).to(torch.float64)).reshape(-1)
    fm, fs = fb.mean(), fb.std().clamp_min(1e-8)
    return (lambda Z: (f(Z).reshape(-1) - fm) / fs), D


ALL_FUNCTIONS = list(SYNTHETIC) + list(UCI)
