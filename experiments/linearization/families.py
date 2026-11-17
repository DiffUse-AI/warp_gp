"""Test-function families for the linearization robustness study (see the experiment plan).

Every length is expressed in units of h_D, the mean nearest-neighbour distance of a scrambled Sobol design
of size N0[D] + 50, the primary budget of 50 rounds (measured, 5 scrambles). Every feature depends on x1 (the first
Knothe-Rosenblatt coordinate) except the texture, which depends on all coordinates.

A  sharpening step:  f = tanh((x1 - 0.5) / (m h_D))                      dial m (edge width / spacing)
B  global texture:   f = S(x1) + a R_{lam}(x),  lam = 0.5 h_D            dial a (texture amplitude)
C  decoy texture:    f = S(x1) + a_C W(x1) R_{m h_D}(x)                  dial m (texture wavelength / spacing)
   a_C is set per dimension so that the window holds about half of a 50-round Sobol design's 1-NN error
   once the texture is unresolved. S(x1) = tanh((x1 - 0.2) / h_D), the anchoring step; W a logistic
   window on x1 in [0.65, 0.95] (edge width 0.02); S is nearly but not exactly flat there (in 4D its slope
   at x1 = 0.65 is 10% of its maximum). R_lam(x) = D^(-1/2) sum_d sin(2 pi x_d / lam + phi_d).
Standardization (mean, std): family A per member (raw standard deviations differ, e.g. 0.28 to 0.97 in 4D,
so the standardized amplitude and signal-to-noise ratio are constant along A); families B and C by their
clean member (a = 0, a_C = 0), so the noise variance is fixed along them. Parameters, normalization and
phases are fixed at this 50-round calibration for every budget.
"""
import math

import torch

H = {2: 0.0824, 3: 0.1627, 4: 0.2500}  # mean NN distance, Sobol, N0 + 50 points (measured 2026-11-17)
PHI = [0.0, 1.9, 3.7, 5.3, 0.8, 2.6]


def texture(U, lam):
    D = U.shape[1]
    return sum(torch.sin(2 * math.pi * U[:, d] / lam + PHI[d]) for d in range(D)) / math.sqrt(D)


def window(x1, lo=0.65, hi=0.95, edge=0.02):
    return torch.sigmoid((x1 - lo) / edge) * torch.sigmoid((hi - x1) / edge)


A_C = {2: 0.32, 3: 0.46, 4: 0.49}  # window holds about half of the 50-round Sobol 1-NN error


def family(name, D, p, a_C=None):
    a_C = A_C[D] if a_C is None else a_C
    h = H[D]
    if name == "A":
        return lambda U: torch.tanh((U[:, 0] - 0.5) / (p * h))
    S = lambda U: torch.tanh((U[:, 0] - 0.2) / h)
    if name == "B":
        return lambda U: S(U) + p * texture(U, 0.5 * h)
    if name == "C":
        return lambda U: S(U) + a_C * window(U[:, 0]) * texture(U, p * h)
    raise ValueError(name)


def clean(name, D, p, a_C=None):
    """The member whose standardization is used for the whole dial."""
    if name == "A":
        return family("A", D, p)
    return family(name, D, 0.0) if name == "B" else family("C", D, 1.0, a_C=0.0)


DIALS = {"A": [4.0, 2.0, 1.0, 0.5, 0.25, 0.125],
         "B": [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3],
         "C": [8.0, 4.0, 2.0, 1.4, 1.0, 0.7, 0.5, 0.25]}
