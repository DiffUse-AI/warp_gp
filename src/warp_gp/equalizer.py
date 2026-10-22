"""Geometric equalizer g(x) = min(1, vbar_N(centre) / vbar_N(x)), where vbar_N is the posterior
variance averaged over scrambled Sobol designs of size N. Reflections and permutations of the
cube are used to reduce the Monte Carlo variance.
"""
import itertools

import numpy as np
import torch

from .gp import DT, matern52


def _k(A, B, ell):
    return matern52(torch.cdist(A, B) / ell)


def geometric_equalizer(C, n, D, ell, noise, S=None, SYM=16):
    S = S or max(8, 64 // 2 ** (D - 2))
    q = torch.cat([C, torch.full((1, D), 0.5)])  # last row: centre of the domain
    gen = torch.Generator().manual_seed(2024)
    vbar = torch.zeros(len(q))
    for s in range(S):
        Xd = torch.quasirandom.SobolEngine(D, scramble=True, seed=101 + s).draw(n).to(DT)
        Ki = torch.linalg.inv(_k(Xd, Xd, ell) + noise * torch.eye(n))
        qs = []
        for _ in range(SYM):
            perm = torch.randperm(D, generator=gen)
            flip = torch.randint(0, 2, (D,), generator=gen).to(DT)
            qs.append(flip + (1.0 - 2.0 * flip) * q[:, perm])
        kc = _k(torch.cat(qs), Xd, ell)
        vbar += (1.0 - (kc @ Ki * kc).sum(1)).reshape(SYM, -1).mean(0)
    return (vbar[-1] / vbar[:-1].clamp_min(1e-12)).clamp(max=1.0)


class EqualizerTable:
    """Geometric equalizer tabulated on the sorted coordinates min(x, 1 - x) and interpolated
    multilinearly. The table is rebuilt when N has grown by 10%."""

    LEVELS = (0.0, 0.0125, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5)

    def __init__(self, D, ell, noise, S=512):
        self.D, self.ell, self.noise, self.S = D, ell, noise, S
        self.lev = torch.tensor(self.LEVELS)
        self.n = None

    def _build(self, n):
        D, L = self.D, len(self.lev)
        idx = np.array(list(itertools.combinations_with_replacement(range(L), D)))
        g = geometric_equalizer(self.lev[torch.as_tensor(idx)], n, D, self.ell, self.noise, S=self.S)
        grid = np.stack(np.meshgrid(*[np.arange(L)] * D, indexing="ij"), -1).reshape(-1, D)
        poly = L ** np.arange(D)[::-1]
        keys = idx @ poly
        order = np.argsort(keys)
        pos = np.searchsorted(keys[order], np.sort(grid, 1) @ poly)
        self.G = torch.as_tensor(g.numpy()[order[pos]])
        self.n = n

    def __call__(self, C, n):
        if self.n is None or n >= 1.1 * self.n:
            self._build(n)
        lev, L = self.lev, len(self.lev)
        z = torch.sort(torch.minimum(C, 1 - C), 1).values.clamp(0, 0.5)
        hi = torch.searchsorted(lev, z.contiguous()).clamp(1, L - 1)
        lo = hi - 1
        w = ((z - lev[lo]) / (lev[hi] - lev[lo])).clamp(0, 1)
        out = torch.zeros(len(C))
        for corner in itertools.product((0, 1), repeat=self.D):
            cw, flat = torch.ones(len(C)), torch.zeros(len(C), dtype=torch.long)
            for j, b in enumerate(corner):
                cw = cw * (w[:, j] if b else 1 - w[:, j])
                flat = flat * L + (hi if b else lo)[:, j]
            out += cw * self.G[flat]
        return out
