"""Reconstruction warp: a Knothe-Rosenblatt map that transports the design density
rho(x) = 1 + (||grad mu(x)|| / c)^(2D/(D+2)) to the uniform distribution on [0,1]^D."""
import itertools

import torch

from .gp import DT, grad_posterior_mean


def design_density(grad_norm, D, floor=1.0):
    c = grad_norm.pow(2).mean().sqrt().clamp_min(1e-8)
    return floor + (grad_norm / c).pow(2.0 * D / (D + 2.0))


class KRWarp(torch.nn.Module):
    """Triangular transport map. T_d(x) is the conditional CDF of rho along x_d given
    x_1, ..., x_{d-1}. The CDFs are piecewise linear and interpolated multilinearly between
    conditioning cells."""

    def __init__(self, edges, tables, n_cond):
        super().__init__()
        self.edges, self.tables, self.n_cond = edges, tables, n_cond

    def _cdf(self, rows, x):
        nb = len(self.edges) - 1
        i = (torch.bucketize(x.contiguous(), self.edges) - 1).clamp(0, nb - 1)
        x0, x1 = self.edges[i], self.edges[i + 1]
        y0, y1 = rows.gather(1, i[:, None]).squeeze(1), rows.gather(1, (i + 1)[:, None]).squeeze(1)
        return y0 + (x - x0) / (x1 - x0 + 1e-12) * (y1 - y0)

    def forward(self, x):
        x = x.detach().clamp(1e-6, 1 - 1e-6)
        B, nc = len(x), self.n_cond
        out = []
        for d, table in enumerate(self.tables):
            if d == 0:
                rows = table.expand(B, -1)
            else:
                pos = (x[:, :d] * nc - 0.5).clamp(0, nc - 1)
                i0 = pos.floor().long().clamp(0, nc - 2)
                w = (pos - i0).clamp(0, 1)
                rows = torch.zeros(B, len(self.edges))
                for corner in itertools.product((0, 1), repeat=d):
                    cw, flat = torch.ones(B), torch.zeros(B, dtype=torch.long)
                    for j, b in enumerate(corner):
                        cw = cw * (w[:, j] if b else 1 - w[:, j])
                        flat = flat * nc + (i0[:, j] + b)
                    rows = rows + cw[:, None] * table[flat]
            out.append(self._cdf(rows, x[:, d]))
        return torch.stack(out, 1)


def kr_from_density(U, rho, D, n_bins=16, n_cond=4, reg=0.05):
    """Build the transport map for a density given by the weights rho at the probe points U."""
    edges = torch.linspace(0, 1, n_bins + 1, dtype=DT)
    cells = (U * n_cond).long().clamp(0, n_cond - 1)
    bins = [(torch.bucketize(U[:, d].contiguous(), edges) - 1).clamp(0, n_bins - 1) for d in range(D)]

    def cdf(d, mask):
        w = rho * mask
        h = torch.zeros(n_bins).scatter_add_(0, bins[d], w)
        h = h + (w.sum() / n_bins).clamp_min(1e-8) * reg + 1e-10
        c = torch.cat([torch.zeros(1), h.cumsum(0)])
        return c / c[-1]

    tables = [cdf(0, torch.ones(len(U)))[None]]
    for d in range(1, D):
        t = []
        for combo in itertools.product(range(n_cond), repeat=d):
            mask = torch.ones(len(U), dtype=torch.bool)
            for j, c in enumerate(combo):
                mask &= cells[:, j] == c
            t.append(cdf(d, mask.to(DT)))
        tables.append(torch.stack(t))
    return KRWarp(edges, tables, n_cond)


def build_reconstruction_warp(X, y, noise, ell, D):
    n_probe = 8192 if D <= 3 else 16384
    U = torch.quasirandom.SobolEngine(D, scramble=True, seed=7).draw(n_probe).to(DT).clamp(1e-4, 1 - 1e-4)
    g = grad_posterior_mean(X, y, U, noise, ell).norm(dim=1)
    return kr_from_density(U, design_density(g, D), D, n_cond=3 if D >= 5 else 4)
