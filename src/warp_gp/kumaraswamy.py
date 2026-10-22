"""Kumaraswamy input warp (baseline), refitted by marginal likelihood in every round."""
import torch

from .gp import log_marginal_likelihood

LO, HI = 0.2, 3.0  # range of both shape parameters


def _inv(p):
    z = torch.tensor((p - LO) / (HI - LO))
    return torch.log(z) - torch.log1p(-z)


class KumaraswamyWarp(torch.nn.Module):
    def __init__(self, D):
        super().__init__()
        self.raw_a = torch.nn.Parameter(torch.full((D,), float(_inv(1.0))))
        self.raw_b = torch.nn.Parameter(torch.full((D,), float(_inv(1.0))))

    def forward(self, X):
        a = LO + (HI - LO) * torch.sigmoid(self.raw_a)
        b = LO + (HI - LO) * torch.sigmoid(self.raw_b)
        cdf = lambda t: 1.0 - (1.0 - t.clamp(0, 1).pow(a)).pow(b)
        lo, hi = 1e-2, 1.0 - 1e-2
        u = lo + (hi - lo) * X
        g_lo, g_hi = cdf(torch.full_like(u, lo)), cdf(torch.full_like(u, hi))
        return (cdf(u) - g_lo) / (g_hi - g_lo + torch.finfo(X.dtype).eps)


def fit_kumaraswamy(X, y, noise, ell, iters=100, lr=1e-2):
    w = KumaraswamyWarp(X.shape[1])
    opt = torch.optim.Adam(w.parameters(), lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        (-log_marginal_likelihood(X, y, noise, ell, warp=w)).backward()
        opt.step()
    return w
