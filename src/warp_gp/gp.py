"""Matern-5/2 GP with fixed hyperparameters: zero mean, unit output scale and known noise."""
import math

import torch

DT = torch.float64
torch.set_default_dtype(DT)

ACQ_JITTER = 1e-4  # added to the noise variance when computing the acquisition variance
_EPS = 1e-12


def default_lengthscale(D):
    return math.sqrt(0.12 * 0.6) * math.sqrt(D / 2.0)


def matern52(r):
    z = math.sqrt(5.0) * r
    return (1.0 + z + z * z / 3.0) * torch.exp(-z)


def _dist(A, B):
    sq = (A ** 2).sum(-1, keepdim=True) + (B ** 2).sum(-1, keepdim=True).T - 2.0 * (A @ B.T)
    return (sq.clamp_min(0.0) + _EPS).sqrt()


def kernel(A, B, ell, warp=None):
    if warp is not None:
        A, B = warp(A), warp(B)
    return matern52(_dist(A, B) / ell)


def kernel_diag(A, ell):
    return matern52(torch.full((A.shape[0],), _EPS ** 0.5) / ell)


def posterior_variance(X, C, noise, ell, warp=None):
    """Posterior variance at C given the design X, with inputs transformed by the warp if one is given."""
    K = kernel(X, X, ell, warp) + (noise + ACQ_JITTER) * torch.eye(len(X))
    L = torch.linalg.cholesky(K)
    KXC = kernel(X, C, ell, warp)
    return (kernel_diag(C, ell) - (KXC * torch.cholesky_solve(KXC, L)).sum(0)).clamp_min(1e-8)


def posterior_mean(X, y, U, noise, ell):
    L = torch.linalg.cholesky(kernel(X, X, ell) + noise * torch.eye(len(X)))
    return kernel(U, X, ell) @ torch.cholesky_solve(y[:, None], L).squeeze(-1)


def grad_posterior_mean(X, y, U, noise, ell):
    """Gradient of the posterior mean at U, divided by the standard deviation of the mean over U."""
    U = U.detach().clone().requires_grad_(True)
    mu = posterior_mean(X, y, U, noise, ell)
    g = torch.autograd.grad(mu.sum(), U)[0]
    return g / mu.detach().std().clamp_min(1e-8)


def log_marginal_likelihood(X, y, noise, ell, warp=None):
    """Log marginal likelihood divided by the number of observations."""
    n = len(X)
    L = torch.linalg.cholesky(kernel(X, X, ell, warp) + noise * torch.eye(n))
    a = torch.cholesky_solve(y[:, None], L).squeeze(-1)
    return (-0.5 * (y * a).sum() - L.diagonal().log().sum() - 0.5 * n * math.log(2 * math.pi)) / n
