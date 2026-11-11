import math
import torch
torch.set_default_dtype(torch.float64)
SQ5 = math.sqrt(5.0)


def _parts(Delta, ls, s2):
    """Delta (..., D). Returns k, g = (dk/dDelta)/(s2*Delta/ls^2), e = exp(-sqrt5 r)."""
    u = Delta / ls
    r = (u * u).sum(-1).add(1e-24).sqrt()
    z = SQ5 * r
    e = torch.exp(-z)
    k = s2 * (1.0 + z + z * z / 3.0) * e
    g = -(5.0 / 3.0) * (1.0 + z) * e          # phi'(r)/r for phi(r) = (1+z+z^2/3)e^{-z}
    return k, g, e


def kernel_cross(A, B, ls, s2):
    """k(a, b) for all pairs, shape (|A|, |B|)."""
    Delta = A[:, None, :] - B[None, :, :]
    k, _, _ = _parts(Delta, ls, s2)
    return k


def grad_cross(A, B, ls, s2):
    """d k(a, b) / d a_d for all pairs, shape (|A|, |B|, D)."""
    Delta = A[:, None, :] - B[None, :, :]
    _, g, _ = _parts(Delta, ls, s2)
    return s2 * g[..., None] * Delta / ls ** 2


def var_normal_abs(mu, var):
    """Variance of |X| for X ~ N(mu, var). Same formula as the attached file."""
    var = var.clamp_min(1e-12)
    sd = var.sqrt()
    Phi = torch.special.ndtr(-mu / sd)
    mean_abs = sd * math.sqrt(2.0 / math.pi) * torch.exp(-0.5 * mu * mu / var) + mu * (1.0 - 2.0 * Phi)
    return mu * mu + var - mean_abs * mean_abs


def abs_var_reduction(X, y, C, noise, ls, s2=1.0, obs_var=None, Q=None):
    """Score each candidate in C.

    X (n, D), y (n,): labeled data (y on the same scale as the GP was trained on).
    C (m, D): candidates. 
    Q (p, D): query set for the global version; None = local (q = c).
    ls (D,): lengthscales. 
    s2: signal variance. 
    noise: GP noise variance.
    obs_var: variance of the lookahead observation (default: noise).
    Returns (m,) scores; larger is better.
    """
    obs_var = noise if obs_var is None else obs_var
    n = len(X)
    Kn = kernel_cross(X, X, ls, s2) + noise * torch.eye(n)
    L = torch.linalg.cholesky(Kn)
    alpha = torch.cholesky_solve(y[:, None], L).squeeze(-1)

    KXC = kernel_cross(X, C, ls, s2)                   # (n, m)
    Z = torch.cholesky_solve(KXC, L)                    # Kn^-1 k(X, c)
    vC = s2 + obs_var - (KXC * Z).sum(0)                # Var(f(c) | data) + obs_var, (m,)

    local = Q is None
    Qp = C if local else Q
    dKQX = grad_cross(Qp, X, ls, s2)                    # (p, n, D): Cov(grad f(q), f(X))
    p, D = Qp.shape[0], Qp.shape[1]

    # posterior mean of grad f(q)
    muQ = torch.einsum("pnd,n->pd", dKQX, alpha)        # (p, D)

    # posterior marginal variance of grad f(q)
    prior_var = 5.0 * s2 / (3.0 * ls ** 2)              # Var(d_d f) prior for Matern-5/2
    W = torch.cholesky_solve(dKQX.permute(1, 0, 2).reshape(n, p * D), L).reshape(n, p, D)
    varQ = prior_var - (dKQX * W.permute(1, 0, 2)).sum(1)   # (p, D)
    varQ = varQ.clamp_min(1e-12)

    if local:
        # Cov(grad f(c), f(c)) = 0 (zero at Delta = 0), so the cross term is only the data part.
        b = -torch.einsum("cnd,nc->cd", dKQX, Z)        # (m, D)
        after = (varQ - b * b / vC[:, None]).clamp_min(1e-12)
        red = var_normal_abs(muQ, varQ) - var_normal_abs(muQ, after)
        return red.sum(-1)                              # (m,)

    # global: cross covariance between grad f(q) and f(c), posterior given data
    dKQC = grad_cross(Qp, C, ls, s2)                    # (p, m, D)
    b = dKQC - torch.einsum("pnd,nc->pcd", dKQX, Z)     # (p, m, D)
    after = (varQ[:, None, :] - b * b / vC[None, :, None]).clamp_min(1e-12)
    red = var_normal_abs(muQ[:, None, :], varQ[:, None, :]) - var_normal_abs(
        muQ[:, None, :], after
    )
    return red.sum(-1).mean(0)