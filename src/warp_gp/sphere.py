"""Sequential acquisition on the spheres S^2 and S^3, represented as unit vectors in R^{d+1}.

The GP uses a chordal Matern-5/2 kernel with two lengthscales. The reconstruction warp
reparameterizes the angle to one or two centres so that the design density becomes
1 + (||tangential grad mu|| / c)^(2d/(d+2)).
"""
import math

import numpy as np
import torch

DT = torch.float64
torch.set_default_dtype(DT)

NOISE = 0.05 ** 2
N0 = {2: 20, 3: 40}
NCAND = {2: 32768, 3: 65536}
# lengthscales of the acquisition kernel and of the posterior-mean kernel, per dimension
ELLS = {2: ((0.32, 0.79), (0.32, 2.0)), 3: ((0.52, 0.79), (0.52, 2.0))}
FUNCS = {"cap_s2": 2, "twocaps_s2": 2, "cap_s3": 3, "twocaps_s3": 3}
METHODS = ("fps", "eig", "warp", "sobol", "random")


def uniform(n, dim, gen):
    z = torch.randn(n, dim + 1, generator=gen, dtype=DT)
    return z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)


def sobol(n, dim, seed):
    """Scrambled Sobol points on S^dim, mapped by the Lambert equal-area map on S^2 and by the
    inverse normal CDF on S^3."""
    m = max(1, math.ceil(math.log2(max(n, 2))))
    eng = torch.quasirandom.SobolEngine(2 if dim == 2 else dim + 1, scramble=True, seed=seed)
    U = eng.draw(2 ** m).to(DT)[:n].clamp(1e-9, 1 - 1e-9)
    if dim == 2:
        z, phi = 1.0 - 2.0 * U[:, 0], 2.0 * math.pi * U[:, 1]
        r = (1.0 - z * z).clamp_min(0.0).sqrt()
        return torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], 1)
    g = torch.erfinv(2 * U - 1) * math.sqrt(2.0)
    return g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)


def grid(n, dim):
    """Deterministic, nearly uniform candidate grid: a Fibonacci lattice on S^2 and a Kronecker
    lattice in Hopf coordinates on S^3."""
    i = torch.arange(n, dtype=DT)
    if dim == 2:
        phi = math.pi * (3.0 - math.sqrt(5.0)) * i
        z = 1.0 - 2.0 * (i + 0.5) / n
        r = (1 - z * z).clamp_min(0).sqrt()
        return torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], 1)
    rho = 1.32471795724474602596  # plastic number
    eta = torch.asin(((i + 0.5) / n).clamp(0, 1).sqrt())
    xi1, xi2 = 2 * math.pi * torch.frac(i * (1 / rho)), 2 * math.pi * torch.frac(i * (1 / rho ** 2))
    return torch.stack([torch.cos(eta) * torch.cos(xi1), torch.cos(eta) * torch.sin(xi1),
                        torch.sin(eta) * torch.cos(xi2), torch.sin(eta) * torch.sin(xi2)], 1)


def make_func(name):
    dim = FUNCS[name]
    unit = lambda v: v / v.norm().clamp_min(1e-12)
    e = torch.ones(dim + 1, dtype=DT)
    c1 = unit(e)
    e2 = e.clone(); e2[0], e2[1] = -1.0, 2.0
    c2 = unit(e2)
    if name.startswith("cap"):
        return lambda X: torch.sigmoid(20.0 * (X @ c1 - math.cos(math.radians(60))))
    return lambda X: (torch.sigmoid(25.0 * (X @ c1 - math.cos(math.radians(50))))
                      - 0.8 * torch.sigmoid(25.0 * (X @ c2 - math.cos(math.radians(55)))))


def m52(r, l):
    a = math.sqrt(5.0) / l
    return (1 + a * r + (a * r) ** 2 / 3) * torch.exp(-a * r)


def dm52(r, l):
    a = math.sqrt(5.0) / l
    return -(a ** 2 * r / 3) * (1 + a * r) * torch.exp(-a * r)


def kern(A, B, ells):
    r = torch.cdist(A, B).clamp_min(0)
    return sum(m52(r, l) / len(ells) for l in ells)


def chol(X, ells):
    return torch.linalg.cholesky(kern(X, X, ells) + (NOISE + 1e-8) * torch.eye(len(X)))


def variance(X, C, ells, warp=None):
    if warp is not None:
        X, C = warp(X), warp(C)
    K = kern(X, C, ells)
    return (1.0 - (K * torch.cholesky_solve(K, chol(X, ells))).sum(0)).clamp_min(1e-10)


def grad_norm(X, alpha, U, ells):
    """Norm of the tangential gradient of the posterior mean at U."""
    diff = U[:, None, :] - X[None, :, :]
    r = diff.norm(dim=2).clamp_min(1e-9)
    dk = sum(dm52(r, l) / len(ells) for l in ells)
    g = ((alpha[None, :] * dk / r)[:, :, None] * diff).sum(1)
    return (g - (g * U).sum(1, keepdim=True) * U).norm(dim=1)


class CapWarp:
    """Composition of angle reparameterizations a -> G^{-1}(F(a)) about each centre, where F is the
    CDF of the design density along the angle and G is the uniform CDF on the sphere."""

    def __init__(self, centers, gmaps, agrid):
        self.centers, self.gmaps, self.agrid = centers, gmaps, agrid

    def _one(self, X, c, gv):
        cosa = (X @ c).clamp(-1 + 1e-9, 1 - 1e-9)
        a = torch.acos(cosa)
        i = torch.searchsorted(self.agrid, a.contiguous()).clamp(1, len(self.agrid) - 1)
        a0, a1 = self.agrid[i - 1], self.agrid[i]
        anew = gv[i - 1] + (a - a0) / (a1 - a0 + 1e-12) * (gv[i] - gv[i - 1])
        e = X - cosa[:, None] * c[None, :]
        en = e.norm(dim=1, keepdim=True)
        e = torch.where(en > 1e-9, e / en.clamp_min(1e-12), e)
        out = torch.cos(anew)[:, None] * c[None, :] + torch.sin(anew)[:, None] * e
        pole = en.squeeze(1) <= 1e-9
        out[pole] = X[pole]
        return out / out.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def __call__(self, X):
        for c, g in zip(self.centers, self.gmaps):
            X = self._one(X, c, g)
        return X


def build_warp(X, alpha, U, dim, ells, n_grid=33):
    g = grad_norm(X, alpha, U, ells)
    rho = 1.0 + (g / g.pow(2).mean().sqrt().clamp_min(1e-12)).pow(2.0 * dim / (dim + 2.0))
    excess = (rho - 1.0).clamp_min(0)
    # centres: the density-weighted mean direction and, if present, a second peak at least 60 degrees away
    centers = []
    c1 = (excess[:, None] * U).sum(0)
    c1 = c1 / c1.norm().clamp_min(1e-12)
    if c1.norm() > 1e-6:
        centers.append(c1)
    far = (U @ (centers[0] if centers else U[0])) < math.cos(math.radians(60))
    if far.any() and excess[far].max() > 0.5 * excess.max():
        centers.append(U[far][int(excess[far].argmax())])
    agrid = torch.linspace(0, math.pi, n_grid, dtype=DT)
    mid = 0.5 * (agrid[:-1] + agrid[1:])
    gmaps = []
    for c in centers:
        a = torch.acos((U @ c).clamp(-1 + 1e-9, 1 - 1e-9))
        i = torch.bucketize(a.contiguous(), agrid).clamp(1, n_grid - 1) - 1
        dens = torch.zeros(n_grid - 1).scatter_add_(0, i, rho)
        dens = dens / torch.zeros(n_grid - 1).scatter_add_(0, i, torch.ones_like(rho)).clamp_min(1)
        F = torch.cat([torch.zeros(1), torch.cumsum(dens * torch.sin(mid), 0)])
        F = F / F[-1].clamp_min(1e-12)
        gmaps.append(torch.acos((1 - 2 * F).clamp(-1, 1)))
    return CapWarp(centers, gmaps, agrid)


def standardized(name):
    dim = FUNCS[name]
    f = make_func(name)
    fb = f(uniform(8192, dim, torch.Generator().manual_seed(999)))
    fm, fs = fb.mean(), fb.std().clamp_min(1e-8)
    return lambda Z: (f(Z) - fm) / fs, dim


def run(name, seed, method, rounds=100):
    """Run one method and return the initial design and the selected points."""
    fn, dim = standardized(name)
    acq_ells, mean_ells = ELLS[dim]
    X0 = sobol(N0[dim], dim, seed)
    y = fn(X0) + math.sqrt(NOISE) * torch.randn(len(X0), generator=torch.Generator().manual_seed(1000 + seed))
    if method == "sobol":
        return X0, sobol(len(X0) + rounds, dim, seed)[len(X0):]
    if method == "random":
        gen = torch.Generator().manual_seed(9000 + seed)
        return X0, torch.cat([uniform(1, dim, gen) for _ in range(rounds)])
    C = grid(NCAND[dim], dim)
    U = uniform(4096, dim, torch.Generator().manual_seed(4000 + seed))
    noise_gen = torch.Generator().manual_seed(8000 + seed)
    X = X0
    for _ in range(rounds):
        if method == "fps":
            s = torch.cdist(C, X).min(1).values
        else:
            warp = None
            if method == "warp":
                alpha = torch.cholesky_solve(y[:, None], chol(X, mean_ells)).squeeze(-1)
                warp = build_warp(X, alpha, U, dim, mean_ells)
            s = variance(X, C, acq_ells, warp)
        x = C[int(s.argmax())][None]
        X = torch.cat([X, x])
        y = torch.cat([y, fn(x) + math.sqrt(NOISE) * torch.randn(1, generator=noise_gen)])
    return X0, X[len(X0):]


def onenn_curve(name, X0, picks, rounds, step=2, n_test=16384):
    """Mean squared error of the 1-NN reconstruction after rounds 0, step, 2 * step, ..., evaluated
    on a fixed Sobol test set on the sphere."""
    fn, dim = standardized(name)
    T = sobol(n_test, dim, 12321)
    fT = fn(T)
    X = torch.cat([torch.as_tensor(X0), torch.as_tensor(picks)])
    fX = fn(X)
    out = []
    for b in range(0, rounds + 1, step):
        idx = torch.cdist(T, X[:len(X0) + b]).argmin(1)
        out.append(float(((fX[idx] - fT) ** 2).mean()))
    return np.array(out)
