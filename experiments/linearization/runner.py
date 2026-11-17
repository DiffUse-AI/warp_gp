"""Acquisition runs for the linearization robustness study (see the experiment plan).

Mirrors warp_gp.loop.run (same initial Sobol design, candidate pool, noise draws and acquisition GP), with
three differences: the test function is a family member from families.py, its standardization is fixed to
the clean member of the family, and the noise variance is fixed to 0.0025 (5% of the clean standard
deviation, squared), so that only the dial changes along a family. The equalizer is tabulated.
Arms:
  eq       equalizer alone (observation-independent baseline)
  warp     equalizer + reconstruction warp (the paper's method), floor rho0
  oracle   equalizer + warp built from the TRUE gradient (same density law, floor and transport)
  safe     warp + equalized reference safeguard at level tau
No repository file is modified.
"""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)
from families import clean, family  # noqa: E402
from warp_gp.equalizer import EqualizerTable  # noqa: E402
from warp_gp.evaluate import onenn_trajectory  # noqa: E402
from warp_gp.gp import DT, default_lengthscale, grad_posterior_mean, posterior_variance  # noqa: E402
from warp_gp.loop import N0, NCAND, TABLE_NOISE, sobol  # noqa: E402
from warp_gp.warp import design_density, kr_from_density  # noqa: E402

NOISE = TABLE_NOISE


def standardized(name, D, p):
    f, f0 = family(name, D, p), clean(name, D, p)
    z = f0(sobol(D, 8192, 999))
    m, s = z.mean(), z.std()
    return lambda U: (f(U.reshape(1, -1) if U.ndim == 1 else U) - m) / s


def probes(D):
    n = 8192 if D <= 3 else 16384
    return torch.quasirandom.SobolEngine(D, scramble=True, seed=7).draw(n).to(DT).clamp(1e-4, 1 - 1e-4)


def warp_from(gnorm, U, D, rho0):
    return kr_from_density(U, design_density(gnorm, D, floor=rho0), D, n_cond=3 if D >= 5 else 4)


def run(name, D, p, seed, arm, rounds=200, rho0=1.0, tau=0.1, table=None):
    fn = standardized(name, D, p)
    ell = default_lengthscale(D)
    g = table or EqualizerTable(D, ell, TABLE_NOISE)
    X0 = sobol(D, N0[D], seed)
    y0 = fn(X0) + NOISE ** 0.5 * torch.randn(len(X0), generator=torch.Generator().manual_seed(1000 + seed))
    C = sobol(D, NCAND[D], 2000 + seed).clamp(1e-4, 1 - 1e-4)
    U = probes(D)
    oracle_T = None
    if arm == "oracle":
        Ug = U.clone().requires_grad_(True)
        (gr,) = torch.autograd.grad(fn(Ug).sum(), Ug)
        oracle_T = warp_from(gr.norm(dim=1), U, D, rho0)
    gen = torch.Generator().manual_seed(8000 + seed)
    X, y, hits = X0, y0, 0
    for _ in range(rounds):
        geq = g(C, len(X))
        if arm == "eq":
            s = posterior_variance(X, C, NOISE, ell) * geq
        else:
            T = oracle_T if arm == "oracle" else warp_from(
                grad_posterior_mean(X, y, U, NOISE, ell).norm(dim=1), U, D, rho0)
            with torch.no_grad():
                s = posterior_variance(X, C, NOISE, ell, T) * geq
        i = int(s.argmax())
        if arm == "safe":
            ref = posterior_variance(X, C, NOISE, ell) * geq
            ok = ref >= tau * ref.max()
            if not ok[i]:
                hits += 1
                i = int(torch.where(ok, s, torch.full_like(s, -1.0)).argmax())
        x = C[i][None]
        X = torch.cat([X, x])
        y = torch.cat([y, fn(x) + NOISE ** 0.5 * torch.randn(1, generator=gen)])
    return fn, X0, X[len(X0):], hits


def errors(fn, D, X0, picks):
    T = torch.quasirandom.SobolEngine(D, scramble=True, seed=12321).draw(16384).to(DT)
    with torch.no_grad():
        fT = fn(T).numpy()
    return onenn_trajectory(X0.numpy(), picks.numpy(), T, fT, lambda Z: fn(torch.as_tensor(Z)).detach().numpy())


if __name__ == "__main__":
    # timed pilot: one dial value per family, D = 2 and 4, one seed, every arm
    for D in (2, 4):
        for name, p in (("A", 0.5), ("B", 0.02), ("C", 1.0)):
            for arm in ("eq", "warp", "oracle", "safe"):
                t = time.time()
                fn, X0, picks, hits = run(name, D, p, 0, arm)
                e = errors(fn, D, X0, picks)
                print(f"D={D} {name}({p}) {arm:6s}: {time.time() - t:6.1f} s, final error {e[-1]:.4f}, "
                      f"area {e.sum():.2f}" + (f", safeguard interventions {hits}" if arm == "safe" else ""), flush=True)
