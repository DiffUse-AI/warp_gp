"""Sequential acquisition on [0,1]^D."""
import torch

from .acquisition import score
from .functions import get_function, standardized
from .gp import DT, default_lengthscale
from .equalizer import EqualizerTable

N0 = {2: 20, 3: 40, 4: 60, 5: 80, 6: 100}
NCAND = {2: 2048, 3: 4096, 4: 8192, 5: 16384, 6: 32768}


def sobol(D, n, seed):
    return torch.quasirandom.SobolEngine(D, scramble=True, seed=seed).draw(n).to(DT)


def setup(func_name, seed, ncand=None):
    """Return the standardized function, its dimension, the noise variance (5% of the function's
    standard deviation), the initial design with noisy observations, and the candidate pool."""
    f, D = get_function(func_name)
    fn, _ = standardized(func_name)
    pilot = f(sobol(D, 4096, 0)).reshape(-1)
    sig = max(0.05 * float(pilot.std(unbiased=False)), 1e-12)
    fs = f(sobol(D, 8192, 999)).reshape(-1).std().clamp_min(1e-8)
    noise = max((sig / float(fs)) ** 2, 1e-4)
    X0 = sobol(D, N0[D], seed)
    y0 = fn(X0) + noise ** 0.5 * torch.randn(len(X0), generator=torch.Generator().manual_seed(1000 + seed))
    C = sobol(D, ncand or NCAND[D], 2000 + seed).clamp(1e-4, 1 - 1e-4)
    return fn, D, noise, X0, y0, C


def run(func_name, seed, method, rounds=200, ncand=None, table=False):
    """Run one method and return the initial design, the selected points and their observations."""
    fn, D, noise, X0, y0, C = setup(func_name, seed, ncand)
    ell = default_lengthscale(D)
    g = EqualizerTable(D, ell, noise) if table else None
    gen = torch.Generator().manual_seed(8000 + seed)
    X, y = X0, y0
    for _ in range(rounds):
        if method == "fps":
            i = torch.cdist(C, X).min(1).values.argmax()
        else:
            i = score(method, X, y, C, noise, ell, g).argmax()
        x = C[int(i)][None]
        X = torch.cat([X, x])
        y = torch.cat([y, fn(x) + noise ** 0.5 * torch.randn(1, generator=gen)])
    return X0, X[len(X0):], y[len(X0):]
