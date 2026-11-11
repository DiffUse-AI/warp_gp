import numpy as np


def _score(rel, d):
    """Neighbourhood score S = R / C, Eqs. (4.3)-(4.7). `rel` holds offsets from the reference point."""
    if len(rel) < 2:
        return 0.0
    C = np.linalg.norm(rel, axis=1).mean()                        # cohesion, Eq. (4.3)
    if C <= 0:
        return 0.0
    if d == 1:                                                     # Eq. (4.6), two neighbours
        a, b = rel[0, 0], rel[1, 0]
        den = abs(a) + abs(b) + abs(a - b)
        R = 1.0 - abs(a + b) / den if den > 0 else 0.0
    else:
        D = np.linalg.norm(rel[:, None, :] - rel[None, :, :], axis=-1)
        np.fill_diagonal(D, np.inf)
        A = D.min(axis=1).mean()                                   # adhesion, Eq. (4.4)
        R = A / (np.sqrt(2.0) * C)                                 # cross-polytope ratio, Eq. (4.5)
    return R / C                                                   # Eq. (4.7)


def _try_add(X, nb, r, q, m, d):
    """Offer candidate q to the neighbourhood of r (neighbourhood step of Algorithm 3)."""
    cur = nb[r]
    if q in cur:
        return
    if len(cur) < m:
        nb[r] = cur + [q]
        return
    best, best_s = None, _score(X[cur] - X[r], d)
    for j in range(len(cur)):
        cand = cur[:j] + cur[j + 1:] + [q]                         # replace neighbour j by q
        s = _score(X[cand] - X[r], d)
        if s > best_s:
            best, best_s = cand, s
    if best is not None:
        nb[r] = best


def _neighbourhoods(X, m, d):
    """Incremental neighbourhood construction, in arrival order, as in Algorithm 3."""
    nb = [[] for _ in range(len(X))]
    for k in range(len(X)):
        for r in range(k):
            _try_add(X, nb, r, k, m, d)                            # k as candidate neighbour of r
            _try_add(X, nb, k, r, m, d)                            # r as candidate neighbour of k
    return nb


def _nonlinearity(X, y, nb):
    """E(p_r) from Eqs. (4.8)-(4.9): hyperplane through p_r, summed absolute residuals of neighbours."""
    E = np.zeros(len(X))
    for r, idx in enumerate(nb):
        if not idx:
            continue
        rel = X[idx] - X[r]
        db = y[idx] - y[r]
        g = np.linalg.lstsq(rel, db, rcond=None)[0]
        E[r] = np.abs(db - rel @ g).sum()
    return E


def lola_pick(X, y, C):
    """Index into candidate pool C chosen by LOLA-Voronoi (Crombecq et al., 2011), pool-based."""
    Xn = X.detach().cpu().double().numpy()
    yn = y.detach().cpu().double().reshape(-1).numpy()
    Cn = C.detach().cpu().double().numpy()
    n, d = Xn.shape
    m = 2 * d

    # Exploration: Voronoi mass of each design point, estimated on the candidate pool
    owner = np.linalg.norm(Cn[:, None, :] - Xn[None, :, :], axis=-1).argmin(axis=1)
    V = np.bincount(owner, minlength=n) / len(Cn)

    # Exploitation: local nonlinearity from observed values and neighbourhood geometry
    nb = _neighbourhoods(Xn, m, d)
    E = _nonlinearity(Xn, yn, nb)
    H = V + (E / E.sum() if E.sum() > 0 else 0.0)            # Eq. (5.1)

    # Highest-ranked cell that contains candidates; take the candidate farthest from p_i and its neighbours
    for i in np.argsort(-H):
        idx = np.where(owner == i)[0]
        if len(idx) == 0:
            continue
        ref = Xn[[i] + nb[i]]
        dist = np.linalg.norm(Cn[idx][:, None, :] - ref[None, :, :], axis=-1).min(axis=1)
        return int(idx[dist.argmax()])
    raise RuntimeError("no candidate lies in any Voronoi cell")