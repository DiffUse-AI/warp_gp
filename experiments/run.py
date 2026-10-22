"""Run one method on one benchmark for one seed and save the result to results/.

    python experiments/run.py --func grlee08 --seed 0 --method eq_warp  # cube, 200 rounds
    python experiments/run.py --func cap_s2 --seed 0 --method warp        # sphere, 100 rounds
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from warp_gp import acquisition, loop, sphere  # noqa: E402
from warp_gp.functions import ALL_FUNCTIONS, UCI  # noqa: E402

CUBE_METHODS = ("fps",) + acquisition.METHODS

ap = argparse.ArgumentParser()
ap.add_argument("--func", required=True, choices=ALL_FUNCTIONS + list(sphere.FUNCS))
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--method", required=True, choices=sorted(set(CUBE_METHODS + sphere.METHODS)))
ap.add_argument("--rounds", type=int, default=None, help="default: 200 (cube), 100 (sphere)")
ap.add_argument("--out", default="results")
a = ap.parse_args()

path = os.path.join(a.out, f"{a.func}_s{a.seed}_{a.method}.npz")
if not os.path.exists(path):
    if a.func in sphere.FUNCS:
        assert a.method in sphere.METHODS, f"sphere methods: {sphere.METHODS}"
        X0, picks = sphere.run(a.func, a.seed, a.method, a.rounds or 100)
    else:
        assert a.method in CUBE_METHODS, f"cube methods: {CUBE_METHODS}"
        # the real-data benchmarks use the tabulated equalizer
        X0, picks, _ = loop.run(a.func, a.seed, a.method, a.rounds or 200, table=a.func in UCI)
    os.makedirs(a.out, exist_ok=True)
    np.savez(path, X0=X0.numpy(), picks=picks.numpy())
print(path)
