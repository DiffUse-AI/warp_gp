"""Area reduction relative to FPS (mean ± standard error over seeds) for all saved runs.

    python experiments/table.py                    # cube: 50, 100, 200 rounds
    python experiments/table.py --budgets 20,100   # sphere
"""
import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from warp_gp import sphere  # noqa: E402
from warp_gp.evaluate import onenn_trajectory, true_field  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--budgets", default="50,100,200")
a = ap.parse_args()
budgets = [int(b) for b in a.budgets.split(",")]

runs = defaultdict(dict)
for p in glob.glob(os.path.join(a.results, "*.npz")):
    m = re.match(r"(.+)_s(\d+)_([a-z_]+)\.npz$", os.path.basename(p))
    if m:
        with np.load(p) as z:
            runs[m[1]][int(m[2]), m[3]] = {k: z[k] for k in z.files}


field = {f: true_field(f) for f in runs if f not in sphere.FUNCS}


def area(func, z, b):
    """Error summed over rounds up to b (cube: rounds 1..b; sphere: rounds 0, 2, ..., b)."""
    if func in sphere.FUNCS:
        return sphere.onenn_curve(func, z["X0"], z["picks"][:b], b).sum()
    return onenn_trajectory(z["X0"], z["picks"][:b], *field[func]).sum()


for func, files in sorted(runs.items()):
    for method in sorted({k[1] for k in files} - {"fps"}):
        cells = []
        for b in budgets:
            ar = [100 * (1 - area(func, z, b) / area(func, files[s, "fps"], b))
                  for (s, mm), z in files.items()
                  if mm == method and (s, "fps") in files and len(z["picks"]) >= b]
            cells.append(f"{np.mean(ar):6.1f} ± {np.std(ar, ddof=1) / np.sqrt(len(ar)):.1f}"
                         if len(ar) > 1 else "     n/a")
        print(f"{func:20s} {method:10s} " + "  ".join(cells))
