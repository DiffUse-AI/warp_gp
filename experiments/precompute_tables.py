"""Precompute the geometric-equalizer tables for D = 2, ..., 6 at the benchmark noise level and save
them to data/equalizer/. The tables cover the default budget of 200 rounds.

    python experiments/precompute_tables.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from warp_gp import loop  # noqa: E402
from warp_gp.equalizer import EqualizerTable  # noqa: E402
from warp_gp.gp import default_lengthscale  # noqa: E402

for D in range(2, 7):
    table = EqualizerTable(D, default_lengthscale(D), loop.TABLE_NOISE)
    table.precompute(loop.N0[D], loop.N0[D] + 199)
    print(D, table.path, flush=True)
