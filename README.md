# Correcting Boundary Bias and Observation Independence in Bayesian Experimental Design

This repository contains the implementation and experiments for the paper
[*Correcting Boundary Bias and Observation Independence in Bayesian Experimental Design*](https://arxiv.org/abs/2602.01898).

## Overview

For a Gaussian process with fixed hyperparameters, the posterior variance depends on where
measurements were taken but not on the measured values. Acquisition rules that maximize the
variance, such as expected information gain (EIG), therefore cannot place more measurements
where the function varies rapidly. On bounded domains they are also biased towards the
boundary, where the variance is inflated.

We address the two effects separately and modify only the GP used for acquisition:

- **Geometric equalizer.** The equalizer `g(x)` down-weights the boundary. It is the ratio of
  the reference variance at the centre of the domain to the reference variance at `x`, capped
  at one, where the reference variance is the posterior variance averaged over space-filling
  designs of the current size. It does not depend on the observations and can be precomputed.
- **Reconstruction warp.** The warp `T` makes acquisition depend on the observations. It is a
  Knothe–Rosenblatt transport map that realizes the design density
  `ρ(x) = 1 + (‖∇μ(x)‖ / c)^(2D/(D+2))`, where `μ` is the posterior mean and `c` is its
  root-mean-square gradient. The map is rebuilt from `μ` in every round and requires no training.

The next measurement maximizes `g(x) · s²_T(x)`, where `s²_T` is the posterior variance under
the warped kernel `k(T(x), T(x'))`.

## Installation

The code requires PyTorch and NumPy. The demo notebook also requires Matplotlib.

```bash
pip install torch numpy matplotlib
```

## Demo

The notebook `notebooks/demo.ipynb` illustrates both effects and both corrections on a
two-dimensional example and compares the acquisition rules over 60 rounds.

## Reproducing the experiments

`experiments/run.py` runs one acquisition method on one benchmark for one seed and saves the
selected points in `results/`. `experiments/table.py` evaluates all saved runs with
nearest-neighbour reconstruction and reports the area reduction (AR) relative to farthest-point
sampling (FPS) as mean ± standard error over seeds.

```bash
python experiments/run.py --func grlee08 --seed 0 --method eq_warp
python experiments/table.py --budgets 50,100,200
```

The paper uses seeds 0–9. All runs are deterministic.

### Bounded domains

The benchmarks are defined on `[0,1]^D` and run for 200 rounds by default.

| Method | Description |
|---|---|
| `fps` | Farthest-point sampling (reference) |
| `eig` | Posterior-variance maximization (EIG) |
| `eq` | EIG with the geometric equalizer |
| `eq_warp` | Geometric equalizer and reconstruction warp (proposed method) |
| `warp` | Reconstruction warp without the equalizer |
| `eq_kumar` | Geometric equalizer and a Kumaraswamy input warp trained by marginal likelihood |

| D | Benchmarks |
|---|---|
| 2 | `peaks`, `smooth_box_2d`, `grlee08`, `levy2d`, `zhou98`, `beale` |
| 3 | `smooth_box_3d`, `hartmann_3d`, `levy_3d`, `styblinski_tang_3d` |
| 4 | `smooth_box_4d`, `hartmann_4d`, `levy_4d`, `power_plant` |
| 5 | `friedman_5d`, `levy_5d`, `airfoil` |
| 6 | `hartmann_6d` |

`power_plant` and `airfoil` are real-data benchmarks. Their ground truth is the posterior mean
of a GP fitted to the UCI Combined Cycle Power Plant and Airfoil Self-Noise datasets
(CC BY 4.0), stored in `data/`.

### Sphere

The spheres S² and S³ have no boundary and therefore isolate the effect of observation
dependence. The benchmarks are `cap_s2`, `twocaps_s2`, `cap_s3` and `twocaps_s3`, the methods
are `fps`, `eig`, `warp`, `sobol` and `random`, and each run has 100 rounds by default.
The paper reports AR after 20 and 100 rounds (`table.py --budgets 20,100`).

## Repository structure

```
src/warp_gp/
  gp.py            GP with fixed hyperparameters used for acquisition
  equalizer.py    geometric equalizer
  warp.py          reconstruction warp
  kumaraswamy.py   trained Kumaraswamy warp (baseline)
  acquisition.py   acquisition scores
  loop.py          sequential acquisition on [0,1]^D
  sphere.py        sequential acquisition on S² and S³
  functions.py     benchmark functions
  uci.py           real-data benchmarks
  evaluate.py      nearest-neighbour reconstruction error
experiments/       run.py and table.py
notebooks/         demo.ipynb
data/              GP surrogates of the UCI datasets
```

## Citation

```bibtex
@article{jarl2026correcting,
  title   = {Correcting Boundary Bias and Observation Independence in Bayesian Experimental Design},
  author  = {Jarl, Sanna and Sj{\"o}lund, Jens and Scragg, Jonathan J. S. and B{\aa}nkestad, Maria},
  journal = {arXiv preprint arXiv:2602.01898},
  year    = {2026}
}
```

## License

MIT; see `LICENSE`.
