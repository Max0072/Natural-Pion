# Natural Pion

Spectrum-preserving rotations along the natural gradient. A variant of [Pion](https://arxiv.org/abs/2605.12492).

Pion trains each weight matrix only by rotating it on both sides, `W <- R_out W R_in`, so the singular values of
`W` never change, and it drives the rotation with the raw gradient. **Natural Pion keeps that constraint and changes
the direction:** it turns `W` along the Kronecker-factored natural gradient on the rotation group, and it sets the
step as a fixed rotation angle, in radians, of each matrix's largest rotation plane.

**Result** (LLaMA-60M, C4, 9.6B tokens; `val` = held-out cross-entropy, lower is better):

| | steps | wall time | val |
|---|---|---|---|
| Pion (published configuration, lr 1e-3) | 73 242 | 34 063 s | 3.3706 |
| **Natural Pion, same wall time** | 56 900 | 34 012 s | **3.3468** |
| **Natural Pion, same steps** | 73 242 | 43 402 s | **3.3296** |

At equal wall time it is ahead of Pion on all four seeds at every horizon measured (mean gap 0.055 at 1 800 s,
0.025 at 8 900 s). To reach Pion's final loss it needs at least ~1.45x fewer steps. A heavy-tailed initial spectrum
(power-law tail, exponent 3) helps Natural Pion and does not help Pion. Full write-up, ablations and limitations:
**[`docs/paper/FINAL.md`](docs/paper/FINAL.md)**.

## The method

For a weight `W` (m x n) with gradient `G`:

```
G_in  = skew(W^T G),   G_out = skew(G W^T)          the gradient with respect to a rotation, skew(M) = (M - M^T)/2
M     = beta1 M + (1 - beta1) G                     Lie momentum on each side (as in Pion)
X     = F^-1(M)                                     natural gradient, each side
sigma = max(||X_in||_2, ||X_out||_2)
c     = theta_t / sigma                             the largest rotation plane turns by theta_t
W    <- Cay(X_out, c) W Cay(X_in, c)                Cay(X, c) = (I + c/2 X)^-1 (I - c/2 X), exactly orthogonal
```

**Why `F` is cheap.** For one token the generator is a bivector, `skew(W^T d x^T)`. The covariance of a bivector of
two independent vectors is fixed by their two covariances, so with `A = E[x x^T]` (layer input) and
`D = E[d d^T]` (gradient at the layer output) the Fisher operator on skew `X` is

```
F(X) = 2 (B X C + C X B),   in-side (B, C) = (A, W^T D W),   out-side (B, C) = (D, W A W^T)
```

A congruence `P` with `P^T B P = I`, `P^T C P = diag(lam)` makes it elementwise:
`X = P [ (P^T G P) / (2 (lam_i + lam_j)) ] P^T`. One generalized eigendecomposition per side, reused for `t_fac`
steps; every inverted spectrum is floored at `1e-4 * lam_max`, the only regularisation.

**Why the angle.** `F^-1` gives a good direction and a bad scale: it amplifies by four or more orders of magnitude,
unevenly across layers. Taking the direction from `F^-1 G` and the length from a fixed angle is what makes it work
(FINAL.md section 6.2: the more evenly the matrices turn, the lower the loss).

## Use

```python
import torch
from ngd_pion import NaturalPionFast, attach, attach_backward

linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear) and m is not model.head]
opt = NaturalPionFast([m.weight for m in linears], lr=0.05)   # lr = peak rotation angle, radians
attach(linears, opt)              # feeds A = E[x x^T] from the forward pass
attach_backward(linears, opt)     # feeds D = E[d d^T] from the backward pass
# embeddings, head and norm gains: AdamW, as in Pion. Schedule opt's lr (cosine in our runs).
```

`NaturalPion` is the exact form (basis and statistics every step); `NaturalPionFast` the form we run (basis every
25 steps, statistics in the 6 steps before each rebuild, batched over equal shapes; 1.28x Pion's step time).

In the harness:

```bash
pytest -q
python scripts/prepare_data.py --out data --target-tokens 1e10
python -m harness.run --optimizer natural-pion-fast --lr 0.05 --train-steps 56900 --adamw-lr 8e-3 \
    --batch-sequences 512 --micro-batch 256 --precision bf16
python -m harness.run --optimizer pion --lr 1e-3 --train-steps 73242 --adamw-lr 8e-3 ...   # the baseline
```

`RunConfig`'s hash names the output directory and covers every field that can change a result.

## Layout

```
ngd_pion/
  natural_pion.py    the method: NaturalPion, NaturalPionFast
  covariance.py      A and D, exponential moving averages in exact fp32
  direction.py       generators, F^-1 in a basis
  factorization.py   the bases (identity anchor, congruence) and the spectral floor
  linalg.py          skew, Cayley, power iteration, exact fp32
  hooks.py           forward hook (A); with_s.attach_backward is the backward hook (D)
  pion_baseline.py   published Pion, the baseline
  optimizer.py, with_s.py, unified.py   the reference implementations the work grew from (alpha trust ratio)
  reference.py       numpy oracle for the tests
  legacy/            variants tried and set aside (damping, F^-p, Shampoo on so(n), ...)
harness/             LLaMA-60M, C4 corpus, the training loop, run configuration
scripts/
  paper/             every job behind docs/paper/FINAL.md, indexed by section
  analysis/          final_tables.py regenerates FINAL.md's tables from runs-record/
  probes/            single-question measurements; step_norm.py is where the method was developed
  rl/                preparation for the RLVR experiment
  cluster/           infrastructure jobs (data, preflight, throughput, resume, train)
  archive/sbatch/    job scripts of the earlier work
runs-record/         manifest.json and log.jsonl of every run: the source of every number
docs/
  paper/             FINAL.md (method and results), RLVR_PLAN.md (next experiment)
  RESUME.md          what is running now and what comes next
  decisions/         decision records
  CLUSTER.md, LITERATURE.md
  history/           JOURNAL.md (every decision, in order), PLAN.md, VALIDATION.md, ALGORITHM.md (the earlier spec)
```

## Status and limits

* One model size (60M) and one data set; full length on one seed.
* Muon in the same harness is ahead of us at 3000 steps; its full-length run is in progress.
* All comparisons are inside one harness: our Pion is 0.013 above the Pion paper's own 60M number (Megatron).
* Next: RLVR, where a step costs minutes of generation and fewer steps matter most
  ([`docs/paper/RLVR_PLAN.md`](docs/paper/RLVR_PLAN.md)).

The history of how the method was found, including every negative result, is in
[`docs/history/JOURNAL.md`](docs/history/JOURNAL.md).
