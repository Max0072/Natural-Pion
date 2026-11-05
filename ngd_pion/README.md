# ngd_pion

**The method is `natural_pion.py`.** Four tiers of one algorithm:

| class | harness name | tier | what it is |
|---|---|---|---|
| `NaturalPion` | `natural-pion` | reference | Algorithm 1: exact form, statistics every step, basis every step (`t_fac` 1), exact Cayley solve, fp32 |
| `NaturalPionFast` | `natural-pion-fast` | the method | Algorithm 2: basis every 25 steps, A and D in the 6 steps before a refresh, one A per input, warm batched sigma, Cayley by Newton-Schulz in Z - I, bf16 statistics inputs. Every speed-up standard and explainable, loss within 0.010 of the reference; only Natural Pion's own parts. Also carries the ablation switches of Section 5.4 (`direction`, `length`, `sides`, `planes`, `attn_mult`, `a_window`, `d_window`, `a_every`), each defaulting to the method |
| `NaturalPionTurbo` | `natural-pion-turbo` | aggressive | anything faster; loss cost measured; in no comparison |
| `NaturalPionStiefel` | `natural-pion-stiefel` | aggressive | Turbo with the wide side of a rectangular matrix in its k-dimensional Stiefel form |

Primitives: `covariance` (EMA second moments A, D), `direction` (generators, the Fisher operator, its solve in a
basis), `factorization` (the generalized eigenbasis), `linalg` (Cayley, spectral norm, exact-fp32 context). `hooks`
records activations (A) and output gradients (D). `pion_baseline` is published Pion, the only baseline. `reference`
is a numpy oracle for the tests.

The earlier NGD-Pion optimizers (`ngd-pion-*`, the alpha trust region, `shampoo-pion`) were removed on 2026-11-05;
they are in git history and described in `docs/history/`.
