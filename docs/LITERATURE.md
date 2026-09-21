# Literature check, 2026-09-22

The question asked: **has anyone already built a curvature-preconditioned
(Fisher / K-FAC-style) version of a spectrum-preserving rotational optimizer such
as Pion?** This is what was searched, what turned up, and what the check cannot
show. It is a check, not a survey; redo it before anything is submitted.

## Verdict

**Nothing found that does what NGD-Pion does.** No paper turned up that
preconditions the rotation generators of Pion or POET with a Fisher, K-FAC or
Shampoo-type operator. That is "not found by this search", not "does not exist"
(see the limits below), and the word "first" should not go into the text until the
re-check has been done by a human with Google Scholar.

The Pion paper itself never mentions curvature, Fisher, natural gradient, K-FAC,
Shampoo or second-order preconditioning, including in its limitations section
(read via WebFetch of the arXiv HTML, 2026-09-22).

## Closest works, and how each relates

| work | id | what it does | overlap with NGD-Pion |
|---|---|---|---|
| **Pion**, Shi et al., 12 May 2026 | [2605.12492](https://arxiv.org/abs/2605.12492) | left and right orthogonal updates, spectrum frozen, driven by the raw generator; Lie-algebra momentum; RMS scaling; truncated exponential (Cayley "considered, modest gains") | **the baseline.** Reported 60M results: 3.3575 bilateral, 3.3654 alternate |
| **POET**, Qiu et al., NeurIPS 2025 | [2506.08001](https://arxiv.org/abs/2506.08001) | reparameterises `W = R W0 P` with learnable orthogonal `R`, `P` | same principle, reparameterised; no curvature seen in the abstract |
| **POET-X** | [2603.05500](https://arxiv.org/abs/2603.05500) | memory-efficient POET | abstract only; whether it preconditions the rotations was **not verifiable** from the page fetched |
| **Cayley SGD / Cayley Adam**, Li, Li, Todorovic, ICLR 2020 | [2002.01113](https://arxiv.org/abs/2002.01113) | Cayley retraction and momentum on the Stiefel manifold | **prior art for the retraction.** Our exact Cayley is not new; using it to keep a frozen spectrum exact against Pion's truncated exponential is the point |
| **Riemannian Natural Gradient Methods** | [2207.07287](https://arxiv.org/abs/2207.07287) | Fisher metric on general manifolds, finite-sum negative-log-probability losses, convergence theory | **prior art for the idea "natural gradient on a manifold".** Not applied to LLM pretraining or to the two-sided rotation of weight matrices, as far as its abstract says |
| **ISO, Isospectral Optimization** | [2607.19331](https://arxiv.org/abs/2607.19331) | keeps the base model's spectra, changes input and output singular frames (RLVR) | close in spirit, concurrent; no curvature preconditioning per its abstract |
| **K-FAC**, Martens and Grosse, ICML 2015 | [1503.05671](https://arxiv.org/abs/1503.05671) | Kronecker-factored Fisher on the weights | **the approximation we use** (independence of `u` and `x`), on a different object: the rotation generators, not the weights |
| **Shampoo**, Gupta, Koren, Singer, ICML 2018; **SOAP**, **KL-Shampoo** | [PMLR v80](https://proceedings.mlr.press/v80/gupta18a/gupta18a.pdf) | Kronecker-factored preconditioning of the weight gradient | on the weights. `shampoo-pion` in this repository is the transfer to `so(n)`; the Pion authors' own `pion_msign` is its memoryless case (journal, 2026-08-28) |
| **ONG, Orthogonal Natural Gradient Descent** | [2508.17169](https://arxiv.org/abs/2508.17169) | continual learning, projection orthogonal to earlier tasks | different problem; title seen, **paper not read** |
| **Isometron / Iso**, Jacobsen | [2307.12979](https://arxiv.org/abs/2307.12979) | update norm invariant to linear transformations of inputs and outputs | different mechanism |

From memory, **not verified in this session**: natural gradient on the orthogonal
group is old in signal processing (Amari's natural gradient for blind source
separation; learning on the Stiefel manifold for ICA, Nishimori, Plumbley's Lie-group
methods), and PSGD learns a preconditioner constrained to a Lie group, which is a
different object (a preconditioner, not a rotation of the weights). Cite after
checking, not from here.

## What is plausibly new

The combination, none of whose parts is new alone: the Fisher operator on the
**bivector** tangent space of the two-sided rotation, obtained in closed form
because the per-sample generator is a wedge of two vectors and the K-FAC
independence approximation then factorises it through the activation covariance
(`F(X) = 2(SXA + AXS)`, one eigendecomposition per side); an exact Cayley
retraction that keeps the frozen spectrum exact; applied to LLM pretraining on top
of Pion, with tuned baselines. None of the works above does this, per what could
be read.

## Limits of this check

* **The search tool is not an index.** Semantic Scholar's citation list for
  Pion returned three papers (POET-X, ISO, and one on online-to-nonconvex
  conversion). Pion is four months old and that list is certainly incomplete.
* **Papers were read through summaries.** Abstracts and pages were fetched and
  answered by a small model, not read; a relevant section inside a paper whose
  abstract does not mention curvature would be missed. POET-X's method section was
  not seen at all.
* **The first searches returned this repository.** `Max0072/Natural-Pion` is
  public on GitHub and shows up for "Fisher-preconditioned Pion". It is not
  independent evidence, and it does mean the idea has a public, dated trail
  (commits from 2026-08-24).
* Nothing after about mid-September 2026 could be assumed indexed.

## How to redo it properly

1. Google Scholar "cited by" on Pion (2605.12492) and on POET (2506.08001),
   filtered to 2026, read for any that touch preconditioning.
2. arXiv listing search, title and abstract, for each of: "Pion" with
   "Fisher", "natural gradient", "K-FAC", "Shampoo", "preconditioned"; "orthogonal
   equivalence" with "curvature"; "Lie algebra" with "natural gradient" and
   "language model".
3. Read the method sections of POET-X and ISO, not the abstracts.
4. Repeat one week before any submission: this area is moving monthly.
