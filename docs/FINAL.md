# NGD-Pion: spectrum-preserving rotations along the natural gradient

Status: final write-up of the method and its results, 2026-11-03. It is the basis for the paper.
The full reasoning is in [`JOURNAL.md`](JOURNAL.md), the current state in [`RESUME.md`](RESUME.md),
and every run's manifest and log in [`../runs-record/`](../runs-record/).

**One-sentence claim.** Pion (Shi et al., arXiv:2605.12492) changes each weight matrix only by
rotations, so its singular values never change. Our method keeps exactly that, but turns the matrix
along the Kronecker-factored natural gradient instead of the raw gradient. It sets the step as a fixed
rotation angle of the matrix's largest rotation plane. On LLaMA-60M / C4 it reaches a lower validation
loss than published Pion in 1.29x fewer steps or less. At the same wall-clock time it is ahead of
Pion on every seed and every horizon measured.

---

## 1. Setting

| item | value |
|---|---|
| model | LLaMA-60M: hidden 512, 8 layers, 8 heads, FFN 1376, RoPE, RMSNorm, vocabulary 32 100 (T5-base tokenizer) |
| data | C4, sequence length 256, batch 512 sequences (131 072 tokens per step) |
| full length | 73 242 steps = 9.6B tokens, the Pion paper's 60M budget |
| precision | bf16 model; optimizer statistics and rotations in exact fp32 (TF32 off inside them) |
| matrices | every 2-D weight inside the transformer blocks is rotated: 56 matrices, 43.5% of the parameters |
| the rest | embedding, head and norms (56.5%) use AdamW: lr 8e-3, betas (0.9, 0.95), weight decay 0.1, cosine, no warmup, gradient clip 1.0. The same for every method compared |
| metric | `val`: mean token cross-entropy (nats) on held-out C4, 20 batches. Lower is better; `exp(val)` is perplexity |
| hardware | one NVIDIA RTX PRO 6000 Blackwell per run. Wall-clock time is measured on the same node type for all methods |

**Baseline.** The baseline is published Pion and nothing else: Lie-algebra momentum 0.9, RMS step
scaling to 0.2, truncated-exponential retraction, query matrix split per head, lr 1e-3, cosine. In this
setting it costs **0.466 s per step**. lr 1e-3 is the paper's choice from the grid
{1e-4, 5e-4, 1e-3, 5e-3, 1e-2}. We checked it ourselves:

| Pion lr | 5e-4 | 1e-3 | 2e-3 | 4e-3 |
|---|---|---|---|---|
| 3 840 steps | 3.8397 | 3.7804 | **3.7668** | 4.1523 |
| 19 200 steps | - | **3.4960** | 3.5138 | 3.8802 |

At the long horizons 1e-3 is the best. At 3 840 steps 2e-3 is better by 0.014, and the comparison
there is reported against both.

---

## 2. Notation

For one weight matrix `W` (m x n), with loss gradient `G = dL/dW`:

* `skew(M) = (M - M^T) / 2`.
* **Generators.** `G_in = skew(W^T G)` (n x n) and `G_out = skew(G W^T)` (m x m). These are the
  Riemannian gradients of the loss with respect to a rotation applied on the right and on the left of `W`.
* **Factors** (K-FAC statistics). `A = E[x x^T]` is the second moment of the layer's input `x`
  (n x n). `D = E[d d^T]` is that of the gradient `d` at the layer's output (m x m). Both are
  exponential moving averages over steps.
* **Fisher operator on rotations.** Under K-FAC, the Fisher metric restricted to rotations on one side
  of `W` is the linear operator `F(X) = 2 (B X C + C X B)` on skew `X`, with
  * input side: `(B, C) = (A, W^T D W)`;
  * output side: `(B, C) = (D, W A W^T)`.
* **Natural direction.** `X = F^-1(G_skew)`, solved exactly in a basis `P` with
  `P^T B P = I` and `P^T C P = diag(lam)`. In that basis `F` acts elementwise:
  `X = skew( P [ (P^T G_skew P) / (2 (lam_i + lam_j)) ] P^T )`.
  The eigenvalues of `B` and `C` are floored at `eps * lam_max` with `eps = 1e-4`. That floor is the only
  regularisation.
* **sigma.** `sigma = max(||X_in||_2, ||X_out||_2)`, the largest singular value over both sides.
* **Cayley retraction.** `Cay(X, c) = (I + c/2 X)^-1 (I - c/2 X)`. It is exactly orthogonal for skew
  `X`, so `W <- R_out W R_in` keeps the singular values of `W` unchanged.
* **theta_t.** The scheduled rotation angle at step `t`: cosine from `theta_peak` down to `1e-5`.
  With `c = theta_t / sigma` the largest rotation plane turns by about `theta_t` radians every step,
  in every matrix.

---

## 3. Algorithm 1: NGD-Pion, the method in its exact form

This is the method as a mathematical object. Every quantity is fresh at every step.

```
Input: rotated matrices {W}, peak angle theta_peak, momentum beta1 = 0.9, factor EMAs beta_A, beta_D,
       floor eps = 1e-4, steps T.
State per W: A, D (EMAs), M_in, M_out (momentum on the generators).

for t = 0 .. T-1:
    forward + backward; hooks fold x x^T into A and d d^T into D (fp32)
    theta_t = cosine(t; theta_peak -> 1e-5 over T)
    for each W with gradient G:
        G_in  = skew(W^T G);           G_out = skew(G W^T)
        M_in  = beta1 M_in  + (1 - beta1) G_in            # Lie-algebra momentum, as in Pion
        M_out = beta1 M_out + (1 - beta1) G_out
        P_in,  lam_in  = gen_eig(A, W^T D W)              # P^T B P = I, P^T C P = diag(lam)
        P_out, lam_out = gen_eig(D, W A W^T)
        X_in  = F^-1(M_in)  in (P_in,  lam_in)            # exact natural gradient on each side
        X_out = F^-1(M_out) in (P_out, lam_out)
        sigma = max(||X_in||_2, ||X_out||_2)              # one scale shared by both sides
        c     = theta_t / sigma
        W    <- Cay(X_out, c) W Cay(X_in, c)              # spectrum of W unchanged
    AdamW step on embedding, head, norms
```

What each element is and why it is there. Every line is an ablation in §6.

1. **Direction: the natural gradient, `X = F^-1 G`.** Pion uses the raw generator `G`. Replacing it with
   `F^-1 G` is the main source of the gain: 0.12 of val at 3000 steps with momentum, 0.46 without.
2. **Length: a fixed angle, `c = theta / sigma(X)`.** The norm is taken after `F^-1`. `F^-1` amplifies by
   four or more orders of magnitude, and unevenly across layers. Without a length control, single
   attention matrices run away (`c sigma` up to 1e5). Fixing the largest plane's angle removes that, so
   every matrix turns by the same angle.
3. **One shared `sigma` for both sides.** This keeps the ratio between input and output rotations that
   `F^-1` sets. Normalising each side by its own `sigma` is worse.
4. **Lie momentum on the generators, before `F^-1`.** As in Pion.
5. **Cayley retraction.** It is exactly orthogonal, so it keeps Pion's invariant: the singular values of
   `W` are frozen at initialisation. In fp32 they drift by about 0.2% over 3000 steps (fp64: 1e-15).
6. **Cosine schedule on the angle.** The best peak angle falls with the horizon: 0.2 at 3000 steps,
   0.05-0.1 at 15 000, 0.05 at 56 900 and 73 242.

Cost in our implementation: 1.75 s per step (Pion 0.466 s). Most of it is the per-step generalized
eigendecompositions and the spectral norms.

---

## 4. Algorithm 2: NGD-Pion, efficient form (the one we run)

Algorithm 1 with the same direction, length rule and retraction. Only the implementation changes, and
every change was validated against the exact form (§6.4).

```
as Algorithm 1, except:
    (a) the bases (P, lam) are recomputed every T_fac = 25 steps, not every step;
        between refactors F^-1 uses the last basis, with the current W, G and momentum
    (b) A and D are collected only in the last 6 steps before each refactor (EMA factor 0.5 for both)
    (c) ||X||_2 by power iteration, warm-started from the previous step's vector:
        2 iterations per step, 50 right after a refactor
    (d) F^-1 and the Cayley solves are batched across matrices of equal size: one call per shape, not 56
    (e) no diagnostics: no quad/curv, no per-layer probes
```

| | Algorithm 1 | Algorithm 2 |
|---|---|---|
| bases recomputed | every step | every 25 steps |
| A, D collected | every step | 6 of every 25 steps |
| s per step | 1.75 | **0.596** (Pion 0.466; ratio 1.28) |
| val, 3000 steps, seeds 0-3 (mean) | 3.7089 (angle 0.1) | 3.7216 (angle 0.2) |
| val, 15 000 steps, seed 0 | 3.4788 (angle 0.1) | 3.4720 (angle 0.05) |

* **Cost of the approximation.** At 3000 steps the approximation costs 0.013 of val. At 15 000 it is
  within noise.
* **Midway option.** `T_fac = 10` costs 0.71 s per step and gives 3.7032 at 3000 steps (seeds mean).
* **The limit on (a).** The fresh basis matters on its own. Without momentum, `T_fac` 25 against 1 loses
  0.040. Momentum makes up most of that.

---

## 5. Results

### 5.1 Full length, 73 242 steps (9.6B tokens), seed 0

| run | steps | wall time | val | vs Pion |
|---|---|---|---|---|
| Pion, lr 1e-3 | 73 242 | 34 063 s | 3.3706 | - |
| **ours, same wall time**, theta 0.05 | 56 900 | 34 012 s | **3.3468** | **-0.024** |
| ours, same wall time, theta 0.025 | 56 900 | ~34 000 s | 3.3577 | -0.013 |
| **ours, same steps**, theta 0.05 | 73 242 | 43 402 s | **3.3296** | **-0.041** |
| Muon, same cell (running, job 397404) | 73 242 | ~32 700 s | *pending* | |
| *Pion paper, Megatron, bilateral* | 73 242 | | *3.3575* | different code base, context only |

### 5.2 Same wall time, four seeds

Pion runs as many steps as fit in our wall time. Gap = Pion val - ours (positive: we are ahead).

| budget | Pion steps | our steps (theta) | seed 0 | seed 1 | seed 2 | seed 3 | mean |
|---|---|---|---|---|---|---|---|
| ~1 800 s | 3 840 | 3 000 (0.2) | 0.056 | 0.059 | 0.049 | 0.057 | **0.055** |
| ~8 900 s | 19 200 | 15 000 (0.05) | 0.024 | 0.023 | 0.020 | 0.034 | **0.025** |
| ~34 000 s | 73 242 | 56 900 (0.05) | 0.024 | | | | |

Raw val, seeds 0-3:

* Pion 3 840: 3.7804 / 3.7767 / 3.7728 / 3.7775
* ours 3 000: 3.7242 / 3.7176 / 3.7241 / 3.7205
* Pion 19 200: 3.4960 / 3.4961 / 3.4944 / 3.5046
* ours 15 000: 3.4720 / 3.4735 / 3.4747 / 3.4702

Against Pion at its best short-horizon lr (2e-3, seed 0) the 1 800 s gap is 0.043.

### 5.3 Same number of steps

| steps | Pion | ours | gap |
|---|---|---|---|
| 3 000 | 3.8394 (seeds mean) | 3.7216 (seeds mean, Alg. 2) | 0.118 |
| 15 000 | 3.5254 | 3.4720 | 0.053 |
| 73 242 | 3.3706 | 3.3296 | 0.041 |

**Steps to equal loss.** At every horizon ours with fewer steps is below Pion with more steps:
3 000 vs 3 840, 15 000 vs 19 200, 56 900 vs 73 242. So ours needs **at least 1.28-1.29x fewer steps**
to reach Pion's loss. That is a lower bound; the exact ratio was not measured.

### 5.4 Wall-clock cost per step

| part (s per step, GPU-synchronised timers) | Alg. 1 | Alg. 2 before (d), (e) |
|---|---|---|
| refactor (eigendecompositions) | 0.545 | 0.025 |
| Cayley retraction | 0.163 | 0.157 |
| spectral norm (power iteration) | 0.415 | 0.042 |
| F^-1 and the rest of the step | 0.052 | 0.050 |
| collecting A and D | 0.136 | 0.136 |
| **whole step** | **1.75** | 0.85 -> 0.744 with (b) -> 0.697 with (e) -> **0.596** with (d) |

Forward, backward and AdamW are common to all methods. Pion's step is 0.466 s.

---

## 6. What each part contributes (ablations)

All at 3000 steps, seed 0, unless stated. "d" is val minus the reference row of each block. Every grid's
best value is interior: the next angle on both sides is worse.

### 6.1 Direction: natural gradient vs raw generator

Same normalisation, retraction and schedule. Only `X = F^-1 G` vs `X = G` changes. Best angle of each.

| | no momentum | Lie momentum 0.9 |
|---|---|---|
| X = F^-1 G | 3.7399 | 3.7032 |
| X = G | 4.2007 | 3.8247 |
| **Fisher's gain** | **0.461** | **0.122** |

Momentum supplies part of what `F^-1` does, but not all of it. `X = G` with momentum is roughly Pion:
3.8247 vs Pion 3.8412.

### 6.2 How the step length is set (Algorithm 1 without momentum)

| length rule, all after F^-1 unless stated | val | spread of real angle across layers (p90/p10) |
|---|---|---|
| **spectral: `c = theta / sigma(X)`** | **3.7399** | 1 |
| RMS of the update, as in Pion | 3.7540 (gap over seeds 0-3: 0.014 / 0.009 / 0.011 / 0.006) | 1.7 |
| old trust ratio `alpha = quad / curv` | 3.7843 | 131 |
| Fisher norm `c = theta / sqrt(<X, F X>)` | <= 3.8281 (edge of grid) | 5e4 |
| none (fixed `c`) | 3.8419 | 4e6 |
| spectral, but measured **before** F^-1 (`sigma(G)`) | 3.9951 | 3e11 |

The more evenly the matrices turn, the better the loss: monotone over six points. One seed; this is a
correlation, not a measured cause.

### 6.3 Other choices tested and rejected

| change | val | reference | verdict |
|---|---|---|---|
| each side by its own sigma | 3.7412 (best) | 3.7227 | worse: shared sigma kept |
| attention angle x0.25 / x0.5 / x2 / x4 | 3.7876 / 3.7467 / 3.7600 / 3.7789 | 3.7399 | equal angles kept |
| flatten the step's spectrum (p = 0, 0.5) | +0.011 / +0.012 at step 2499 | | worse and 5-15x dearer |
| constant angle | 3.8313 | 3.7399 | worse |
| linear decay | 3.7311 / 3.7229 | 3.7399 / 3.7227 | tie: cosine kept |
| warmup + cosine | 3.7688 | 3.7399 | worse |
| WSD (decay over last 20% / 10%), peak 0.2 | 3.7578 / 3.7763 | 3.7227 | worse |
| early decay `(1 - t/T) / sqrt(1 + t/T0)`, 15 000 steps, peak 0.2 / 0.1 | 3.4750 / 3.4818 | 3.4720 | tie / worse |
| RProp-style global multiplier on the angle | 3.7497 | 3.7227 | worse |
| AdamW lr 4e-3 / 1.6e-2 | 3.7367 / 3.7195 | 3.7227 | 8e-3 kept |

### 6.4 The efficient form against the exact one

| change from Alg. 1 toward Alg. 2 | val | reference | s per step |
|---|---|---|---|
| T_fac 25 (no momentum) | 3.7798 | 3.7399 (T_fac 1) | 0.85 |
| T_fac 25 (momentum) | 3.7227 | 3.7032 (T_fac 1) | 0.84 |
| D only in last 6 steps | 3.7272 | 3.7227 | 0.788 |
| A only in last 6 steps | 3.7205 | 3.7227 | 0.806 |
| A and D only in last 6 steps | 3.7227 | 3.7227 | 0.744 |
| EMA of D at 0.9 / 0.95 instead of 0.5 | 3.7206 / 3.7201 | 3.7227 | |
| no diagnostics (e) | bit-identical over 300 steps | | 0.697 |
| batched F^-1 and Cayley (d) | 3.7242 | 3.7227 | 0.596 |

---

## 7. Relation to other work (first pass; recheck arXiv before submission)

* **Pion** (Shi et al. 2026, arXiv:2605.12492) and **POET** (arXiv:2506.08001): orthogonal
  transformations that preserve the spectrum, driven by the raw generator. No Fisher.
* **K-FAC** (Martens & Grosse 2015) and its use in RL, **ACKTR** (Wu et al. 2017). Natural policy
  gradient (Kakade 2001), TRPO.
* **Riemannian natural gradient** (arXiv:2207.07287): Kronecker inverses on Grassmann rows. It has no
  two-sided rotations and no step normalisation.
* **FISMO** (arXiv:2601.21750), **Go-MUON** (arXiv:2608.09763): K-FAC combined with a polar step,
  additive rather than rotational.
* **Grafting** (arXiv:2002.11803): the precedent for taking the direction from one method and the length
  from another.
* We found no comparison of normalising before vs after `F^-1`. Do not claim "first K-FAC plus spectral
  normalisation"; the combination as a whole was not found.

**Published Muon near this setting**, as context only: no one reports Muon at 60M / 9.6B tokens.
* At 60M, C4, seq 256, batch 512, 10K steps (1.0-1.3B tokens), Muon reaches `ln(ppl)` 3.36-3.40: HTMuon
  arXiv:2603.10067, MoLS 2605.05794, Magma 2602.15322, Conda 2509.24218.
* At 1.3B / 54B tokens the Pion paper reports Muon 2.7225, Pion 2.7350, AdamW 2.7700.

---

## 8. Limitations and open items

1. **Muon at full length in our harness** is running (job 397404). At 3000 steps Muon in the same cell
   is ahead of us: 3.6505 vs 3.7216 by steps, at 0.452 s per step.
2. **Full length is one seed.** The seed spread at shorter horizons (§5.2) suggests the gap is stable,
   but the main table needs seeds.
3. **One model size** (60M) and one data set.
4. **Code bases differ.** Our Pion is 0.013 above the paper's Pion at full length (3.3706 vs 3.3575,
   Megatron). This is why every comparison here is inside one harness.
5. **The spectrum is exactly frozen only in exact arithmetic.** In fp32 it drifts by ~0.2% per 3000 steps,
   with a consistent sign. Whether that costs anything is unmeasured.
6. **Steps-to-loss is a lower bound** (1.28-1.29x), not a measured ratio.
7. **Hypothesis, not a result: an advantage in RL.** There each step needs fresh generations, so cost
   goes with the number of steps, not the time of one step. Pion's RLVR experiment (its Table 3) is the
   natural test.

---

## 9. Reproduction

* **Algorithm 2** is `scripts/probes/step_norm.py`, a subclass of the reference `NGDPionUnified`, which is
  left unchanged. Run with:

  ```
  STEP_NORM_FAST=1 STEP_NORM_BATCH=1 STEP_NORM_A_LAST=6 STEP_NORM_D_LAST=6 \
  python scripts/probes/step_norm.py --optimizer ngd-pion-m --ngd-momentum lie --ngd-beta1 0.9 \
      --lr <theta_peak> --ngd-trust none --ngd-beta 0.5 --ngd-beta-backward 0.5 --ngd-t-fac 25 \
      --batch-sequences 512 --micro-batch 256 --precision bf16 --adamw-lr 8e-3 --train-steps <T> ...
  ```
* **Algorithm 1** is the same command without `FAST`, `BATCH`, `A_LAST`, `D_LAST`, with `--ngd-t-fac 1`
  and the default factor EMAs.
* **Jobs.** The exact jobs are in `scripts/sbatch/step_norm_long.sbatch` (full length, seeds, Pion lr) and
  `step_norm_final.sbatch`.
* **Run records.** `runs-record/stepnormlong/`, `runs-record/stepnormfinal/` and the other `stepnorm*`
  directories.
* **Planned clean-up for publication.** Promote Algorithm 2 from the probe into a named optimizer class
  with explicit arguments, next to the reference, with a test that it reproduces the probe.
