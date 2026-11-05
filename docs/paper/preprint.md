---
title: "Natural Pion: Spectrum-Preserving Rotations along the Natural Gradient"
author: "[authors]"
date: "Draft, November 2026"
---

## Abstract

Pion trains each weight matrix of a language model only by rotating it on both sides,
$W \leftarrow R_{\mathrm{out}} W R_{\mathrm{in}}$, so the singular values of every matrix stay fixed for the
whole run. We keep this constraint and change the direction of the rotation from the gradient to the
natural gradient. On LLaMA-60M trained on 9.6B tokens of C4, the resulting optimizer, **Natural Pion**,
reaches a validation loss of {{np_full_steps:.4f}} in Pion's number of steps, against Pion's {{pion_full:.4f}}. At Pion's
wall-clock time it reaches {{np_full_clock:.4f}}. With each method on its own full schedule, it needs {{budget_lo:.2f}}–{{budget_hi:.2f}}x fewer steps to reach the losses Pion
reaches; on one GPU it is also ahead at equal wall-clock time on every seed and horizon we measured.

The method rests on three results. First, on the rotation generators the Kronecker-factored (K-FAC)
Fisher information takes a closed form, $F(X) = \tfrac12(SXA + AXS)$. Here $A$ is the second moment of
the layer's inputs and $S$ that of its output gradients pulled back through $W$. A single generalized
eigendecomposition per side inverts this operator exactly. Second, the natural gradient is a good
direction with a bad scale: its magnitude varies over orders of magnitude between layers. We keep the
direction and set the length so that every matrix turns its fastest rotation plane by the same scheduled
angle. Third, an exact Cayley retraction keeps Pion's invariant, which Pion's own truncated exponential lets drift. A fast
form of the method costs {{flops_60M_ratio:.2f}}x Pion's floating-point operations per step ({{fast2_ratio:.2f}}x its time on one GPU in
our implementation).

---

## 1 Introduction

Most optimizers for neural networks move a weight matrix additively, $W \leftarrow W - \eta\,\Delta$.
They differ in how $\Delta$ is preconditioned: per coordinate (Adam), by Kronecker factors (K-FAC,
Shampoo), or by orthogonalizing it (Muon). Pion (Shi et al., 2026) takes a different route. It
updates $W$ multiplicatively, by an orthogonal equivalence transformation

$$W \leftarrow R_{\mathrm{out}}\, W\, R_{\mathrm{in}}, \qquad R_{\mathrm{out}} \in O(m),\; R_{\mathrm{in}} \in O(n),$$

so the singular values of $W$ are those it was initialised with, for the whole of training. Only
the singular vectors are learned. Pion reports that this gives spectral control without explicit
normalization, stable training, and performance competitive with AdamW and Muon.

Pion chooses the rotation by steepest descent on the rotation group: it follows the Riemannian
gradient of the loss with respect to the two generators. Steepest descent depends on the metric, and
Pion's is the Euclidean one on the generators, which treats every direction of rotation alike. But the curvature of the loss along
different rotation directions, and across layers, differs by orders of magnitude. The classical remedy
is the natural gradient (Amari, 1998): steepest descent in the metric of the model's output
distribution, the Fisher information. This paper derives the natural gradient for Pion's parameterization, makes
it exact and cheap, and pairs it with a step-length rule that keeps the stability Pion's normalisation
provides.

**Contributions.**

1. **The Fisher on rotation generators, in closed form.** Under the K-FAC independence
   approximation, the Fisher information restricted to the rotations of $W$ is
   $F(X) = \tfrac12(SXA + AXS)$ on each side (Proposition 1). Its factors are the input second
   moment $A$ and the output-gradient second moment $D$, pulled through $W$. The structure has a
   simple origin: a single token's rotation gradient is a bivector, the wedge product of two vectors.
2. **Exact inversion.** One generalized eigendecomposition per side diagonalises $F$, and the
   solve becomes an elementwise division (Proposition 2). No damping is needed beyond a relative
   floor on the eigenvalues.
3. **Direction from the Fisher, length from an angle.** $F^{-1}$ scales gradients very unevenly across
   layers. We take only its direction. The step length is set so
   that the largest rotation plane of every matrix turns by the same scheduled angle $\theta_t$
   (Proposition 3). With the same angle rule and the same momentum, the natural gradient reaches
   {{dir_fisher_mom:.4f}} at 3 000 steps against {{dir_raw_mom:.4f}} for Pion's raw generator (Pion itself:
   {{pion3000_s0:.4f}}). The angle rule is what lets the natural gradient through: without momentum and with a
   fixed step size, $F^{-1}G$ reaches only {{len_fixed:.4f}}, against {{dir_fisher_nomom:.4f}} with the angle (Section 5.4).
4. **A retraction that is exact at large angles.** Natural Pion turns every matrix by 0.05–0.2 rad per step.
   After 3 000 steps the Cayley transform has moved the spectrum by {{rd_cayley_med:.1e}} of each
   matrix's scale, Pion's truncated exponential by {{rd_trunc_med:.1e}}, and Cayley computed by Newton–Schulz in
   correction form by {{rd_ns_delta_med:.1e}} (Sections 3.4, 4.8).
5. **A fast form.** Amortised refreshes, statistics collected only where they are read, a warm and batched
   power iteration, batching, a Newton–Schulz Cayley solve and bf16 statistics take the step from
   {{tf1_mom_sps:.2f}} s to {{fast2_sps:.3f}} s (Pion: {{pion3000_sps:.3f}} s). Each change is measured separately for time and loss;
   the cost is also given in FLOPs (Section 4).
6. **Experiments.** On LLaMA-60M / C4 at Pion's full 9.6B-token budget, Natural Pion is ahead of
   Pion at equal steps by {{gap_full_steps:.3f}} and at equal wall-clock time by {{gap_full_clock:.3f}} in validation loss. It is ahead
   on four of four seeds at two shorter horizons, and needs {{budget_lo:.2f}}–{{budget_hi:.2f}}x fewer steps. Ablations
   isolate each component.

---

## 2 Background

### 2.1 Pion: training by orthogonal equivalence

Let $W \in \mathbb{R}^{m \times n}$ be a weight matrix with loss gradient $G = \partial L / \partial W$.
Pion parameterises an update by two skew-symmetric generators,
$\Omega_{\mathrm{in}} \in \mathfrak{so}(n)$ and $\Omega_{\mathrm{out}} \in \mathfrak{so}(m)$, and moves

$$W(\Omega) = e^{\Omega_{\mathrm{out}}}\, W\, e^{\Omega_{\mathrm{in}}}.$$

The derivative at $\Omega = 0$ is $\Omega_{\mathrm{out}} W + W \Omega_{\mathrm{in}}$. Write
$\langle M, N \rangle = \operatorname{tr}(M^\top N)$ and $\operatorname{skew}(M) = \tfrac12(M - M^\top)$.
For skew $\Omega$, $\langle M, \Omega\rangle = \langle \operatorname{skew}(M), \Omega\rangle$, so

$$dL = \langle G W^\top, \Omega_{\mathrm{out}}\rangle + \langle W^\top G, \Omega_{\mathrm{in}}\rangle
\quad\Longrightarrow\quad
G_{\mathrm{in}} = \operatorname{skew}(W^\top G), \qquad G_{\mathrm{out}} = \operatorname{skew}(G W^\top)$$

are the gradients of the loss with respect to the two generators: the Riemannian gradient on
$O(m) \times O(n)$ at the identity.

Pion steps along $-G_{\mathrm{in}}$ and $-G_{\mathrm{out}}$ with three further ingredients:

* **Lie-algebra momentum.** An exponential moving average of $G_{\mathrm{in}}$ and $G_{\mathrm{out}}$.
* **RMS scaling.** The step is scaled so that the RMS of the induced change
  $W\Omega_{\mathrm{in}} + \Omega_{\mathrm{out}} W$ equals a target (0.2 times the learning rate).
* **A truncated exponential retraction** $e^{\Omega} \approx I + \Omega + \Omega^2/2$.

Both rotations are applied at every step (the *bilateral* update; Pion also describes an alternating
variant, which it finds slightly worse at 60M). The query projection is rotated per attention head: each
head's block of rows is treated as a separate matrix. Embeddings, the output head and normalization
gains are trained by AdamW. Only the 2-D weights inside the transformer blocks are rotated.

**Why preserve the spectrum.** It fixes every weight's spectral norm without weight decay or
normalisation; Pion reports stable training without normalisation layers, learning-rate transfer across
widths, and results within 0.0125 of Muon at 1.3B parameters. This paper is about the direction of
motion inside the constraint.

### 2.2 The natural gradient and K-FAC

For a model with predictive distribution $p_\phi(y \mid x)$, the Fisher information
$F = \mathbb{E}[\nabla_\phi \log p\, \nabla_\phi \log p^\top]$ defines a metric on parameter space. The
natural gradient $F^{-1}\nabla L$ is the steepest-descent direction in that metric (Amari, 1998):

$$-F^{-1}\nabla L \;\propto\; \arg\min_{\Delta}\; \langle \nabla L, \Delta\rangle
\quad \text{s.t.} \quad \tfrac12 \Delta^\top F \Delta \le \varepsilon .$$

In a language model the "examples" of this expectation are token positions. A linear layer $y = Wx$
sees one input vector $x_t$ and receives one output gradient $\delta_t = \partial \ell_t / \partial y_t$
per token $t$, and its weight gradient is the average of the per-token outer products
$\delta_t x_t^\top$. K-FAC (Martens & Grosse, 2015) assumes $x$ and $\delta$ are independent, which gives
$F_W \approx A \otimes D$ with $A = \mathbb{E}_t[x_t x_t^\top]$ and $D = \mathbb{E}_t[\delta_t\delta_t^\top]$.
As an operator on matrices, $F_W(V) = D V A$.

Strictly, the Fisher averages over outputs $y$ sampled from the model. Like most K-FAC
implementations, we use the **empirical Fisher**: $\delta_t$ is the gradient of the training loss at
the observed next token. In an earlier variant of our method, sampling the targets from the model
instead gave {{fisher_true:.4f}} against {{fisher_empirical:.4f}} at 3 000 steps.

---

## 3 Method

### 3.1 The Fisher on rotation generators

We derive the Fisher of the input-side generator $\Omega_{\mathrm{in}} \in \mathfrak{so}(n)$ in four
steps: write it as the covariance of the per-token gradients in the skew space, expand it, apply the
K-FAC assumption, and simplify. The output side follows by symmetry.

**Step 1: the Fisher as a covariance on $\mathfrak{so}(n)$.** By Section 2.1, the gradient of token
$t$'s loss with respect to $\Omega_{\mathrm{in}}$ is

$$g_t = \operatorname{skew}(W^\top \delta_t\, x_t^\top) \in \mathfrak{so}(n).$$

The (empirical) Fisher on $\mathfrak{so}(n)$ is the second moment of $g_t$. A second moment of
elements of a vector space is a linear operator on that space:

$$F_{\mathrm{in}}(X) = \mathbb{E}_t\big[\langle g_t, X\rangle\, g_t\big], \qquad X \in \mathfrak{so}(n),$$

so that $\langle X, F_{\mathrm{in}}(X)\rangle = \mathbb{E}_t[\langle g_t, X\rangle^2]$ is the variance of
the directional derivative along $X$. In coordinates this is an $\tfrac{n(n-1)}{2} \times
\tfrac{n(n-1)}{2}$ matrix, about $1.7\cdot10^{10}$ entries for $n = 512$, and it cannot be stored, let
alone inverted.

**Step 2: each token's generator is a bivector.** Write $u_t = W^\top \delta_t \in \mathbb{R}^n$, the
output gradient pulled back to the input space. Then

$$g_t = \tfrac12\big(u_t x_t^\top - x_t u_t^\top\big) = \tfrac12\, u_t \wedge x_t,$$

the antisymmetrised outer product of two vectors. Its pairing with a skew $X$ is a single bilinear
form: since $x^\top X u = -u^\top X x$,

$$\langle g_t, X\rangle = \tfrac12\big(u_t^\top X x_t - x_t^\top X u_t\big) = u_t^\top X x_t .$$

**Step 3: expand.** Substituting both expressions into Step 1 and moving the scalar
$u_t^\top X x_t$ next to the vector it pairs with:

$$F_{\mathrm{in}}(X) = \tfrac12\,\mathbb{E}_t\big[(u_t^\top X x_t)\, u_t x_t^\top\big]
- \tfrac12\,\mathbb{E}_t\big[(u_t^\top X x_t)\, x_t u_t^\top\big]
= \tfrac12\,\mathbb{E}_t\big[u_t u_t^\top\, X\, x_t x_t^\top\big]
- \tfrac12\,\mathbb{E}_t\big[x_t x_t^\top\, X^\top u_t u_t^\top\big].$$

Each term is now a product of a quantity that depends on $u_t$ only, $X$, and a quantity that depends
on $x_t$ only.

**Step 4: the K-FAC assumption.** Assume, as K-FAC does, that the layer's input $x_t$ and its output
gradient $\delta_t$ (hence $u_t$) are independent. The expectation of each product then factorises:

$$\mathbb{E}_t\big[u u^\top X x x^\top\big] = \mathbb{E}[u u^\top]\, X\, \mathbb{E}[x x^\top] = S X A,
\qquad
\mathbb{E}_t\big[x x^\top X^\top u u^\top\big] = A X^\top S,$$

with $A = \mathbb{E}[x x^\top]$ and $S = \mathbb{E}[u u^\top] = W^\top \mathbb{E}[\delta\delta^\top] W = W^\top D W$.

**Step 5: simplify.** With $X^\top = -X$ the two terms add:

$$F_{\mathrm{in}}(X) = \tfrac12\big(S X A - A X^\top S\big) = \tfrac12\big(S X A + A X S\big).$$

**The output side.** For $\Omega_{\mathrm{out}} \in \mathfrak{so}(m)$, token $t$'s generator is
$\operatorname{skew}(\delta_t x_t^\top W^\top) = \tfrac12\,\delta_t \wedge v_t$ with $v_t = W x_t$, the input
pushed forward to the output space. Steps 2-5 are unchanged with $(u, x)$ replaced by $(v, \delta)$, and
$\mathbb{E}[v v^\top] = W A W^\top$.

**Proposition 1 (Fisher on the generators).** Under the K-FAC independence assumption, and treating
the two sides separately,

$$F_{\mathrm{in}}(X) = \tfrac12\big(S X A + A X S\big), \qquad S = W^\top D W,$$
$$F_{\mathrm{out}}(Y) = \tfrac12\big(D Y V + V Y D\big), \qquad V = W A W^\top,$$

for $X \in \mathfrak{so}(n)$, $Y \in \mathfrak{so}(m)$. Both operators are self-adjoint and positive
semi-definite on the skew matrices: $\langle Z, SXA + AXS\rangle$ is symmetric in $X$ and $Z$, and
$\langle X, F_{\mathrm{in}}(X)\rangle = \mathbb{E}[(u^\top X x)^2] \ge 0$ before the factorisation and
$\tfrac12\langle X, SXA + AXS\rangle = \|S^{1/2} X A^{1/2}\|_F^2 \ge 0$ after it. $\square$

**Relation to K-FAC.** The operator we derived turns out to be K-FAC itself, written on the rotation
generators. The change of $W$ caused by a generator is $J(\Omega) = W\Omega$, whose adjoint onto
$\mathfrak{so}(n)$ is $J^*(M) = \operatorname{skew}(W^\top M)$. Pulling K-FAC's Fisher on $W$,
$F_W(V) = DVA$, back through this map gives

$$J^* F_W J(X) = \operatorname{skew}(W^\top D\, W X\, A) = \operatorname{skew}(S X A) = \tfrac12\big(SXA + AXS\big),$$

which is Proposition 1. So nothing beyond K-FAC's own assumption enters. What the derivation adds is
the form the operator takes on the generators, and the reason it keeps a Kronecker-like structure
there: each token contributes a bivector of two independent vectors. That form decides how the operator
is inverted (Section 3.2). It is not a Kronecker product, so K-FAC's inverse
$D^{-1} V A^{-1}$ does not apply.

**Remarks.**

* **What is approximated.** Three things: the independence of $x$ and $\delta$ (Step 4, as in K-FAC);
  the coupling between the two sides and between layers (dropped); and the empirical Fisher in place
  of the true one (Section 2.2). Everything after Step 4 is exact algebra.
* **Size.** The operator of Step 1 has $O(n^4)$ entries. Proposition 1 fixes it by two $n \times n$
  matrices.
* **The output gradient enters both sides.** It appears on the input side through $S = W^\top D W$,
  and the input moment appears on the output side through $V = W A W^\top$. If $D$ is replaced by $I$
  (no backward statistic), the two sides receive the wrong relative scale (Appendix B).
* **Numerical check.** We checked Proposition 1 against a Monte Carlo estimate with independent $x$
  and $\delta$: relative error {{mc_err_in:.1%}} (in) and {{mc_err_out:.1%}} (out) at $4 \times 10^5$ samples, which is
  sampling noise. The pull-back identity holds to {{pullback_err:.0e}} (`scripts/analysis/check_fisher.py`).

### 3.2 Exact inversion

$F_{\mathrm{in}}$ acts on a space of dimension $n(n-1)/2$. Solving $F_{\mathrm{in}}(X) = G$ as a dense
linear system costs $O(n^6)$, about $10^{16}$ operations for $n = 512$, at every refresh of every matrix.
Nor is $F_{\mathrm{in}}$ the map $X \mapsto SXA$, whose inverse would be $S^{-1} G A^{-1}$: the sum of the
two orderings does not factor unless $S$ and $A$ commute. It is nonetheless diagonalised by a
congruence, which brings the cost to $O(n^3)$.

**Proposition 2 (Inversion).** Let $A \succ 0$ and $S \succeq 0$, and let $P$ solve the generalized
eigenproblem $P^\top A P = I$, $P^\top S P = \Lambda = \operatorname{diag}(\lambda)$. For skew $G$, the
unique skew solution of $F_{\mathrm{in}}(X) = G$ with $\lambda_i + \lambda_j > 0$ is

$$X = P\, \big[ (P^\top G P) \oslash K \big]\, P^\top, \qquad K_{ij} = \tfrac12(\lambda_i + \lambda_j),$$

where $\oslash$ is elementwise division.

*Proof.* Substitute $X = PYP^\top$:
$P^\top (SXA + AXS) P = (P^\top S P)\, Y\, (P^\top A P) + (P^\top A P)\, Y\, (P^\top S P) = \Lambda Y + Y \Lambda$.
So $F_{\mathrm{in}}(X) = G$ becomes $\tfrac12(\lambda_i + \lambda_j)\, Y_{ij} = (P^\top G P)_{ij}$. Since
$P^\top G P$ is skew, so are $Y$ and $X$. $P$ is invertible, so the solution is unique. $\square$

**Computation.** We compute $P = A^{-1/2} Q$, with $Q$ the eigenvectors of $A^{-1/2} S A^{-1/2}$. The
symmetric root, rather than a Cholesky factor, exposes the eigenvalues so they can be floored.

**Regularisation.** It is a single relative floor: every spectrum that reaches a denominator ($A$, $S$
and $\Lambda$) is raised to at least $\varepsilon \lambda_{\max}$, with $\varepsilon = 10^{-4}$. A
floor leaves every eigenvalue above it exact, which an additive shift does not.

**Cost.** One generalized eigendecomposition per side, $O(n^3)$, recomputed every $T_{\mathrm{fac}}$
steps. Each step then costs four matrix products per side.

The same construction applies to the output side, with $A$ replaced by $D$ and $S$ by $V = WAW^\top$.

### 3.3 Step length as a rotation angle

The natural gradient fixes a direction $X = F^{-1}G$. Its length in the Fisher metric,
$\sqrt{\langle X, F X\rangle}$, is the classical trust-region measure, and it is a poor one here. In
our model, $F^{-1}$ scales the gradient very unevenly across layers. With a fixed step size, the
rotation angle actually taken spans {{angle_spread_fixed_c:.2g}} across the 56 matrices (90th over 10th
percentile, Section 5.4): some matrices barely move while others turn by large angles.

The scale is also where K-FAC is least reliable: along our step, its curvature is {{kfac_vs_exact:.2g}} of the
exact per-token curvature on a trained model (Section 7). We use the geometry of the parameterization instead. The step is a rotation, and a rotation has an
angle.

**Proposition 3 (Angle of a Cayley step).** Let $X$ be skew with largest singular value $\sigma$ and
$c > 0$. The rotation $R = \big(I + \tfrac c2 X\big)^{-1}\big(I - \tfrac c2 X\big)$ turns its planes by
the angles $2\arctan(c\,\sigma_k/2)$, where $\sigma_k$ are the singular values of $X$. Its largest
angle is $2\arctan(c\sigma/2) = c\sigma + O((c\sigma)^3)$.

*Proof.* A real skew matrix has eigenvalues $\pm i\sigma_k$, in planes that are mutually orthogonal.
On such a plane $R$ acts by $(1 - ic\sigma_k/2)/(1 + ic\sigma_k/2) = e^{-2i\arctan(c\sigma_k/2)}$.
$\square$

We set

$$c_t = \frac{\theta_t}{\sigma}, \qquad \sigma = \max\big(\|X_{\mathrm{in}}\|_2,\ \|X_{\mathrm{out}}\|_2\big),$$

so the fastest rotation plane of every matrix turns by $2\arctan(\theta_t/2) \approx \theta_t$ radians
at step $t$. Because $2\arctan(\cdot/2) < \pi$, no Cayley step turns any plane by $\pi$ or more,
however large $c$ is: an outsized step saturates instead of overshooting.

**Corollary (the angle bounds the change of $W$).** With $c_t = \theta_t/\sigma$,
$$\|R_{\mathrm{out}} W R_{\mathrm{in}} - W\|_2 \le 2\,\theta_t\,\|W\|_2 .$$

*Proof.* $R$ is orthogonal with eigenvalues $e^{\pm i\varphi_k}$, so
$\|R - I\|_2 = \max_k |e^{i\varphi_k} - 1| = 2\sin(\varphi_{\max}/2) \le \varphi_{\max} \le \theta_t$ for both
sides (both share $c_t$, and each side's largest singular value is at most $\sigma$). Then
$R_{\mathrm{out}} W R_{\mathrm{in}} - W = (R_{\mathrm{out}} - I)\,W R_{\mathrm{in}} + W\,(R_{\mathrm{in}} - I)$. $\square$

So the angle rule is a bound on the relative change of every weight in spectral norm, the quantity that
Muon-style methods and the modular norm (Large et al., 2024) control directly. Here it
follows from the geometry of the rotation. The
other planes turn in proportion to their singular values, so the natural gradient's own ranking of
directions within a matrix is kept. $\theta_t$ follows a cosine schedule from a peak $\theta_0$.

Two design points matter, each tested in Section 5.4:

* **The norm is taken after $F^{-1}$, not before.** Normalising the raw gradient and then applying
  $F^{-1}$ lets $F^{-1}$'s uneven scale back into the step.
* **One $\sigma$ for both sides of a matrix.** The two sides come from one block-diagonal Fisher, and
  the ratio $F^{-1}$ sets between them is part of the direction. Normalising each side by its own
  $\sigma$ discards that ratio.

### 3.4 Retraction and exact spectrum preservation

**Proposition 4.** For skew $\Omega$, the Cayley map $R = (I + \Omega)^{-1}(I - \Omega)$ is orthogonal
for every scale of $\Omega$. Hence $R_{\mathrm{out}} W R_{\mathrm{in}}$ has exactly the singular values
of $W$. The degree-2 truncated exponential $T = I + \Omega + \Omega^2/2$ satisfies
$T^\top T = I + \Omega^4/4$.

*Proof.* The eigenvalues of a real skew $\Omega$ are $\pm i\sigma_k$, so those of $I + \Omega$ are
$1 \pm i\sigma_k \neq 0$ and $I + \Omega$ is invertible. $I + \Omega$ and $I - \Omega$ commute, and
$(I \pm \Omega)^\top = I \mp \Omega$, so
$R^\top R = (I + \Omega)(I - \Omega)^{-1}(I + \Omega)^{-1}(I - \Omega) = I$. For $T$, expand with
$\Omega^\top = -\Omega$: the odd and quadratic terms cancel, leaving $\Omega^4/4$. $\square$

**What the retraction must do.** Stay orthogonal to round-off at 0.05–0.2 rad, on every matrix and every
step. Pion's angles are smaller and uneven: typical block {{pion_angle_block_median:.2g}} rad per step (p90/p10
{{pion_angle_spread:.1f}}), largest {{pion_angle_max:.2g}} (`scripts/probes/pion_angles.py`, first {{pion_angle_steps:,.0f}}
steps). Per step, the truncated exponential changes singular values by up to {{trunc_err_0.2:.0e}} at 0.2,
{{trunc_err_0.05:.0e}} at 0.05, and about {{trunc_err_pion:.0e}} at Pion's typical angle.

*Natural Pion Fast, seed 0. Drift: $\max_i |s_i - s_i^0| / \max_i s_i^0$ after 3 000 steps, median / worst of the
56 matrices (`scripts/probes/retraction_drift.py`).*

| retraction | val, 3 000 steps ($\theta_0$ 0.2) | val, 15 000 steps ($\theta_0$ 0.05) | drift |
|---|---|---|---|
| Cayley | {{batch_val:.4f}} | {{np15k:.4f}} | {{rd_cayley_med:.1e}} / {{rd_cayley_max:.1e}} |
| truncated exponential (Pion's) | {{ret_trunc_3000:.4f}} | {{ret_trunc_15000:.4f}} | {{rd_trunc_med:.1e}} / {{rd_trunc_max:.1e}} |

The loss does not separate them at these horizons (same run on RTX and B200: {{ret_hw_noise:.3f}}); the spectrum does.

**Precision.** The solve must run in true fp32: TF32 moves the spectrum by {{tf32_drift:.2%}} in 200 steps at angle
$10^{-2}$, fp32 by {{fp32_drift_200:.3%}}.

### 3.5 The algorithm

**Algorithm 1: Natural Pion** (exact form; one rotated matrix $W \in \mathbb{R}^{m\times n}$, all of
them processed alike). Everything is recomputed at every step; Section 4 makes it fast.

```
state:  A, D (EMAs of x x^T and of d d^T, fp32; initialised by their first observation),
        M_in, M_out (momentum, initialised at zero), bases (P, lambda) per side
every step t:
    theta_t <- theta_min + (theta_0 - theta_min) (1 + cos(pi t / T)) / 2         cosine, theta_min = 1e-5
    forward: A <- beta_A A + (1 - beta_A) mean_tokens(x x^T)
    backward: D <- beta_D D + (1 - beta_D) mean_tokens(d d^T)     d = per-token output gradient
    (P_in, lambda_in)   <- gen_eig(A, W^T D W)                      the bases (Prop. 2); every spectrum
                                                                    floored at eps * lambda_max
    (P_out, lambda_out) <- gen_eig(D, W A W^T)
    G_in  <- skew(W^T G);   G_out <- skew(G W^T)
    M_in  <- beta1 M_in  + (1 - beta1) G_in
    M_out <- beta1 M_out + (1 - beta1) G_out
    X_in  <- P_in  [ (P_in^T  M_in  P_in)  / K_in  ] P_in^T          K_ij = (lambda_i + lambda_j)/2
    X_out <- P_out [ (P_out^T M_out P_out) / K_out ] P_out^T
    sigma <- max(||X_in||_2, ||X_out||_2)                           warm-started power iteration
    c     <- theta_t / sigma
    W     <- Cay(c X_out) W Cay(c X_in),   Cay(Z) = (I + Z/2)^-1 (I - Z/2)
```

Embedding, output head and normalization gains use AdamW, as in Pion. The rotated matrices get no
weight decay: a rotation cannot change their norm, and their spectrum is fixed by construction.

A constant factor in $A$ or $D$ (e.g. autograd's $1/N$) cancels in $cX$. The floor $10^{-4}$ is the best
setting tried (Section 5.4).

---

## 4 Making it fast: Natural Pion Fast

Algorithm 1 costs {{tf1_mom_sps:.2f}} s per step in our setting, against Pion's {{pion3000_sps:.3f}} s. This section takes it to
{{fast2_sps:.3f}} s, one change at a time, and gives for each change what it saves and what it costs in loss. All
numbers are at 3 000 steps, seed 0, Lie momentum 0.9, on RTX unless marked. Seconds per step are wall-clock time over steps of
the training runs themselves, evaluation every 500 steps included for every method, unless marked as
profiled.

### 4.1 Where the time goes

*Table 1. Seconds per step by part, GPU-synchronised timers (profiled; the synchronisation inflates
the parts slightly).*

| part | Algorithm 1 ($T_{\mathrm{fac}} = 1$) | $T_{\mathrm{fac}} = 25$ |
|---|---|---|
| basis refresh (generalized eigendecompositions) | {{prof_tfac1_refactor:.3f}} | {{prof_tfac25_refactor:.3f}} |
| spectral norm (power iteration) | {{prof_tfac1_angle:.3f}} | {{prof_tfac25_angle:.3f}} |
| Cayley retraction | {{prof_tfac1_retract:.3f}} | {{prof_tfac25_retract:.3f}} |
| $F^{-1}$ and the rest of the step | {{prof_tfac1_rest:.3f}} | {{prof_tfac25_rest:.3f}} |
| accumulating $A$ and $D$ | {{prof_tfac1_stats:.3f}} | {{prof_tfac25_stats:.3f}} |
| forward, backward, AdamW, data (not profiled) | {{prof_tfac1_other:.3f}} | {{prof_tfac25_other:.3f}} |
| whole step (profiled run) | {{prof_tfac1_step:.2f}} | {{prof_tfac25_step:.2f}} |

Forward, backward and AdamW are common to all methods. The profiled runs are 400 steps long and
synchronise the GPU around every part, so their whole-step times differ slightly from the training runs
quoted in the rest of this section.

### 4.2 Amortised refresh

The bases $(P, \lambda)$ are recomputed every $T_{\mathrm{fac}}$ steps instead of every step. In
between, $F^{-1}$ uses the last bases with the current $W$, $G$ and momentum. This is the standard
amortisation of K-FAC. It also removes most of the power-iteration cost: $X$ jumps at a refresh, so the
power iteration needs 50 iterations right after one and 2 otherwise.

| refresh | momentum | val | s / step |
|---|---|---|---|
| every step | none | {{tf1_nomom:.4f}} | {{tf1_nomom_sps:.2f}} |
| every 25 steps | none | {{tf25_nomom:.4f}} (+{{tf25_nomom_d:.3f}}) | {{tf25_nomom_sps:.2f}} |
| every step | 0.9 | {{tf1_mom:.4f}} | {{tf1_mom_sps:.2f}} |
| every 10 steps | 0.9 | {{tf10_mom:.4f}} (+{{tf10_mom_d:.3f}}) | {{tf10_mom_sps:.2f}} |
| every 25 steps | 0.9 | {{tf25_mom:.4f}} (+{{tf25_mom_d:.3f}}) | {{tf25_mom_sps:.2f}} |

Each row's angle is tuned separately. Without momentum a stale basis costs {{tf25_nomom_d:.3f}}; momentum absorbs
half of it at $T_{\mathrm{fac}} = 25$ and all of it at $T_{\mathrm{fac}} = 10$. Over four seeds,
$T_{\mathrm{fac}} = 1$ averages {{tf1_mom_mean:.4f}}, $T_{\mathrm{fac}} = 10$ {{tf10_mom_mean:.4f}} and $T_{\mathrm{fac}} = 25$ {{np3000_mean:.4f}}.
At 15 000 steps the difference is gone: {{tf1_mom_15k:.4f}} for $T_{\mathrm{fac}} = 1$, {{np15k:.4f}} for 25.

### 4.3 Statistics only where they are read

Nothing but a refresh reads $A$ and $D$. So we accumulate them only in the 6 steps before each refresh,
which is 6 of every 25 steps, and shorten the EMA of $A$ from 0.95 to 0.5 so that the window dominates
it: an observation more than 6 steps old then carries less than 2% of the weight. The EMA of $D$ is 0.5
throughout.

| change ($T_{\mathrm{fac}} = 25$) | EMA of $A$ / $D$ | val | s / step |
|---|---|---|---|
| both every step (reference) | 0.95 / 0.5 | {{win_ref:.4f}} | {{win_ref_sps:.3f}} |
| $D$ only in the last 6 steps | 0.95 / 0.5 | {{win_dlast6:.4f}} | {{win_dlast6_sps:.3f}} |
| $A$ only in the last 6 steps | 0.5 / 0.5 | {{win_alast6:.4f}} | {{win_alast6_sps:.3f}} |
| **$A$ and $D$ only in the last 6 steps** | 0.5 / 0.5 | **{{win_adlast6:.4f}}** | **{{win_adlast6_sps:.3f}}** |
| $A$ every second step | 0.9025 / 0.5 | {{win_aevery2:.4f}} | {{win_aevery2_sps:.3f}} |
| $D$ with a longer EMA | 0.95 / 0.9, 0.95 | {{win_betad09:.4f}} / {{win_betad095:.4f}} | {{win_betad09_sps:.3f}} / {{win_betad095_sps:.3f}} |

Every change is within {{win_maxdev:.3f}} of the reference, inside run-to-run noise. Neither the window nor the
factors' memory matters here, so we take the cheapest option. ($A$ every second step uses $0.95^2$ to keep
its time constant.)

### 4.4 A warm power iteration for $\sigma$

The exact spectral norm (`matrix_norm(X, 2)`) took {{svd_share_of_step:.0%}} of the whole optimizer step on these shapes:
cuSOLVER falls back to a slow path on $1376 \times 1376$ matrices. Power iteration on $X^\top X$ costs
matrix-vector products only, {{power_speedup:.0f}}x less. Cold power iteration converges badly on skew matrices,
whose singular values come in equal pairs: on a random $512 \times 512$ skew matrix its relative error
is {{power_cold_1:.1%}} after one iteration, {{power_cold_5:.1%}} after five and {{power_cold_20:.1%}} after twenty. Warm-started from the previous step's
vector, one iteration reaches {{power_warm_1:.0e}}. We run 50 iterations after a refresh and 2 otherwise. Batched over
matrices of one size, with $c$ kept on the GPU (no host synchronisation per matrix), it costs {{bench_new_sigma:.3f}} s per optimizer
step against {{bench_old_sigma:.3f}} s (Table 2b).

### 4.5 Computing only what the step uses

The reference implementation also computed $\langle G, X\rangle$ and $\langle X, FX\rangle$ (four extra
products per side) for a trust ratio the final method does not use, plus per-layer diagnostics and a
backward probe for logging. Removing them leaves the trajectory **bit-identical**: training loss at
every logged step and validation loss over 300 steps match exactly. The step goes from {{win_adlast6_sps:.3f}} s to
**{{fast_sps:.3f}} s**.

### 4.6 Batching by shape

The 56 rotated matrices come in a few shapes. $F^{-1}$ and the Cayley solves are run as one batched call
per size instead of one call per matrix. The mathematics is unchanged and only round-off differs:
{{batch_val:.4f}} against {{ref_tf25:.4f}}. The step goes from {{fast_sps:.3f}} s to **{{batch_sps:.3f}} s**.

### 4.7 Cheaper statistics

In the 6 steps before a refresh every rotated matrix forms two grams, $A$ over its inputs and $D$ over
its output gradients, each over all {{tokens_per_step:,}} tokens of the step and in true fp32, so without tensor
cores.

**One $A$ per input.** The query, key and value projections read the same normalised input, and so do the
gate and up projections; their $A$ is one matrix, formed five times. Natural Pion Fast forms it once per
input tensor. The trajectory is bit-identical (`tests/test_natural_pion.py`).

**bf16 inputs.** Rounding the *assembled* $A$ to bf16 is not usable: on a trained model it pushes the
smallest eigenvalues through zero (Appendix A). Rounding the *inputs* is different: the gram of bf16
vectors is still a gram, so $A$ stays positive semidefinite, and the rounding adds a noise covariance of
relative size about $(2^{-9})^2 \approx 4 \cdot 10^{-6}$, far below the floor $10^{-4} \lambda_{\max}$. We round
activations and output gradients to bf16 and multiply them with fp32 accumulation and an fp32 result.

*On B200, 3 000 steps, seed 0, both with one $A$ per input:*

| statistics | val | s / step |
|---|---|---|
| fp32 inputs | {{stats_fp32_val:.4f}} | {{stats_fp32_sps:.3f}} |
| bf16 inputs | {{stats_bf16_val:.4f}} ({{stats_bf16_d:+.3f}}) | {{stats_bf16_sps:.3f}} |

bf16 inputs save {{stats_bf16_saving:.3f}} s per step at no measurable cost at 3 000 steps; at 15 000 steps
(angle 0.05) the run gives {{stats_bf16_15k:.4f}} against {{np15k:.4f}} for fp32.

### 4.8 The Cayley solve by Newton–Schulz

$\mathrm{Cay} = 2(I + A)^{-1} - I$ with $A = \tfrac c2 X$, and $(I + A)^{-1}$ can be computed by Newton–Schulz,
$Z \leftarrow Z(2I - (I + A)Z)$ from $Z_0 = I - A$, with matrix products only. Two iterations leave
$R = \mathrm{Cay} - 2(I+A)^{-1}A^8$: {{ns_err_02:.0e}} at angle 0.2, {{ns_err_04:.0e}} at 0.4, {{ns_err_08:.0e}} at 0.8 (fp64).

In fp32 the standard form is unstable: it rounds entries next to 1 with a sign, and $\|W\|_F^2$, which a
rotation keeps, grows by a median {{diag_ns_step:.1e}} per training step (exact solve {{diag_cayley_step:.1e}};
`scripts/probes/ns_diag.py`). Carried in $z = Z - I$ ($r = A + z + Az$, $z \leftarrow z - r - zr$,
$R = I + 2z$) nothing is next to 1, at the same cost. Over random rotations ($1376 \times 1376$, angle 0.05),
$\operatorname{tr}(R^\top R - I)/n$ averages {{rb_solve_mean:+.1e}} for the exact solve ($z = {{rb_solve_z:.0f}}$ standard
errors), {{rb_ns_mean:+.1e}} for standard Newton–Schulz and {{rb_ns_delta_mean:+.1e}} ($z = {{rb_ns_delta_z:.1f}}$) in this
form (`scripts/probes/retraction_bias.py`). 200 steps of a $1376 \times 1376$ rotation at angle 0.05
(`scripts/probes/ns_rounding.py`):

| Cayley computed by | $\Delta\|W\|_F^2$, bare matrix | val, 3 000 steps | drift (as in Section 3.4) | s / step, RTX | s / step, B200 |
|---|---|---|---|---|---|
| exact solve | {{rnd_cayley_0.05:+.1e}} | {{batch_val:.4f}} | {{rd_cayley_med:.1e}} / {{rd_cayley_max:.1e}} | {{batch_sps:.3f}} | — |
| Newton–Schulz, standard | {{rnd_ns_0.05:+.1e}} | {{ret_ns_3000:.4f}} | {{rd_ns_med:.1e}} / {{rd_ns_max:.1e}} | {{ret_ns_sps:.3f}} | {{ret_ns_sps_b200:.3f}} |
| Newton–Schulz in $Z - I$ | {{rnd_ns_delta_0.05:+.1e}} | {{ret_nsd_3000:.4f}} (B200) | {{rd_ns_delta_med:.1e}} / {{rd_ns_delta_max:.1e}} | — | {{ret_nsd_sps_b200:.3f}} |

Natural Pion Fast uses the $Z - I$ form from here on; every run in Section 5 used the exact solve.

### 4.9 Tried and not adopted

* **Exact SVD to reshape the step's spectrum** (the polar step in Section 5.4): {{svd_sps_lo:.1f}}–{{svd_sps_hi:.1f}} s per step, and worse in
  loss.
* **A larger $T_{\mathrm{fac}}$ beyond 25** saves little: the refresh is already {{prof_tfac25_refactor:.3f}} s per step.

### 4.10 Summary

*Table 2. From Algorithm 1 to Natural Pion Fast.*

| step | s / step | val (3 000 steps, seed 0) |
|---|---|---|
| Algorithm 1 | {{tf1_mom_sps:.2f}} | {{tf1_mom:.4f}} |
| refresh every 25 steps (4.2) | {{tf25_mom_sps:.3f}} | {{tf25_mom:.4f}} |
| + statistics only before a refresh (4.3) | {{win_adlast6_sps:.3f}} | {{win_adlast6:.4f}} |
| + compute only what the step uses (4.5) | {{fast_sps:.3f}} | {{ref_tf25:.4f}} (bit-identical: {{fast_bit_identical}}) |
| + batching by shape (4.6) | {{batch_sps:.3f}} | {{batch_val:.4f}} |
| **+ batched $\sigma$, one $A$ per input, Newton–Schulz Cayley, bf16 statistics (4.4, 4.7, 4.8)**: Natural Pion Fast | **{{fast2_sps:.3f}}** | **{{fast2_val:.4f}}** |
| Pion | {{pion3000_sps:.3f}} | {{pion3000_s0:.4f}} |

*Table 2b. The optimizer step alone, by part (s; synthetic statistics on the shapes of the model, GPU-synchronised
timers; `scripts/probes/optimizer_step_bench.py`).*

| part | after 4.6 | Natural Pion Fast |
|---|---|---|
| statistics $A$, $D$ | {{bench_old_statistics:.3f}} | {{bench_new_statistics:.3f}} |
| basis refresh (per step) | {{bench_old_refresh:.3f}} | {{bench_new_refresh:.3f}} |
| Cayley | {{bench_old_cayley:.3f}} | {{bench_new_cayley:.3f}} |
| $F^{-1}$ | {{bench_old_Finv:.3f}} | {{bench_new_Finv:.3f}} |
| $\sigma$ | {{bench_old_sigma:.3f}} | {{bench_new_sigma:.3f}} |
| generators, momentum, rotation (as in Pion) | {{bench_old_shared:.3f}} | {{bench_new_shared:.3f}} |
| **total** | {{bench_old_total_synchronised:.3f}} | **{{bench_new_total_synchronised:.3f}}** |
| Pion, whole optimizer step | | {{bench_pion:.3f}} |

Part of the optimizer's time overlaps the next step's host-side work, so the difference in training
({{fast2_sps:.3f}} against {{pion3000_sps:.3f}} s) is smaller than in this table.

**Algorithm 2: Natural Pion Fast.** It is Algorithm 1 with the bases refreshed every 25 steps; $A$ and
$D$ accumulated (EMA 0.5) only in the 6 steps before a refresh; $\sigma$ by warm power iteration (2
iterations, 50 after a refresh, batched); $F^{-1}$ batched by shape; Cayley by Newton–Schulz in $Z - I$; one $A$ per input
tensor, its inputs and those of $D$ in bf16 with fp32 accumulation. At 3 000 steps the form of 4.6 is {{exact_vs_fast_3000:.3f}}
behind Algorithm 1 over four seeds, and at 15 000 steps level or ahead; the final form matches it ({{fast2_val:.4f}} against
{{batch_val:.4f}} at 3 000 steps, {{fast2_15k:.4f}} against {{np15k:.4f}} at 15 000). It is {{fast2_ratio:.2f}}x slower than Pion per step
on our GPU.[^cost] Its peak GPU memory is {{peak_np:.1f}} GB against Pion's {{peak_pion:.1f}} GB (+{{peak_d:.1f}} GB for $A$, $D$, the bases
and the momentum). The runs of Section 5 were made with the form of 4.6 ({{batch_sps:.3f}} s per step), before 4.7–4.8.

[^cost]: One number per quantity throughout: seconds per step are wall-clock time over steps of 3 000-
step runs (Natural Pion Fast, {{batch_sps:.3f}} s) and the 3 840-step Pion run ({{pion3000_sps:.3f}} s), both
evaluating every 500 steps; Algorithm 1 is the 3 000-step run with momentum ({{tf1_mom_sps:.2f}} s). The
full-length runs, which evaluate every 2 000 steps, give {{np_sps:.3f}} s and {{pion_sps:.3f}} s, a ratio of
{{cost_ratio:.2f}}; their wall times are the ones in Table 4.

### 4.11 Cost in floating-point operations

Seconds depend on the GPU and on the implementation; operations do not. Optimizer FLOPs per step as a share
of the model's forward and backward pass ($6 \times$ tokens $\times$ matrix parameters, plus attention;
`scripts/analysis/flops.py`, eigendecompositions counted at $9n^3$):

| | LLaMA-60M | 1.3B (Pion's) |
|---|---|---|
| Pion | {{flops_60M_pion:.1%}} | {{flops_1_3B_pion:.1%}} |
| Natural Pion Fast | {{flops_60M_np:.1%}} | {{flops_1_3B_np:.1%}} |
| of which statistics / $F^{-1}$ / Cayley / refresh | {{flops_60M_part_statistics:.1%}} / {{flops_60M_part_Finv:.1%}} / {{flops_60M_part_Cayley:.1%}} / {{flops_60M_part_refresh:.1%}} | {{flops_1_3B_part_statistics:.1%}} / {{flops_1_3B_part_Finv:.1%}} / {{flops_1_3B_part_Cayley:.1%}} / {{flops_1_3B_part_refresh:.1%}} |
| step, Natural Pion Fast / Pion | {{flops_60M_ratio:.2f}} | {{flops_1_3B_ratio:.2f}} |

$F^{-1}$ and Cayley cost $O(n^3)$ per matrix and step against the model's $O(Tn^2)$ for $T$ tokens, so their share
grows with width at a fixed batch. With data parallelism the forward and backward pass divides among the GPUs;
the optimizer's matrices must be divided among them too (one GPU per matrix), or, replicated on every GPU as
Pion's step is, their share grows with the number of GPUs. Statistics are the largest term in FLOPs but run on tensor cores in bf16; eigendecompositions are the
smallest but latency-bound, which is why Table 2b and this table rank the parts differently.

---

## 5 Experiments

### 5.1 Setup

| | |
|---|---|
| model | LLaMA-style, 60M parameters: hidden 512, 8 layers, 8 heads, SwiGLU FFN 1376, RoPE, RMSNorm, vocabulary 32 100 (T5 tokenizer) |
| data | C4, sequence length 256, batch 512 sequences (131 072 tokens per step) |
| budget | 73 242 steps = 9.6B tokens: Pion's 60M budget |
| rotated | the 56 matrices inside the blocks (43.5% of parameters) |
| the rest | AdamW, lr 8e-3 (the Pion script: 1e-3), betas (0.9, 0.95), weight decay 0.1, cosine; the same for every method |
| precision | bf16 model; statistics, eigendecompositions and retractions in exact fp32 |
| metric | validation loss: token cross-entropy on held-out C4, in nats |
| hardware | one NVIDIA RTX PRO 6000 Blackwell per run; B200 where marked |

**Baseline.** Pion in its published configuration: Lie momentum 0.9, RMS scale 0.2, truncated
exponential, query matrix split per head, bilateral update, lr $10^{-3}$ with a cosine schedule. The
Pion paper chose lr $10^{-3}$ from $\{10^{-4}, 5\cdot10^{-4}, 10^{-3}, 5\cdot10^{-3}, 10^{-2}\}$. We
checked it in our setting (Table 3). $10^{-3}$ is best at 19 200 steps. At 3 840 steps $2\cdot10^{-3}$
is better by {{pion_lr_3840_better:.3f}}, and we report that comparison too.

All comparisons are within one code base and one data pipeline. In our setting Pion ends at a training loss
of {{pion_full_train:.4f}} (validation {{pion_full:.4f}}); the Pion paper reports 3.3575 in its own Megatron-LM setting.

**Natural Pion.** Natural Pion Fast (Algorithm 2, Section 4): $T_{\mathrm{fac}} = 25$; $A$ and $D$
accumulated (EMA 0.5) over the 6 steps before each refresh, one $A$ per input tensor; Lie momentum 0.9; $\varepsilon = 10^{-4}$; cosine schedule on $\theta$ down to
$10^{-5}$. The peak angle $\theta_0$ is 0.2 at 3 000 steps and 0.05 from 15 000 steps on. The best peak
falls with the horizon; 0.05 was chosen on 15 000-step runs and kept for the full length, where the one
other value we ran, 0.025, is worse ({{np_full_clock_0025:.4f}} against {{np_full_clock:.4f}}).

**Equal wall-clock time.** The step counts compared at equal time were fixed before the runs, from the
step times measured at the time (Pion {{clock_pion_sps:.3f}} s, Natural Pion Fast {{batch_sps:.3f}} s): Pion's 73 242 steps
correspond to {{clock_steps_planned:,.0f}} of ours, run as 56 900; 3 840 and 19 200 Pion steps correspond to
3 000 and 15 000. The measured wall times (Table 4) agree.

**The query projection.** Pion rotates the query projection per attention head; all our Natural Pion
runs rotate it whole. With the per-head split, Natural Pion Fast reaches {{qhead_val:.4f}} at 3 000 steps
against {{batch_val:.4f}} without it ({{qhead_d:+.3f}}, inside run-to-run noise), so the difference in
settings does not favour us.

Pion's learning rate was checked up to 19 200 steps; at the full length it is the published $10^{-3}$, not
re-tuned.

*Table 3. Pion learning rate (validation loss, seed 0).*

| Pion steps | 5e-4 | 1e-3 | 2e-3 | 4e-3 |
|---|---|---|---|---|
| 3 840 | {{pion_lr_3840_5e-4:.4f}} | {{pion_lr_3840_1e-3:.4f}} | **{{pion_lr_3840_2e-3:.4f}}** | {{pion_lr_3840_4e-3:.4f}} |
| 19 200 | - | **{{pion_lr_19200_1e-3:.4f}}** | {{pion_lr_19200_2e-3:.4f}} | {{pion_lr_19200_4e-3:.4f}} |

### 5.2 Main result

*Table 4. Full length, seed 0.*

| run | steps | wall-clock | val loss |
|---|---|---|---|
| Pion | 73 242 | {{pion_full_wall:,.0f}} s | {{pion_full:.4f}} |
| Natural Pion, Pion's wall-clock | 56 900 | {{np_full_clock_wall:,.0f}} s | **{{np_full_clock:.4f}}** (-{{gap_full_clock:.3f}}) |
| Natural Pion, Pion's steps | 73 242 | {{np_full_steps_wall:,.0f}} s | **{{np_full_steps:.4f}}** (-{{gap_full_steps:.3f}}) |

![Validation loss against optimizer steps.](figures/preprint/val_vs_steps.png)

![Validation loss against wall-clock time.](figures/preprint/val_vs_time.png)

*Table 5. Equal wall-clock time on one GPU type (RTX PRO 6000) with the form of 4.6, four seeds. Gap = val(Pion) -
val(Natural Pion); positive means Natural Pion is ahead. Pion runs as many steps as fit in Natural Pion's time.*

| budget | Pion steps | Natural Pion steps | seed 0 | seed 1 | seed 2 | seed 3 | mean |
|---|---|---|---|---|---|---|---|
| ~1 800 s | 3 840 | 3 000 | {{gap_1800_s0:.3f}} | {{gap_1800_s1:.3f}} | {{gap_1800_s2:.3f}} | {{gap_1800_s3:.3f}} | **{{gap_1800_mean:.3f}}** |
| ~8 900 s | 19 200 | 15 000 | {{gap_8900_s0:.3f}} | {{gap_8900_s1:.3f}} | {{gap_8900_s2:.3f}} | {{gap_8900_s3:.3f}} | **{{gap_8900_mean:.3f}}** |
| ~34 000 s | 73 242 | 56 900 | {{gap_full_clock:.3f}} | | | | |

Against Pion at its best short-horizon learning rate ($2\cdot10^{-3}$, seed 0), the 1 800 s gap is {{gap_1800_vs_pion2e3:.3f}}.

*Table 6. Equal number of steps.*

| steps | Pion | Natural Pion | gap |
|---|---|---|---|
| 3 000 (mean of 4 seeds) | {{pion3000_mean:.4f}} | {{np3000_mean:.4f}} | {{gap3000:.3f}} |
| 15 000 | {{pion15k:.4f}} | {{np15k:.4f}} | {{gap15k:.3f}} |
| 73 242 | {{pion_full:.4f}} | {{np_full_steps:.4f}} | {{gap_full_steps:.3f}} |

### 5.3 Steps to reach Pion's loss

Every run below has its own full cosine schedule ending at its budget $B$, so no loss is read off the middle of
a schedule. For each method the final validation loss is fitted as $L(B) = a + bB^{-c}$ ({{budget_n_pion}} Pion budgets,
{{budget_n_np}} Natural Pion budgets, four seeds where available; worst residual {{budget_resid:.4f}}), and the saving at Pion's
loss $L$ is $B_{\mathrm{Pion}} / B_{\mathrm{NP}}(L)$, with $B_{\mathrm{NP}}$ from the fit or from linear interpolation in $\log B$
between Natural Pion's budgets (`scripts/analysis/budget_fit.py`).

*Table 7. Steps saved at Pion's loss: fit / interpolation.*

| Pion steps | Pion val | saving |
|---|---|---|
| 5 300 | {{budget_5300_L:.4f}} | {{budget_5300_pow:.2f}} / {{budget_5300_int:.2f}} |
| 15 000 | {{budget_15000_L:.4f}} | {{budget_15000_pow:.2f}} / {{budget_15000_int:.2f}} |
| 19 200 | {{budget_19200_L:.4f}} | {{budget_19200_pow:.2f}} / {{budget_19200_int:.2f}} |
| 30 000 | {{budget_30000_L:.4f}} | {{budget_30000_pow:.2f}} / {{budget_30000_int:.2f}} |
| 45 000 | {{budget_45000_L:.4f}} | {{budget_45000_pow:.2f}} / {{budget_45000_int:.2f}} |
| 73 242 | {{budget_73242_L:.4f}} | {{budget_73242_pow:.2f}} / {{budget_73242_int:.2f}} |

The saving does not shrink with length. Where a step is expensive for reasons other than the optimizer
(large distributed batches, reinforcement learning with generation), steps rather than optimizer time are
the cost that matters.

### 5.4 Ablations

All ablations are at 3 000 steps, seed 0. They were run on the exact form or on the fast form as they
were developed, so each block states its setting and its reference. Where a row says "best", it is the
best of an angle grid; every such optimum is interior to its grid except where marked.

**The direction: natural gradient against the raw generator.** Algorithm 1 ($T_{\mathrm{fac}} = 1$),
best angle of each grid. Same normalisation, retraction and schedule; only $X = F^{-1}G$ against $X = G$
changes.

| | no momentum | Lie momentum 0.9 |
|---|---|---|
| $X = F^{-1}G$ | **{{dir_fisher_nomom:.4f}}** | **{{dir_fisher_mom:.4f}}** |
| $X = G$ | {{dir_raw_nomom:.4f}} | {{dir_raw_mom:.4f}} |
| gain from $F^{-1}$ | {{gain_fisher_nomom:.3f}} | {{gain_fisher_mom:.3f}} |

Momentum supplies part of what $F^{-1}$ does, but not all of it. $X = G$ with momentum is close to Pion
({{pion3000_s0:.4f}}), as it should be.

**The length.** Algorithm 1 without momentum, best of each grid. All rules use the direction $F^{-1}G$. The
last column is the spread of the realised rotation angle across the 56 matrices (90th over 10th
percentile, after step 1000).

| length rule | val | angle spread |
|---|---|---|
| **spectral, after $F^{-1}$: $c = \theta/\sigma(F^{-1}G)$** | **{{dir_fisher_nomom:.4f}}** | {{angle_spread_spectral:.2g}} |
| RMS of the induced update, as in Pion | {{len_rms:.4f}} | {{angle_spread_rms:.2g}} |
| quadratic-model ratio $\min(1, \langle G, X\rangle / \langle X, FX\rangle)$ | {{len_alpha:.4f}} | {{angle_spread_alpha:.2g}} |
| Fisher norm, $c = \theta / \sqrt{\langle X, FX\rangle}$ | $\le$ {{len_fishernorm:.4f}} | {{angle_spread_fishernorm:.2g}} |
| fixed $c$ | {{len_fixed:.4f}} | {{angle_spread_fixed:.2g}} |
| spectral, before $F^{-1}$: $c = \theta/\sigma(G)$ | {{len_before:.4f}} | {{angle_spread_before:.2g}} |

Across six rules, the more evenly the matrices turn, the lower the loss. The spectral rule beats the
RMS rule on all four seeds (gaps {{rms_gap_s0:.3f}}, {{rms_gap_s1:.3f}}, {{rms_gap_s2:.3f}}, {{rms_gap_s3:.3f}}). The Fisher-norm row is the best of its
grid at the edge of the range.

**Regularisation of the Fisher.** Natural Pion Fast, 3 000 steps, angle 0.2. The default is a relative floor
$\varepsilon = 10^{-4}$ ({{batch_val:.4f}}).

| floor $\varepsilon$ (no damping) | $10^{-6}$ | $10^{-5}$ | $10^{-3}$ | $10^{-2}$ |
|---|---|---|---|---|
| val | {{reg_eps_1e-6:.4f}} | {{reg_eps_1e-5:.4f}} | {{reg_eps_1e-3:.4f}} | {{reg_eps_1e-2:.4f}} |

| damping $\delta$ (floor $10^{-6}$; $F + \delta\lambda_{\max} BXB$) | $10^{-4}$ | $10^{-3}$ | $10^{-2}$ | $10^{-1}$ |
|---|---|---|---|---|
| val | {{reg_damp_1e-4:.4f}} | {{reg_damp_1e-3:.4f}} | {{reg_damp_1e-2:.4f}} | {{reg_damp_1e-1:.4f}} |

The floor at $10^{-4}$ is an interior optimum, and the best alternative is {{reg_best_other_d:+.3f}} worse.
Too little regularisation is as harmful as too much: at $10^{-6}$ the step follows directions the
statistics barely determine, and at $10^{-2}$ $F^{-1}$ moves toward the raw generator.

**Other choices.** Single runs at the reference's angle unless stated; the reference value is in brackets.
Rows with reference {{ref_tf25:.4f}} are on the fast form ($T_{\mathrm{fac}} = 25$, momentum, angle 0.2); rows with
reference {{dir_fisher_nomom:.4f}} on Algorithm 1 without momentum (angle 0.4).

| change | val |
|---|---|
| each side normalised by its own $\sigma$, best of angles 0.1 and 0.2 ({{ref_tf25:.4f}}) | {{sides_best:.4f}} |
| attention matrices at $\times 0.25 / 0.5 / 2 / 4$ the angle ({{dir_fisher_nomom:.4f}}) | {{attn_0.25:.4f}} / {{attn_0.5:.4f}} / {{attn_2:.4f}} / {{attn_4:.4f}} |
| constant angle; linear decay; warmup + cosine, best of 3 angles each (cosine {{dir_fisher_nomom:.4f}}) | {{sched_constant:.4f}}; {{sched_linear:.4f}}; {{sched_warmcos:.4f}} |
| linear decay on the fast form (cosine {{ref_tf25:.4f}}) | {{sched_linear_fast:.4f}} |
| warmup-stable-decay, decay over the last 20% / 10% ({{ref_tf25:.4f}}) | {{wsd_0.2:.4f}} / {{wsd_0.1:.4f}} |
| equal angle on every plane of $X$ (polar step), at step 2499 ({{polar_ref_2499:.4f}}) | {{polar_2499:.4f}} |

Linear decay and cosine are within noise of each other on both forms; we keep the cosine. The polar step
needs an SVD per matrix per step ({{svd_sps_lo:.1f}}–{{svd_sps_hi:.1f}} s per step), and its runs could not finish within the
time limit, so it is compared at the last evaluation they all reached, step 2499, against Algorithm 1 with
momentum at the same step.

---

## 6 Related work

<!-- Checked 2026-11-03 against the method sections of FISMO, Go-MUON, ISO, POET-X, POET and 2207.07287
(arXiv HTML); the newer items (Mousse) from abstracts only. References and claims re-checked 2026-11-04 against
arXiv full texts; novelty search re-run the same day (nearest: COEC 2608.21142, Symo 2606.00442, neither a
Fisher on the orbit). Re-run before submission. -->

**Orthogonal equivalence training.** POET (Qiu et al., 2025), following orthogonal fine-tuning (Qiu et al.,
2023), reparameterises $W = R W_0 P$ with orthogonal $R$, $P$ built from skew-symmetric parameters by a
truncated Cayley–Neumann series; POET-X (Qiu et al., 2026) makes it memory- and compute-efficient. Pion (Shi et al.,
2026) turns the same principle into an optimizer, with updates in the Lie algebra. ISO (Zhu et al.,
2026) keeps the spectrum of a pretrained model fixed through a frame parameterisation $W = U\Sigma_0 V^\top$,
applies AdamW or Muon to the frames and retracts them by a polar map, for reinforcement learning from
verifiable rewards, and reports reaching Pion's final accuracy faster on Qwen3-4B. None of these
uses a curvature metric on the orbit: they move along the gradient with a first-order or orthogonalised
preconditioner. Natural Pion keeps Pion's manifold and changes the metric to the Fisher.

**Natural gradient on manifolds and Kronecker factorisations.** The natural gradient (Amari, 1998) has a
long line of approximations. K-FAC (Martens & Grosse, 2015) factorises the Fisher of a layer into input
and output-gradient moments; Shampoo (Gupta et al., 2018) builds Kronecker factors from the gradient
itself. On orthogonality-constrained parameters the natural Riemannian gradient goes back to ICA on
Stiefel manifolds (Amari, 1999), and gradient flows that preserve eigenvalues or singular values are
classical (Helmke & Moore, 1994, Ch. 2–3). Riemannian natural gradient methods (Hu et al., 2024) define the Fisher on a manifold's tangent
space, use a retraction, and work out a K-FAC-style Kronecker approximation for Grassmann manifolds.
Natural Pion is an instance of that framework on a different manifold, the orbit of a matrix under
two-sided rotations. There the K-FAC Fisher on the generators is the two-term operator of Proposition 1
rather than a Kronecker product, and Section 3.2 inverts it exactly. We did not find this operator, or a
natural gradient for orthogonal-equivalence updates, in prior work.

**Curvature-aware orthogonalised updates.** A recent family puts Kronecker curvature into Muon's
additive step. FISMO (Xu et al., 2026) fits Kronecker factors to gradient outer products and takes
$W \leftarrow W - \eta P^{-1/2}\,\mathrm{Polar}(\tilde G)\, Q^{-1/2}$ under a spectral-norm trust region in that
metric. Go-MUON (Che, 2026) applies Muon between inverse fourth roots of K-FAC's activation and
output-gradient moments. Mousse (Zhang et al., 2026) runs
Muon in Shampoo-whitened coordinates. All three are additive and change the weight spectrum; Natural
Pion's update is a rotation and keeps it.

**Orthogonalised additive updates.** Muon (Jordan et al., 2024) replaces the update by its orthogonal
polar factor, $W \leftarrow W - \eta\, UV^\top$, which equalises the update's singular values but
leaves the weight's spectrum free.

**Step size from geometry.** Grafting (Agarwal et al., 2020) separates the direction of one optimizer
from the step size of another. Muon and the modular norm (Large et al., 2024; Bernstein & Newhouse, 2024) set the
step in the spectral norm. Our length rule is spectral as well, but applied to the natural-gradient
generator. Its unit is the rotation angle, which the multiplicative update makes literal.

**Retractions on the orthogonal group.** The Cayley transform is a standard retraction on the
orthogonal and Stiefel manifolds (e.g. Wen & Yin, 2013; Li, Li & Todorovic, 2020). Pion tested it and
reports modest gains over its truncated exponential; at Natural Pion's angles it is what keeps the spectrum
(Section 3.4).

---

## 7 Limitations

* **Scale.** One model size (60M) and one data set. The full-length comparison is one seed; four seeds
  at shorter horizons agree in sign and size.
* **One code base.** Our Pion ends {{pion_vs_paper:.3f}} above the Pion paper's training loss; we compare within our
  setting only.
* **Exactness in practice.** In fp32 the exact Cayley solve inflates the spectrum by {{rd_cayley_signed:+.2%}} over 3 000 steps;
  the Newton–Schulz form of Section 4.8 does not.
* **Only Pion as a baseline.** We compare against the published Pion configuration only, not against
  AdamW or Muon, and only on validation loss, not on downstream tasks.
* **The peak angle depends on the horizon.** The best peak falls with the training length (0.2 at 3 000
  steps, 0.05 from 15 000 on), so it has to be chosen for the budget, as a learning rate is.
* **The K-FAC assumption.** Along our step, the K-FAC curvature is {{kfac_vs_exact:.2g}} of the exact per-token
  curvature (median of three layers of a trained model): the independence of $x$ and $\delta$ fails by more
  than an order of magnitude in scale. This is consistent with Section 3.3: we keep the direction of
  $F^{-1}G$ but not its length.
* **Cost.** $F^{-1}$, Cayley and the refresh are $O(n^3)$ per matrix: {{flops_60M_ratio:.2f}}x Pion's step FLOPs at 60M,
  {{flops_1_3B_ratio:.2f}}x at 1.3B (Section 4.11); on several GPUs the matrices have to be divided among them to keep
  that share.

**Future work.** Reinforcement learning from verifiable rewards, where a step costs minutes of
generation and the step efficiency of Section 5.3 is the relevant measure. The natural policy gradient
lineage (Kakade, 2001) makes it the natural test. Larger models, where the optimizer's $O(n^3)$ work has to be
divided across devices (at the shapes of a 1.7B model, one refresh of all matrices takes {{rl_refactor_b200:.0f}} s on one B200)
and can be cut further: on a rectangular matrix the wide side need not form the block of its generator that
cannot move $W$, which brings $F^{-1}$ and the Cayley step from $O(m^3)$ to $O(m^2 n)$ and a Woodbury solve of size $2n$.

---

## References

- Che, T. *Second-order Muon done right: a principled marriage of spectral geometry and curvature.* arXiv:2608.09763, 2026.
- Helmke, U., Moore, J. B. *Optimization and Dynamical Systems.* Springer, 1994.
- Hu, J., Ao, R., So, A. M.-C., Yang, M., Wen, Z. *Riemannian natural gradient methods.* SIAM J. Sci. Comput. 46(1):A204–A231, 2024; arXiv:2207.07287.
- Zhang, Y., Xing, S., Huang, J., Lv, K., Zhou, Y., Qiu, X., Guo, Q., Chen, K. *Mousse: rectifying the geometry of Muon with curvature-aware preconditioning.* arXiv:2603.09697, 2026.
- Qiu, Z., Liu, L., Weller, A., Shi, H., Liu, W. *POET-X: memory-efficient LLM training by scaling orthogonal transformation.* ICML 2026; arXiv:2603.05500.
- Xu, C., Yan, W., Zhang, Y.-J. A. *FISMO: Fisher-structured momentum-orthogonalized optimizer.* arXiv:2601.21750, 2026.
- Zhu, H., Cong, W., Sha, Z., et al. *ISO: an RLVR-native optimization stack.* arXiv:2607.19331, 2026.
- Agarwal, N., Anil, R., Hazan, E., Koren, T., Zhang, C. *Disentangling adaptive gradient methods from learning rates.* arXiv:2002.11803, 2020.
- Amari, S. *Natural gradient works efficiently in learning.* Neural Computation 10(2):251–276, 1998.
- Amari, S. *Natural gradient learning for over- and under-complete bases in ICA.* Neural Computation 11(8):1875–1883, 1999.
- Bernstein, J., Newhouse, L. *Old optimizer, new norm: an anthology.* OPT Workshop 2024; arXiv:2409.20325.
- Gupta, V., Koren, T., Singer, Y. *Shampoo: preconditioned stochastic tensor optimization.* ICML 2018.
- Jordan, K., Jin, Y., Boza, V., You, J., Cesista, F., Newhouse, L., Bernstein, J. *Muon: an optimizer for hidden layers in neural networks.* 2024. https://kellerjordan.github.io/posts/muon/
- Kakade, S. *A natural policy gradient.* NeurIPS 2001.
- Li, J., Li, F., Todorovic, S. *Efficient Riemannian optimization on the Stiefel manifold via the Cayley transform.* ICLR 2020, arXiv:2002.01113.
- Martens, J., Grosse, R. *Optimizing neural networks with Kronecker-factored approximate curvature.* ICML 2015, arXiv:1503.05671.
- Martin, C. H., Mahoney, M. W. *Implicit self-regularization in deep neural networks: evidence from random matrix theory and implications for learning.* JMLR 22(165):1–73, 2021.
- Qiu, Z., Buchholz, S., Xiao, T. Z., Dax, M., Schölkopf, B., Liu, W. *Reparameterized LLM training via orthogonal equivalence transformation (POET).* NeurIPS 2025; arXiv:2506.08001.
- Shi, K., Li, H., Qiu, Z., Wen, Y., Buchholz, S., Liu, W. *Pion: a spectrum-preserving optimizer via orthogonal equivalence transformation.* arXiv:2605.12492, 2026.
- Wen, Z., Yin, W. *A feasible method for optimization with orthogonality constraints.* Mathematical Programming 142:397–434, 2013.
- Large, T., Liu, Y., Huh, M., Bahng, H., Isola, P., Bernstein, J. *Scalable optimization in the modular norm.* NeurIPS 2024; arXiv:2405.14813.
- Qiu, Z., Liu, W., Feng, H., Xue, Y., Feng, Y., Liu, Z., Zhang, D., Weller, A., Schölkopf, B. *Controlling text-to-image diffusion by orthogonal finetuning.* NeurIPS 2023; arXiv:2306.07280.
- Raffel, C., et al. *Exploring the limits of transfer learning with a unified text-to-text transformer (C4, T5).* JMLR 2020.

---

## Appendix A: implementation details

* **Statistics.** $A$ is fed by a forward pre-hook on each rotated linear layer and $D$ by a hook on its
  output gradient, from one micro-batch per step. Both accumulate in fp32 with autocast disabled: a
  bf16 product that returns a bf16 matrix pushes the near-zero eigenvalues of $A$ below zero (measured
  at step 499: thousands of negative eigenvalues across the 56 layers). Rounding only the inputs and
  keeping fp32 accumulation and result (Section 4.7) does not, since the result is still a gram. The $1/N$ that autograd puts on a
  mean loss is undone by scaling $D$ by $N^2$; Algorithm 1 is invariant to it anyway.
* **Power iteration.** $\|X\|_2$ is estimated by power iteration on $X^\top X$, warm-started from the
  previous step's vector: 2 iterations per step and 50 after a refresh, where $X$ jumps. It is
  {{power_speedup:.0f}}x cheaper than an exact spectral norm on these shapes.
* **Floors.** $A$, $S$ (or $D$, $V$) and the pencil eigenvalues are floored at
  $10^{-4}\,\lambda_{\max}$ before any division.
* **Code.** `ngd_pion/natural_pion.py` (`NaturalPion`); every job and run record is listed in
  `scripts/paper/README.md` and `runs-record/`. Every number in this paper is computed from the run
  records by `scripts/analysis/paper_numbers.py` (the list, with sources, is `docs/paper/numbers.md`).

## Appendix B: why the output-gradient statistic is needed

If $D$ is taken to be the identity, $S = W^\top W$ and the step's scale per layer goes as
$\|\delta\| / (\|W\|\,\|x\|)$ instead of $1 / (\|W\|\,\|\delta\|\,\|x\|)$. The step is then largest
where the backward signal is strongest, the opposite of what curvature prescribes.

## Appendix C: hyperparameters

| | Natural Pion | Pion |
|---|---|---|
| rotated-matrix step | peak angle $\theta_0$ 0.2 (3k steps), 0.05 ($\ge$ 15k), cosine to $10^{-5}$ | lr $10^{-3}$, cosine to $10^{-5}$, RMS 0.2 |
| momentum | Lie, 0.9 | Lie, 0.9 |
| curvature | $T_{\mathrm{fac}}$ 25; $A$, $D$ EMA 0.5 over the 6 steps before a refresh; floor $10^{-4}$ | - |
| retraction | Cayley (exact fp32 solve) | truncated exponential, degree 2 |
| query projection | rotated whole (per head: Section 5.1) | rotated per head |
| update | both sides every step | both sides every step |
| other parameters | AdamW 8e-3, (0.9, 0.95), wd 0.1, cosine | same |
| gradient clipping | 1.0 | 1.0 |
| warmup | none | none |

## Appendix D: remarks

**The initial spectrum.** Under Pion and Natural Pion the singular values of a rotated matrix stay those
it was initialised with, so the initial spectrum is part of the trained model. We ran a short check with
a power-law tail in the initial spectrum: the largest 20% of the singular values of a
$\mathcal N(0, 0.02^2)$ matrix are replaced by $s_i \propto i^{-1/(2(\alpha-1))}$, the shape reported for
trained layers (Martin & Mahoney, 2021). It changes energy as well as shape.

| initial spectrum (3 000 steps, seed 0) | val | vs normal |
|---|---|---|
| $\mathcal N(0, 0.02^2)$ | {{init_normal:.4f}} | |
| tail, $\alpha$ = 2 / 2.5 / 3 / 4 / 5 | {{init_2:.4f}} / {{init_2.5:.4f}} / {{init_3:.4f}} / {{init_4:.4f}} / {{init_5:.4f}} | best {{init_best_d:.3f}} |
| flat, all singular values $\sqrt{m/n}$ | {{init_flat:.4f}} | +{{init_flat_d:.3f}} |
| orthogonal, all singular values 1 | {{init_orth:.4f}} | +{{init_orth_d:.3f}} |

*Gain from $\alpha = 3$: val with $\mathcal N(0, 0.02^2)$ minus val with $\alpha = 3$; positive means $\alpha = 3$ helps.*

| | 3 000 steps (4 seeds) | 15 000 steps | full length |
|---|---|---|---|
| Natural Pion | {{init_gain_3000_mean:+.3f}} | {{init_gain_15k:+.3f}} | {{init_gain_full:+.3f}} |
| Pion | {{init_pion_gain_3000:+.3f}} (seed 0) | {{init_pion_gain_15k:+.3f}} | {{init_pion_gain_full:+.3f}} |

Every other experiment in the paper uses $\mathcal N(0, 0.02^2)$.
