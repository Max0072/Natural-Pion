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
reaches a validation loss of **[v2_full?]** in Pion's number of steps, against Pion's 3.3706. With each
method on its own full schedule, it needs **[budget_lo?]**–**[budget_hi?]**x fewer steps to reach the losses Pion reaches.

The method rests on three results. First, on the rotation generators the Kronecker-factored (K-FAC)
Fisher information takes a closed form, $F(X) = \tfrac12(SXA + AXS)$. Here $A$ is the second moment of
the layer's inputs and $S$ that of its output gradients pulled back through $W$. A single generalized
eigendecomposition per side inverts this operator exactly. Second, the natural gradient is a good
direction with a bad scale: its magnitude varies widely between layers. We keep the
direction and set the length so that every matrix turns its fastest rotation plane by the same scheduled
angle. Third, these angles are large, so the rotation must be exact at large angles: a Cayley retraction, computed
by Newton–Schulz, keeps the spectrum to $2.3 \cdot 10^{-6}$ of each matrix's scale. The method costs 1.09x Pion's floating-point operations per step (1.08x its time on one GPU in
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
different rotation directions, and across layers, can differ widely. The classical remedy
is the natural gradient (Amari, 1998): steepest descent in the metric of the model's output
distribution, the Fisher information. This paper derives the natural gradient for Pion's parameterization, makes
it exact and cheap, and pairs it with a step-length rule in the spirit of Pion's normalisation.

**Contributions.**

1. **The Fisher on rotation generators, in closed form.** Under the K-FAC independence
   approximation, the Fisher information restricted to the rotations of $W$ is
   $F(X) = \tfrac12(SXA + AXS)$ on each side (Proposition 1). Its factors are the input second
   moment $A$ and the output-gradient second moment $D$ pulled through $W$, $S = W^\top D W$. The structure has a
   simple origin: a single token's rotation gradient is a bivector, the wedge product of two vectors.
2. **Exact inversion.** One generalized eigendecomposition per side diagonalises $F$, and the
   solve becomes an elementwise division (Proposition 2). The only regularisation is a relative
   floor on the eigenvalues.
3. **Direction from the Fisher, length from an angle.** We take only the direction of $F^{-1}G$. The step length is set so
   that the largest rotation plane of every matrix turns by the same scheduled angle $\theta_t$
   (Proposition 3). With the same angle rule and the same momentum, the natural gradient reaches
   **[dir_fisher_mom?]** at 3 000 steps against **[dir_raw_mom?]** for Pion's raw generator (Pion itself:
   3.8412). With a fixed step size instead of the angle, $F^{-1}G$ gives
   **[len_fixed?]** (Section 5.4).
4. **A retraction that is exact at large angles.** Natural Pion turns each matrix by up to 0.05–0.2 rad per step.
   Cayley computed by Newton–Schulz in $Z - I$ keeps the spectrum to $2.3 \cdot 10^{-6}$ of each matrix's scale after
   3 000 steps; Pion's degree-2 truncated exponential, accurate at Pion's small angles, moves it by $5.8 \cdot 10^{-4}$
   at ours (Sections 3.4, 4.7).
5. **A fast implementation, Natural Pion Fast.** Amortised refreshes, statistics collected only where they are read, a warm and batched
   power iteration, batching, a Newton–Schulz Cayley solve and bf16 statistics take the step from
   2.10 s to 0.507 s (Pion: 0.468 s). Section 4 gives the time and loss along this chain
   and the cost in FLOPs.
6. **Experiments.** On LLaMA-60M / C4 at Pion's full 9.6B-token budget, the validation-loss gap to
   Pion at equal steps is **[v2_gap_full?]** (0.127 at 3 000 steps over four seeds), and Natural Pion needs
   **[budget_lo?]**–**[budget_hi?]**x fewer steps. Ablations vary one component at a time.

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
variant, which it finds slightly worse at 60M). Embeddings, the output head and normalization
gains are trained by AdamW. Only the 2-D weights inside the transformer blocks are rotated.

**Why preserve the spectrum.** It fixes every weight's spectral norm without weight decay or
normalisation; Pion reports stable training without normalisation layers, learning-rate transfer across
widths, and results close to Muon at 1.3B parameters (Shi et al., 2026). This paper is about the direction of
motion inside the constraint.

### 2.2 The natural gradient and K-FAC

For a model with predictive distribution $p_\phi(y \mid x)$, the Fisher information
$F = \mathbb{E}[\nabla_\phi \log p\, \nabla_\phi \log p^\top]$ defines a metric on parameter space. The
natural gradient $F^{-1}\nabla L$ is the steepest-descent direction in that metric (Amari, 1998):

$$-F^{-1}\nabla L \;\propto\; \arg\min_{\Delta}\; \langle \nabla L, \Delta\rangle
\quad \text{s.t.} \quad \tfrac12 \Delta^\top F \Delta \le r .$$

In a language model the "examples" of this expectation are token positions. A linear layer $y = Wx$
sees one input vector $x_t$ and receives one output gradient $\delta_t = \partial \ell_t / \partial y_t$
per token $t$, and its weight gradient is the average of the per-token outer products
$\delta_t x_t^\top$. K-FAC (Martens & Grosse, 2015) assumes $x$ and $\delta$ are independent, which gives
$F_W \approx A \otimes D$ with $A = \mathbb{E}_t[x_t x_t^\top]$ and $D = \mathbb{E}_t[\delta_t\delta_t^\top]$.
As an operator on matrices, $F_W(Z) = D Z A$.

Strictly, the Fisher averages over outputs $y$ sampled from the model. Like most K-FAC
implementations, we use the **empirical Fisher**: $\delta_t$ is the gradient of the training loss at
the observed next token (Section 5.4 compares it with the true Fisher).

---

## 3 Method

### 3.1 The Fisher on rotation generators

We derive the Fisher of the input-side generator $\Omega_{\mathrm{in}} \in \mathfrak{so}(n)$ in five
steps; the output side follows by symmetry.

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
pushed forward to the output space. Steps 2-5 are unchanged with $(u, x)$ replaced by $(\delta, v)$, and
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
$F_W(Z) = DZA$, back through this map gives

$$J^* F_W J(X) = \operatorname{skew}(W^\top D\, W X\, A) = \operatorname{skew}(S X A) = \tfrac12\big(SXA + AXS\big),$$

which is Proposition 1: K-FAC written on the generators. It is not a Kronecker product, so K-FAC's
inverse $D^{-1} Z A^{-1}$ does not apply; Section 3.2 inverts it.

**Remarks.**

* **What is approximated.** Three things: the independence of $x$ and $\delta$ (Step 4, as in K-FAC);
  the coupling between the two sides and between layers (dropped); and the empirical Fisher in place
  of the true one (Section 2.2). Everything after Step 4 is exact algebra.
* **Size.** The operator of Step 1 has $O(n^4)$ entries. Proposition 1 fixes it by two $n \times n$
  matrices.
* **The output gradient enters both sides.** It appears on the input side through $S = W^\top D W$,
  and the input moment appears on the output side through $V = W A W^\top$. If $D$ is replaced by $I$
  (no backward statistic), the step's scale across layers follows the backward signal instead of the
  curvature (Appendix B).
* **Numerical check.** We checked Proposition 1 against a Monte Carlo estimate with independent $x$
  and $\delta$: relative error 0.7% (in) and 0.4% (out) at $4 \times 10^5$ samples. The pull-back identity holds to $4 \cdot 10^{-15}$ (`scripts/analysis/check_fisher.py`).

### 3.2 Exact inversion

$F_{\mathrm{in}}$ acts on a space of dimension $n(n-1)/2$. Solving $F_{\mathrm{in}}(X) = G$ as a dense
linear system costs $O(n^6)$, about $10^{15}$ operations for $n = 512$, at every refresh of every matrix.
Nor is $F_{\mathrm{in}}$ the map $X \mapsto SXA$, whose inverse would be $S^{-1} G A^{-1}$: the sum of the
two orderings does not factor unless $S$ and $A$ commute. It is nonetheless diagonalised by a
congruence, which brings the cost to $O(n^3)$.

**Proposition 2 (Inversion).** Let $A \succ 0$ and $S \succeq 0$, and let $P$ solve the generalized
eigenproblem $P^\top A P = I$, $P^\top S P = \Lambda = \operatorname{diag}(\lambda)$, with $\lambda_i + \lambda_j > 0$
for all $i, j$. For skew $G$, the unique skew solution of $F_{\mathrm{in}}(X) = G$ is

$$X = P\, \big[ (P^\top G P) \oslash K \big]\, P^\top, \qquad K_{ij} = \tfrac12(\lambda_i + \lambda_j),$$

where $\oslash$ is elementwise division.

*Proof.* Substitute $X = PYP^\top$:
$P^\top (SXA + AXS) P = (P^\top S P)\, Y\, (P^\top A P) + (P^\top A P)\, Y\, (P^\top S P) = \Lambda Y + Y \Lambda$.
So $F_{\mathrm{in}}(X) = G$ becomes $\tfrac12(\lambda_i + \lambda_j)\, Y_{ij} = (P^\top G P)_{ij}$. Since
$P^\top G P$ is skew, so are $Y$ and $X$. $P$ is invertible, so the solution is unique. $\square$

**Computation.** We compute $P = A^{-1/2} Q$, with $Q$ the eigenvectors of $A^{-1/2} S A^{-1/2}$. The
symmetric root, rather than a Cholesky factor, exposes the eigenvalues so they can be floored.

**Regularisation.** It is a single relative floor: every spectrum the solve uses ($A$, $S$
and $\Lambda$) is raised to at least $\varepsilon \lambda_{\max}$, with $\varepsilon = 10^{-4}$. A
floor leaves every eigenvalue above it exact, which an additive shift does not.

**Cost.** One generalized eigendecomposition per side, $O(n^3)$, at every step (Section 4.2 amortises it). Each step then costs four matrix products per side.

The same construction applies to the output side, with $A$ replaced by $D$ and $S$ by $V = WAW^\top$.

### 3.3 Step length as a rotation angle

The natural gradient fixes a direction $X = F^{-1}G$. Its length in the Fisher metric,
$\sqrt{\langle X, F X\rangle}$, is the classical trust-region measure, and it is a poor one here. In
our model, $F^{-1}$ scales the gradient very unevenly across layers. With a fixed step size, the
rotation angle actually taken spans **[angle_spread_fixed?]** across the rotated matrices (90th over 10th
percentile, Section 5.4): some matrices barely move while others turn by large angles.

The scale is also where K-FAC is least reliable: along our step, its curvature is 0.034 of the
exact per-token curvature on a trained model (`scripts/probes/paper_checks.py`). We use the geometry of the parameterization instead. The step is a rotation, and a rotation has an
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
at step $t$. $\sigma$ is computed by power iteration on $X^\top X$, warm-started from the previous step's vector
(Section 4.5). Because $2\arctan(\cdot/2) < \pi$, no Cayley step turns any plane by $\pi$ or more,
however large $c$ is: an outsized step saturates instead of overshooting.

**Corollary (the angle bounds the change of $W$).** With $c_t = \theta_t/\sigma$,
$$\|R_{\mathrm{out}} W R_{\mathrm{in}} - W\|_2 \le 2\,\theta_t\,\|W\|_2 .$$

*Proof.* $R$ is orthogonal with eigenvalues $e^{\pm i\varphi_k}$, so
$\|R - I\|_2 = \max_k |e^{i\varphi_k} - 1| = 2\sin(\varphi_{\max}/2) \le \varphi_{\max} \le \theta_t$ for both
sides (both share $c_t$, and each side's largest singular value is at most $\sigma$). Then
$R_{\mathrm{out}} W R_{\mathrm{in}} - W = (R_{\mathrm{out}} - I)\,W R_{\mathrm{in}} + W\,(R_{\mathrm{in}} - I)$. $\square$

So the angle rule is a bound on the relative change of every weight in spectral norm, the quantity that
Muon-style methods and the modular norm (Large et al., 2024) control directly. The other planes turn in proportion to their singular values, so the natural gradient's own ranking of
directions within a matrix is kept. $\theta_t$ follows a cosine schedule from a peak $\theta_0$.

Two design choices, each tested in Section 5.4:

* **The norm is taken after $F^{-1}$, not before.** Normalising the raw gradient and then applying
  $F^{-1}$ lets $F^{-1}$'s uneven scale back into the step.
* **One $\sigma$ for both sides of a matrix.** The two sides come from one block-diagonal Fisher, and
  the ratio $F^{-1}$ sets between them is part of the direction. Normalising each side by its own
  $\sigma$ discards that ratio.

### 3.4 Retraction and exact spectrum preservation

**Proposition 4.** For skew $\Psi$, the Cayley map $R = (I + \Psi)^{-1}(I - \Psi)$ is orthogonal
for every scale of $\Psi$. Hence $R_{\mathrm{out}} W R_{\mathrm{in}}$ has exactly the singular values
of $W$. The degree-2 truncated exponential $T = I + \Psi + \Psi^2/2$ satisfies
$T^\top T = I + \Psi^4/4$.

*Proof.* The eigenvalues of a real skew $\Psi$ are $\pm i\sigma_k$, so those of $I + \Psi$ are
$1 \pm i\sigma_k \neq 0$ and $I + \Psi$ is invertible. $I + \Psi$ and $I - \Psi$ commute, and
$(I \pm \Psi)^\top = I \mp \Psi$, so
$R^\top R = (I + \Psi)(I - \Psi)^{-1}(I + \Psi)^{-1}(I - \Psi) = I$. For $T$, expand with
$\Psi^\top = -\Psi$: the odd and quadratic terms cancel, leaving $\Psi^4/4$. $\square$

**What the retraction must do.** Stay orthogonal to round-off at 0.05–0.2 rad, on every matrix and every
step. Pion's angles are smaller and uneven: typical block 0.026 rad per step (p90/p10
2.0), largest 0.097 (`scripts/probes/pion_angles.py`, first 1 058
steps). By Proposition 4, one step of the truncated exponential changes singular values by about $\theta^4/8$:
$2 \cdot 10^{-4}$ at 0.2,
$8 \cdot 10^{-7}$ at 0.05, and about $6 \cdot 10^{-8}$ at Pion's typical angle.

In training (Section 4.7), after 3 000 steps, Cayley computed by Newton–Schulz keeps each matrix's spectrum to
$2.3 \cdot 10^{-6}$ of its scale (median over matrices); the truncated exponential moves it by $5.8 \cdot 10^{-4}$.

**Precision.** The solve must run in true fp32: TF32 moves a singular value by up to 0.37% (relative) in 200 steps at angle
$10^{-2}$, fp32 by 0.012%.

### 3.5 The algorithm

**Algorithm 1: Natural Pion** (exact form; one rotated matrix $W \in \mathbb{R}^{m\times n}$, all of
them processed alike). Everything is recomputed at every step; Section 4 makes it fast.

```
state:  A, D (EMAs of x x^T and of d d^T, fp32; initialised by their first observation),
        beta_A = 0.95, beta_D = 0.5, beta1 = 0.9, eps = 1e-4,
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
    sigma <- max(||X_in||_2, ||X_out||_2)                           power iteration, warm-started, 50 iterations
    c     <- theta_t / sigma
    W     <- Cay(c X_out) W Cay(c X_in),   Cay(cX) = (I + cX/2)^-1 (I - cX/2)
```

Embedding, output head and normalization gains use AdamW, as in Pion. The rotated matrices get no
weight decay: a rotation cannot change their norm, and their spectrum is fixed by construction.

A constant factor in $A$ or $D$ (e.g. autograd's $1/N$) cancels in $cX$.

---

## 4 Making it fast: Natural Pion Fast

Algorithm 1 costs 2.10 s per step in our setting, against Pion's 0.468 s. This section takes it to
0.507 s, one change at a time, and gives for each change what it saves and what it costs in loss. Unless stated
otherwise, numbers are at 3 000 steps, seed 0, Lie momentum 0.9, on one RTX PRO 6000. Seconds per step are wall-clock time over steps of
the training runs themselves, evaluation every 500 steps included for every method, unless marked as
profiled.

### 4.1 Where the time goes

*Table 1. Seconds per step by part, over 400 profiled steps with GPU-synchronised timers (the
synchronisation inflates the parts slightly).*

| part | Algorithm 1 ($T_{\mathrm{fac}} = 1$) | $T_{\mathrm{fac}} = 25$ |
|---|---|---|
| basis refresh (generalized eigendecompositions) | 0.601 | 0.026 |
| spectral norm (power iteration) | 0.837 | 0.078 |
| Cayley retraction | 0.055 | 0.055 |
| $F^{-1}$ | 0.014 | 0.015 |
| generators, momentum, rotation (as in Pion) | 0.011 | 0.011 |
| forward and backward (with the hooks that accumulate $A$ and $D$), AdamW, data | 0.566 | 0.566 |
| whole step (profiled run) | 2.08 | 0.75 |

### 4.2 Amortised refresh

The bases $(P, \lambda)$ are recomputed every $T_{\mathrm{fac}}$ steps instead of every step. In
between, $F^{-1}$ uses the last bases with the current $W$, $G$ and momentum (the standard amortisation
of K-FAC).

*Table 2. Refresh cadence, each row at its best angle.*

| refresh | momentum | val | s / step |
|---|---|---|---|
| every step | none | 3.7253 | 2.05 |
| every 25 steps | none | 3.7686 (+0.043) | 0.74 |
| every step | 0.9 | 3.7006 | 2.10 |
| every 10 steps | 0.9 | 3.6925 (-0.008) | 0.85 |
| every 25 steps | 0.9 | **[tf25_mom?]** (**[tf25_mom_d?]**) | **[tf25_mom_sps?]** |

Each row's angle is tuned separately ($T_{\mathrm{fac}} = 10$: angle 0.2). Over four seeds, $T_{\mathrm{fac}} = 1$ averages
**[tf1_mom_mean?]**, $T_{\mathrm{fac}} = 10$ **[tf10_mom_mean?]** and Natural Pion Fast ($T_{\mathrm{fac}} = 25$)
3.7122. At 15 000 steps: **[tf1_mom_15k?]** for $T_{\mathrm{fac}} = 1$, 3.4742 for Natural Pion Fast.

### 4.3 Statistics only where they are read

Nothing but a refresh reads $A$ and $D$. So we accumulate them only in the 6 steps before each refresh,
which is 6 of every 25 steps, and shorten the EMA of $A$ from 0.95 to 0.5 so that the window dominates
it: an observation more than 6 steps old then carries less than 2% of the weight. The EMA of $D$ is 0.5
throughout.

*Table 3. Statistics windows ($T_{\mathrm{fac}} = 25$, angle 0.2).*

| change ($T_{\mathrm{fac}} = 25$) | EMA of $A$ / $D$ | val | s / step |
|---|---|---|---|
| both every step (reference) | 0.95 / 0.5 | 3.7186 | 0.742 |
| $D$ only in the last 6 steps | 0.95 / 0.5 | **[win_dlast6?]** | **[win_dlast6_sps?]** |
| $A$ only in the last 6 steps | 0.5 / 0.5 | **[win_alast6?]** | **[win_alast6_sps?]** |
| **$A$ and $D$ only in the last 6 steps** | 0.5 / 0.5 | ****[win_adlast6?]**** | ****[win_adlast6_sps?]**** |
| $A$ every second step | 0.9025 / 0.5 | **[win_aevery2?]** | **[win_aevery2_sps?]** |
| $D$ with a longer EMA | 0.95 / 0.9, 0.95 | **[win_betad09?]** / **[win_betad095?]** | **[win_betad09_sps?]** / **[win_betad095_sps?]** |

The largest change from the reference is **[win_maxdev?]**; we take the cheapest option. ($A$ every second step uses $0.95^2$ to keep
its time constant.)

### 4.4 Batching by shape

The 56 rotated matrices come in a few shapes. $F^{-1}$ and the Cayley solves are run as one batched call
per size instead of one call per matrix. Only round-off changes:
3.7146 against **[win_adlast6?]**. The step goes from **[win_adlast6_sps?]** s to **0.641 s**.

### 4.5 Fewer and batched power iterations for $\sigma$

The exact spectral norm (`matrix_norm(X, 2)`) would take 97% of the optimizer step on these
shapes; power iteration on $X^\top X$ costs matrix-vector products only, 189x less. Cold, it converges
badly on skew matrices, whose singular values come in equal pairs: on a random $512 \times 512$ skew matrix its
relative error is 21.0% after one iteration, 6.1% after five and 0.5% after twenty.
Warm-started from the previous step's vector, one iteration reaches $2 \cdot 10^{-5}$. Algorithm 1 runs 50 warm
iterations every step; between refreshes $X$ changes little, so Natural Pion Fast runs 2, and 50 only right after a
refresh, when $X$ jumps. Batched over matrices of one size, with $c$ kept on the GPU (no host synchronisation per
matrix), $\sigma$ costs 0.002 s per optimizer step against 0.039 s (Table 7).

### 4.6 Cheaper statistics

In the 6 steps before a refresh every rotated matrix forms two grams, $A$ over its inputs and $D$ over
its output gradients, each over the 65 536 tokens of the step's first micro-batch and in true fp32, so without tensor
cores.

**One $A$ per input.** The query, key and value projections read the same normalised input, and so do the
gate and up projections; $q$, $k$, $v$ share one $A$, formed three times, and gate and up another, formed twice. Natural Pion Fast forms it once per
input tensor. The trajectory is bit-identical (`tests/test_natural_pion.py`).

**bf16 inputs.** Rounding the *assembled* $A$ to bf16 is not usable: during training it pushes the
smallest eigenvalues through zero (Appendix A). Rounding the *inputs* is different: the gram of bf16
vectors is still a gram, so $A$ stays positive semidefinite, and the rounding adds a noise covariance of
relative size about $(2^{-8})^2 \approx 1.5 \cdot 10^{-5}$, far below the floor $10^{-4} \lambda_{\max}$. We round
activations and output gradients to bf16 and multiply them with fp32 accumulation and an fp32 result.

*Table 4. Statistics inputs (seed 0, one $A$ per input).*

| statistics | val, 3 000 steps | val, 15 000 steps | s / step |
|---|---|---|---|
| fp32 inputs | 3.7165 | **[stats_fp32_15k?]** | 0.529 |
| bf16 inputs (Natural Pion Fast) | 3.7162 (-0.000) | 3.4742 | 0.507 |

bf16 inputs save 0.022 s per step.

### 4.7 The Cayley solve by Newton–Schulz

$\mathrm{Cay} = 2(I + \Psi)^{-1} - I$ with $\Psi = \tfrac c2 X$, and $(I + \Psi)^{-1}$ can be computed by Newton–Schulz,
$Z \leftarrow Z(2I - (I + \Psi)Z)$ from $Z_0 = I - \Psi$, with matrix products only. After two iterations the error is
$2(I+\Psi)^{-1}\Psi^8$: $5 \cdot 10^{-9}$ at angle 0.2, $10^{-6}$ at 0.4, $3 \cdot 10^{-4}$ at 0.8 (fp64).

In fp32 this standard form is biased: entries next to 1 lose their low bits, always in the same direction, and
$\|W\|_F^2$, which a rotation keeps, grows by a median $9.5 \cdot 10^{-6}$ per training step (exact solve
$2.6 \cdot 10^{-7}$; `scripts/probes/ns_diag.py`). Carried in $Y = Z - I$ from $Y_0 = -\Psi$ (residual
$E = \Psi + Y + \Psi Y$, $Y \leftarrow Y - E - YE$, rotation $I + 2Y$) nothing is next to 1, with the same number of
products. Over random rotations ($1376 \times 1376$, angle 0.05), $\operatorname{tr}(R^\top R - I)/n$ averages
$+1.7 \cdot 10^{-7}$ for the exact solve (171 standard errors from zero), $+1.4 \cdot 10^{-6}$ for standard
Newton–Schulz and $+6.3 \cdot 10^{-11}$ (0.7 standard errors) in this form
(`scripts/probes/retraction_bias.py`).

*Table 5. Retractions: Natural Pion Fast with each, seed 0. First column: change of $\|W\|_F^2$ over 200 steps of a bare
$1376 \times 1376$ rotation at angle 0.05 (`scripts/probes/ns_rounding.py`). Drift: $\max_i |s_i - s_i^0| / \max_i s_i^0$
after 3 000 training steps, median / worst over the rotated matrices (`scripts/probes/retraction_drift.py`).*

| retraction | $\Delta\|W\|_F^2$, bare matrix | val, 3 000 steps | val, 15 000 steps | drift | s / step |
|---|---|---|---|---|---|
| Cayley, exact solve | $+3.4 \cdot 10^{-5}$ | 3.7166 | | $3.6 \cdot 10^{-4}$ / $4.5 \cdot 10^{-4}$ | 0.542 |
| Cayley, Newton–Schulz | $+2.8 \cdot 10^{-4}$ | 3.7182 | | $4.4 \cdot 10^{-3}$ / $1.7 \cdot 10^{-2}$ | 0.501 |
| Cayley, Newton–Schulz in $Z - I$ (Natural Pion Fast) | $+1.6 \cdot 10^{-7}$ | 3.7162 | 3.4742 | $2.3 \cdot 10^{-6}$ / $6.3 \cdot 10^{-6}$ | 0.507 |
| truncated exponential, degree 2 (Pion's) | | 3.7167 | 3.4754 | $5.8 \cdot 10^{-4}$ / $3.0 \cdot 10^{-3}$ | 0.493 |

### 4.8 Summary

*Table 6. From Algorithm 1 to Natural Pion Fast.*

| step | s / step | val (3 000 steps, seed 0) |
|---|---|---|
| Algorithm 1 | 2.10 | 3.7006 |
| refresh every 25 steps (4.2) | **[tf25_mom_sps?]** | **[tf25_mom?]** |
| + statistics only before a refresh (4.3) | **[win_adlast6_sps?]** | **[win_adlast6?]** |
| + batching by shape (4.4) | 0.641 | 3.7146 |
| **+ batched $\sigma$, one $A$ per input, Newton–Schulz Cayley, bf16 statistics (4.5, 4.6, 4.7)**: Natural Pion Fast | **0.507** | **3.7162** |
| Pion | 0.468 | 3.8412 |

*Table 7. The optimizer step alone, by part (s; synthetic statistics on the shapes of the model, GPU-synchronised
timers; `scripts/probes/optimizer_step_bench.py`).*

| part | after 4.4 | Natural Pion Fast |
|---|---|---|
| statistics $A$, $D$ | 0.060 | 0.018 |
| basis refresh (per step) | 0.021 | 0.021 |
| Cayley | 0.049 | 0.015 |
| $F^{-1}$ | 0.013 | 0.013 |
| $\sigma$ | 0.039 | 0.002 |
| generators, momentum, rotation (as in Pion) | 0.009 | 0.008 |
| **total** | 0.191 | **0.078** |
| Pion, whole optimizer step | | 0.018 |

Part of the optimizer's time overlaps the next step's host-side work, so the difference in training
(0.507 against 0.468 s) is smaller than in this table.

**Algorithm 2: Natural Pion Fast.** It is Algorithm 1 with the bases refreshed every 25 steps; $A$ and
$D$ accumulated (EMA 0.5) only in the 6 steps before a refresh; $\sigma$ by warm power iteration (2
iterations, 50 after a refresh, batched); $F^{-1}$ batched by shape; Cayley by Newton–Schulz in $Z - I$; one $A$ per input
tensor, its inputs and those of $D$ in bf16 with fp32 accumulation. Over four seeds at 3 000 steps it gives
3.7122 against Algorithm 1's **[tf1_mom_mean?]** (**[exact_vs_fast_3000?]**), and at 15 000 steps 3.4742
against **[tf1_mom_15k?]**. A step takes 1.08x Pion's time on our GPU.[^cost] Its peak GPU memory is
43.0 GB against Pion's 42.3 GB (+0.8 GB for $A$, $D$, the bases and the momentum).

[^cost]: Seconds per step are wall-clock time over steps of 3 000-step runs on RTX PRO 6000 (Natural Pion
Fast 0.507 s) and of the 3 840-step Pion run (0.468 s), all evaluating
every 500 steps; Algorithm 1 is the 3 000-step run with momentum (2.10 s). The full-length runs of
Section 5.2, which evaluate every 2 000 steps, give **[v2_full_sps?]** s and 0.465 s, a ratio of
**[v2_full_ratio?]**.

### 4.9 Cost in floating-point operations

Seconds depend on the GPU and on the implementation; operations do not. Table 8 counts them (model:
$6 \times$ tokens $\times$ matrix parameters, plus attention; statistics over the first micro-batch; eigendecompositions
at $9n^3$; `scripts/analysis/flops.py`).

*Table 8. Optimizer FLOPs per step as a share of the model's forward and backward pass.*

| | LLaMA-60M |
|---|---|
| Pion | 1.2% |
| Natural Pion Fast | 10.3% |
| of which statistics / $F^{-1}$ / Cayley / refresh | 5.7% / 1.9% / 1.9% / 0.4% |
| of which generators and rotation (as Pion) | 0.5% |
| step, Natural Pion Fast / Pion | 1.09 |

$F^{-1}$ and Cayley cost $O(n^3)$ per matrix and step against the model's $O(Nn^2)$ for $N$ tokens, so their share
grows with width at a fixed batch. With data parallelism the forward and backward pass divides among the GPUs;
the optimizer's matrices must be divided among them too (one GPU per matrix), or, replicated on every GPU as
Pion's step is, their share grows with the number of GPUs. Statistics are the largest term in FLOPs but run on tensor cores in bf16; eigendecompositions are the
smallest but latency-bound, which is why Tables 7 and 8 rank the parts differently.

---

## 5 Experiments

### 5.1 Setup

| | |
|---|---|
| model | LLaMA-style (Touvron et al., 2023), 60M parameters: hidden 512, 8 layers, 8 heads, SwiGLU FFN 1376 (Shazeer, 2020), RoPE (Su et al., 2024), RMSNorm (Zhang & Sennrich, 2019), vocabulary 32 100 (T5 tokenizer) |
| data | C4 (Raffel et al., 2020), sequence length 256, batch 512 sequences (131 072 tokens per step) |
| budget | 73 242 steps = 9.6B tokens: Pion's 60M budget |
| rotated | the 56 matrices inside the blocks (43.5% of parameters) |
| the rest | AdamW (Loshchilov & Hutter, 2019), lr 8e-3 (the Pion script: 1e-3), betas (0.9, 0.95), weight decay 0.1, cosine; the same for every method |
| precision | bf16 model; statistics from bf16 inputs with fp32 accumulation; eigendecompositions and retractions in fp32 |
| metric | validation loss: token cross-entropy on held-out C4, in nats |
| hardware | one NVIDIA RTX PRO 6000 Blackwell per run |

**Baseline.** Pion's rotation update in its published configuration: Lie momentum 0.9, RMS scale 0.2,
truncated exponential, bilateral update, lr $10^{-3}$ with a cosine schedule; the other parameters use AdamW as
in the table above, the same for both methods. The Pion paper chose lr $10^{-3}$ from
$\{10^{-4}, 5\cdot10^{-4}, 10^{-3}, 5\cdot10^{-3}, 10^{-2}\}$. We checked it in our setting up to 19 200 steps
(Table 9) and did not re-tune it at full length.

*Table 9. Pion learning rate (validation loss, seed 0).*

| Pion steps | 5e-4 | 1e-3 | 2e-3 | 4e-3 |
|---|---|---|---|---|
| 3 840 | 3.8397 | 3.7804 | 3.7668 | 4.1523 |
| 19 200 | - | 3.4960 | 3.5138 | 3.8802 |

All comparisons are within one code base and one data pipeline. In our setting Pion ends at a training loss
of 3.3735 (validation 3.3706); the Pion paper reports 3.3575 in its own Megatron-LM (Shoeybi et al., 2019) setting.

**Natural Pion.** Natural Pion Fast (Algorithm 2), Lie momentum 0.9, cosine schedule on $\theta$ down to $10^{-5}$.
The peak angle $\theta_0$ is 0.2 at 3 000 steps and 0.05 from 9 000 steps on; 0.05 was chosen on 15 000-step runs.

### 5.2 Main result

*Table 10. Equal number of steps (seed 0 unless marked). Gap = val(Pion) - val(Natural Pion).*

| steps | Pion | Natural Pion | gap |
|---|---|---|---|
| 3 000 (mean of 4 seeds) | 3.8394 | 3.7122 | 0.127 |
| 15 000 | 3.5254 | 3.4742 | 0.051 |
| 73 242 (full length) | 3.3706 | **[v2_full?]** | **[v2_gap_full?]** |

At 3 000 steps the smallest per-seed gap is 0.123.

![Figure 1. Validation loss against optimizer steps, full length, seed 0.](figures/preprint/val_vs_steps.png)

### 5.3 Steps to reach Pion's loss

Every run below has its own full cosine schedule ending at its budget $B$, so no loss is read off the middle of
a schedule. For each method the final validation loss is fitted as $L(B) = a + bB^{-\gamma}$ (fitted constants $a$, $b$, $\gamma$) (**[budget_n_pion?]** Pion budgets,
**[budget_n_np?]** Natural Pion budgets, four seeds where available; worst residual **[budget_resid?]**), and the saving at Pion's
loss $L$ is $B_{\mathrm{Pion}} / B_{\mathrm{NP}}(L)$, with $B_{\mathrm{NP}}$ from the fit or from linear interpolation in $\log B$
between Natural Pion's budgets (`scripts/analysis/budget_fit.py`).

*Table 11. Steps saved at Pion's loss (mean over seeds where there are four): fit / interpolation.*

| Pion steps | Pion val | saving |
|---|---|---|
| 5 300 | **[budget_5300_L?]** | **[budget_5300_pow?]** / **[budget_5300_int?]** |
| 15 000 | **[budget_15000_L?]** | **[budget_15000_pow?]** / **[budget_15000_int?]** |
| 19 200 | **[budget_19200_L?]** | **[budget_19200_pow?]** / **[budget_19200_int?]** |
| 30 000 | **[budget_30000_L?]** | **[budget_30000_pow?]** / **[budget_30000_int?]** |
| 45 000 | **[budget_45000_L?]** | **[budget_45000_pow?]** / **[budget_45000_int?]** |
| 73 242 | **[budget_73242_L?]** | **[budget_73242_pow?]** / **[budget_73242_int?]** |

Pion's targets are at its published learning rate $10^{-3}$. At short budgets $2\cdot10^{-3}$ is better (Table 9:
3.7668 against 3.7804 at 3 840 steps), so the savings at the shortest budgets are
measured against a Pion that is not at its best learning rate there.

**Equal wall-clock time.** On one GPU a Natural Pion step costs 1.08x Pion's in the 3 000-step runs
(0.507 s against 0.468 s; **[v2_full_ratio?]**x in the full-length runs), so in the time of Pion's $B$
steps it takes $B/1.08$. These are estimates read off the budget curve above, not separate runs: the gap
to Pion at Pion's full-length wall-clock time is **[eqt_73242_gap?]** (fit; **[eqt_73242_gap_int?]** by
interpolation), at 19 200 steps' time **[eqt_19200_gap?]**, and at 3 840 steps' time **[eqt_3840_gap?]**.

### 5.4 Ablations

All ablations are on Natural Pion Fast at 3 000 steps, seed 0, and change one thing. The reference is the method
itself: angle 0.2, Lie momentum 0.9, cosine schedule (3.7162). Where a row says "best", it is the best
over a grid of that row's step-size parameter; such rows are compared with the method at its own best angle,
**[dir_fisher_mom_lr?]** (**[dir_fisher_mom?]**).

**The direction: natural gradient against the raw generator.** Best angle of each grid. Same normalisation,
retraction and schedule; only $X = F^{-1}G$ against $X = G$ changes.

*Table 12. The direction.*

| | no momentum | Lie momentum 0.9 |
|---|---|---|
| $X = F^{-1}G$ | **[dir_fisher_nomom?]** | **[dir_fisher_mom?]** |
| $X = G$ | **[dir_raw_nomom?]** | **[dir_raw_mom?]** |
| gain from $F^{-1}$ | **[gain_fisher_nomom?]** | **[gain_fisher_mom?]** |

Pion at 3 000 steps: 3.8412.

**The length.** Direction $F^{-1}G$, momentum 0.9, best of each grid; $\theta$ is each rule's grid parameter. The last
column is the spread of the realised rotation angle across the rotated matrices (90th over 10th percentile, median
over the steps after 1 000). $F$ in the quadratic-model and Fisher-norm rules is the Fisher at the current $W$ with
the current statistics; with a fresh basis $\langle G, X\rangle = \langle X, FX\rangle$, so the quadratic-model rule
differs from a fixed $c$ only as far as the basis has gone stale since the last refresh.

*Table 13. The length rule.*

| length rule | val | angle spread |
|---|---|---|
| spectral, after $F^{-1}$: $c = \theta/\sigma(F^{-1}G)$ (the method) | **[dir_fisher_mom?]** | 1 |
| RMS of the induced update, as in Pion | **[len_rms?]** | **[angle_spread_rms?]** |
| quadratic-model ratio, $c = \theta \min(1, \langle G, X\rangle / \langle X, FX\rangle)$ | **[len_quad?]** | **[angle_spread_quad?]** |
| Fisher norm, $c = \theta / \sqrt{\langle X, FX\rangle}$ | **[len_fisher?]** | **[angle_spread_fisher?]** |
| fixed, $c = \theta$ | **[len_fixed?]** | **[angle_spread_fixed?]** |
| spectral, before $F^{-1}$: $c = \theta/\sigma(G)$ | **[len_before?]** | **[angle_spread_before?]** |

RMS rule minus spectral rule over four seeds: **[rms_gap_s0?]**, **[rms_gap_s1?]**, **[rms_gap_s2?]**, **[rms_gap_s3?]**.

**Regularisation of the Fisher.** The default is a relative floor $\varepsilon = 10^{-4}$ (3.7162).

*Table 14. Regularisation of the Fisher.*

| floor $\varepsilon$ (no damping) | $10^{-6}$ | $10^{-5}$ | $10^{-3}$ | $10^{-2}$ |
|---|---|---|---|---|
| val | 3.9148 | 3.7966 | 3.7317 | 3.8568 |

| damping $\mu$ (floor $10^{-6}$; $F + \mu\lambda_{\max}(B)\, BXB$, $B$ = $A$ in, $D$ out) | $10^{-4}$ | $10^{-3}$ | $10^{-2}$ | $10^{-1}$ |
|---|---|---|---|---|
| val | 3.7336 | 3.7206 | 3.7437 | 3.7695 |

The best alternative differs from the default by +0.004.

**Other choices.** Single runs against the reference (3.7162) unless stated.

*Table 15. Other choices.*

| change | val |
|---|---|
| each side normalised by its own $\sigma$, best of angles 0.1 and 0.2 | **[sides_best?]** |
| attention matrices at $\times 0.25 / 0.5 / 2 / 4$ the angle | **[attn_0.25?]** / **[attn_0.5?]** / **[attn_2?]** / **[attn_4?]** |
| constant angle; linear decay; warmup + cosine, best of 3 angles each (cosine, at its best angle: **[dir_fisher_mom?]**) | **[sched_constant?]**; **[sched_linear?]**; **[sched_warmcos?]** |
| warmup-stable-decay, decay over the last 20% / 10% | **[wsd_0.2?]** / **[wsd_0.1?]** |
| true Fisher ($D$ from targets drawn from the model, one more forward and backward pass), best of its angle grid (empirical: **[fisher_empirical?]**) | 3.7063 |
| equal angle on every plane of $X$ (polar step), best of angles 0.1 and 0.2 | **[polar?]** |

The polar step needs an SVD per matrix per step (**[svd_sps_lo?]**–**[svd_sps_hi?]** s per step). Appendix D varies the
initial spectrum.

---

## 6 Related work

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

* **Scale.** One model size (60M) and one data set. The full-length comparison is one seed; Table 10 gives four
  seeds at 3 000 steps.
* **One code base.** Our Pion ends 0.016 above the Pion paper's training loss; we compare within our
  setting only.
* **Exactness in practice.** In fp32 the exact Cayley solve changes the singular values by $+3.6 \cdot 10^{-4}$ (relative) on
  average over 3 000 steps, the Newton–Schulz form of Section 4.7 by $+2.1 \cdot 10^{-6}$.
* **Only Pion as a baseline.** We compare against the published Pion configuration only, not against
  AdamW or Muon, and only on validation loss, not on downstream tasks.
* **The peak angle depends on the horizon.** It is set per training length (Section 5.1), as a learning rate is.
* **The K-FAC assumption.** Along our step, the K-FAC curvature is 0.034 of the exact per-token
  curvature (median of three layers of a trained model); we keep the direction of $F^{-1}G$ but not its length
  (Section 3.3).
* **Cost.** 1.09x Pion's step FLOPs at 60M; the share grows with width and, unless the matrices are
  divided among GPUs, with their number (Section 4.9).

**Future work.** Reinforcement learning from verifiable rewards, where a step costs minutes of
generation and the step efficiency of Section 5.3 is the relevant measure. The natural policy gradient
lineage (Kakade, 2001) makes it the natural test. Larger models, where the optimizer's $O(n^3)$ work has to be
divided across devices (at the shapes of a 1.7B model, one refresh of all matrices takes 58 s on one B200)
and can be cut further: on a rectangular $m \times n$ matrix ($m > n$) the wide side can be solved in $O(m^2 n)$
instead of $O(m^3)$, since rotations outside $W$'s column space do not move it.

---

## References

- Agarwal, N., Anil, R., Hazan, E., Koren, T., Zhang, C. *Disentangling adaptive gradient methods from learning rates.* arXiv:2002.11803, 2020.
- Amari, S. *Natural gradient works efficiently in learning.* Neural Computation 10(2):251–276, 1998.
- Amari, S. *Natural gradient learning for over- and under-complete bases in ICA.* Neural Computation 11(8):1875–1883, 1999.
- Bernstein, J., Newhouse, L. *Old optimizer, new norm: an anthology.* OPT Workshop 2024; arXiv:2409.20325.
- Che, T. *Second-order Muon done right: a principled marriage of spectral geometry and curvature.* arXiv:2608.09763, 2026.
- Gupta, V., Koren, T., Singer, Y. *Shampoo: preconditioned stochastic tensor optimization.* ICML 2018.
- Helmke, U., Moore, J. B. *Optimization and Dynamical Systems.* Springer, 1994.
- Hu, J., Ao, R., So, A. M.-C., Yang, M., Wen, Z. *Riemannian natural gradient methods.* SIAM J. Sci. Comput. 46(1):A204–A231, 2024; arXiv:2207.07287.
- Jordan, K., Jin, Y., Boza, V., You, J., Cesista, F., Newhouse, L., Bernstein, J. *Muon: an optimizer for hidden layers in neural networks.* 2024. https://kellerjordan.github.io/posts/muon/
- Kakade, S. *A natural policy gradient.* NeurIPS 2001.
- Large, T., Liu, Y., Huh, M., Bahng, H., Isola, P., Bernstein, J. *Scalable optimization in the modular norm.* NeurIPS 2024; arXiv:2405.14813.
- Li, J., Li, F., Todorovic, S. *Efficient Riemannian optimization on the Stiefel manifold via the Cayley transform.* ICLR 2020, arXiv:2002.01113.
- Loshchilov, I., Hutter, F. *Decoupled weight decay regularization.* ICLR 2019; arXiv:1711.05101.
- Martens, J., Grosse, R. *Optimizing neural networks with Kronecker-factored approximate curvature.* ICML 2015, arXiv:1503.05671.
- Martin, C. H., Mahoney, M. W. *Implicit self-regularization in deep neural networks: evidence from random matrix theory and implications for learning.* JMLR 22(165):1–73, 2021.
- Qiu, Z., Liu, W., Feng, H., Xue, Y., Feng, Y., Liu, Z., Zhang, D., Weller, A., Schölkopf, B. *Controlling text-to-image diffusion by orthogonal finetuning.* NeurIPS 2023; arXiv:2306.07280.
- Qiu, Z., Buchholz, S., Xiao, T. Z., Dax, M., Schölkopf, B., Liu, W. *Reparameterized LLM training via orthogonal equivalence transformation (POET).* NeurIPS 2025; arXiv:2506.08001.
- Qiu, Z., Liu, L., Weller, A., Shi, H., Liu, W. *POET-X: memory-efficient LLM training by scaling orthogonal transformation.* ICML 2026; arXiv:2603.05500.
- Raffel, C., et al. *Exploring the limits of transfer learning with a unified text-to-text transformer (C4, T5).* JMLR 2020.
- Shazeer, N. *GLU variants improve Transformer.* arXiv:2002.05202, 2020.
- Shi, K., Li, H., Qiu, Z., Wen, Y., Buchholz, S., Liu, W. *Pion: a spectrum-preserving optimizer via orthogonal equivalence transformation.* arXiv:2605.12492, 2026.
- Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., Catanzaro, B. *Megatron-LM: training multi-billion parameter language models using model parallelism.* arXiv:1909.08053, 2019.
- Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., Liu, Y. *RoFormer: enhanced Transformer with rotary position embedding.* Neurocomputing 568:127063, 2024; arXiv:2104.09864.
- Touvron, H., Lavril, T., Izacard, G., et al. *LLaMA: open and efficient foundation language models.* arXiv:2302.13971, 2023.
- Wen, Z., Yin, W. *A feasible method for optimization with orthogonality constraints.* Mathematical Programming 142:397–434, 2013.
- Xu, C., Yan, W., Zhang, Y.-J. A. *FISMO: Fisher-structured momentum-orthogonalized optimizer.* arXiv:2601.21750, 2026.
- Zhang, B., Sennrich, R. *Root mean square layer normalization.* NeurIPS 2019; arXiv:1910.07467.
- Zhang, Y., Xing, S., Huang, J., Lv, K., Zhou, Y., Qiu, X., Guo, Q., Chen, K. *Mousse: rectifying the geometry of Muon with curvature-aware preconditioning.* arXiv:2603.09697, 2026.
- Zhu, H., Cong, W., Sha, Z., et al. *ISO: an RLVR-native optimization stack.* arXiv:2607.19331, 2026.

---

## Appendix A: implementation details

* **Statistics.** $A$ is fed by a forward pre-hook on each rotated linear layer and $D$ by a hook on its
  output gradient, from one micro-batch per step. Both accumulate in fp32 with autocast disabled: a
  bf16 product that returns a bf16 matrix pushes the near-zero eigenvalues of $A$ below zero (measured
  at step 499 of a training run: thousands of negative eigenvalues over the 56 matrices). Rounding only the
  inputs (Section 4.6) does not. The $1/N$ that autograd puts on a
  mean loss is undone by scaling $D$ by $N^2$; Algorithm 1 is invariant to it anyway.
* **Code.** `ngd_pion/natural_pion.py`: `NaturalPion` is Algorithm 1, `NaturalPionFast` Algorithm 2, the method
  of every experiment; every job and run record is listed in
  `scripts/paper/README.md` and `runs-record/`. Every measured number in this paper is computed from the run
  records by `scripts/analysis/paper_numbers.py` (the list, with sources, is `docs/paper/numbers.md`).

## Appendix B: why the output-gradient statistic is needed

If $D$ is taken to be the identity, $S = W^\top W$ and the step's scale per layer goes as
$\|\delta\| / (\|W\|\,\|x\|)$ instead of $1 / (\|W\|\,\|\delta\|\,\|x\|)$. The step is then largest
where the backward signal is strongest, the opposite of what curvature prescribes.

## Appendix C: hyperparameters

| | Natural Pion | Pion |
|---|---|---|
| rotated-matrix step | peak angle $\theta_0$ 0.2 (3k steps), 0.05 ($\ge$ 9k), cosine to $10^{-5}$ | lr $10^{-3}$, cosine to $10^{-5}$, RMS 0.2 |
| momentum | Lie, 0.9 | Lie, 0.9 |
| curvature | $T_{\mathrm{fac}}$ 25; $A$, $D$ EMA 0.5 over the 6 steps before a refresh; floor $10^{-4}$ | - |
| retraction | Cayley by Newton–Schulz in $Z - I$ | truncated exponential, degree 2 |
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
| $\mathcal N(0, 0.02^2)$ | 3.7162 | |
| tail, $\alpha$ = 2 / 2.5 / 3 / 4 / 5 | **[init_2?]** / **[init_2.5?]** / **[init_3?]** / **[init_4?]** / **[init_5?]** | best **[init_best_d?]** |
| flat, all singular values $\sqrt{m/n}$ | **[init_flat?]** | **[init_flat_d?]** |
| orthogonal, all singular values 1 | **[init_orth?]** | **[init_orth_d?]** |

*Gain from $\alpha = 3$: val with $\mathcal N(0, 0.02^2)$ minus val with $\alpha = 3$; positive means $\alpha = 3$ helps.*

| | 3 000 steps (4 seeds) | 15 000 steps | full length |
|---|---|---|---|
| Natural Pion | **[init_gain_3000_mean?]** | **[init_gain_15k?]** | **[init_gain_full?]** |
| Pion | -0.006 (seed 0) | -0.019 | -0.007 |

Every other experiment in the paper uses $\mathcal N(0, 0.02^2)$.
