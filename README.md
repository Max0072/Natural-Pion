# NGD-Pion

A curvature-preconditioned variant of [Pion](https://arxiv.org/abs/2605.12492).

Pion updates a weight matrix by orthogonal rotations on both sides, which leaves
its singular values untouched for the whole of training, and drives the rotation
with the raw gradient. NGD-Pion preconditions that rotation by the Fisher
operator on the bivector tangent space instead.

The idea in one line: **take the covariance of the gradient on the Lie algebra,
and it turns out to be expressible through the covariance of the activations.**
A general covariance operator on `so(512)` would carry about `8.6e9` free
numbers; this one carries `131,328`, because the generator is a bivector and the
covariance of a bivector factorises.

## The algorithm

**Pion's geometry.** A weight matrix is updated by rotating both of its spaces:

```
W  <-  R_out · W · R_in
```

Both rotations are orthogonal, so the singular values of `W` never move — the
spectrum you initialise with is the spectrum you finish with. What is learned is
the rotation, and rotations near the identity are parameterised by
skew-symmetric matrices: the Lie algebra `so(n)`.

**The gradient there.** Differentiate the loss along
`W(t) = exp(t·Ω_out) W exp(t·Ω_in)`. For any skew `Ω`,

```
<G, W·Ω_in>  = ½ <G_in,  Ω_in>       G_in  = Wᵀ G − Gᵀ W
<G, Ω_out·W> = ½ <G_out, Ω_out>      G_out = G Wᵀ − W Gᵀ
```

So `G_in` and `G_out` *are* the Riemannian gradient with respect to the
rotation, doubled. Pion steps along them directly.

**The idea.** Step along the *natural* gradient instead — `F⁻¹G`, where `F` is
the covariance of the gradient on the algebra. Steepest descent in the metric
the noise actually has, rather than the Euclidean one.

**Why that is even possible.** For a single sample `G = δxᵀ` the generator is

```
G_in = (Wᵀδ) ∧ x = δ'xᵀ − xδ'ᵀ
```

a **bivector** — an antisymmetrised outer product of two vectors. The covariance
of such an object, when the two vectors are independent, is fixed entirely by
their two covariances. Writing `A = E[xxᵀ]` and `S' = Wᵀ E[δδᵀ] W`, the
covariance as an operator on skew `X` is

```
F(X) = 2 ( A X S'  +  S' X A )
```

That is the whole content of the method. A general covariance operator on
`so(512)` would carry about `8.6·10⁹` free numbers; this one carries `131,328`,
and not by luck — it follows from the generator being built out of two vectors.

**Taking `S = I`.** The backward covariance is assumed isotropic. The out-side
operator then becomes `2(XA' + A'X)` with `A' = W A Wᵀ`, a plain symmetric
eigenproblem, and `A = E[xxᵀ]` is the only statistic the method keeps —
no backward hook has to exist.

**Inverting `F`.** It is *not* the map `X ↦ AXS'`, which would invert as
`A⁻¹GS'⁻¹`; the sum of the two orderings does not factor unless `A` and `S'`
commute, and they do not. But `A` is positive definite, so there is a congruence
`P` with

```
Pᵀ A P = I        Pᵀ S' P = diag(λ)
```

and in those coordinates the operator is elementwise. Substituting `X = P Y Pᵀ`
turns `F(X) = G` into

```
2 (λᵢ + λⱼ) · Yᵢⱼ = (Pᵀ G P)ᵢⱼ
```

Divide, transform back. One eigendecomposition per side, reused for `T_fac`
steps. For a square or tall `W` under an orthogonal initialisation the
congruence is unnecessary and `λ` is simply the spectrum of `A`.

**The step.**

```
A   <- β·A + (1−β)·xᵀx/n                    forward activations only
G_in, G_out                                 the generators above
X   =  P [ (Pᵀ G P) / (2(λᵢ+λⱼ)) ] Pᵀ       both sides
α   =  min(1, quad/curv)                    reads out basis staleness
W   <- Cayley(−η·α·X_out) · W · Cayley(−η·α·X_in)
```

with `Cayley(−cX) = (I + c/2·X)⁻¹ (I − c/2·X)`, which is *exactly* orthogonal
for skew `X` at any step size. That is why the spectrum is preserved exactly
here and only approximately in Pion, whose truncated exponential satisfies
`RᵀR = I + A⁴/4`.

**The approximations, named.** One is `x ⊥ δ'`, the K-FAC independence
assumption, which is not true in a deep network.

The other is `S = I`, and it is less of a concession than it looks. The
measured `S` is strongly anisotropic — no simple model fits it — but *using* it
makes the step worse: on a toy transformer the direction built with `S = I`
reduced held-out loss **1.4-1.9x more** than the one built with the measured
`S`. A covariance estimated from a finite sample is too noisy to invert, and
isotropising it acts as regularisation rather than simplification. Whether that
survives the far larger samples of a real run is **untested**; see
[`AGENTS.md`](AGENTS.md).

Everything downstream of the two is exact algebra, not a further approximation.

**Regularisation is one knob.** Every spectrum that reaches a denominator is
floored at `max(λ, ε·λ_max)`, and nothing else is damped anywhere.

## Where to look

| | |
|---|---|
| **[`ALGORITHM.md`](ALGORITHM.md)** | the specification — every decision with the measurement behind it |
| **[`AGENTS.md`](AGENTS.md)** | state of play, decisions not to reopen, traps already hit |
| **[`docs/CLUSTER.md`](docs/CLUSTER.md)** | the cluster sequence, in order |

Read `ALGORITHM.md` before changing anything. Each module implements one of its
sections and names it in the docstring.

## Layout

```
ngd_pion/
  reference.py       numpy transcription of the spec — the oracle, not for training
  linalg.py          skew, the spectral floor, Cayley
  covariance.py      §3  the input covariance
  factorization.py   §4  the bases
  direction.py       §1 §5 §6  generators through trust region
  optimizer.py       §7  orchestration and retraction
  pion_baseline.py   vanilla Pion, with the switches an ablation needs
harness/
  model.py           LLaMA-60M in their configuration
  data.py            memmapped token corpus
  config.py          run configuration; its hash names the run
  train.py           the loop
  instrument.py      per-layer diagnostics
  anchor.py          reproduce their published figure
```

## Use

```python
from ngd_pion import NGDPion, attach

linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
opt = NGDPion([m.weight for m in linears], lr=1e-3)
recorder = attach(linears, opt)          # feeds the input covariance
```

The optimizer takes parameters rather than modules, and the covariance is
supplied from outside. `attach` covers `nn.Linear`; anything else — Megatron's
parallel linears, a fused QKV projection — writes its own adapter of the same
size without this package changing.

Give it 2-D weights only. Embeddings, the output head, norm gains and biases
belong to another optimizer, which is how Pion splits parameters too.

## Running

```bash
pytest -q                                        # 117 tests, seconds
python scripts/prepare_data.py --out data --target-tokens 1e10
python -m harness.run --optimizer ngd --lr 1e-3
python -m harness.run --anchor bilateral         # calibration; read AGENTS.md first
```

`RunConfig`'s hash names the output directory and covers every field that can
change a result, so a sweep is a job array over flags.

## Comparison design

The measurement is `pion_ablated` against `ngd`: identical but for `F^-1`.
Published Pion runs alongside as context, so the ablated baseline cannot be
called a straw man.

Ablating Pion's RMS scaling **forces** an exact retraction. Their degree-2
truncated exponential satisfies `R^T R = I + A^4/4`, so it inflates every step;
the scaling is what holds the rotation angle small enough for that to stay
negligible. Switch the scaling off and it diverges within tens of steps. Cayley
is exactly orthogonal at any angle, which makes it a precondition of the
ablation rather than a preference — and removes the confound instead of adding
one.

## Status

The mathematics is verified against independent routes: the Fisher operator
against Monte Carlo, the closed-form solve against an explicit Kronecker
system, the descent lemma, the sign, Cayley's exactness, spectrum preservation.

**Nothing has been trained at scale.** The only evidence the method helps is a
toy least-squares with an exactly reachable target, where natural gradient wins
almost tautologically. It was a kill criterion, not a result.

## Tests

The torch path is correct insofar as it reproduces `reference.py`, and the tests
pin that. Several also pin findings that would otherwise silently regress — that
the spectral floor's lower bound is set by the working precision, that the step
is invariant to whatever `eigh` happens to return, that the truncated
exponential diverges unscaled. Their docstrings say which.
