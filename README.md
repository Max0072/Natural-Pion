# NGD-Pion

A curvature-preconditioned variant of [Pion](https://arxiv.org/abs/2605.12492).
Pion updates a weight matrix by orthogonal rotations on both sides, which
leaves its singular values untouched for the whole of training; the rotation
itself is driven by the raw gradient. NGD-Pion preconditions that rotation by
the Fisher operator on the bivector tangent space instead.

**[`ALGORITHM.md`](ALGORITHM.md) is the specification.** It states every design
decision together with the measurement behind it, and each module implements
one of its sections. Read it before changing anything here.

## Layout

| | |
|---|---|
| `ngd_pion/reference.py` | numpy transcription of the spec -- the oracle, not for training |
| `ngd_pion/linalg.py` | `skew`, the spectral floor, Cayley |
| `ngd_pion/covariance.py` | §3, the input covariance |
| `ngd_pion/factorization.py` | §4, the bases |
| `ngd_pion/direction.py` | §1, §5, §6, generators through trust region |
| `ngd_pion/optimizer.py` | §7, orchestration and retraction |
| `ngd_pion/pion_baseline.py` | vanilla Pion, with the switches an ablation needs |
| `harness/` | LLaMA-60M in Pion's configuration, data, training loop |

## Use

```python
from ngd_pion import NGDPion, attach

linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
opt = NGDPion([m.weight for m in linears], lr=1e-3)
recorder = attach(linears, opt)          # feeds the input covariance
```

The optimizer takes parameters rather than modules, and the covariance is
supplied from outside. `attach` covers `nn.Linear`; anything else -- Megatron's
parallel linears, a fused QKV projection -- writes its own adapter of the same
size without this package changing.

Give it 2-D weights only. Embeddings, the output head, norm gains and biases
belong to another optimizer, which is how Pion splits parameters too.

## Running an experiment

```bash
python scripts/prepare_data.py --out data --target-tokens 5e9
python -m harness.run --optimizer ngd --lr 1e-3 --seed 0
```

`RunConfig`'s hash names the output directory and covers every field that can
change a result, so a sweep is a job array over flags.

## Comparison design

The point of measurement is `pion_ablated` against `ngd`: identical but for
`F^-1`. Published Pion runs alongside as context, so the ablated baseline
cannot be called a straw man.

Ablating Pion's RMS scaling forces an exact retraction. Their degree-2
truncated exponential satisfies `R^T R = I + A^4/4`, so it inflates every step
and the scaling is what holds the rotation angle small enough for that to stay
negligible; switch the scaling off and it diverges within tens of steps. Cayley
is exactly orthogonal at any angle, which makes it a precondition of the
ablation rather than a preference.

## Tests

```bash
pytest -q
```

The torch path is correct insofar as it reproduces `reference.py`, and the
tests pin that. They also pin the findings that are easy to regress: that the
spectral floor's lower bound is set by the working precision, that the step is
invariant to what `eigh` happens to return, and that the truncated exponential
diverges unscaled.
