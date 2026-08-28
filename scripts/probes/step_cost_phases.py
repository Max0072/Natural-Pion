"""Where inside `opt.step()` the 240 ms goes.

`step_cost_with_s.py` splits the optimizer's work three ways -- `opt.step()`
240 ms, `A` accumulation 106 ms, `D` accumulation 151 ms -- and that is enough
to say the statistics cost more than the step. It is not enough to decide what
to optimise *inside* the step, and two of the candidates hinge on it:

* raising `T_fac`, which only pays if the factorisation is a real share.
  `alpha` sits at 0.98-1.00 across a whole full-length run, so the basis does
  not go stale in 25 steps and `T_fac` is over-conservative -- but if `eigh` is
  5% of the step there is nothing there to win.
* replacing the Cayley solve with Newton-Schulz, which only pays if the solve
  is a real share. `linalg.cayley_newton_schulz` exists and was measured and
  dropped once already.

So this times the phases separately, on the real LLaMA-60M shapes, with the
device drained before the clock stops. The refactorisation is reported both raw
and divided by `T_fac`, since that is what a step actually pays.

    python scripts/probes/step_cost_phases.py
"""

from __future__ import annotations

import torch

from ngd_pion.direction import fisher_apply, generators, natural_gradient
from ngd_pion.linalg import cayley, cayley_newton_schulz, spectral_norm
from ngd_pion.with_s_fast import FastNGDPionS
from step_cost_with_s import SHAPES, build, timed

T_FAC = 25


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("needs a GPU: the point is the cost on the card the runs use")
    print(f"  {torch.cuda.get_device_name(0)}, 56 weights, T_fac = {T_FAC}\n")

    torch.manual_seed(0)
    opt, params, mats, by_shape = build(FastNGDPionS, device, beta_backward=0.5)
    group = opt.param_groups[0]
    opt.step()                                    # force the first factorisation

    # --- refactorisation, the whole set at once, as `_refactor` is called ----
    refac = timed(lambda: opt._refactor(params, group), warmup=1, iters=3)

    # --- the per-step phases, timed per shape group and weighted by count ----
    phases = {k: 0.0 for k in
              ("generators", "natural_gradient", "quad+curv", "angle", "cayley", "cayley (NS)")}
    for (m, n), count in SHAPES:
        p = params[by_shape[(m, n)]]
        st = opt.state[p]
        W, G = p.detach().float(), p.grad.detach().float()
        bi, bo = st["bases"]
        A = st["cov"].matrix.float()
        D = st["cov_backward"].matrix.float()
        G_in, G_out = generators(W, G)
        X_in = natural_gradient(G_in, bi)
        X_out = natural_gradient(G_out, bo)
        Wt = W.T

        def add(key, fn, iters=10):
            phases[key] += count * timed(fn, iters=iters)

        add("generators", lambda: generators(W, G))
        add("natural_gradient",
            lambda: (natural_gradient(G_in, bi), natural_gradient(G_out, bo)))
        add("quad+curv", lambda: (
            (G_in * X_in).sum() + (G_out * X_out).sum(),
            (X_in * fisher_apply(A, Wt @ D @ W, X_in)).sum()
            + (X_out * fisher_apply(D, W @ A @ Wt, X_out)).sum()))
        add("angle", lambda: (spectral_norm(X_in, 2, st.get("angle_v_in")),
                              spectral_norm(X_out, 2, st.get("angle_v_out"))))
        add("cayley", lambda: cayley(X_out, 0.01) @ W @ cayley(X_in, 0.01))
        add("cayley (NS)",
            lambda: cayley_newton_schulz(X_out, 0.01, 2) @ W @ cayley_newton_schulz(X_in, 0.01, 2))

    per_step = dict(phases)
    per_step["refactor / T_fac"] = refac / T_FAC
    order = ["refactor / T_fac", "generators", "natural_gradient", "quad+curv", "angle", "cayley"]
    total = sum(per_step[k] for k in order)

    print(f"  {'phase':<22}{'per step':>12}{'share':>10}")
    print("  " + "-" * 44)
    for k in order:
        print(f"  {k:<22}{per_step[k]*1000:11.1f}ms{100*per_step[k]/total:9.1f}%")
    print("  " + "-" * 44)
    print(f"  {'sum of phases':<22}{total*1000:11.1f}ms")
    print(f"\n  refactorisation raw: {refac*1000:.1f} ms every {T_FAC} steps")
    print(f"  Newton-Schulz retraction would be {phases['cayley (NS)']*1000:.1f} ms "
          f"against Cayley's {phases['cayley']*1000:.1f} ms "
          f"({phases['cayley (NS)']/phases['cayley']:.2f}x)")
    print("\n  For reference, step_cost_with_s.py: opt.step() 240 ms, "
          "A accum 106 ms, D accum 151 ms.")
    print("  Phases timed in isolation do not sum to a launched step -- kernel launch")
    print("  overhead and the loop over 56 weights are not attributed here -- so read")
    print("  the shares, not the total.")


if __name__ == "__main__":
    main()
