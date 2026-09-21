# ADR 0011 -- delete `pion_ablated` from the code?

- **Status:** Proposed (waiting on the user)
- **Date:** 2026-09-22
- **Decided by:** --

## Context

ADR 0008 withdraws `pion_ablated` from the plan. The code is untouched. It is
referenced from `harness/config.py`, `harness/train.py`, `tests/test_harness.py`,
`tests/test_pion_baseline.py`, `tests/test_shampoo.py`, `ngd_pion/direction.py`,
`ngd_pion/shampoo.py` (where `power = 0` is described as exactly that arm) and
six sbatch scripts that produced the 150-step sweeps still on disk
(`runs/ablated/`).

## Options

1. **Delete it** from config, harness, tests and the `shampoo.py` control. Cost:
   the old sbatch scripts stop being runnable and their run directories lose the
   code that made them; a few tests that pin its behaviour go too.
2. **Keep it, mark it** (done: `ngd_pion/README.md` says "not a baseline"). Cost: a
   third optimizer stays reachable by name.

## Decision

Not taken.
