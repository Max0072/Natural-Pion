# ADR 0011 -- delete `pion_ablated` from the code?

- **Status:** Accepted
- **Date:** 2026-09-22
- **Decided by:** the user

## Context

ADR 0008 withdraws `pion_ablated` from the plan. The code is untouched. It is
referenced from `harness/config.py`, `harness/train.py`, `tests/test_harness.py`,
`tests/test_pion_baseline.py`, `tests/test_shampoo.py`, `ngd_pion/direction.py`,
`ngd_pion/shampoo.py` (where `power = 0` is described as exactly that arm) and
six sbatch scripts that produced the 150-step sweeps still on disk
(`runs/ablated/`).

## Options considered

1. **Delete it** from config, harness, tests and the `shampoo.py` control. Cost:
   the old sbatch scripts stop being runnable and their run directories lose the
   code that made them; a few tests that pin its behaviour go too.
2. **Keep it, mark it** (done: `ngd_pion/README.md` says "not a baseline"). Cost: a
   third optimizer stays reachable by name.

## Decision

Option 1: delete it. Done 2026-09-22 in the same commit as this record.

Removed: the `pion_ablated` branch of `build_optimizers` (the `pion` branch now
takes every setting from `RunConfig` and nothing is substituted), the name from
the `RunConfig` comment and the error message, and the tests that wired and ran
it. Added: `test_pion_is_wired_as_published` and `test_pion_ablated_is_gone`
(asking for the old name raises `unknown optimizer`).

**Kept, on purpose:** the `Pion` class and its `scaling="none"`,
`momentum="none"` and Cayley switches, because the class API and two tests that
pin findings need them (the truncated exponential diverging once scaling is off;
Shampoo's `power = 0` limit, renamed `test_power_zero_is_the_raw_generator_step`).
No `RunConfig` field was added or removed, so the configuration hash and the
directory name of every `pion` run are unchanged.

## Consequences

`scripts/sbatch/sweep.sbatch` (which lists `pion_ablated`, and a name `ngd` that
was never registered) is marked stale and must not be submitted. The other five
old scripts mention the arm only in comments about void results and are left as
records. `runs/ablated/` and its five 150-step runs stay on disk; their numbers are
void and their manifests record the opposite of what ran. Test count 262 -> 261.
