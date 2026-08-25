# Start here

A single page for picking this up cold — after a lost session, a killed
terminal, or three weeks away. `AGENTS.md` is the settled state of play,
`docs/JOURNAL.md` is the sequence that produced it (newest entry last, and long
by now), `docs/CLUSTER.md` is the machine. This file is only ever *now*: what is
in flight, what to look at first, and what was already decided so it is not
re-litigated.

**Rewrite this file rather than appending to it.** It is a snapshot, not a log.

---

## As of 2026-08-25 16:40

### In flight

Two anchor runs, the **third** attempt, on `rtx6001`, code `cf73187`:

| job | side | run directory | target |
|---|---|---|---|
| 246482 | bilateral | `$DATA_p330/runs/rtx/pion-lr0.001-s0-76ecc39a71` | 3.3575 |
| 246483 | alternate | `$DATA_p330/runs/rtx/pion-lr0.001-s0-21d352699d` | 3.3654 |

About 9 hours left from 16:40, so expect them around **01:45 on the 26th**. The
24 h limit is comfortable. Each writes `anchor.json` on completion; that file
appearing *is* the finish signal.

    squeue -u $USER
    cat $DATA_p330/runs/rtx/pion-lr0.001-s0-*/anchor.json

They went to `rtx` rather than `b200` because a job took all eight B200s for
24 h minutes before submission; `rtx` was idle and the partition does not change
the result (round 1 gave 3.3997 on both pools, to four decimals).

### Read the verdict against criteria fixed *before* the numbers

Do not move these after seeing the result. That is what they are for.

1. **Gap.** `alternate - bilateral` within **0.005** of their **0.0079**.
2. **Level.** Both sides `matched: true`, i.e. `|delta| <= 0.02`.
3. Bilateral is readable against **3.4021** (round 2) and **3.3997** (round 1)
   to see what this round's two fixes did on their own.

History, so the trend is visible:

| round | code | bilateral | alternate | gap | vs their 0.0079 |
|---|---|---|---|---|---|
| 1 | `7c9783fe` / `3bade81` | 3.3997 | 3.4352 / 3.4369 | +0.0355 | 4.5x |
| 2 | `6e738bf` | 3.4021 | 3.4161 | +0.0140 | 1.8x |
| 3 | `cf73187` | — | — | — | — |

### What was fixed in each round, so a miss is diagnosable

**Round 2 — three differences in our Pion, all read off their source, not guessed.**
Their `_scale_update_matrix_rms` normalises the side it is *about to apply*;
ours always normalised both, which under `alternate` calibrated an RMS target
against a step twice the size of the one taken. Their second moment is off by
default; ours was hard-wired on. Their Q is rotated per head; ours was rotated
whole. Only the first moved anything — the gap — which is the signature it was
predicted to have.

**Round 3 — two differences in the half of the model Pion does not own.**
The embedding, head and norm gains are 32.9M of 58.2M parameters and had never
been audited. `build_optimizers` passed no betas to AdamW and took torch's
`(0.9, 0.999)` where their script sets `--adam-beta2 0.95`; and it decayed every
parameter at 0.1 where Megatron exempts every 1-D parameter and bias, which over
this schedule shrinks an RMSNorm gain by **40x**. That bites harder under Pion
than elsewhere: a network normally absorbs decayed gains by growing the linear
weights, and a spectrum-preserving optimizer forbids exactly that.

## If it misses again — decided in advance, do not improvise

Three rounds of "found a difference, fixed it, still missing" have each cost
~10 GPU-hours and a day. A fourth guess has the same expected value. So:

**Do not hunt for a fifth difference. Run the control.**

Take their repository at `$DATA_p330/reference/pion` **as it is** — their fork
of Megatron-LM with their Pion, nothing of ours added — and run it on **our
tokens**, on our cluster. Two outcomes, both terminal:

- it reaches 3.3575 → the fault is our harness, and we now have a reference to
  diff against step by step rather than by final loss;
- it lands near 3.40 like ours → the offset is the data or the machine, our
  harness is vindicated, and the anchor is passed with a documented offset.

`docs/CLUSTER.md`'s "not on the critical path" note defers *integrating NGD-Pion
into Megatron* — parallel linears, fused QKV, an all-reduced covariance. The
control needs none of that: their code is already written.

### What is already known about the control

- Their script wants **2 GPUs**, `TP=1`, `PP=1`, plain data parallel. `b201` has
  eight.
- Their Megatron reads a `.bin`/`.idx` pair, not our flat array — but
  **`$DATA_p330/c4/c4_train.bin` already *is* their `.bin`**: the same `uint16`
  stream of documents laid end to end. Only the `.idx` has to be built, from EOS
  positions. That gives the control byte-identical tokens and removes the data
  as a variable by construction, with no re-tokenisation.
- `megatron/core/datasets/indexed_dataset.py` has `_IndexWriter(idx_path, dtype)`
  and `DType` supports `uint16`. Their fork's `tools/` is stripped, so there is
  no `preprocess_data.py` to lean on.
- **Untested:** whether their code imports in our container at all. It needs
  `transformer_engine`, and the one check ran on the login node without `--nv`
  and failed on `libcuda.so.1`, which proves nothing. A ten-minute `srun` on one
  card settles it. Do that before building any data.

### The fallback if the control does not separate them either

State the result narrowly and move on, rather than spending another week:
the harness reproduces the published *relative* structure to within X, the
absolute level sits ~0.045 high as a common offset, and the comparison this
project reports is internal — same harness, same data, same seed — where a
common offset cancels. The quantity that bounds it is our own run-to-run
reproducibility, **measured at 0.002 across two GPU architectures**.

## Open, and not blocking

- **The residual arm asymmetry, 0.0061.** Three times the machine-to-machine
  spread, so real. Not explained by anything found so far; the round-3 fixes act
  on both arms alike and cannot touch it.
- **A fixed single-side update** is absent from their paper and their scripts —
  `in`/`out` exist only as the internals of `alternate`. A colleague is on this;
  see commit `c29b5f9`, "Neither rotation side is redundant, and they do not
  overlap".
- **The RMS normaliser couples update scheme to Lie-algebra step size.**
  Measured on a trained checkpoint: `c_alt/c_bi` is 2.11 on the in-side and 1.64
  on the out-side, so `alternate` rotates the in-side 5% *more* and the out-side
  18% *less*, for 0.89 of bilateral's total rotation. Their "alternating retains
  most of the benefit" is then arithmetic — half as many rotations, each ~1.8x
  larger — rather than a fact about geometry. **Explicitly parked**, not
  interesting enough to pursue.
- **NGD-Pion throughput.** 1,788 tokens/s against Pion's 283,000 — 158x slower,
  which would put a 9.6B run at 62 days. Suspect: `optimizer.py:236` takes two
  full SVDs per weight per step for a diagnostic that is logged every 50. Parked
  until the anchor lands, but it blocks step 6 and is not small.

## Where things are

| | |
|---|---|
| repository | `$DATA_p330/Natural-Pion`, branch `main`, push straight to it |
| their code | `$DATA_p330/reference/pion` |
| corpus | `$DATA_p330/c4/c4_{train,val}.bin`, 10.0B tokens, 64 of 1024 shards |
| runs | `$DATA_p330/runs/{,b200/,rtx/}` |
| retired | `$DATA_p330/attic/<date>/`, with `register.tsv` |
| container | `$DATA_p330/containers/ngd-pion.sif` |

`pytest -q` in the container: 140 pass, 1 skipped by design, ~24 s.

## Habits that cost something to relearn

- **Commit journal entries as they are written.** `docs/JOURNAL.md` is tracked,
  so an uncommitted edit marks the tree dirty, and a run started mid-edit
  inherits a `-dirty` stamp that is technically true and practically misleading.
- **Everything goes to `main`.** This is the working repository, not the
  published one; no topic branches.
- **The repository is bind-mounted live into jobs.** A job imports whatever the
  working tree holds *at the moment it starts*, so editing `harness/` or
  `ngd_pion/` while something is queued changes what it will run. Editing
  `docs/` is always safe, and SLURM copies the batch script at submission so
  `scripts/sbatch/` is safe too.
- **Measure the old path before believing the new one is slow.** The sampler
  looked like a 10 s/batch regression on the login node; the *old* sampler
  measured the same there. It was cold NFS under an 8 GB cgroup, not a
  regression.
- **Read their source instead of inferring from their paper.** Every difference
  that actually moved a number was found by reading `pion.py`, and every one
  that was inferred from prose turned out to be either wrong or not a difference
  at all.
