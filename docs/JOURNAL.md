# Journal

A running record of what was decided, what was submitted, and what came back.
`AGENTS.md` holds the settled state of play; this file holds the sequence that
produced it, so that an interrupted session can be picked up from here. Newest
entry last. Every entry names job IDs, because the queue is the only thing that
remembers what was actually run.

---

## 2026-08-24, evening — the two anchor pairs

### What is in flight

| job | partition | what | state |
|---|---|---|---|
| 241310 | rtx | `train.sbatch --anchor bilateral --micro-batch 512` | RUNNING on rtx6004 since 19:52 |
| 241311 | rtx | `train.sbatch --anchor alternate --micro-batch 512` | RUNNING on rtx6004 since 19:52 |
| 241338 | b200 | same, `--out-dir $DATA_p330/runs/b200` | PENDING |
| 241339 | b200 | same, alternate | PENDING |

The b200 pair duplicates the rtx pair on the other partition. Different
`--out-dir`, so the two never share a run directory even though the
configuration hash is identical — `out_dir` is excluded from the hash, and
`train.py` creates the directory itself, so nothing has to be made by hand.

### The rtx pair is healthy

At 20:20, both at step 3450 of 73242, 283k tokens/s by the window counter,
peak 83.31 GB of 97.9, checkpoint written every 500 steps (~4 min) at eval
time. Which run is which:

| job | side | run directory | train loss @3450 |
|---|---|---|---|
| 241310 | bilateral (`pion_alternate: false`) | `pion-lr0.001-s0-8aff0f95a9` | 3.9389 |
| 241311 | alternate (`pion_alternate: true`) | `pion-lr0.001-s0-51458a70e2` | 3.9586 |

Bilateral is ahead by 0.0197, the same direction as their published 0.0079 gap.
That is not evidence of anything at 4.7% of the run — the gap is a converged
quantity and `anchor.check` refuses to judge an incomplete run — but it would
have been worth noticing if the sign had come out backwards.

**Projected wall clock: 9.8 h**, from 2600 steps in 1254 s. That agrees with
the corrected throughput probe (9.9 h) and confirms the withdrawn 17.8 h figure
was the instrument, not the machine. Both finish around 06:00 on the 25th,
well inside the 24 h cap, so no requeue is expected.

### Why the b200 pair has not started

Not a defect in the submission. The b200 partition is two nodes and neither is
available:

- **b201** — all 8 GPUs allocated to `241244` (another user, 10 h limit,
  3.5 h elapsed at 20:15). Frees around 02:37 on the 25th.
- **b202** — inside the open-ended `migration` reservation
  (`b202,cn14,gpu06,ne[35,53],rtx6006`, until 2027), so it is not schedulable
  for us at all regardless of the two long-running jobs on it.

So b200 is effectively a single node for this project. Slurm's projection has
241338 starting 03:18 and 241339 only at 19:18 the following day — i.e. it
currently plans them *serially*. That is a backfill estimate, not a promise,
and once b201 frees all 8 GPUs at once both should start together; but if
241339 really does wait for 241338, the b200 arm finishes a day later than the
rtx arm rather than beside it. Worth re-checking after 03:18 rather than
assuming.

### Is a 16 h limit too long?

No. The measured rate puts a full 9.6B-token run at 9.8 h on rtx, and a B200
will not be slower, so 16 h leaves about 60% headroom — enough to absorb a
slower step or a restart from checkpoint without losing the run to the wall.
The limit is a ceiling, not a reservation: accounting is on elapsed time, so an
unused hour costs nothing from the 1000 B200 hours. The only price of a
generous limit is scheduling — a shorter request backfills into gaps a longer
one cannot fit. Given that b200 is blocked on a whole-node occupancy rather
than on gap-fitting, that price is currently zero.

### Audit of the logs, 20:20

Checked end to end, because a run that is quietly wrong for ten hours is worse
than one that fails in the first minute. Nothing needed fixing.

- **Config matches `anchor_config` exactly** in both manifests: `optimizer
  pion`, lr 1e-3 → 1e-5 cosine, **no warmup**, 73242 steps, batch 512
  sequences × 256 tokens = 131072 per step, weight decay 0.1, clip 1.0,
  `pion_scaling rms`, `pion_rms 0.2`, `pion_momentum lie`, `pion_retraction
  trunc`, `precision bf16`. That is their published-number configuration and
  not their shell script's, which is the point of the anchor.
- **The two sides are not crossed.** `pion_alternate` is `false` under 241310
  and `true` under 241311, and each manifest records its own `slurm_job`.
- **stderr is empty** for both jobs. The only non-empty `.err` files in `logs/`
  belong to the corpus build and the earlier probes: HF rate limits and a
  `cusolver` SVD-non-convergence warning that falls back to an accurate method
  and touches only the NGD path, not these Pion runs.
- **Evaluation is running**, every 500 steps: val loss 3.988 (bilateral) and
  4.013 (alternate) at the last eval, falling monotonically. Checkpoint and
  lock heartbeat ride on the same eval boundary.
- **No NaN or Inf** in either log.
- **The corpus is whole.** `c4_train.bin` is 20,015,547,418 bytes = 10.0B
  uint16 tokens, and each run logs `passes 0.959` against the 9.6B it needs —
  so no token is seen twice. The shard failures and 429s in
  `ngd-data-241285/241288/241289.err` are from earlier attempts that resumed;
  the final build succeeded.
- **Logs land where they should.** Slurm stdout/stderr in
  `Natural-Pion/logs/ngd-pion-<jobid>.{out,err}`, per-run `log.jsonl`,
  `manifest.json` and `checkpoint.pt` in the run directory. The Slurm log file
  keeps the `ngd-pion-` prefix because `-J` is set before the run knows what it
  is; the rename via `scontrol update` only fixes the queue's display name.
  Cosmetic, and already documented in `train.sbatch`.

### One thing to be careful about before 02:37

The container binds the repository live (`--bind $REPO:$REPO`, `--pwd $REPO`),
so a job imports whatever is in the working tree **at the moment it starts**,
not at the moment it was submitted. The rtx pair is running commit `7c9783fe`;
the b200 pair will start at 02:37 against whatever HEAD is then.

Right now that is harmless: `7c9783fe..3bade81` touches `harness/run.py` and
`harness/train.py` only to add the run-directory lock and its `--force`
escape. No arithmetic changed, and `manifest.json` records the commit either
way, so the difference stays visible. But **any edit to `harness/` or
`ngd_pion/` before 02:37 goes straight into the b200 runs.** Until they start,
keep changes to `docs/`.

### A gap, not a fault: resume is still unverified

`241301` was to prove that a killed run resumes from its checkpoint rather than
restarting. Leg 1 passed — SIGKILL at step 900, a complete 1.04 GB checkpoint,
no stray `.tmp` — but the job hit its own limit during leg 2, so the resume
half never reported. It does not block anything here: 9.8 h against a 24 h cap
on rtx and a 16 h cap on b200 means neither pair should ever need to resume.
Worth closing before any run that does.

### Two jobs in the queue that are not part of this

- **241323** — `tput` on rtx, `OPT=ngd`, 40 min cap, running on rtx6004
  alongside the anchors. Legitimate: it measures NGD-Pion throughput, which
  nothing has measured yet. Micro-batch 512 already failed with CUDA OOM and it
  fell back to 256 — expected, since NGD carries the covariance and both bases
  on top of what Pion carries. Ends by 20:37 on its own.
- **241298** — `tput` on b200, PENDING, submitted 19:11 from a previous
  session. Its script is `/tmp/claude-.../scratchpad/throughput.sbatch`, a copy
  taken **before** the logging-overhead fix; the scratchpad directory is now
  empty, though Slurm keeps its own copy and would run it. It is the b200 half
  of the pair whose rtx half (241297) produced the discredited 149,697
  tokens/s. Running it would produce a b200 number by the same broken method,
  and it sits ahead of the real b200 runs in the queue with a 40 min limit.
  **Cancelled at 20:17** (`scancel 241298`), on the reasoning that a number
  produced by a method already known to be wrong is worse than no number. If a
  b200 throughput figure is wanted, resubmit from the repository script, which
  logs every 50 steps — but the two real b200 runs will report the same rate as
  a side effect, so it may not be worth a separate job.

  Removing it moved 241338 forty minutes earlier, to 02:37: it had been holding
  the first slot on b201.

### Next

1. **After 02:37**, confirm both b200 jobs started rather than serialised.
   Slurm still projects 241339 for 18:38 on the 25th, one full run behind
   241338. That is most likely a conservative estimate — Slurm computes an
   honest start time for the top pending job and projects the rest as "after
   the one ahead" — and the node frees all 8 GPUs at once, so both should fit.
   But it is the one thing here that could quietly cost a day.
2. Around 06:00, `anchor.check` on the rtx pair — the level against 3.3575 /
   3.3654, and more importantly the 0.0079 gap.
3. 241323 will have left an NGD throughput number and a workable micro-batch
   (512 OOMs, 256 fits). That feeds the allocation plan for the real
   comparison runs, which are still unsubmitted.

---

## 2026-08-24, 20:45 — what the runs actually cost, and an NGD number that changes the plan

### The anchors, in resources

Per job: **1 GPU, 8 CPUs**, and the memory Slurm hands out with a GPU — 128750 M
on rtx, 257920 M on b200, which in both cases is exactly one eighth of the
node, so it is the per-GPU share and not an over-request. `TRESBillingWeights`
on both partitions is `gres/gpu=1.0`, so the allocation is billed in straight
GPU-hours and nothing else we request moves the meter.

Spent from the allocation since 2026-08-01, all probes and false starts
included: **5.15 rtx GPU-hours and 0.98 b200 GPU-hours.** The anchor
calibration will add 2 × 9.8 h ≈ 19.6 rtx GPU-hours (about 1% of 2000) and at
most 2 × 16 h = 32 b200 GPU-hours (about 3% of 1000, and less in practice since
the runs finish early). The whole calibration therefore costs under 2% of the
allocation.

### How hard the card is actually working

A step is 131072 tokens and takes 0.463 s, so 283k tokens/s. The model does
6 × 41.73 M matmul parameters per token — the 25.30 M transformer body plus the
16.44 M output head, the embedding being a lookup — plus 12 × 8 × 256 × 512 for
the attention scores, so 2.63e8 FLOPs per token, and

    2.63e8 × 2.83e5 = 7.4e13 = **74 TFLOP/s per card**,

with Pion's own rotation work on top and uncounted. Peak memory is 83.31 GB of
97.9, so the card is 85% full at micro-batch 512.

There is no honest denominator for that yet. The matmul figures in
`preflight-241238.out` — 15-16 TFLOPS fp32, 26-30 TFLOPS bf16 — were taken
before the warm-up fix and are the instrument, not the card; the one warmed
number on record is **184 TFLOPS** for the fp32/TF32 matmul. Against that, 74
is about 40%, and since bf16 peak is higher than TF32 peak the true fraction is
lower. A warmed bf16 matmul number would settle it and is worth taking the next
time anything runs preflight.

### 241323: NGD-Pion is 158x slower than Pion

The probe hit its 40 min limit before the 200-step measurement finished, but
the 12-step probe at micro-batch 256 is enough:

| | tokens/s | s/step | 9.6B tokens |
|---|---|---|---|
| Pion (anchor, mb 512) | 283,000 | 0.46 | 9.8 h |
| NGD-Pion (mb 256) | 1,788 | 73.5 | **1,491 h = 62 days** |

Steps 1-11 took 808 s of the 880 s elapsed, so 73.5 s/step is steady state, not
start-up, and `t_fac` is 100 so no refactorisation happened in that window
either. One NGD run at this speed would consume three quarters of the entire
2000-hour rtx allocation. **The comparison the whole project is for cannot be
run until this is fixed.**

### The suspect, and it is diagnostics

`optimizer.py:236`, inside `_apply`, on **every step for every 2-D weight**:

    state["angle"] = c * float(torch.maximum(
        torch.linalg.matrix_norm(X_in, 2), torch.linalg.matrix_norm(X_out, 2)))

`matrix_norm(·, 2)` is a full SVD. The model has 8 layers × 7 matrices = 56
two-dimensional weights, so that is **112 SVDs per step** at 512×512 and
1376×1376 — and on this card cusolver does not converge on them and silently
falls back to a slower algorithm, which is what the `UserWarning` in
`smoke-241252.err` and `smoke-241254.err` has been reporting all along. 73.5 s
over 112 SVDs is 0.66 s each, which is the right order for that fallback.

Its own comment says it is **recorded for diagnostics** — it answers the open
question "do rotation angles stay bounded without RMS", and nothing in the step
reads it. It is logged only every `log_every` = 50 steps, and computed every
one of them.

This is a hypothesis with an arithmetic argument behind it, not a measurement.
Before changing anything, time the two halves of `_apply` separately on a card:
if the SVD is not most of the 73.5 s, the cost is somewhere else and the fix
would be aimed at the wrong place.

If it is confirmed, the cheap fix is to compute `angle` only on the steps that
log it, which needs the harness to tell the optimizer that a step is a logging
step; and, independently, to get the norm from a few power iterations rather
than a full SVD, since only the largest singular value is wanted and a skew
matrix is the easy case for it. Either alone is a large factor; both together
should put NGD within a small multiple of Pion.

**Nothing about this touches the anchors.** They run `pion_baseline`, which
takes a Frobenius norm at `pion_baseline.py:164` and never calls an SVD.

---

## 2026-08-24, 22:10 — the b200 pair started, and step 1 finally has its second number

### They started together, not serially

Slurm's projection had 241338 at 03:18 and 241339 a full run behind it at 19:18
on the 25th. Both actually started at **21:25 on the 24th**, side by side on
`b201`: the 8-GPU job that held the whole node had a 10 h limit and released
early. The serial projection was the conservative estimate it looked like, and
the b200 arm is now running beside the rtx arm rather than a day behind it.

Both are healthy — `NVIDIA B200, 183359 MiB`, empty stderr, `.run.lock` held,
checkpoint written, and each landed in its own directory under `runs/b200/`.

### The ratio: B200 is 1.94x RTX, exactly as this document warned

| | tokens/s (window) | s/step | 9.6B run |
|---|---|---|---|
| rtx, RTX PRO 6000 Blackwell | 284,750 | 0.460 | 9.4 h |
| b200, B200 | 551,600 | 0.238 | 4.8 h |

`CLUSTER.md` step 1 predicted this: *"Expect the B200 to disappoint... the
peak-FLOPS ratio (roughly 25x) will not appear."* It did not. A 60M model's
512x512 and 512x1376 matrices do not fill a B200, and the observed ratio is
**1.94x**, not 25x. Peak memory is 83.31 GB on both, i.e. identical work per
card; the B200's 183 GB simply goes unused at micro-batch 512.

**What that decides.** Measured in whole 9.6B runs, the two allocations are
worth almost exactly the same:

    rtx   2000 h / 9.4 h  = 213 runs
    b200  1000 h / 4.8 h  = 207 runs

So the choice is not about which pool is cheaper — it is about shape. `rtx` has
two usable nodes (16 cards) against b200's one (8), so it is where **breadth**
lives: step 5's twelve-run sweep and the ablations, as arrays. `b200` turns a
single run around in half the wall clock, so it is where **latency** lives: the
runs whose answer gates the next decision. The placeholder in `CLUSTER.md`
("long runs to b200, sweeps to rtx") happens to survive, but for the second
reason rather than the first, and the document's own rule of thumb — *"if the
observed ratio is only 2-3x, the long runs belong on the RTX pool, which has
twice the hours"* — turns out to be a wash rather than a win, because the hours
and the speed cancel almost exactly.

This also means the 16 h limit on the b200 pair is 3.3x what the run needs.
Harmless, since billing is on elapsed time, but a 7 h request would backfill
into gaps a 16 h one cannot.

### `-dirty` in the b200 manifests is mine, and it is only this file

Both `runs/b200/*/manifest.json` record `git_commit: 3bade81715d1-dirty`.
`config.git_commit()` calls `git status --porcelain`, which lists **untracked**
files, and the only untracked file in the tree is `docs/JOURNAL.md` — this one.
The code the b200 runs execute is exactly commit `3bade81715d1`, clean. The rtx
pair, started before this file existed, records `7c9783fefa01` with no marker.

Recorded here because a `-dirty` provenance stamp on an anchor run is exactly
the kind of thing that costs an hour of suspicion later. Committing the journal
would stop it recurring; it cannot un-stamp the two manifests already written.

---

## 2026-08-24, 22:30 — fairshare, what a GPU costs, and the dirty marker fixed

### Our fairshare number, and why it moved

`sshare -U` reports one association per partition:

| partition | RawShares | NormShares | RawUsage | EffectvUsage | FairShare |
|---|---|---|---|---|---|
| b200 | 1 | 0.25 | 5,974 | 0.176 | **0.788** |
| rtx | 1 | 0.25 | 27,928 | 0.824 | **0.788** |

`FairShare` runs 0 to 1 and says how far *under* our entitlement we are: 1 is
"used nothing", 0 is "used far more than our share". `NormShares 0.25` is the
entitlement — four equal associations under `p330`, so a quarter each.
`RawUsage` is billing-weighted seconds, decayed with a **7-day half-life**
(`PriorityDecayHalfLife`), so last week's jobs count half and the number
recovers on its own once we stop.

It was 0.903 at 20:15 and is 0.788 at 22:30: that is the four GPUs we are now
holding, showing up. It matters because `PriorityWeightFairShare = 10000`
against `PriorityWeightAge = 1000` — fairshare is roughly ten times age in the
priority sum, and only QOS (11000) weighs more. Being under-used is most of why
we took `b201` the moment it freed.

### Billing is MAX over TRES, not the sum — and memory can exceed the GPU

`PriorityFlags = MAX_TRES` with `TRESBillingWeights = cpu=0.0625,
Mem=0.007767G, gres/gpu=1.0`. A job bills the **largest** of its weighted
components, not their total. For our runs:

    gpu    1 x 1.0                  = 1.000   <- binds
    cpu    8 x 0.0625               = 0.500
    mem    125.7 G x 0.007767       = 0.976

So the eight cores are genuinely free, and our memory sits just under the GPU
term — 128750 M is one eighth of a 1030000 M node by construction, and the
weights are tuned so that one eighth of the node bills as one card. Ask for
more memory than the per-GPU share and **memory becomes the billed term**:
`241225` on rtx6004 asked for 400 G and bills 3, `236864` asked for 300 G and
bills 2, both while holding 2 GPUs.

Two consequences. Lowering our `--mem` would save nothing — the GPU term still
binds at 1.0 — so the reason to do it is courtesy, not allocation: right now
rtx6004 has 2 free GPUs and 32 free cores that nobody can use because only
54 GB of memory is left. And `CLUSTER.md`'s line that "the allocation is spent
by GPU-hour and the eight cores each job asks for cost almost nothing" is right
for the shape of job we run, but it is right by arithmetic rather than by rule,
and would stop being right on a fatter memory request.

### A GPU is allocated whole. There is no half card

`GresTypes = gpu` — no `mps`, no `shard`. `SelectType = select/cons_tres` with
`CR_CORE_MEMORY` makes **cores and memory** consumable and divisible between
jobs on one node, and that is the whole list: a GPU is an indivisible unit.
One job holds one entire card whether it uses 5% of it or 100%, and no second
job can be placed on it.

That is why unused cores are cheap and an under-used card is not. Leaving 6 of
8 cores idle costs the cluster nothing that anyone can reclaim; leaving a card
at 40% utilisation wastes something nobody else can have. It also settles the
question of whether to ask for more cores "since they are there" — a fatter CPU
request cannot make our single-GPU run faster, and it can strand other people's
cards on the same node exactly the way the memory requests above are stranding
two right now.

### `git_commit()` no longer fires on untracked files

`config.git_commit()` decided dirtiness with `git status --porcelain`, which
includes untracked files, so `docs/JOURNAL.md` — a notes file that no run
imports — stamped both b200 anchor manifests `-dirty` while the code they ran
was clean.

Changed to `--untracked-files=no`, so `-dirty` now means what it should: a
tracked file differs from the commit, and the code that ran is not the code the
hash names. Untracked **modules** are the one case that could still matter, so
`untracked_modules()` reports `.py` files separately in a new manifest field
rather than folding them into a flag that cannot say which file it means.

126 tests pass, 1 skipped by design, 21 s in the container.

The two manifests already written are left alone — rewriting a record after the
fact is worse than annotating it — and each b200 run directory now carries a
`PROVENANCE.md` saying exactly what the marker meant and how it was verified.

---

## 2026-08-24, 22:50 — the fix committed, the running pair left alone

Considered stopping the b200 pair and restarting it to clear the `-dirty`
stamp, and decided against it. `train.py:212` does rewrite `manifest.json` at
every start, so it would have worked mechanically — but the pair was at step
11,400 of 73,242, and a restart resumes from the checkpoint rather than
starting over. The rewritten manifest would then have named a commit under
which the first 15.6% of the run was *not* executed, with nothing in the
directory to say so. That trades a documented error for an invisible one.

Two further reasons. The rtx pair records `7c9783fefa01` with no marker and
runs the identical configuration, so both sides of the anchor already have a
clean-provenance witness; the b200 pair is the duplicate. And `scancel` leaves
the run lock behind — 900 s grace, refreshed every ~2 min — so an immediate
resubmit would have exited on `RunLock.take` and handed back the node for
nothing, unless forced past a lock that was not actually stale.

Committed instead:

    57a20da  A journal, so an interrupted session can be picked up
    76fcf23  A notes file is not a dirty working tree

`git_commit()` now returns `76fcf234d9c4` with no marker. Every run started
from here — the step 5 sweeps and the nine comparison runs of step 6, which are
the ones that go in the paper — carries an honest stamp. The two b200 anchors
keep their footnote in `PROVENANCE.md`.

**A habit this creates.** `docs/JOURNAL.md` is tracked now, so an uncommitted
journal entry marks the tree dirty — correctly, since the marker no longer
distinguishes a notes file from a code change once that file is tracked.
Commit journal entries as they are written, or a run started mid-edit inherits
a marker that is technically true and practically misleading.

Also noticed and not acted on: `AGENTS.md` and `README.md` both say "121 tests,
14 s". It is 126 tests, 1 skipped, 21 s in the container as of `76fcf23`.

---

## 2026-08-25, ~05:30 — the anchor missed, and the gap missed by more

All four runs completed the full 73242 steps. `anchor.check` says `matched:
false` on every one.

| side | rtx | b200 | target | miss |
|---|---|---|---|---|
| bilateral | 3.3997 | 3.3997 | 3.3575 | **+0.0422** |
| alternate | 3.4352 | 3.4369 | 3.3654 | **+0.0698 / +0.0715** |

Tolerance is 0.02, so bilateral misses by 2.1x it and alternate by 3.5x.
Validation agrees with training throughout — 3.4059 and 3.4414 on rtx — and the
corpus is 0.959 passes, so nothing here is overfitting.

### The duplicate pair paid for itself

Running the same anchor on both partitions was two runs more than step 4 asks
for. It is what makes the next sentence sayable: **the miss is systematic, not
noise.** Bilateral came out at 3.3997 on an RTX PRO 6000 and on a B200 — equal
to four decimals — and alternate differed by 0.0017 between them. Hardware-to-
hardware spread is therefore ~0.002 against misses of 0.042 and 0.070, twenty
to thirty times larger. No amount of re-running fixes this.

### The gap is the real finding

`AGENTS.md` says reproducing the 0.0079 bilateral-to-alternate gap is the
sharper test, *because the gap is insensitive to data order and initialisation
in a way the level is not*. Ours:

    rtx    3.4352 - 3.3997 = 0.0355     4.5x theirs
    b200   3.4369 - 3.3997 = 0.0372     4.7x theirs

The direction is right — bilateral beats alternate, as in their table — but the
size is wrong by a factor of four and a half, on the quantity specifically
chosen because harness differences should not move it. Every entry in
`KNOWN_DIFFERENCES` is a Megatron or data difference that acts on the *level*;
none of them explains a gap that is four times too wide.

### First suspect, with a mechanism

`pion_baseline._step`, lines 192-203. The step scale is computed once, from
**both** generators:

    c = self._scale(W, g_in, g_out, group)     # base = W @ g_in + g_out @ W
    if group["alternate"]:
        W = W @ right if state["step"] % 2 else left @ W

`_scale` is their `_scale_update_matrix_rms`: it normalises by the Frobenius
norm of the *two-sided* update and targets an RMS of 0.2. In `alternate` mode
only one of those two sides is then applied. So the scale is calibrated against
a step twice the size of the one taken, and alternate is systematically
under-stepped relative to bilateral — which is exactly the direction and
roughly the shape of an inflated gap.

Note where the arithmetic lands: bilateral misses by 0.0422 and alternate by
0.0698, and the difference between those two misses, 0.0276, *is* the gap
inflation. A single mechanism that penalises only the alternate arm would
account for the whole discrepancy while leaving bilateral's miss to the ordinary
harness differences.

This is a hypothesis with a mechanism, not a measurement. It is checkable
without spending an hour of GPU: read their released `_scale_update_matrix_rms`
and its call site and see whether the scale there is computed two-sided when
`alternate` is on. Their code is not stored on this filesystem; the login node
has network.

### Where that leaves the project

Bilateral's +0.0422 sits inside the ~0.05 that `AGENTS.md` calls "more likely
the flat token stream or the C4 subset than a defect", and the three data
differences at the top of `KNOWN_DIFFERENCES` remain the natural explanation
for a level that lands slightly high. If the scale hypothesis holds and is
fixed, the expected picture is both arms high by a similar ~0.04 with a gap
near 0.0079 — which would be a passing anchor in everything but absolute level,
and a much more defensible position than either number alone.

Until then, step 5 does not start. `AGENTS.md`: *nothing this harness produces
means anything until this lands.*

---

## 2026-08-25 — the six differences, read from the code rather than listed

Went through `KNOWN_DIFFERENCES` against the source. Two entries were weaker
than the list implies, one is more interesting than the list implies, and the
list's framing of the sixth was wrong.

**1. Flat token stream.** `prepare_data.py:220` appends `eos` after every
document, so the boundaries **are in the corpus**. What ignores them is
`TokenCorpus.batch`, which draws uniform random starts. So matching Megatron
here needs no corpus rebuild: one CPU pass over the 20 GB `c4_train.bin`
recording EOS positions gives a document index, and the sampler can then pack
from documents. Cheaper than the entry implies, and worth reconsidering — the
earlier note that we would simply never adopt their sampling was too quick.

**2. Our C4 subset.** 197 of 1024 `en` train shards fetched, 10.0B tokens
written against a corpus of roughly 156B. Their run draws its 9.6B from the
whole stream. And `TokenCorpus` samples windows **with replacement** and has no
epoch, so coverage is random rather than a shuffled pass: some windows repeat,
some never appear.

**3. Master weights and clipping — the weakest entry.** We do clip:
`train.py:277`, `clip_grad_norm_(model.parameters(), 1.0)`, one global norm
over every parameter, before the step. Megatron clips its fp32 master
gradients; our parameters *are* fp32 and autocast only changes what the forward
computes in, so the two are the same arrangement described twice. This entry
should be demoted.

**4. Separate Q, K, V — more than bookkeeping, for this project.** `model.py:88`
carries `wq`, `wk`, `wv` as three 512x512 matrices; Megatron fuses them into one
512x1536. Pion rotates *each weight matrix*, so fused means one pair of
rotations preserving one spectrum where we have three independent pairs
preserving three. That is a difference in the geometry the method acts on, not
a difference in layout. It moves the level; no mechanism is identified by which
it would inflate the bilateral-to-alternate gap.

**5. Weight decay — ours is principled, theirs is unread.** `build_optimizers`
gives AdamW only `rest` (embedding, head, norm gains) with
`weight_decay=0.1`; the Pion-owned 2-D weights receive **no decay at all**.
That is deliberate and load-bearing: decay shrinks singular values, and Pion's
premise is that singular values never move. Whether their Megatron
configuration also exempts the rotated weights is not known here and is worth
reading off their script — if theirs decays them, their spectra drift and ours
do not.

**6. The transcription — and it is not Megatron's doing.** Their Pion optimizer
is a standalone piece of code. Nothing about Megatron prevents matching it line
for line, which is exactly why this is the one difference that is cheap to
close and the one that can move the gap.

### What this reorders

Level: 1, 2 and 4 are the real contributors; 3 is noise; 5 is unknown until
their script is read. Gap: still only 6, with `_scale` normalising the
two-sided update while `alternate` applies one side as the named mechanism.

---

## 2026-08-25 — their code read. Three real differences found, two entries closed

Cloned `github.com/Sphere-AI-Lab/pion` to `$DATA_p330/reference/pion` (40 MB,
outside our tree). The optimizer is
`megatron-lm/megatron/core/optimizer/pion.py`. What it says:

### 1. The gap hypothesis is confirmed, line for line

`_scale_update_matrix_rms` takes `update_side` and normalises **the side it is
about to apply**:

    if update_side == "in":    base = p_data @ A_in
    elif update_side == "out": base = A_out @ p_data
    else:                      base = p_data @ A_in + A_out @ p_data

and `_effective_update_side` resolves `alternate` to `"in"`/`"out"` *before*
scaling — `"in" if step % 2 == 1 else "out"`, which is our phase exactly.

`pion_baseline._scale` always builds `W @ g_in + g_out @ W`. So under
`alternate` we calibrate against a two-sided step and take a one-sided one.
This is a defect in our transcription, not a Megatron difference, and it is the
named cause of a bilateral-to-alternate gap 4.5x too wide.

### 2. We apply a second moment; they do not

`--pion-use-second-momentum` is `action="store_true", default=None`, and
`_use_second_momentum` falls through to `False` when it is absent.
`opt_llama_60M_pion.sh` only passes it when `USE_SECOND_MOMENTUM` is set in the
environment, so their default run normalises the Lie momentum by its first
moment alone.

`pion_baseline.__init__` has `beta2: float | None = 0.95`, and `train.py`
constructs `Pion(...)` without passing betas, so **we divide by
`sqrt(v) + 1e-8`** where they do not. That is an Adam-shaped change to the
generator, and it acts on both arms — a level difference, not a gap one.

Worse, `beta1` and `beta2` are **not fields of `RunConfig` at all**. They
cannot be set, they are not in the configuration hash, and they are not in the
manifest — against `config.py`'s own opening rule that every field which can
change a result lives there. Two runs differing in `beta2` would produce the
same hash and share a directory.

### 3. They rotate Q per head; we rotate it whole

`pion_qkv_split_granularity` defaults to `"head"`
(`pion_split_qkv_per_head=True`), and the 60M script does not override it. In
that branch Q is sliced into `q_per_group // head_dim` blocks of
`head_dim x in_dim` — for this model, **8 blocks of 64x512, each rotated
independently, each with its own RMS scaling** — while K and V are rotated
whole. They also split the fused FC1 into up and gate, which we already have as
separate matrices.

So the entry "separate Q, K, V rather than a fused QKV matrix" had it
backwards. The fused-versus-separate part is not the difference; the
granularity is. We rotate one square 512x512 Q where they rotate eight wide
64x512 blocks — a different geometry, with a kernel on the in-side that a
square matrix does not have.

### Closed: two entries that are not differences

**Weight decay.** Their `Pion` takes `weight_decay`, stores it in `defaults`,
and **never reads it again** — no decay term appears anywhere in the update.
Rotated weights are undecayed in their implementation exactly as in ours. The
entry should come out of `KNOWN_DIFFERENCES`.

**Lie momentum under alternate.** `_lie_lie_generators` advances *both* buffers
every step regardless of `update_side`, which is what `_smooth_lie` does. Not a
difference. The worry recorded earlier was unfounded.

### Where this leaves the six

| # | was | now |
|---|---|---|
| 1 | flat token stream | stands; cheaper to fix than thought (EOS already in corpus) |
| 2 | our C4 subset | stands |
| 3 | master weights / clipping | demoted — we clip globally, our fp32 params are the master |
| 4 | separate vs fused QKV | **restated**: per-head vs whole-matrix rotation of Q |
| 5 | weight decay reach | **closed** — no difference |
| 6 | our transcription | **confirmed**, and now two named defects, not one |

Level: 1, 2, 4 and the second moment. Gap: the scaling side, alone.

---

## 2026-08-25 — all four changes made, `980e18c`

**Scaling side.** `_scale` takes the side and normalises `W @ g_in`,
`g_out @ W` or their sum; `_step` resolves `alternate` to a side *before*
scaling and retracts only that side. This is the gap fix.

**Betas.** `pion_beta1`, `pion_beta2` and `pion_second_moment` are `RunConfig`
fields now, so they are in the hash and the manifest. The second moment
defaults **off**, which is what their published run does.

**Per-head Q.** `Pion` takes `row_blocks: {id(param): n}` and rotates each row
block on its own, with its own scale and its own momentum buffers, sharing the
parameter's step counter so blocks alternate in phase — their arrangement
exactly. `build_optimizers` applies it to Q only, `heads` blocks, and **only
for `optimizer == "pion"`**: `NGDPion` rotates whole matrices, so switching it
on for `pion_ablated` would give the two arms of the comparison different
geometry and put a second variable in it. Lifting that means teaching `NGDPion`
about blocks, not flipping the flag.

**Sampler.** Windows partition the stream and a per-epoch permutation orders
them, seeded by `(seed, epoch)` and rebuilt on resume rather than checkpointed —
39M windows would be 313 MB of `int64` per checkpoint. `rng_state` is now
`{seed, epoch, cursor}` and **refuses** an old bit-generator state rather than
resuming into a different data distribution.

Also corrected in passing: `--use-same-init-for-output-layers`, which their
script sets, makes O and down initialise at the same 0.02 as everything else
instead of Megatron's `0.02/sqrt(2*layers)`. This harness initialises uniformly
at `init_std`, so it matches — checked rather than assumed.

136 tests, 1 skipped, 22 s. Ten are new. That the old suite passed the scaling
change untouched is itself the finding: nothing pinned `alternate`'s scale, so
the defect could sit there through a full 9.6-hour run and four completed
anchors without a single test going red.

### The re-run

`anchor_config` hashes moved, so nothing overwrites anything:

| side | was | now |
|---|---|---|
| bilateral | `8aff0f95a9` | `1795e6ddb3` |
| alternate | `51458a70e2` | `65f5f2acdb` |

Two runs, ~4.9 h each on b200, side by side on one node: about 10 b200
GPU-hours, 1% of that allocation. What to look for, in order of what it settles:

1. **The gap.** Their 0.0079. If the scaling fix is the whole story it should
   land near it; the pre-registered criterion is within 0.005, decided now
   rather than after seeing the number.
2. **Both levels.** Expect them still high and *by a common amount* — the C4
   subset is unchanged. A common offset is what licenses the comparison; two
   different offsets would not.
3. Whether the second-moment and per-head changes moved bilateral at all, which
   the old bilateral number (3.3997) makes readable.

---

## 2026-08-25, 09:45 — the anchor re-run is in flight

`245312` bilateral and `245313` alternate, both on `b201`, submitted from a
clean `main` at `6e738bf19b1a` and pushed, so the manifest hash resolves to a
commit anyone can fetch. `b201` was `IDLE`, so neither queued.

| job | side | directory |
|---|---|---|
| 245312 | bilateral | `runs/b200/pion-lr0.001-s0-1795e6ddb3` |
| 245313 | alternate | `runs/b200/pion-lr0.001-s0-65f5f2acdb` |

Manifests confirm what the fixes were supposed to change: `pion_second_moment
false`, `pion_split_q_per_head true`, `pion_alternate` false and true
respectively, `beta1/beta2` present as fields at all. Both start at loss
10.49246 — the same initialisation and the same first batch, which is what two
runs differing only in update side should do. Peak memory 82.47 GB against
83.31 before, consistent with dropping the second-moment buffers.

### Checks run before submitting, and one scare

The one worth recording: on the login node the sampler took **10 s per batch**,
which would have turned a 4.9 h run into 200 h. It is not a regression. The
*old* uniform-with-replacement sampler measured 9-20 s per batch on the same
machine, while sequential reads were instant — this is a cold NFS read under
the login node's 8 GB cgroup cap, not the permutation. On a compute node the
20 GB corpus fits in page cache, and the evidence is already on disk: the four
completed anchors sustained 280k tokens/s with the *cumulative* counter at 271k
by step 2600, so there was never a slow start to explain away.

Measuring the old path on the same machine, rather than reasoning about why the
new one might be slow, is what settled it in two minutes.

The rest: 39,092,866 windows and 76,353 steps to an epoch against 73,242
needed, so the run never crosses an epoch boundary and every window is seen
once; targets are the inputs shifted by one; every id inside the vocabulary;
peak sampler RSS 0.90 GB. One optimizer step on the real model puts the weight
displacement exactly on the RMS target — `wq` 1.024e-01 against
`lr*rms*sqrt(mn) = 0.1024`, `down` 1.679e-01 against 0.1679 — and bilateral and
alternate now move by the *same* amount, which is the scaling fix doing its job:
before it, the alternate arm was normalised against a step twice the size of the
one it took. Per-head spectra hold to 1e-6.

### The criterion, restated before the numbers arrive

1. Gap within **0.005** of their 0.0079.
2. Both levels still high, and high by a **common** amount.
3. Bilateral readable against its old 3.3997 to see what the second-moment and
   per-head changes did on their own.

---

## 2026-08-25, ~10:50 — `genoa` access arrived mid-morning

Checked `cpu` at 10:02: refused, and `genoa` with it. Checked again at 10:52
because the user said access had been granted, and `genoa` now works — the
association appeared between the two checks. `cpu` is still refused.

    p330  b200   normal
    p330  genoa  normal      <- new
    p330  rtx    normal

Two things worth carrying:

**The block on `cpu` is an accounting association, not partition policy.**
`scontrol show partition cpu` says `AllowGroups=ALL AllowAccounts=ALL
AllowQos=ALL State=UP`, 17 nodes and 680 cores. So the request to make is
narrow and specific: associate account `p330` with partition `cpu`.

**`genoa` is not cheap CPU.** `TRESBillingWeights=CPU=1.0,Mem=0.375G` against
`cpu=0.0625` on both GPU partitions, and with `PriorityFlags=MAX_TRES` a job
bills the largest weighted component. One genoa core-hour therefore bills the
same as one whole GPU-hour on `rtx`, and a genoa node bills 192 an hour where
an rtx node bills 8. No association carries a hard quota, so nothing is
enforced — what it spends is fairshare, in the same `p330` pool that decides
when our GPU jobs start.

That inverts the obvious plan. `genoa` is the right *place* for CPU work
because it strands nobody's cards, but at these weights a 96-core tokenisation
run costs 96 billing units an hour there against 6 on `rtx`. Whether the
institute charges `genoa` against the 2000/1000 GPU grant or a separate CPU
budget is not visible from SLURM and should be asked before anything large goes
there.

---

## 2026-08-25 — the corpus is the same corpus, and the entry is gone

Challenged on why the dataset would be tokenised again when it already had
been, and the answer is that it should not be. The entry read "a 10B-token C4
subset in our own order against their full stream", which sounds like a
difference and is not.

Both runs consume **9.6B tokens of C4 with the T5 vocabulary**. C4 `en` is
roughly 156B tokens, so theirs is about 6% of the corpus and ours is about 6%
of the corpus. Neither sees a "full stream". Checked what we actually hold:
**64 shards of 1024, numbers 0 through 63** — an unbroken prefix. (An earlier
note here said 197; that was a `find` over every file under `downloads/`,
service files included. 64 is the shard count.) C4's shards are a
deterministic partition of an already-shuffled crawl, with no ordering by
source, date or quality, so a prefix of 64 is exchangeable with any other 64.
Tokenising the remaining 960 would produce a *different draw from the same
distribution*, not a more representative one, and the size of that effect is
already measured: two runs of the same configuration on different hardware
differed by 0.002.

Priced for the record, since it will look tempting again: ~156 core-hours at
about 1 core-hour per 1B tokens, 312 GB of disk against 977 GB free, and a
fairshare hit of 561,600 billing-seconds against the 119,844 this project has
spent in total. For nothing measurable.

`KNOWN_DIFFERENCES` is down to two, and the note above it says the corpus is
the same one and tells the next reader not to put the entry back. Of what
remains, one is where the master weights sit — which we established is largely
the same arrangement described twice — and the other is storage layout, whose
geometry `pion_split_q_per_head` already matches.

The lesson worth keeping: a list of caveats accumulates entries that were
plausible when written and were never re-checked against what they would
actually cost or buy. This one had been quoted three times in this journal
before anybody asked whether 6% against 6% was a difference at all.

---

## 2026-08-25, 15:30 — the re-run: gap 4.5x -> 1.8x, criterion missed by 0.0011

| side | target | before | after | miss |
|---|---|---|---|---|
| bilateral | 3.3575 | 3.3997 | **3.4021** | +0.0446 |
| alternate | 3.3654 | 3.4352 | **3.4161** | +0.0507 |

Gap **+0.0140** against their 0.0079, down from +0.0355. The criterion set
before the numbers — `|gap - 0.0079| <= 0.005` — gives 0.0061 and is **not
met**. It stays where it was put; missing by 0.0011 is missing.

Two readings, both clean:

**The scaling fix did what it was diagnosed to do.** Alternate improved by
0.019 and bilateral moved 0.0024, which is our noise floor. Since `_scale`
only misbehaved under `alternate`, an improvement confined to that arm is
exactly the predicted signature.

**The other two fixes did nothing to the level.** Dropping the second moment
and rotating Q per head left bilateral where it was, inside noise. So the
remaining common offset of ~0.045 is not explained by either, and the two
surviving `KNOWN_DIFFERENCES` entries are thin: master-weight placement, which
we established is the same arrangement described twice, and storage layout,
whose geometry is now matched.

**The residual arm asymmetry is real.** The two offsets differ by 0.0061,
three times the 0.0020 spread two machines showed. Something still treats the
sides differently from the way their code does, and it is not the scale, the
alternation phase, the momentum buffers or the per-head granularity — all four
were read off their source and matched.

### Their own experiment stops one step short

Asked whether one rotator might contribute nothing, and whether they ever
trained a single side. Checked both the code and the paper.

Their `_configured_update_side` accepts `in` and `out`, but those are the
internals of `alternate` — `_effective_update_side` resolves the alternating
mode into one of them per step. No script sets them, and `README_PION.md`
documents `--pion-update-side {both, alternate}` only. Section 2.4.3 compares
exactly two configurations, and their conclusion is:

> "updating the two transformations simultaneously is not always necessary to
> obtain strong optimization performance, and much of the benefit can be
> retained through a more lightweight alternating scheme"

**A fixed single side, held for a whole run, is not in the paper.**

### Whether a side can be dead — measured

Partly, and structurally rather than empirically. For a wide `W` the in-side
generator has a block that provably cannot move it: `down` is 512x1376, its
kernel is 864-dimensional, and a skew supported entirely inside that kernel
gives `||W·Cayley(Omega) - W|| = 1.0e-10`. That block is **372,816 of 946,000
generator parameters, 39.4%**, dead by construction. The same holds for the
out-side of the tall `gate` and `up`. For the square `wq/wk/wv/wo` neither
side has a dead block.

But dead is not the same as redundant. With `W = U S V^T`, the left rotation
moves `U` and the right moves `V`; a single side reaches a strictly smaller
set, it does not reach the same set by another route. So the honest statement
is that one side is *not* useless, while their own alternate result says the
two are largely interchangeable in practice — and nobody has asked whether one
alone suffices.

That question is cheap here and has something at stake for this project
specifically: a single side would mean one eigendecomposition per step instead
of two, and `S = I` would only ever apply to the out-side. It needs
`update_side` added to `Pion` and `NGDPion`, mirroring their
`_configured_update_side`, and four short runs at the 1.2B budget step 5 uses.

---

## 2026-08-25 — the setting, not the optimizer: two differences on 56.5% of the model

With the optimizer matched and the gap down to 1.8x, the remaining ~0.045
common offset had to be somewhere we had never looked. It was: the parameters
Pion does **not** own. The embedding, the output head and the 17 norm gains are
**32,879,104 of 58,176,000 parameters, 56.5% of the model**, and every audit so
far had been about the 43.5% Pion rotates.

Their non-matrix parameters go to Megatron's ordinary Adam (`pion.py` flips
`matrix_params.requires_grad` off, calls `get_megatron_optimizer`, flips it
back). `build_optimizers` differed from it twice.

**Adam betas.** We built `torch.optim.AdamW(rest, lr=..., weight_decay=...)`
with no betas and took torch's `(0.9, 0.999)`. Their script sets
`--adam-beta1 0.9 --adam-beta2 0.95`, and `--pion-beta2 0.95` beside it, so 0.95
under either reading. This is the third time a hyperparameter that changes
results was outside `RunConfig` and therefore outside the hash.

**Weight decay on 1-D parameters.** `get_standard_config_overrides` assigns
`ParamGroupOverride(wd_mult=0.0)` to every parameter with `len(shape) == 1` or a
name ending `.bias`, so their RMSNorm gains are never decayed. We decayed
everything in `rest` at 0.1. Measured against the anchor's own schedule, the
AdamW decoupled factor over 73242 steps is **0.0248** — a gain that starts at
1.0 ends at 0.025, a **40x shrink**.

The second is the one to believe, and the reason is specific to this family of
optimizers. A network normally absorbs shrinking norm gains by growing the
linear weights that follow them. Under Pion it cannot: the singular values of
every rotated matrix are frozen for the entire run, by construction. The only
free capacity left is the embedding and the head.

Both differences act on bilateral and alternate alike, so they move the
**common offset** and not the gap — precisely the part that was unexplained,
and precisely not the part that already had an explanation. The 0.0061
asymmetry between the two arms is untouched by this and stays open.

**Checked and found matching, so nobody re-checks them:** gradient clipping
(`ChainedOptimizer.step` computes one norm over all chained optimizers and
applies that single coefficient to each group, which is our one global
`clip_grad_norm_`); `--untie-embeddings-and-output-weights`; `--swiglu`;
`--disable-bias-linear`; `--init-method-std 0.02` with
`--use-same-init-for-output-layers`; RMSNorm at `--norm-epsilon 1e-6`; rotary
base 10000; `--kv-channels 64`; and the parameter split itself — 2-D and not
embedding-or-output goes to Pion, as ours does.

### Fixed

`adam_beta1 = 0.9`, `adam_beta2 = 0.95`, `adam_eps = 1e-8` and
`decay_norms_and_biases = False` are `RunConfig` fields, so they are in the hash
and the manifest. `_adamw` builds two groups, 1-D at `wd = 0`. The escape hatch
defaults to `False` because their runs do not decay those; it exists so the
completed anchors remain reproducible.

140 tests, 1 skipped. Four are new. One of them failed first at `lr=0`, which
was the test being wrong rather than the code: decoupled decay is proportional
to the learning rate, so at `lr=0` nothing decays and a broken split would have
passed.

---

## 2026-08-25, 16:35 — tidying, and what was deliberately left alone

`runs/` went from 8.1 GB to 3.0 GB. Nothing was deleted; `scripts/retire.sh`
moved everything to `attic/2026-08-25/` with `register.tsv` recording where each
thing came from. Within one filesystem that is a rename, so it cost nothing and
is reversible.

**Retired.** `runs/tput-241323/` and its three logs — the NGD throughput probe
that hit its 40-minute limit before finishing the 200-step measurement; its one
useful reading (1,788 tokens/s at micro-batch 256, OOM at 512) is in this
journal already, and it carried a stale `.run.lock`. Two zero-byte
`resumecheck-241301.leg*` files from the resume test that was cut off. And the
**four round-1 anchor checkpoints**, renamed first so the attic is readable
without consulting the register:

    checkpoint-rtx-bilateral-7c9783fe.pt     checkpoint-b200-bilateral-3bade817.pt
    checkpoint-rtx-alternate-7c9783fe.pt     checkpoint-b200-alternate-3bade817.pt

The reason those weights are dead and not merely old: the code that produced
them has three defects we have since named and fixed — the scaling side under
`alternate`, an unwanted second moment, and Q rotated whole. Nobody should draw
a conclusion from those weights. Their **results** stay in place: every
`anchor.json`, `log.jsonl`, `manifest.json` and `PROVENANCE.md` is untouched,
and those are the record.

**Kept.** The two round-2 checkpoints (`1795e6ddb3`, `65f5f2acdb`, code
`6e738bf`), because their Pion is correct and only the AdamW half was wrong, and
because one of them is the weight-level reference used today for the rotation
measurement. The two round-3 runs are in flight. `logs/` — 54 files, 215 KB, the
operational record `AGENTS.md` says not to thin out, and two of them are being
written right now. `reference/pion`, needed for the control. `c4/downloads`,
19 GB of source shards kept on purpose so a rebuild fetches nothing.

**Space was never the reason.** 974 GB free of 1 TB, 5% used. This was for
legibility, and the restraint above is the point: on a shared filesystem with no
pressure, the case for removing something has to be that it misleads, not that
it is large.

One detail that checks itself: round-2 checkpoints are 821 MB where round-1's
are 1044 MB. The 223 MB difference is exactly the second-moment buffers the
`pion_second_moment` fix removed — the change is visible in the file size.

---

## 2026-08-25 — the 158x, measured rather than argued

The entry above named `optimizer.py:236` on an arithmetic argument and said so
in as many words: "a hypothesis with an arithmetic argument behind it, not a
measurement". `scripts/probes/step_cost.py` now times every stage of `_apply`
on the real shapes of LLaMA-60M, on one RTX PRO 6000 Blackwell. Full table in
`docs/measurements/step-cost-rtx.txt`.

### The hypothesis was right, and by a wider margin than expected

**SVD: 69.5 s of the 73.5 s/step measured in job 241323.** 95%. The cusolver
non-convergence warning fired in the probe too, so the fallback path the
journal guessed at is now observed rather than inferred.

The cost is not spread evenly. A 512x512 pair costs 40 ms; a pair involving a
1376x1376 costs **2.84 s**. Twenty-four of the 56 weights have a 1376 side —
`gate` and `up` through `X_out`, `down` through `X_in` — and 24 x 2.84 s is
essentially the whole 69.5 s. The diagnostic was cheap on attention and
catastrophic on the feed-forward block.

**Power iteration replaces it at 14.4 ms/step**, five iterations, against
69 499 ms. A factor of 4 830. For the question the diagnostic answers — does
the rotation angle stay bounded without RMS scaling — three digits on the top
singular value is far more than enough.

### What becomes the bottleneck once it is gone

| stage | ms/step |
|---|---|
| `cayley` x2 | 160.0 |
| `natural_gradient` x2 | 16.4 |
| `curv` via `fisher_apply` | 15.6 |
| power iteration (SVD replacement) | 14.4 |
| `build_bases`, amortised over `t_fac`=100 | 9.5 |
| `generators` | 5.4 |
| grams | 3.8 |
| host syncs | 1.8 |
| **total** | **~227** |

Against Pion's 460 ms/step that is **+50%**: a 9.8 h anchor run becomes about
14.7 h. Runnable, where 62 days was not.

**`cayley` is now 71% of what remains, and it is latency, not arithmetic.** One
512x512 solve with 512 right-hand sides is 0.36 GFLOP and takes 1 ms, which is
roughly 28x off what the card does on work of that size. `_refactor` already
fuses layers of equal shape into one `eigh`; `_apply` does not, and loops over
all 56 weights instead. Three shape groups — 32 at 512x512, 16 at 1376x512, 8
at 512x1376 — would collapse about 1100 kernel launches into 60. If batching
recovers even 10x on the solve, the optimizer lands near 80 ms/step, or +18%.

### Two of my own suspects died on the numbers

**Narrowing `exact_fp32` to Cayley alone is not worth it.** `natural_gradient`
runs 16.4 ms with TF32 off and 8.7 ms with it on. The whole prize is 7.7
ms/step — 3% of the step — in exchange for touching the guarantee the method
rests on. Not a trade worth making. Killed, and it should stay killed.

**Host syncs are not the problem either.** `float(alpha)` and `float(angle)`
across all 56 weights cost 1.8 ms/step, not the pipeline stall I predicted.
Worth removing when `_apply` is rewritten for batching, since `_cayley` already
accepts a tensor `c`, but not on its own account.

### One free and exact saving

`curv` builds a full matrix to produce a scalar. For skew `X` and symmetric
`B`, `C` the two terms of `<X, 2(BXC + CXB)>` are equal — cyclic permutation
gives `tr(XBXC) = tr(XCXB)` — so

    <X, F(X)> = 4 tr(X^T B X C) = 4 sum((BX) * (XC))

Two matmuls instead of four and no intermediate. Checked against the current
expression on five random draws and on the identity-anchor case: relative error
0 to 2e-16. Measured saving 15.6 -> 8.3 ms/step. Small now, but it is exact
algebra with no approximation attached, so there is no reason not to take it.

### What the probe does not cover

Covariance accumulation (`observe`, `x^T x` per layer) is absent from the
table, and it is the only part of the step that scales with micro-batch. The
table accounts for 69.9 s of the 73.5 s/step measured in 241323; `observe` is
the likeliest home of the remaining 3.6 s. Measure it before quoting a final
overhead figure.

### Order of work

1. Power iteration in place of the SVD. This alone takes 158x to about 1.5x.
2. Batch `_apply` by shape. Attacks `cayley`, which is 71% of what is left.
3. The contracted `curv`. Free.
4. Measure `observe`, then quote a real overhead number.

---

## 2026-08-25 — the reference stays the reference

Decided on the user's instruction, and it changes where optimisation work is
allowed to land: **`optimizer.py` is the reference implementation and does not
get modified.** It stays a direct transcription of `ALGORITHM.md`, one level
above the role `reference.py` plays for the mathematics. Speed work goes into
`ngd_pion/fast.py`, a subclass, so the delta is explicit and small.

The reason this is the better arrangement and not merely a tidier one: an
optimised implementation that is edited in place has nothing left to be checked
against. With the split, the check is mechanical — run both from the same
initial state on the same gradients and compare trajectories.

The rule `fast.py` lives under: **every difference is either exactly equivalent
or confined to a diagnostic.** Nothing in it may move the weights.

### First item: power iteration in place of the SVD

`angle` is now estimated by `spectral_norm` — power iteration on `X^T X`,
`O(n^2)` per iteration — instead of `torch.linalg.matrix_norm(X, 2)`.

This is safe for a reason specific to this quantity, not a general tolerance
for approximation: `angle` is read by `harness.instrument` every few hundred
steps and by nothing in the step. It cannot feed back into the trajectory.
`alpha`, which *does* reach the step, is untouched and pinned by its own test.

### Cold convergence is bad, and the design had to answer that

Measured before choosing a default, on a random 512x512 skew:

| iterations | relative error, cold |
|---|---|
| 1 | 2.1e-1 |
| 5 | 6.1e-2 |
| 20 | 5.3e-3 |
| 50 | 1.1e-4 |

That is not an implementation weakness. A skew matrix has its singular values
in equal pairs and the large ones bunch, so the ratio governing convergence
sits near 1. Warm — starting from the vector converged on the previous step's
`X`, with `X` perturbed by a relative `1e-3` — **one iteration gives `1e-4`**.

So the vector is cached in optimizer state, and generous iteration counts are
spent in exactly the two places where no warm vector exists: the first step,
and the step after each refactorisation, where `X` moves discontinuously
because the basis it is expressed in has just been rebuilt. Defaults
`angle_iters=2`, `angle_warmup=50`. Two iterations across all 56 weights is
about 6 ms/step against 69 499 ms for the exact call.

One property worth stating because a test depends on it: the estimate is a
Rayleigh quotient and therefore always a **lower** bound. For a diagnostic
asking whether an angle stays bounded, low is the uncomfortable direction to
err in, which is why the warm-start discipline is a requirement rather than a
refinement.

### Tests

`tests/test_fast.py`, 18 new, suite now 158 passing and 1 skipped.

The equivalence test compares trajectories rather than final weights, with
`t_fac=5` so the run crosses several refactorisations, and asserts
`torch.equal` — bit-identical, not merely close. A single step agreeing would
prove much less: the optimizer carries state, and a divergence introduced at
step 3 can be invisible at step 1 and fatal by step 300.

### Not done yet, on purpose

`FastNGDPion` is not wired into `harness/train.py`. Which implementation
`--optimizer ngd` selects is a decision about what the paper runs, not a
detail, so it waits. Newton-Schulz for the Cayley solve, the batched `_apply`
and the contracted curvature are next, in that order.

---

## 2026-08-25 — `ngd-pion` and `ngd-pion-ref`

`--optimizer ngd` is gone. Two names now:

| name | class | role |
|---|---|---|
| `ngd-pion` | `FastNGDPion` | what runs |
| `ngd-pion-ref` | `NGDPion` | the transcription of `ALGORITHM.md` |

`RunConfig.name` is built from this string, so each implementation naming
itself is what keeps a run directory an honest record of the code behind it.
Checked: the two produce different hashes (`b9ed2b7807` against `ed842a7b07`)
on otherwise identical configurations, so the split is enforced by the naming
scheme rather than by anyone remembering.

Today the two are bit-identical, so the distinction buys nothing yet. It is
made now because the next items — a Newton-Schulz retraction, a batched step —
change rounding, and from then on the same name would mean different numbers
before and after.

**The old bare `ngd` is rejected rather than aliased.** A silent alias defeats
the whole point of the split. Failing loudly costs a puzzled minute; guessing
wrong costs a run.

The fast implementation takes the short name deliberately. The opposite
arrangement fails badly: forgetting a `-fast` suffix buys 62 days of wall-clock
instead of 14 hours, and nothing says so until the job is already queued.

`angle_iters` and `angle_warmup` stay out of `RunConfig`. Everything in the
config enters the hash, and these two move only a diagnostic — putting them in
would give two runs with identical weights different hashes, which is exactly
the confusion the hash exists to prevent.

Suite at 159 passing, 1 skipped.

### Where this leaves the validation phase

The stated goal right now is to get the algorithm runnable at a satisfactory
speed so it can be validated, and to keep the accuracy-trading optimisations
for afterwards, where their cost can be measured against `ngd-pion-ref`.

Worth being clear that the change made so far is **not** in the
accuracy-trading category. Power iteration touches a diagnostic and nothing
else; the weight trajectory is bit-identical. Newton-Schulz and the batched
step are the first entries that genuinely change numbers.

So the open question is narrow: does `ngd-pion` as it now stands already run
fast enough to validate with? The stage timings suggest about 220 ms/step of
optimizer against Pion's 460 ms/step, but that is extrapolation from warmed
micro-benchmarks, and the probe still leaves 3.6 s/step of the original 73.5
unaccounted for. Measure the real thing before deciding anything else.

---

## 2026-08-25 — 246531: the algorithm is runnable

`OPT=ngd-pion sbatch -p rtx scripts/sbatch/throughput.sbatch`, five minutes on
rtx6002, same procedure as 241323 so the numbers compare directly.

| run | micro-batch | tokens/s | s/step | 9.6B tokens |
|---|---|---|---|---|
| Pion, 241338 | 512 | 283,000 | 0.463 | 9.4 h |
| NGD, 241323 | 256 | 1,788 | 73.31 | **1,491 h = 62 days** |
| `ngd-pion`, 246531 | 256 | 121,818 | 1.076 | **21.9 h** |

**A factor of 68**, from one diagnostic. 62 days to under a day.

### The 3.6 s gap resolved, mostly

The stage probe accounted for 69.9 s of the old 73.5 s/step and left 3.6 s
unexplained; the two candidates on record were `observe` and allocator churn
caused by the SVD's own cusolver workspaces. The answer is mostly the second.
Roughly decomposing the 1.076 s/step now:

| | s/step | how known |
|---|---|---|
| optimizer stages | ~0.22 | measured, `step_cost.py` |
| model forward and backward | ~0.48 | Pion's 0.463 at the same tokens |
| remainder, most likely `observe` | ~0.38 | **inferred, not measured** |

So about 3.2 s of the old gap was the SVD's own overhead beyond its arithmetic,
and it left with it. The residual ~0.38 s is consistent with `observe`, whose
`x^T x` at micro-batch 256 is about 3.6e12 FLOP per step. That decomposition is
arithmetic against two measured endpoints, not a measurement, and should be
called an estimate until `observe` is timed directly.

### Micro-batch 512 no longer fits, and that has consequences

Pion fits micro-batch 512 at 83.3 GB of 97.9. `ngd-pion` OOMs there and runs at
256, peaking at 42.28 GB.

The persistent state does not explain it: covariance plus both bases across all
56 weights is about 385 MB. The likely cause is fragmentation from the many
small transient allocations in `_apply` — `X_in`, `X_out`, `eye_out`, the
`fisher_apply` intermediates and Cayley's workspace, several of them 1376x1376
and live at once, 56 times per step with nothing batched.

Two consequences worth recording before they are discovered the hard way:

1. **The comparison arms must share a micro-batch.** `pion_ablated` fits 512,
   `ngd-pion` does not. Running them at different accumulation depths puts a
   second variable into a comparison whose entire claim is that it has one. Fix
   256 for both.
2. ~~**`ngd_beta` means something different at 256.**~~ **Wrong, corrected
   the same day.** I claimed the covariance EMA fires once per micro-batch, so
   accumulation over two would halve its horizon in steps. It does not.
   `ActivationRecorder` carries an `enabled` switch for exactly this reason —
   its docstring says so — and `train.py:331` sets `recorder.enabled = micro ==
   0`, so `observe` runs on the first micro-batch of each step and on no other.
   `beta` means what it says at any accumulation depth.

   What *is* true, and much smaller: the covariance is estimated from one
   micro-batch, so at 256 it sees 65 536 tokens per step where at 512 it would
   see 131 072. The docstring's position is that one micro-batch is already far
   more samples than the covariance needs, which is plausible and untested.

   The lesson is the ordinary one: I inferred a mechanism from the algorithm
   instead of reading the code that implements it, having already been caught
   doing that earlier in this project. It also inverts a prediction — since
   `observe` processes one micro-batch, moving to 512 makes it *more* expensive
   per step, not less, which removes one of the reasons I gave for wanting 512.

### Is it fast enough to validate with

21.9 h against a 24 h partition cap, and the probe ran with `--eval-every
100000`, i.e. with no evaluation at all. A real run evaluates, so a real run
crosses the cap and needs a requeue. `resume.sbatch` exists and the resume path
is tested, so this is friction rather than a blocker — but it is friction on
every validation run, not once.

Against the allocation it is comfortable: one run 21.9 h, a minimal
`ngd-pion` against `pion_ablated` pair about 33 h, a four-point learning-rate
sweep across both arms about 130 h — 6.5% of the 2000 rtx-hour budget.

The honest summary is that speed is no longer what blocks validation, and the
next decision is whether to spend a little more on the memory profile — a
batched `_apply` is mathematically identical and would likely restore
micro-batch 512, removing the requeue and returning `observe` to one call per
step — or to start validating now and leave it.

---

## 2026-08-25 — 246607: I compared the wrong two things

The reported factor was NGD-Pion costing 2.3x Pion. Most of that was not the
optimizer.

| | micro-batch | s/step |
|---|---|---|
| Pion | 512 | 0.463 |
| Pion | 256 | **0.869** |
| `ngd-pion` | 256 | 1.076 |

**The model alone is 1.88x slower at micro-batch 256 than at 512.** Measured at
the same depth, NGD-Pion costs `1.076 / 0.869 = 1.24x` — **+24%**, which is
better than the +50% I had predicted from stage timings and then disowned.

So the earlier post-mortem was itself wrong in its main claim. I said the 1.5x
prediction failed because `observe` was omitted and because micro-benchmarks
underestimate. The real cause was cruder: I compared a run at micro-batch 256
against a run at micro-batch 512 and attributed the whole difference to the
optimizer. `observe` and micro-benchmark optimism are real and still unmeasured,
but they are second-order next to comparing two different things.

This is the same error as the false negative gap earlier in this project, where
the anchor comparison was read across mismatched step counts (52 600 against
56 500). Same shape, second occurrence: **before attributing a difference to the
variable under test, check that nothing else differs.** Cheap to check, and both
times it inverted the conclusion.

## 246607: micro-batch 512 fits, for free

```
expandable_segments off  ->  CUDA out of memory
expandable_segments on   ->  fits
```

Fragmentation, as the budget implied — 385 MB of persistent optimizer state
against about 14.6 GB of headroom under Pion's 83.3 GB peak. The many small
transient allocations in the unbatched `_apply` leave a pool that cannot serve
the 7-30 MB contiguous blocks the 1376x1376 work needs.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is a one-line environment
change with no effect on numerics. The other three remedies on the list —
caching `torch.eye`, the contracted curvature, the batched `_apply` — are not
needed for *this* purpose. They remain worth doing for speed, later, in the
phase where accuracy trades are on the table.

`observe_cost.py` crashed on this run: `Transformer.forward` returns
`(logits, loss)` and the probe used the return value as logits. Fixed; the
decomposition is still unmeasured.

---

## 2026-08-25 — 246613: the missing 0.42 s is warm-up, not the allocator

**This entry replaces one that claimed the allocator flag was worth 1.60x. That
claim was wrong and the error was caught by the user asking a one-line
question: how did micro-batch 512 run without `expandable_segments` when it
does not fit without it? It did not. Job 246610 had the flag on. I compared two
runs that both had it on and labelled the difference as the flag.**

The 2x2, 200 steps a cell, `expandable_segments` on everywhere:

| s/step | mb 256 | mb 512 |
|---|---|---|
| `pion` | 0.869 | 0.460 |
| `ngd-pion` | 0.673 | 0.722 |

### What actually separates the fast numbers from the slow ones

Every measurement taken, with the predicted cost from the measured components
(model + `observe` + optimizer step):

| optimizer | mb | s/step | job | first to read the corpus? | predicted | residual |
|---|---|---|---|---|---|---|
| `pion` | 256 | 0.869 | 246607 | yes (only 12 steps warmed) | 0.427 | **+0.442** |
| `pion` | 256 | 0.869 | 246613 | yes | 0.427 | **+0.442** |
| `pion` | 512 | 0.460 | 246613 | no | 0.438 | +0.022 |
| `ngd-pion` | 256 | 1.076 | 246531 | yes | 0.685 | **+0.391** |
| `ngd-pion` | 256 | 0.673 | 246613 | no | 0.685 | −0.012 |
| `ngd-pion` | 512 | 1.153 | 246610 | yes | 0.730 | **+0.423** |
| `ngd-pion` | 512 | 0.722 | 246613 | no | 0.730 | −0.008 |

The split is exact and has nothing to do with the allocator or the optimizer:
**a run that is the first in its job to touch the corpus pays about +0.42
s/step; one whose windows are already in page cache pays nothing.** With
`--no-resume` and a fixed seed every run draws the same window sequence, so the
second and later runs in a job re-read exactly what the first one read.

### Confirmed against a production run, not just against itself

Anchor 246482, Pion at micro-batch 512, per-window rate:

```
step    50   4.476 s/step     startup
step   100   0.976
step   150   0.881            <- the "cold" number, 0.460 + 0.42
step   200   0.828
...
step 35000   0.469            <- the "warm" number, sustained
```

So the +0.42 s is a **warm-up transient that decays over the first few hundred
steps**, and a 200-step probe logging every 50 measures inside it. The
sustained 0.469 s/step of a 35 000-step run agrees with the grid's warm cell
(0.460) to 2%.

This also dissolves the "reproducible Pion oddity at micro-batch 256" recorded
in the previous entry. Both of its measurements were cold. There is no oddity.

### What survives

* **`expandable_segments` buys no speed.** It is kept because micro-batch 512
  OOMs without it and fits with it — tested back to back inside job 246607,
  which is the only clean evidence anyone has about this flag. The sbatch
  comments have been corrected to say so.
* **The decomposition is right.** Every warm cell sits within 2% of model +
  `observe` + optimizer.
* **The overhead figure survives**, because that comparison was between two
  warm cells in the same job differing only in optimizer:
  **`0.722 / 0.460 = 1.57x`, +57% at micro-batch 512** — 14.7 h against 9.4 h.
* `pion` at micro-batch 256 has no valid measurement; both were cold.

### The methodological failure, third of the day

The first two were comparing the anchor gap across mismatched step counts and
comparing the 2.3x across mismatched micro-batches. This one is worse, because
the grid was built specifically to have one variable per comparison — and it
had a hidden second one, position in the job, which nothing in the design
controlled.

Two things follow for every throughput measurement from here:

1. **A 200-step probe measures warm-up, not steady state.** Either discard the
   first few hundred steps or run long enough that they do not dominate. The
   existing `throughput.sbatch` reports a median over four windows, all of
   which land inside the transient on a cold run.
2. **State the cache condition with the number.** "0.722 s/step" is not a fact
   about the configuration unless it says whether the corpus was warm.

### Where this leaves the validation phase

Speed is settled. Running the comparison at micro-batch 512:

* `ngd-pion` 14.7 h, `pion_ablated` about 9.4 h, both inside the 24 h cap even
  once evaluation is added, so no requeue.
* A pair costs about 24 h, a four-point learning-rate sweep across both arms
  about 97 h — under 5% of the 2000 rtx-hour allocation.

Newton-Schulz, the batched `_apply` and the contracted curvature are all still
worth doing, and all of them now belong where the plan always put them: after
validation, in the phase where trading accuracy is on the table. None of them
is needed to start.
