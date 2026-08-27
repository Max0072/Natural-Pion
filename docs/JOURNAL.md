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

---

## 2026-08-25 — settled: micro-batch 512 with `expandable_segments`

Decided. `RunConfig.micro_batch` defaults to 512, and every GPU sbatch script
sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `preflight` included so
the environment it validates is the one runs use.

The numbers behind it, all on warm page cache, job 246613:

| s/step | mb 256 | mb 512 |
|---|---|---|
| `pion` | 0.869 | **0.460** |
| `ngd-pion` | 0.673 | **0.722** |

For Pion 512 wins outright. For NGD-Pion the two are close, and what settles it
is that at 512 there is no accumulation, so the covariance sees all 131 072
tokens of a step rather than a 65 536-token slice — `observe` fires once per
step either way.

The flag is for the memory, not the speed: `ngd-pion` OOMs at 512 without it and
fits with it, tested back to back in job 246607. Not for lack of room — 385 MB
of optimizer state against 14.6 GB of headroom — but because the unbatched
`_apply` leaves the allocator unable to serve 7-30 MB contiguous blocks.
Anything run outside `scripts/sbatch` has to set the flag itself, and the
comment on `micro_batch` says so.

The run name does not change: `micro_batch` is excluded from the hash, and
`ngd-pion-lr0.001-s0-b9ed2b7807` is the same before and after. Which is the
exclusion working as intended — the depth changes speed and memory, not the
result.

### What a comparison now costs

`ngd-pion` 14.7 h, `pion_ablated` about 9.4 h, both inside the 24 h cap with
evaluation, so no requeue. A pair is about 24 h; a four-point learning-rate
sweep across both arms about 97 h, under 5% of the 2000 rtx-hour allocation.

Suite at 159 passing, 1 skipped. Speed work is done for the validation phase.

---

## 2026-08-25 — Newton-Schulz: measured, then dropped

Written, measured, and left out. `fast.py` is back to bit-identical with the
reference; `cayley_newton_schulz` stays in `linalg.py` as a tested primitive
that nothing calls.

### The error law, confirmed over five orders

Residual after `k` iterations is `||A||^(2^(k+1))` with `A = (c/2)X`. Measured
in fp64 against that prediction:

| `||A||` | NS 1 | NS 2 | NS 3 |
|---|---|---|---|
| 0.010 | `1.1e-08` | `3.2e-15` | `2.4e-15` |
| 0.050 | `6.8e-06` | `3.3e-11` | `2.8e-15` |
| 0.100 | `1.1e-04` | `8.4e-09` | `3.0e-15` |
| 0.250 | `4.0e-03` | `1.2e-05` | `1.5e-10` |
| 0.400 | `2.4e-02` | `4.8e-04` | `2.6e-07` |

The exact solve in fp32 leaves about `1.1e-6` at any angle, so each count stops
being the dominant error at `||A||` of 0.032, 0.180 and 0.424 respectively.

### Why it is out, and it is not about Newton-Schulz

**The rotation angles are one to two radians per step.** Median over 20 steps
on the small test problems:

| shape | `lr=1e-2` | `lr=1e-3` | `lr=1e-4` |
|---|---|---|---|
| 16x16 | 2.86 | 1.00 | 0.18 |
| 24x12 | 3.97 | 1.77 | 0.53 |
| 12x24 | 1.43 | 0.22 | 0.26 |

`lr = 1e-3` is what the anchors run. Newton-Schulz needs `||A|| = angle/2`
under about 0.4, so at those angles it refuses every step and buys nothing.

Two further things came out of trying:

* **The guard cannot be built on `spectral_norm`.** It is a Rayleigh quotient
  and therefore a lower bound, and measured inside the optimizer it ran between
  0.58 and 0.98 of the truth — worst when `X` moves fast and the cached vector
  goes stale. A 42% underestimate turns a nominal `||A|| = 0.35` into a real
  0.60, where NS-3 leaves `3e-4`, silently. The sound upper bounds were checked
  and rejected: for skew `X` both `||X||_inf` and `||X||_F` bound `||X||_2`,
  but on 512x512 they run 4x to 11x loose, so a guard on either refuses always.
* **At small angles a third iteration makes drift slightly worse, not better.**
  The residual is long since under the fp32 floor, so the extra matmuls only
  add rounding: over 1000 steps at angle 0.1 on 512x512, drift was `4.4e-4` at
  two iterations and `8.4e-4` at three. Newton-Schulz at our angles would have
  been rounding-limited, not truncation-limited — its cost to the spectrum is
  two extra matmuls per retraction, roughly doubling drift against the solve.

So `cayley` keeps its 160 ms, and NGD-Pion stays at +57% — 14.7 h against 9.4.
Which does not block validation.

### The finding that outlives the attempt

The angle measurement is the important part, and it is about the method rather
than about the retraction. `ALGORITHM.md` capped its own trust-region grid at
0.1 rad because the quadratic model loses accuracy above it. One to two radians
per step is far outside that, and the open question the whole `angle`
diagnostic exists to answer — whether the angle stays bounded once Pion's RMS
scaling is ablated — now has a preliminary answer, and it is not reassuring.

**Caveat, and it is a large one:** these are 12-to-24 dimensional problems with
random gradients and a fixed covariance. They are evidence about a unit test,
not about a 60M transformer.

Which is now cheap to settle. `ngd-pion` runs at 0.722 s/step, so 500 steps is
six minutes, and `harness.instrument` already logs `angle_min` and `angle_max`.
No NGD run has ever produced them at scale.

---

## 2026-08-25 — 246662/246663: the smoke test earned its keep on the first try

Two 500-step runs, `ngd-pion` and `pion_ablated`, launched through
`scripts/sbatch/train.sbatch` exactly as the overnight runs would be. Both died
in twelve seconds:

```
torch.OutOfMemoryError: Tried to allocate 15.67 GiB
```

15.67 GiB is `512 x 256 x 32100` logits in **fp32**. In bf16 it is 8.4.

`RunConfig.precision` defaulted to `fp32`, and nothing that had ever run used
that default: `anchor_config` sets `bf16`, and `throughput`, `memprobe` and
`grid` all passed `--precision bf16` on the command line. `train.sbatch` passes
no precision at all, and no full run had ever gone through it — every number in
this journal came from a probe script that set the flag itself.

So the configuration we were about to run overnight was one nothing had
measured, in a precision nothing had measured, and it could not have completed
even the first step. The fifteen minutes of smoke test cost less than finding
this at breakfast.

Fixed by making the default `bf16`, which is what their published script uses
(`--bf16` in `opt_llama_60M_pion.sh`) and what everything here already ran. The
comment on the field explained the old default by saying precision was "first
in `anchor.KNOWN_DIFFERENCES` while this is `fp32`" — but precision stopped
being a known difference when `anchor_config` started setting bf16, and the
default was left behind. Stale comments outlive the thing they describe.

`precision` is a scientific field and in the hash, so an fp32 run would at
least have landed in its own directory rather than contaminating one. Small
mercy, and not the one that mattered here.

Suite at 167 passing, 1 skipped.

---

## 2026-08-25 — the default configuration is now the anchor, minus the optimizer

Went through `anchor_config` field by field against `RunConfig()`. Two
disagreed, both of them defaults that nothing actually ran and nothing
defended:

* `precision`, `fp32` against the anchor's `bf16` — the one that killed
  246662/246663.
* `pion_alternate`, `True` against the bilateral anchor's `False`, and the only
  field in that block of `config.py` with no comment justifying it.

`pion_alternate` reaches only the `pion` context arm: `pion_ablated` is forced
to `False` in `train.py` and `NGDPion` never reads it, and `anchor_config` sets
it from `update_side` rather than from the default, so the anchors were never
affected. But the context arm exists so that the ablated baseline cannot be
called a straw man, and that argument only holds if the context is Pion at its
best — 3.3575 bilateral against 3.3654 alternate. Running it in a variant the
measurement arm does not use would have put a second difference between them
for no reason.

With both fixed, `RunConfig()` differs from `anchor_config("bilateral")` in
**one field: `optimizer`.** Every setting the published numbers were produced
under — `lr_min=1e-5`, `warmup_steps=0`, `train_steps=73242`,
`batch_sequences=512`, `weight_decay=0.1`, `grad_clip=1.0`, `pion_rms=0.2`,
`pion_momentum="lie"`, `pion_retraction="trunc"`, `precision="bf16"` — is
inherited rather than restated.

### One inherited value that is *not* justified, and it matters tonight

`lr = 1e-3` is Pion's tuned learning rate. Nothing has tuned it for NGD-Pion,
and inheriting it is a convenience, not a decision.

It is also the value where the toy measurement put rotation angles at one to
two radians per step — far outside the 0.1 rad where `ALGORITHM.md` says the
trust region's quadratic model holds. That measurement was on 12-to-24
dimensional problems and may say nothing about a 60M transformer, but it is the
only evidence there is, and it points at `lr = 1e-3` being too large for this
optimizer rather than at anything being wrong with the optimizer.

So tonight's `ngd-pion` run is **one point of a sweep that has not been run**,
not "the" NGD-Pion result. Whatever it produces should be read that way, and
the `angle_min`/`angle_max` it logs are the thing to look at first.

---

## 2026-08-25 — first learning-rate probes, and a hypothesis cleanly killed

Five 1000-step runs at micro-batch 512, `runs/smoke/`.

| `eta` | warmup | val@999 | `angle@0` | `angle@999` | `alpha_min@999` |
|---|---|---|---|---|---|
| 1e-3 | 0 | 5.4504 | 0.0057 | 0.037 | 0.378 |
| **3e-3** | 0 | **5.0686** | 0.0170 | 0.137 | 0.154 |
| 0.5 | 200 | 5.8628 | 0.0142 | 0.016 | 0.014 |
| 1.0 | 200 | 7.8608 | 0.0283 | 0.000 | 0.020 |
| 1.0 | 0 | 7.6109 | 5.6613 | 0.000 | 0.032 |

### `eta = 3e-3` beats the inherited `1e-3` by 0.38

```
10.49 -> 5.98 -> 5.55 -> 5.49 -> 5.36 -> 5.32 -> 5.30 -> 5.25 -> 5.18 -> 5.12 -> 5.08
```

Monotone, no rebound. Against `1e-3`'s 5.4504, this is the first evidence that
Pion's tuned learning rate is wrong for NGD-Pion, and wrong in the direction of
being too *small*.

### The cold-Fisher hypothesis is dead, and the warmup run is what killed it

The proposal was that `eta = 1` fails because the first steps are taken against
a covariance estimated from a single micro-batch — `condA_max` is `4.8e6` at
step 0 against `2.2e4` by step 50, and the method inverts that matrix. The
mechanism was plausible and specific: ablating Pion's RMS scaling removed the
only thing pinning step length, and `alpha` cannot substitute because §6 makes
it identically 1 on a fresh basis.

Warmup discriminates it, and the answer is no.

```
eta 1.0, warmup 200:  6.67 -> 15.48 -> 26.73 -> 21.02 -> 14.38 -> 8.35 -> ...
eta 1.0, no warmup:   8.44 ->  7.62 ->  7.75 ->  7.71 ->  7.30 -> 8.00 -> ...
```

It is **worse** with warmup, 7.86 against 7.61, and it blows up to 26.7 at steps
200-300 — exactly where the 200-step warmup ends and full `eta` arrives. By then
the covariance has had two hundred EMA updates at `beta = 0.95`, ten times its
own horizon. The Fisher is as warm as it gets and the run still detonates the
moment the step reaches full size.

Warmup did not prevent the catastrophe, it postponed it. So the problem is the
magnitude of `eta`, not the state of `A` when the first step is taken.

### `eta = 0.5` does not blow up but does not converge either

Stuck between 5.87 and 6.20 for 800 steps, val 5.8628 — worse than `1e-3`. Its
`condA_max` reaches `4.1e8` by step 999, against `7e6` at the working step
sizes. Possible feedback: a large rotation moves the activation distribution,
which degrades `A`, whose inverse then amplifies more noise. Hypothesis, not
diagnosis, but the number is striking.

### Correcting something I had been asserting

I had been using `ALGORITHM.md`'s 0.1 rad — the angle above which its
trust-region grid stops being accurate — as a ceiling on admissible `eta`. The
best run has `angle@999 = 0.137`, above that line, and it is the smoothest
curve of the five. The 0.1 rad figure bounds where the *quadratic model* is
accurate; it does not bound where the optimizer works. My inference from it
about the usable range of `eta` was unfounded.

### Open

* `angle_max` is **exactly** `0.00000` from step 700 onward in both `eta = 1`
  runs, while `alpha_max` is 1.0 and the loss is finite. It accompanies the
  collapsed regime specifically. Either a real degeneracy — `G_in = W^T G -
  G^T W` vanishes when `W^T G` is symmetric, which would mean the gradient has
  no rotational component left — or an underflow. Unexplained.
* **`pion_ablated` has only ever run at `1e-3`.** Its toy optimum was `2.2e+01`.
  Comparing the arms while one of them has never been given its own regime
  measures nothing, and at `1e-3` they are tied to 0.016, which is what that
  looks like.

---

## 2026-08-26 — anchor round 3, and the decision to accept it

Jobs 246482/246483 finished.

| | target | measured | delta | relative |
|---|---|---|---|---|
| bilateral | 3.3575 | **3.3805** | 0.0230 | 0.69% |
| alternate | 3.3654 | **3.3954** | 0.0300 | 0.89% |

Gap between the arms **0.0149** against their published 0.0079.

| round | bilateral | alternate | gap | level delta |
|---|---|---|---|---|
| 1 | 3.3997 | 3.4352 | 0.0355 | 0.042 / 0.070 |
| 2 | 3.4021 | 3.4161 | 0.0140 | 0.045 / 0.051 |
| **3** | **3.3805** | **3.3954** | **0.0149** | **0.023 / 0.030** |

Round 3's two fixes — AdamW betas `(0.9, 0.999) -> (0.9, 0.95)` and no weight
decay on 1-D parameters — **halved the level error** and **left the gap
untouched**. That is consistent with what they change: both act on the half of
the model Pion does not own, and both act on the two arms identically. So
whatever remains of the gap discrepancy sits on the Pion side, not the AdamW
side. Worth knowing if anyone returns to this.

### Accepted, against the pre-registered criteria, deliberately

`docs/RESUME.md` fixed the criteria before the numbers existed: gap within
0.005 of 0.0079, and `matched: true` on both levels. Round 3 meets neither —
the gap is off by 0.0070, and the 0.02 absolute tolerance is exceeded by both
levels. The pre-decided fallback was to stop looking for a fifth difference and
run their unmodified code as a control.

**The user overrode that and accepted the anchor**, on the grounds that the
level reproduces to under 1% and the arm ordering and gap are qualitatively
right. Recorded as an override rather than quietly folded in, because the
criteria were written down in advance precisely so that accepting a miss would
have to be a decision someone makes rather than a threshold that drifts.

What it licenses: internal comparisons in this harness. `ngd-pion` against
`pion_ablated` never depended on the anchor anyway — both arms share the
implementation, the data and the sampler, so a common offset is common.

What it does not license: quoting our numbers as reproductions of theirs, or
any claim resting on the *magnitude* of the bilateral/alternate gap, which
comes out 1.9x theirs. In the paper this belongs in the text as a stated
limitation — "reproduced to 0.7% on the level; the arm gap comes out 1.9x" —
not as a footnote and not omitted.

The control run is not cancelled, only deprioritised. It remains the thing that
would separate "our harness differs" from "their number is not reproducible
from the published configuration".

## 2026-08-26 -- where the operator comes from, and what alpha actually measures

### The derivation, from the author rather than from the code

Recorded because the code reads as "Pion with a metric bolted on", and the
order it was actually arrived at is the reverse. The intent was **K-FAC in a
Lie group**: do natural gradient descent where the parameter is a rotation, and
Pion is simply the choice of group.

The chain, verified against `direction.py`:

1. Parametrise by the algebra, `W <- exp(-X_out) W exp(-X_in)`, both `X` skew,
   so `dW = -(X_out W + W X_in)` to first order.
2. Push the weight gradient through: `dL = -<G W^T, X_out> - <W^T G, X_in>`.
3. Substitute the *per-sample* gradient `G = delta x^T`, which is rank one.
   Then `W^T G = u x^T` with `u = W^T delta`, and `G W^T = delta y^T` with
   `y = Wx`. The skew parts are the **bivectors** `u ^ x` and `delta ^ y`. This
   is the step the whole construction rests on: the generator is a wedge of
   exactly two vectors *because* the layer gradient has rank one.
4. The Fisher is the covariance of generators. Using `<u^x, X> = 2 u^T X x`,
       F(X) = 2 E[u u^T X x x^T] - 2 E[x x^T X^T u u^T]
   and the **single approximation of the entire method** -- K-FAC's
   independence of `u` and `x` -- factorises both expectations into
       F(X) = 2 (S X A + A X S),   A = E[x x^T],  S = E[u u^T] = W^T E[dd^T] W
   Everything else above is an identity.

Note what `S = I` actually sets to the identity: `E[delta delta^T]`, the
*output-side* backprop covariance. The pull-back through `W` is kept exactly,
which is why `build_bases` uses the pair `(A, W^T W)` on the in-side and
`(I, W A W^T)` on the out-side. So `S = I` removes the congruence on the
out-side always, and on the in-side only when `W^T W = I`.

### Damping: the gradient's own smallness cancels half the divergence

The design argument for the single floor `max(lam, eps lam_max)`, with no
schedule, was that directions with degenerate eigenvalues also carry small
gradient, so the division is not as violent as it looks. That is right, and it
is right by exactly a factor of one half in the exponent.

In the eigenbasis of `A`, `Y_ij = (u~_i x~_j - x~_i u~_j) / (2(lam_i + lam_j))`.
Since `A = E[x x^T]`, we have `E[x~_i^2] = lam_i`, so the numerator carries
`sqrt(lam)` while the denominator carries `lam`. Hence `||X|| ~ eps^(-1/2)`,
not `eps^(-1)`.

Measured (`scripts/probes/damping_scaling.py`, CPU, fp64, `cond(A) = 1e8`,
against a control with a random skew `G` of equal norm, uncorrelated with `A`):

| eps  | slope, real G | slope, random G | \|\|X\|\| real | \|\|X\|\| random |
|------|---------------|-----------------|----------------|------------------|
| 1e-4 | 0.45          | 0.91            | 1.2e+02        | 5.1e+03          |
| 1e-6 | 0.42          | 0.85            | 8.9e+02        | 2.7e+05          |
| 1e-7 | 0.39          | 0.76            | 2.2e+03        | 1.6e+06          |

Slope is decades of `||X||` gained per decade of `eps` lowered: 0.5 means the
cancellation works, 1.0 means it does not. Real gradient sits at 0.42-0.47,
control at 0.85-0.92; a factor of 720 in step norm at `eps = 1e-7`.

This is worth a sentence in the paper. It explains the four-order plateau in
`eps` -- a naive `1/lam` analysis predicts a sharpness that is not there.

Second structural point: the denominator is `lam_i + lam_j`, a **sum**. A single
degenerate direction is harmless; both indices must be degenerate at once.

### alpha >= 1 is produced by the floor, not by a missing Jacobian

`natural_gradient` divides by `basis.denominator`, built from the **floored**
spectrum. `fisher_apply`, which `curv` calls, uses the **raw** `A` and `W^T W`.
These are two different operators. Writing `lam` raw and `lamt` floored:

    quad = sum g^2 / lamt          (against the floored operator)
    curv = sum g^2 lam / lamt^2    (against the raw one)

The floor only raises, so `lam <= lamt`, so `lam/lamt^2 <= 1/lamt` termwise, so

    curv <= quad  identically   =>   alpha = quad/curv >= 1  by construction

`alpha` is pinned at `alpha_max = 1` structurally. It is not evidence that the
curvature is underestimated, and it does **not** need the missing-Jacobian
explanation to account for it. Confirmed numerically in
`scripts/probes/curv_shapes.py`: `quad/curv -> 1.0000` exactly when the floor
stops being active (`#floored = 0`), for every weight shape.

This corrects the reading in `direction.trust_region_alpha`'s docstring and the
comment block in `fast._apply`. Staleness is a real second mechanism, but it is
the one that dominates only when the step is large: at `eta = 1.0` the basis
falls behind fast, the current `F` outgrows the `F~` that built `X`, `curv`
overtakes `quad`, and `alpha` sits at 0.13-0.22. Measured. So the trust region
works exactly as designed in the regime it was designed for, and is masked by
the floor in the small-step regime where we actually run.

The colleague's H1 stands as a criticism of the Fisher itself -- `tr(D A D^T)`
is not the true GGN -- but it is not what produces `alpha == 1`.

### What did NOT cause the 1e32, and what did

Refuted (`scripts/probes/curv_shapes.py`): rank-deficient grams from the layer
shapes. A wide `W` (192x768) has a 576-dimensional exact kernel in `W^T W`; a
tall `W` (768x192) has one in `W A W^T`. Both were plausible sources of a raw
operator that annihilates `X`. Neither does anything: `quad/curv` comes out
1.04, 1.05 and 1.09 for square, wide and tall, and **fp32 agrees with fp64 to
six digits**, so there is no catastrophic cancellation either.

Found (`scripts/probes/real_spectra.py`, run against
`runs/qoc/ngd-pion-lr0.003-s0-235faf3851/checkpoint.pt`): the real `A` is
rank-deficient on **all 56 layers**. `lam_min` is an exact zero everywhere,
true condition number is infinite, and between 21 and 693 eigenvalues per layer
sit below the floor -- on layer 52 (n = 1376) that is half the spectrum. No toy
reproduced this because every toy used a designed spectrum of 6-8 decades with
no zeros.

With `lam = 0` exactly, `curv` receives `g^2 * 0 / lamt^2 = 0` from those
directions while `quad` receives `g^2 / lamt > 0`. The gradient is non-zero
there because `A` and `G` are not sampled from the same tokens: `A` is an EMA
accumulated from **one micro-batch per step** (`train.py`: `recorder.enabled =
micro == 0`) and decayed over past steps, while `G` is the full current batch.
`A` does not know about directions the current batch does have energy in.

**Also**: the logged `cond_A` of 1e5-1e7 is an artefact. `instrument._cond`
discards every eigenvalue below `lam_max * 1e-12` and reports the condition
number of what remains, so it cannot report the rank deficiency by
construction. A day was spent reading a number that hid the phenomenon. Fixing
it is a diagnostic-only change and has not been made yet.

Not shown: that this mechanism sums to the observed 1e32 across all layers.
That needs the real `G`, i.e. a forward/backward, i.e. GPU. Still pending, and
still requires the user's go-ahead.

### Submitted: the floor probe (jobs 252299, 252302)

Two 500-step runs on rtx, `ngd-pion` at `eta = 3e-3` and `eta = 1.0`, into
`$DATA_p330/runs/floor`. About 7 minutes each, roughly 0.23 rtx-hours in total.
Same command as jobs 246701/246702, only the output directory differs, so the
comparison against those is one move on one axis: the diagnostics changed,
nothing else did.

    sbatch -p rtx scripts/sbatch/train.sbatch --optimizer ngd-pion --lr 3e-3 \
           --max-steps 500 --no-resume --out-dir $DATA_p330/runs/floor
    sbatch -p rtx scripts/sbatch/train.sbatch --optimizer ngd-pion --lr 1.0 \
           --max-steps 500 --no-resume --out-dir $DATA_p330/runs/floor

Three questions, each with a stated prediction, written down before the numbers
arrive so that agreeing with them afterwards means something:

1. **Does `floor_share` go to 1 exactly where `quad_over_curv` blows up?** If
   the floor is the mechanism, the `eta = 3e-3` run should show `floor_share`
   near 1 together with a ratio in the 1e30s, and the `eta = 1.0` run should
   show a much smaller share together with its measured 0.13-0.22. If instead
   `floor_share` is small while the ratio is huge, the floor is not the cause
   and the missing-Jacobian hypothesis comes back into play.
2. **Which of `quad` and `curv` degenerates, in absolute terms?** The ratio
   alone cannot say. The prediction is that `quad` stays ordinary and `curv`
   collapses, since `curv` is the one measured against the raw operator.
3. **Does `angle` still reach exactly zero?** The absorbing state in
   `spectral_norm` is gone, so if the zeros return they are a fact about the
   rotation rather than about the estimator. The prediction is that they do
   not return, and that the layer-by-layer spread between steps 450 and 700
   seen in job 246666 does not reappear.

The trust-region question -- whether `curv` should be measured against the
floored operator instead, so that `quad` and `curv` describe the same geometry
-- is deliberately left untouched until these numbers exist. It changes the
trajectory, and there is no reason to change the trajectory and the diagnostics
in the same step.

### Cancelled: 252299 and 252302, on a synchronisation I introduced

Both jobs were cancelled after 21.7 minutes each -- **0.72 rtx-hours spent for
nothing**, and the fault was in the diagnostic patch, not in the cluster.

The reseed added to `spectral_norm` was written as `if bool(dead.any()):`,
which forces a device-to-host synchronisation on every call. It runs five times
per `spectral_norm` at `angle_iters = 2`, twice per layer, across 56 layers:
560 synchronisations a step, against roughly 170 the step already had, and
11 312 on the steps where `angle_warmup = 50` applies. Throughput went from
0.72 s/step to upwards of 24; step 0 completed and step 50 had not arrived
twenty minutes later.

Rewritten branchless -- `torch.where` on both the vector and its norm, using
`sqrt(n)` for the all-ones reseed so no second reduction is needed. Behaviour
is unchanged: the fp32 sweep over `||X||_2` gives the same estimates as before,
correct to `1e-20` where the old code silently inflated `sigma` by eleven
orders at `1e-16`.

The lesson worth keeping: this patch was submitted to the queue having been
checked for correctness on CPU and not at all for cost. Correctness tests
cannot see a synchronisation. Anything added inside `_apply` runs 56 times a
step and needs a throughput check before it needs a GPU-hours budget.

### What the cancelled runs did show, at step 1

Worth recording, because it changes a diagnostic. `blocks.0.attn.wq`, a square
512x512 weight:

    cond_A = 85.7    lam_min_A = 0.256    n_below_floor = 0    null_frac = 0
    floor_share_in = 0.120    floor_share_out = 0.073    lam_ratio = 10000.0

`A` is in excellent condition and has nothing below the floor -- and the floor
is nevertheless fully active, `lam_ratio` being exactly `1/eps`. The reason is
that `build_bases` takes the congruence path whenever `W^T W != I`, and what
`basis_congruence` floors is the spectrum of the pencil
`A^-1/2 (W^T W) A^-1/2`, not the spectrum of `A`.

So `n_below_floor`, added yesterday and measured on `A`, is the wrong matrix
for the in-side: it would have reported this layer as healthy. `_at_floor` now
counts the floor on `basis.lam` directly, and `orthogonal_in` records which
path built the basis. Also worth noting for its own sake: the in-side is on the
congruence path even for square weights, so `W^T W = I` is not holding in
practice, and why is a separate question.

`qoc` at step 0 was 1.0004 to 1.0297 across layers -- the ratio is 1 on a fresh
basis, exactly as the algebra says, so the 1e32 readings develop with staleness
rather than being present from the start.

### Correction: the synchronisation was most of the slowdown, not all of it

The entry above records the device-to-host sync as *the* cause of the 252299
slowdown. With precise timestamps that is too strong, and the arithmetic is
worth writing down because it also rules a suspect out.

    run                      startup   step 0   steps 1-49        s/step
    246666 old code, rtx6002   14.0 s    4.4 s   64.3 s / 50   1.29 -> 0.72
    252299 with the sync        5.4 s   10.2 s  >1291 s / <49          >26
    252848 sync removed         5.2 s    3.6 s   >198 s / <49           >4

Removing the sync bought a factor of six, 26 s/step to 4, so that diagnosis was
right in substance. But 4 s/step is still five times the baseline, and the
remainder is **not** in `spectral_norm`. Step 0 runs `angle_warmup = 50`, which
is 101 `unit()` calls per `spectral_norm` against 5 on an ordinary step --
twenty times the work -- and step 0 came in at 3.6 s against the baseline's
4.4 s. Whatever costs 4 s on steps 1-49 cannot be the function that step 0 does
twenty times more of and stays fast.

Two suspects remain, and one short run separates them:

* **The node.** Every historically fast run was on rtx6002. Both slow attempts
  were on rtx6001, which is also hosting another tenant's vLLM server -- a
  bursty serving workload that can contend for CPU and bus.
* **Allocator fragmentation caused by `_floor_share` itself.** At step 0 it
  transiently allocates `Gb`, `contrib`, `pair` and `denominator`, 7.6 MB each
  on the 1376-side layers, with 82.5 GB of 97.9 already in use and
  `expandable_segments` on. That has exactly the signature observed: step 0
  cheap, everything after it expensive.

Proposed: 60 steps on the idle rtx6004, about 90 seconds and 0.03 rtx-hours.
Fast means the node, and the real pair goes back in with `--exclude=rtx6001`.
Slow means the diagnostics, and the next measurement is the same run with
`diag_every` off.

**The process lesson, which cost more than the finding.** The fix was checked
for correctness on CPU and not at all for cost, then queued: 0.72 rtx-hours on
an unverified theory. My own estimate of the sync's price was tens of
milliseconds per step against an observation of several seconds, and I did not
notice the two-order gap until afterwards. When an explanation and a
measurement disagree by orders of magnitude, the explanation is wrong, and that
check is free.

### The diagnostics are not the slowdown: measured, old against new, on CPU

Both remaining suspects for the residual 5x were mine, so the cheap thing was to
settle it without the cluster at all. `scripts/probes/bench_step.py` builds
`FastNGDPion` over the real 56 weights of LLaMA-60M, feeds random gradients and
times `step()`, running the old checkout and the current tree as separate
processes with different `sys.path` order.

    configuration            diag_every   step 0    steps 1-5 mean
    old 23edea9                     n/a   19.47 s          5.455 s
    HEAD, diagnostics off             0   20.47 s          5.636 s   (+3.3%)
    HEAD, diagnostics every 50       50   19.68 s          5.755 s   (+5.5%)

Everything added -- the `spectral_norm` rewrite, storing `quad` and `curv`,
`_floor_share` on the logging cadence -- costs **five percent**, not five times.

That also kills the allocator-fragmentation hypothesis, and the arithmetic
should have killed it before it was ever proposed: the 112 extra tensors held
across steps are zero-dimensional, so at the allocator's 512-byte minimum block
they occupy about 57 KB of a 97 GB pool. Estimating that magnitude was free and
would have taken one line.

So the residual is not algorithmic, and it is now established twice over and
independently: by step 0 on the GPU, which does twenty times as many `unit()`
calls as an ordinary step and beats the old baseline, and by this comparison on
CPU. What remains is the execution environment. Hardware is not it -- the nodes
are identical, per the user -- but the *state* of a node is not its hardware,
and the leading candidate is page-cache pressure on the C4 windows from a
co-tenant, against an old note in this journal that a cold cache costs about
0.42 s/step.

Submitted job 253041 to measure it rather than argue about it: 60 steps with
`--log-every 5`, giving twelve throughput points instead of one. Steady at 4
s/step means execution; starting slow and accelerating means the cache, and the
real pair simply needs to be left alone to finish.

**On the guard that cancelled 252848.** It demanded step 50 within 150 seconds,
which silently assumes at least 3 s/step, and it had no baseline to compare
against -- a single hard threshold standing in for a measurement. A throughput
trace compared against the known 0.72 s/step is the right shape for this, and
is what 253041 collects.

### Resolved: throughput here is a page-cache property, not a code property

Half a day and about 1.6 rtx-hours went into a slowdown that turned out to have
nothing to do with NGD-Pion. Written up in full because the failure mode will
recur, and because three of my hypotheses along the way were wrong.

**The mechanism.** `harness.data` memmaps the corpus and samples a *random
permutation* of windows, so one step is 512 scattered reads of 512 bytes each
across a 20 GB file -- 262 KB of payload, but 512 separate page faults. When
those pages are resident the step costs 0.8 s; when they are not, the step is
bounded by small-random-read IOPS against shared storage. At 42 s/step the
implied rate is 6 KB/s, which is not a bandwidth number at all: it is roughly a
hundred IOPS, halved again when two jobs compete.

**The evidence, in the order it actually settled things.**

*Paired control, the one that mattered.* Old checkout `23edea9` and current
HEAD, submitted together, same node, same minute, 60 steps each:

    OLD 23edea9   step 5   42.600 s/step   loss 7.4932
    NEW HEAD      step 5   42.540 s/step   loss 7.4932

Identical to within noise, and the losses agree to the last digit, so the two
are computing the same thing at the same speed. Every previous comparison had
been separated in time, and the environment moves by the minute.

*Co-tenancy is what differs.* On rtx6001, which has hosted another user's
`iboa_vllm_qwen7b` since 10:02, every run sat on a flat 5.0-5.7 s/step plateau
and never improved. On idle rtx6004 the same job started at the same 5.78 and
accelerated monotonically:

    step  5   5.780      step 30   1.980      step 50   1.360
    step 10   3.200      step 40   1.580      step 59   1.300  (still falling)

The cache warms only where nothing is evicting it. A 7B inference server
resident on the node keeps the corpus pages from staying.

**Three hypotheses of mine that were wrong, and why.**

1. *The device-to-host sync is the whole cause.* It was real and worth six of
   the thirty-odd times, but not the rest. Recorded as settled before it was.
2. *Allocator fragmentation from `_floor_share`.* The tensors held across steps
   are zero-dimensional: about 57 KB of a 97 GB pool. One line of arithmetic,
   not done.
3. *Re-running warms the cache, so the second run will be fast.* Falsified
   immediately -- 253044 was slow from step 0 -- because on that node the cache
   was being evicted between runs.

The common thread is that each was proposed at a magnitude I never checked
against the observation. Where an explanation predicts milliseconds and the
measurement shows seconds, the explanation is already dead.

**Standing consequences for this project.**

* Pin long runs to a node with no memory-hungry co-tenant, and check
  `squeue -w <node>` before submitting rather than after.
* Budget the first ~100 steps of any fresh run at 2-5x the warm rate. The
  0.72 s/step figure quoted throughout this journal is a *warm-cache* number and
  should be labelled as such wherever it is used to price a run.
* A throughput guard must compare a trace against a known baseline, not test a
  single hard threshold. The guard that cancelled 252848 demanded step 50 inside
  150 s, which silently assumed 3 s/step and killed a run that was merely cold.

## 2026-08-26 -- jobs 253057 / 253058: what `alpha == 1` actually is

500 steps each, `eta = 3e-3` and `eta = 1.0`, on idle rtx6004, 8m50s and 8m35s,
0.27 rtx-hours for the pair. Throughput converged to 0.93 s/step as the cache
warmed. Losses reproduce the earlier runs, so the diagnostics did not perturb
the trajectory.

### Prediction 3 confirmed: the `angle = 0` readings were the estimator

`#angle == 0` is **0 of 56 at every logged step in both runs**, where job 246666
had all 56 layers reading exactly zero from step 700 on. At `eta = 1.0`, step
500, the smallest angle across layers is 1.155e-06. So the absorbing state in
`spectral_norm` was the whole of it, and "the rotation collapses at `eta = 1`"
was an artefact of how it was measured, not a fact about the method.

### Prediction 2 confirmed, and sharper than predicted: `curv` goes *negative*

    eta 3e-3, median over the second half
    depth      quad         curv          quad/curv
        1   4.03e-03    -8.20e-06         4.86e+32
        3   2.19e-03    -3.54e-04         1.64e+35
        4   1.77e-03    -8.89e-04         1.32e+35

with `curv` reaching **-1.336** across steps. Not underflow: a genuine negative,
six hundred times `quad` in magnitude.

This contradicts the proof recorded above that `<X, F(X)> = 4||B^1/2 X C^1/2||_F^2
>= 0`. The proof is correct, and its hypothesis is that `B` and `C` are PSD.

**They are not.** Read straight off the checkpoints without clamping
(`scripts/probes/negative_eigs.py`):

    eta 3e-3 (working):  6253 negative eigenvalues across 56 layers
                         layer 2: 190 of 512 negative
                         layer 3: 119 negative, lam_min = -2.128, lam_max = 446
    eta 1.0  (broken):   2 negative eigenvalues in total

`A` is an EMA of `x x^T`, a convex combination of PSD matrices, so this is
accumulated fp32 error -- summing 131072 rows against a trace far larger than
`lam_max`, compounded by an EMA of horizon 20. The accumulator is otherwise
careful: `exact_fp32` around the gram, and a dtype check that refuses bf16.

**The chain, complete.** `A` loses positivity -> `floor_eigenvalues` clamps at
zero and applies the floor, so the *basis* is sanitised -> `fisher_apply` is
handed the *raw* `A` -> `X` is largest in exactly the floored directions, which
is where the negative eigenvalues sit -> `curv` goes negative ->
`curv.clamp_min(tiny)` turns it into `+1.18e-38` -> `quad/curv = 4e-3/1.18e-38
= 3e35` -> `alpha` clamps to 1.

So `alpha == 1` in the working regime means neither "the basis is fresh" nor
"the curvature is underestimated". It means the curvature came out negative and
was silently replaced by plus-epsilon. **The trust region does not operate at
all in the regime we actually train in**, and the missing-Jacobian hypothesis
is not needed to explain any part of this.

The defect is one asymmetry: the step is built in a geometry that has been
cleaned (clamp, then floor) and judged in one that has not.

### Prediction 1 not confirmed as stated

`floor_share` does not track `quad/curv` step by step. On the out-side it sits
at 0.77-0.80 throughout the trained phase, at steps where the ratio is 1e35 and
equally at steps where it is 0.17. What does track the ratio is the
refactorisation cycle: steps 101, 201, 301, 401 -- one step past each refactor,
`t_fac = 100` -- carry the huge ratios and negative `curv`, while 151, 251, 351,
451 are moderate and positive. So the floor is the standing condition that puts
`X` into the degenerate directions, and the refactor cycle is what decides
whether the sign flips there. Necessary, not sufficient.

### The counterintuitive one, worth a line in the paper

The *working* run has the degenerate covariance and the *diverging* one does
not: `null_frac` 0.24-0.31 against ~0.00, and 6253 negative eigenvalues against
2. At `eta = 1.0` the weights thrash, activations stay large and diverse, and
`A` keeps full rank. At `eta = 3e-3` the model actually learns, activations
collapse onto a low-dimensional manifold, and `A` degenerates. **The better the
optimisation works, the more degenerate its own curvature estimate becomes.**

### My diagnostics hid this twice

`instrument._cond` discarded everything below `lam_max * 1e-12` and reported the
condition number of the rest. Then `_spectrum`, written to replace it, called
`.clamp_min(0.0)` on the same reasoning `floor_eigenvalues` uses -- that `A` is
PSD by construction -- and so reported `lam_min = 0` for every layer while the
truth was thousands of negatives. Both are now fixed; `lam_min` is raw and
`n_negative` and `neg_frac` are logged. A diagnostic that sanitises its input
cannot find a sanitation bug.

### Open, and it is a trajectory decision

`quad` and `curv` must be evaluated against the same operator. Doing that makes
`alpha` mean what its docstring claims -- a staleness readout -- and makes the
trust region live for the first time in this regime, which changes the
trajectory. Not touched pending a decision on which operator is the right one.

### Root cause: the covariance is accumulated in bf16, because autocast outranks `exact_fp32`

The negative eigenvalues are not accumulated rounding. `A` is **stored in
bf16**, and has been for every NGD-Pion run this project has produced. Read
straight off the checkpoint:

    layer 0  dtype torch.bfloat16   count 68157440   beta 0.95

`observe` is called from a `register_forward_pre_hook`, so it executes inside
the `_autocast` block of the forward pass. Autocast intercepts `matmul` by
*operation*: fp32 operands and an enclosing `exact_fp32()` make no difference,
the product is computed in bf16 and **returned as bf16**. The first
`self._matrix = gram` fixes the dtype, `mul_().add_()` preserves it, and the
covariance stays bf16 for the rest of the run. Demonstrated in
`scripts/probes/autocast_bug.py`:

    matmul outside autocast              torch.float32
    matmul inside exact_fp32()           torch.float32
    matmul inside autocast + exact_fp32  torch.bfloat16
    observe() inside autocast            torch.bfloat16

**Why this destroys positive definiteness, and quantising activations does
not.** Rounding the *inputs* is harmless: the gram of bf16 vectors is still a
gram, hence PSD, and measured it actually lifts `lam_min` because the
quantisation noise contributes `E[e e^T] >= 0`. What happens here is different
in kind -- the assembled matrix is rounded entrywise, *after* the outer
product, and that perturbation has no sign constraint.

The magnitude confirms it. Layer 3: largest diagonal entry 12.0, so the bf16
step is `3.9e-3 * 12 = 0.047` per entry, and a random symmetric error of that
size at `n = 512` has spectral norm about `2 * sqrt(512) * 0.047 = 2.1`.
Measured `lam_min = -2.128`. The spectrum shape agrees too: a broad negative
tail, 119 eigenvalues averaging -0.32, not a cluster of plus-or-minus delta at
zero.

And it explains why the *working* run is the degenerate one. Negative
eigenvalues need eigenvalues near zero for the perturbation to push across, and
only a trained model has them -- its activations collapse onto a
low-dimensional subspace. At `eta = 1.0` nothing is learned, `A` keeps full
rank, and there are 2 negatives in total against 6253.

**The module warned about exactly this and guarded the wrong two doors.**
`covariance.py` states that "perturbing `A` at bf16 level produces a step wrong
by three to four orders of magnitude" and that "fp32 is the floor regardless of
what precision the surrounding model trains in". It defends that with an
`_ALLOWED` dtype check and with `exact_fp32` around the gram, and even notes
"the dtype check above would not have caught it" about TF32. Autocast passes
both, because it acts on the operation rather than on the requested dtype or
the TF32 flags, and `self.dtype` is honestly fp32 the whole time -- it is the
*output* of the matmul that disagrees with its operands.

**Consequence for what is already recorded.** The `alpha == 1` account above
stands as a description: `curv` really is negative, the trust region really is
inert. Its root cause changes. It is not the architectural asymmetry between a
sanitised basis and a raw `fisher_apply` -- that asymmetry is real but with an
honest fp32 covariance it operates at the `1e-8` level and is harmless. It is a
dtype bug, and fixing the dtype most likely repairs the trust region for free.

Every NGD-Pion number this project has produced -- the learning-rate sweep, the
`ngd-pion` against `pion_ablated` comparison, the whole `alpha`/`curv` story --
was produced with a bf16 covariance. None of it is safe to quote until the pair
is re-run.

The fix is one line in `observe`, disabling autocast around the gram. It lives
in the reference implementation and it changes the trajectory, so it is not
being made unilaterally.

### Job 253112: the fix works, and immediately exposes `T_fac`

150 steps on rtx6004, 2m23s, 0.04 rtx-hours, against job 253057 at the same
configuration with the bf16 covariance.

                            bf16 covariance      fp32 covariance
    n_negative per layer          97-190                       0
    rows with curv < 0        323 of 504                0 of 224
    curv    (median)          -5.132e-06              +1.086e-02
    quad    (median)           3.434e-03               3.523e-03
    quad/curv (median)          2.72e+32                   0.378
    alpha   (median)              1.0000                  0.3783

`quad` is unchanged. The entire defect was in `curv`, which flipped sign and
grew two thousandfold in magnitude, exactly as the diagnosis predicted.

`alpha` now does what `trust_region_alpha`'s docstring has always claimed:

    step   0   alpha [1.0000, 1.0000]     fresh basis
    step  20   alpha [0.0003, 0.0194]
    step  80   alpha [0.0004, 0.0226]
    step 100   alpha [0.9784, 1.0000]     refactorised, fresh again
    step 120   alpha [0.3617, 0.8374]
    step 140   alpha [0.0903, 0.6796]

One on a fresh basis, decaying with staleness, one again at the refactor. The
trust region is measuring something for the first time in this project.

**What that exposes.** `alpha` reaches 3e-4 within twenty steps of a refactor,
so at `T_fac = 100` the optimizer spends most of its life taking steps three
thousand times smaller than `eta`. **`T_fac`, not `eta`, is now the dominant
hyperparameter**, and every learning-rate conclusion recorded in this journal
was measured with the trust region inert -- including "3e-3 works, 1.0
detonates". Those runs measured something, but not what they were labelled.

**And the honest part.** Train loss at step 149 is 5.7315, against 5.6619 at
step 150 for the bf16 run. With a working trust region the model is, at this
horizon, slightly *worse*. That is not surprising -- the step is now being
clamped where before it was not -- and 150 steps decides nothing. Recorded
rather than smoothed over, because the next thing to do is find out whether it
persists, and a note saying "the fix improved things" would make that question
invisible.

Next, and in this order: re-run the 500-step pair to replace the numbers this
journal currently carries, then sweep `T_fac`, which has never been varied and
is now the parameter that matters.

### Job 253116: the trust region does not rescue `eta = 1.0`, and cannot

Prediction, written before the run: with `alpha` alive, `eta * alpha` would be
normalised and `eta = 1.0` would behave close to `eta = 3e-3`. **Refuted.**

    step |  eta 3e-3 loss   alpha            |  eta 1.0 loss    alpha
       0 |    10.4924  [1.00e+00, 1.00e+00]  |   10.4924  [1.00e+00, 1.00e+00]
      20 |     7.3516  [2.96e-04, 1.94e-02]  |   49.7066  [6.96e-08, 5.36e-03]
      80 |     6.1524  [4.29e-04, 2.26e-02]  |    8.2705  [1.53e-08, 3.26e-03]
     100 |     5.9799  [9.78e-01, 1.00e+00]  |    9.9630  [6.34e-01, 9.40e-01]
     149 |     5.7315  [9.14e-02, 5.84e-01]  |    8.1820  [1.85e-02, 2.47e-01]

The reason is in the `alpha` column and it is structural, not a defect.
**`alpha` is identically 1 on a fresh basis** -- `X = F^-1 G` gives
`curv = <X, F(X)> = <X, G> = quad` as an identity -- so the step immediately
after every refactorisation is `c = eta * 1`. At `eta = 1.0` that is a rotation
of order a radian, the model is wrecked in the first twenty steps (loss 49.7),
and everything after is thrashing in the wreckage. It recurs on schedule: at
step 100 `alpha` is back to 0.63-0.94 and the loss jumps from 8.27 to 9.96.

**The trust region bounds staleness, not step size, and cannot bound step
size.** On a fresh basis the quadratic model is self-consistent by
construction, at any radius whatever. Self-consistency of the model says
nothing about whether the model is valid at that radius. So `eta` remains a
genuine hyperparameter and this mechanism cannot replace it -- which is what
`ALGORITHM.md` says, and now there is a reason for it.

**This is the colleague's H1, arriving from the other side.** His point was
that `tr(D A D^T)` is not the true GGN. The consequence is exactly the above:
the quantity called curvature is the curvature of *our model*, not of the loss,
so it cannot detect that the step has left the region where the model holds.
Recorded earlier in this journal is a claim that H1 was not needed to explain
`alpha == 1` -- that stands, since that was the dtype bug -- but H1 *is* what
explains this, and he was right on the substance.

**One observation, not a conclusion.** With the covariance fixed, the `eta = 1`
peak is *higher* than it was with the corrupted one: 49.7 here against 8.44 at
step 100 for job 246666. The plausible reading is that bf16 noise was
accidentally damping the inverse, making `X` smaller -- which rhymes with the
earlier measurement that `S = I` beats the measured `S` because inverting a
finite-sample covariance amplifies estimation noise. One run at 150 steps
decides nothing; logged so the question stays visible.

### Jobs 253119 / 253120: shortening `T_fac` does nothing, and refactorisation is free

150 steps each at `eta = 0.5`, one at the default `T_fac = 100` and one at 25,
against the two runs already in hand.

    eta 3e-3  T_fac 100    loss 5.7315   alpha[9.1e-02,5.8e-01]  angle_max 5.69e-02  0.859 s/step
    eta 0.5   T_fac 100    loss 6.7215   alpha[1.0e-02,1.3e-01]  angle_max 8.54e-03  0.858
    eta 0.5   T_fac  25    loss 6.7880   alpha[2.7e-02,1.5e-01]  angle_max 2.15e-02  0.854
    eta 1.0   T_fac 100    loss 8.1820   alpha[1.8e-02,2.5e-01]  angle_max 7.26e-03  0.858

`T_fac = 25` is a hair *worse*, 6.7880 against 6.7215. The two effects cancel,
as predicted before the run: the preconditioner really is more current --
`alpha` holds at 7e-3 to 0.51 mid-cycle instead of 2e-7 to 1e-2, fifty times
higher -- and it buys nothing, because the same change quadruples how often the
unprotected fresh-basis step fires. The two runs agree to the digit at step 20
(15.4499), before the first `T_fac = 25` refactor, and split at step 40 where
the short-cycle arm is worse.

Two things worth keeping.

**Refactorisation is free.** 0.854 s/step against 0.858 for four times as many
`eigh` calls. `T_fac` is not cost-constrained; it is simply not much of a lever.

**The best arm takes the largest rotations.** `angle_max` 5.69e-02 at
`eta = 3e-3` against 7.26e-03 at `eta = 1.0`. A larger `eta` produces a
*smaller* actual rotation, because `alpha` crushes it harder. Step size is
`eta * alpha` and that product is not monotone in `eta`, so `eta` alone cannot
be read as step size anywhere in this project's records.

## 2026-08-26 -- the learning-rate sweep never varied only `eta`

Jobs 253124/253125/253126 capped the rotation at 0.1 rad. The control behaved
exactly as predicted -- `eta = 3e-3` went 5.7315 to 5.7320, the cap never
binding at a measured `angle_max` of 0.057 -- and the other two did not:

    eta 0.5   cap off 6.7215   cap 0.1  7.1253    worse
    eta 1.0   cap off 8.1820   cap 0.1  7.7970    marginally better

The number that matters is neither of those. At `eta = 1.0` the cap cuts the
first step from 5.66 rad to 0.1, a factor of 57, and the loss at step 20 is
**46.81 against 49.71**. Cutting the rotation fifty-sevenfold changed almost
nothing, so the blow-up was never the rotation.

**One learning rate drives both optimizers.**

    _adamw(groups, lr=cfg.lr, ...)
    lr = lr_at(step, cfg)
    for opt in (rot, adamw): ... group["lr"] = lr

`--lr 1.0` therefore runs **AdamW at 1.0** on the embedding, the output head,
the norm gains and the biases. That detonates on its own and says nothing
whatever about NGD-Pion.

### What this invalidates

* **The whole `eta` sweep.** 1e-3, 3e-3, 0.5, 1.0 varied AdamW's learning rate
  in lockstep. The two arms that "diverged" had AdamW at 0.5 and 1.0. The
  conclusion "3e-3 works, 1.0 detonates" is a statement about AdamW.
* **The cold-Fisher warmup experiment**, which found warmup made `eta = 1`
  worse and moved the blow-up to where warmup ended. Same contamination.
* **Today's angle-cap test**, for the two large-`eta` arms. The cap can only
  bound the rotation of the weights Pion owns; the damage was in AdamW's
  parameters, which it cannot touch. The `eta = 3e-3` control stands.
* **`pion_ablated`'s optimum**, which was never explorable: raising `lr` to
  find it would have blown up AdamW first. Its toy optimum was 2.2e+01.

### What survives, and the distinction matters

The algebra does. `alpha` is identically 1 on a fresh basis because
`curv = <X, F(X)> = <X, G> = quad` is an identity, so it measures staleness and
cannot measure step size. That is a proof, and it does not depend on any run.

What does not survive is the *demonstration* I attached to it. I used the
`eta = 1.0` blow-up as evidence that the missing bound has teeth, and that
evidence was contaminated. The gap is still real; its consequences are now
unmeasured.

### The fix, and what has to be re-run

Add `adamw_lr` to `RunConfig`, defaulting to `0.0` meaning "follow `lr`", so
the anchor and every existing configuration reproduce bit-for-bit and only an
explicit setting decouples the two.

Then the sweep that has never actually been run: `eta` varied with AdamW pinned
at a sane, fixed value. Only after that does any statement about NGD-Pion's
learning rate mean anything -- including the angle cap, whose test needs
repeating on an uncontaminated arm.

### Job 253129: `eta = 1.0` does not diverge, and the rotation reaches 2.31 rad

The first run in this project where the rotational learning rate is varied and
AdamW is not. 150 steps, `--lr 1.0 --adamw-lr 1e-3`:

    eta 1,    adamw 1e-3   10.4924  7.2797  7.2213  7.0416  6.7534  6.4959  6.1776  6.0299  6.0241
    eta 1,    adamw 1.0    10.4924 49.7066  9.0467  9.5823  8.2705  9.9630  8.7585  8.9981  8.1820
    eta 3e-3, adamw 3e-3   10.4924  7.3516  7.1256  6.5301  6.1524  5.9799  5.7922  5.7020  5.7315

Monotone descent, no spike anywhere. **"NGD-Pion detonates at large `eta`" was
AdamW throughout**, and every conclusion drawn from that -- the cold-Fisher
warmup experiment, the trust-region urgency, the angle cap's motivation --
rested on it.

And the number this project exists for:

    eta 1,    adamw 1e-3:   angle up to 2.31 rad,  alpha [4.6e-04, 7.3e-01]
    eta 3e-3, adamw 3e-3:   angle up to 0.0569 rad

**A 2.31 radian rotation with the loss descending monotonically**, forty times
the largest rotation the previous best arm ever took. `spectral_norm` returns a
Rayleigh quotient, so the true angle is at least that. This is the first direct
measurement of the claim the method is built on: `F^-1` preconditioning makes a
large step *survivable*, not merely representable. `alpha` spans three orders
across layers meanwhile, so the trust region is working rather than saturated.

**The loss comparison is still confounded and is not being made.** The
`eta = 3e-3` arm ran AdamW at 3e-3, this one at 1e-3. On a 150-step horizon the
embedding and head are most of what moves, so a threefold difference in their
learning rate explains 6.0241 against 5.7315 at least as well as `eta` does.
That is yesterday's mistake mirrored, and job 253132 -- `eta = 3e-3` with AdamW
pinned at 1e-3 -- is the control that makes the two arms differ in one thing.

Note also that the ceiling is unknown again. `eta = 1` was the largest rate ever
tried, it is comfortable, and nothing here says where the limit is.

### Jobs 253132 / 253133: the first honest `eta` curve, and it reverses the ranking

AdamW pinned at 1e-3 in all three, `eta` the only variable, 150 steps:

    eta        0       20       40       60       80      100      120      140      149   angle_max
    3e-3  10.4924   7.4204   7.2563   7.1482   6.9389   6.6273   6.3145   6.1570   6.1498   4.39e-02
    0.5   10.4924   7.2726   7.2315   7.0554   6.7198   6.4615   6.1709   6.0116   6.0190   8.39e-01
    1.0   10.4924   7.2797   7.2213   7.0416   6.7534   6.4959   6.1776   6.0299   6.0241   2.31e+00

**`eta = 3e-3` is the worst of the three.** 0.5 and 1.0 are indistinguishable
from each other and both beat it by about 0.13. The optimum this project has
used since the beginning was AdamW's optimum, found honestly and attributed to
the wrong optimizer.

The size of that confound, measured directly: the same `eta = 3e-3` arm gives
5.7315 with AdamW at 3e-3 and 6.1498 with AdamW at 1e-3. A threefold change in
AdamW's rate moves the 150-step loss by 0.42, three times what a
three-hundredfold change in `eta` moves it. At this horizon the embedding and
head are most of what is learning.

The angle grows with `eta` but sublinearly -- 0.044, 0.84, 2.31 rad for a
`1 : 167 : 333` ratio of rates -- so `alpha` compensates partially and not
nearly fully.

(An earlier reading of "the angle at `eta = 0.5` is two orders smaller" was a
mid-run point at step 40 misread as a final. There is no non-monotonicity.)

### What cannot be concluded, and why that is a hole in the method

**Seed noise has never been measured in this project.** The gap between 0.5 and
1.0 is 0.005 and means nothing. Whether the 0.13 over `eta = 3e-3` means
anything is also unknown. Every comparison from here needs a noise floor and
there isn't one.

**150 steps is very early.** The loss is 6.0 against the anchor's 3.36 at
73 242 steps. Rankings at this horizon invert routinely.

The defensible statement is narrow: **the optimum `eta` is not below 0.5, and
3e-3 was AdamW's optimum.** Everything finer is below the resolution of what
has been run.

Next, in this order: one repeat seed at `eta = 1.0` to price a comparison, then
upward -- 3, 10, 30 -- because a plateau at 0.5 to 1.0 says the ceiling is
higher than anything tried and it has never been seen.

### Jobs 253136 / 253137: `T_fac` is a lever after all, and `alpha` is damage control

The earlier `T_fac` comparison ran at `eta = 0.5` with AdamW driven to 0.5 as
well, so it was contaminated like everything else. Repeated at `eta = 1.0` with
AdamW pinned at 1e-3:

    T_fac      0       20       40       60       80      100      120      140      149
    10    10.4924  7.2790  7.0708  6.5453  6.2778  6.1056  5.9224  5.8455  5.8633
    25    10.4924  7.2797  7.0643  6.6073  6.3338  6.1525  5.9690  5.8643  5.8773
    100   10.4924  7.2797  7.2213  7.0416  6.7534  6.4959  6.1776  6.0299  6.0241

Shorter is better, by 0.16 from `T_fac = 100` to 10 -- larger than the whole
effect of `eta` across three orders of magnitude (0.13). The earlier "not much
of a lever" was an artefact.

**The saw-tooth, read straight off the instrument.** `alpha_max` over time:

    T_fac  10:  1.000  1.000  1.000  0.999  1.000  1.000  1.000  1.000  1.000
    T_fac  25:  1.000  0.515  0.449  0.662  0.929  1.000  0.916  0.992  0.966
    T_fac 100:  1.000  0.515  0.252  0.118  0.058  1.000  0.820  0.775  0.726

At `T_fac = 100` it decays to 0.058 by step 80 and snaps back to exactly 1.000
at step 100. That is the refactorisation, visible as a reset. The staleness
feedback loop proposed earlier today is real and this is its characteristic.

**But the reading of it was wrong.** At `T_fac = 10`, `alpha` sits at 1.000
throughout -- the loop never engages at all -- and that is the *best* arm. So
`alpha` is not a safety mechanism protecting large steps; it is **damage
control for an out-of-date preconditioner**. Remove the staleness and the
compensation is not needed. The correct move is to keep the basis current, not
to lean on the regulator.

Against intuition too: `T_fac = 100` takes a *larger* maximum angle than
`T_fac = 25` (2.31 rad against 1.71) and is the worse arm. Angle magnitude on
its own does not predict quality; agreement between the rotation and the
current geometry does.

**Cost is negligible.** 142 s at `T_fac = 100`, 148 s at 10 -- four percent for
ten times the `eigh` calls. Which makes the limit worth testing: at
`T_fac = 1`, `alpha` is identically 1 forever, the trust region ceases to exist
as a concept, and the method becomes natural gradient with a fully current
Fisher and `eta` as its only parameter. If that is also the best arm, the
paper's statement is that the method works best with its own safeguard
inactive. Jobs 253138/253139/253140 test `T_fac` in {1, 3, 5}.

### Jobs 253138-253140: the full `T_fac` sweep, and why "shorter is better" is the wrong reading

`eta = 1.0`, AdamW 1e-3, 150 steps:

    T_fac    loss@149   alpha_max   angle_max   s/step   vs T=100
        1     5.8287       1.000       9.845     1.583     -0.195
        3     5.8349       1.000       3.908     1.077     -0.189
        5     5.8613       0.998       3.422     1.010     -0.163
       10     5.8633       1.000       3.105     0.901     -0.161
       25     5.8773       0.966       1.708     0.884     -0.147
      100     6.0241       0.726       2.307     0.856      0.000

Monotone, but the gain saturates immediately: **100 to 25 is worth 0.147, and
the whole remaining journey from 25 down to 1 is worth 0.049** spread over five
points. Meanwhile the cost climbs steeply below `T_fac = 10`: 0.884 s/step at
25 against 1.583 at 1, **seventy-nine percent more**.

**Equal steps is the wrong frame.** In the 237 s that `T_fac = 1` needs for 150
steps, `T_fac = 25` completes 268, and the loss is still falling steeply at
that horizon. On a wall-clock budget the short cycle very likely loses.

Confirmed firmly, on five points rather than by argument: `alpha_max` sits at
1.000 across the entire trace for every `T_fac <= 10`. **With a current basis
the trust region never engages at all.** And `T_fac = 1` reaches an angle of
9.85 rad with the loss descending monotonically -- allowing for `Cayley`, that
is roughly 2.8 rad in the leading plane, an enormous rotation that is
nevertheless stable.

**What cannot be said.** The spread across `T_fac` 1, 3, 5, 10 and 25 is 0.049
over five points, and seed noise has still never been measured. It is entirely
possible that the honest result is "100 is bad, everything else is the same".
This is the third comparison tonight to run into the missing noise floor;
proceeding further without one means fitting parameters to fluctuations.

Jobs 253146-253148 measure it: seeds 1, 2, 3 at `eta = 1.0, T_fac = 25`, joining
seed 0 from job 253137.

### Jobs 253146-253148: the noise floor, and what it retracts

Four seeds at `eta = 1.0`, `T_fac = 25`, AdamW 1e-3, 150 steps:

    seed 0   5.8773
    seed 1   5.8455
    seed 2   5.8353
    seed 3   5.8198

    n = 4    mean 5.8444    sd 0.0243    range 0.0575

Applied to tonight's comparisons:

    T_fac 100 -> 25     0.1468     6.0 sd     real
    eta 3e-3 -> 1.0     0.1257     5.2 sd     real
    T_fac  25 -> 1      0.0486     2.0 sd     not resolvable
    eta 0.5 vs 1.0      0.0051     0.2 sd     noise

**The two headline results hold with room to spare.** `T_fac = 100` is
genuinely bad and `eta >= 0.5` genuinely beats `eta = 3e-3`, at five and six
standard deviations.

**"Shorter `T_fac` is better" was me fitting noise.** `T_fac = 1` scored 5.8287
against a four-seed mean of 5.8444 at `T_fac = 25` -- a difference of 0.016,
0.7 sd, nothing -- while costing 79% more per step. And the luck ran the wrong
way: seed 0, the one that happened to be in the sweep, was the *worst* of the
four at 5.8773, which inflated the apparent gain of the short cycle. The
recommendation is `T_fac ~ 25`; everything below it is paid for and not
received.

**The resolution of the standard experiment, worth keeping.** With
`sd = 0.024`, a two-sample comparison at reasonable power needs about
`n = 15.7 sigma^2 / delta^2`:

    to resolve 0.05    4 seeds per arm
    to resolve 0.02   23 seeds per arm

So **a single 150-step run resolves about 0.05 in loss and no better**. Every
grid from here should be planned against that number, and any difference below
it reported as unresolved rather than ranked. Three comparisons tonight were
made before this existed and two of them survived; the third did not.

## 2026-08-26 evening -- the headline result, and what the literature already says

### `ngd-pion` wins in a regime the baseline cannot enter

`pion_ablated` swept over `eta` with AdamW pinned at 1e-3, against `ngd-pion`
at the same settings, 150 steps:

    eta       ngd-pion   pion_ablated
    1e-3            --         6.1729
    3e-3        6.1498             --
    0.5         6.0190         6.1143
    1           6.0241             --
    3         * 5.9113 *        7.1657
    10          5.9818        11.5893
    30              --        10.0755

`pion_ablated` tops out below `eta = 3`. `ngd-pion`'s optimum *is* near 3, and
it is still healthy at 10 where the baseline is destroyed. Best against best is
5.9113 versus 6.1143, a difference of 0.203 -- **6 sd** against the measured
noise floor.

That is the claim the paper needs, and it is qualitative rather than
quantitative: not "we took a bigger step", but "there is a regime the
unpreconditioned method cannot enter at all".

Note also what `pion_ablated` does when it breaks: it *thrashes* -- 9.95, 7.52,
10.95, 8.08, 11.59 -- rather than diverging monotonically. Its retraction is
Cayley, exactly orthogonal, so the spectrum of its weights cannot explode. What
fails is coherence, not scale.

### Spectrum drift under AdamW: the premise for excluding the head is half right

2000 steps of pure AdamW, singular values against a reproduced initialisation
(`scripts/probes/spectrum_drift.py`):

                          ||ds||/||s||   shape drift
    Pion-owned (n=56)           0.889         0.421
    excluded   (n= 2)           1.653         1.479

The head and the embedding do move about twice as far, so excluding them is
justified -- but the inner layers move a great deal too: 89% relative change,
and 42% of it *after* removing the best uniform rescale. "The inner spectra
barely move" is not true.

What rescues it: the best uniform rescale is 1.4-1.9 for inner layers, so most
of the drift is uniform stretching -- and a spectrum-preserving optimizer can
get that indirectly, through the RMSNorm gains, which belong to AdamW.
Stretching a weight and stretching the norm after it are the same thing. Only
the 0.42 of *shape* is genuinely unavailable. That is the real price of
freezing the spectrum, and it is a third of what the first column suggests.

### Heavy-tailed initialisation: one wrong experiment, then one inconclusive one

First attempt matched the Frobenius norm of the `normal(0, init_std)` it
replaced. That is the wrong invariant, and wrong in a way that looks like a
result: holding `||W||_F` fixed, a spikier spectrum puts more of it into the
leading singular value, so the operator norm grows -- 3.9x above the
feature-learning condition at `alpha = 2`, 9.8x at `alpha = 1.25`. The measured
losses ranked monotonically in that violation. It read as "heavy tails hurt"
and was "the scale was wrong". **That sweep tested nothing about tails.**

Rewritten to set the spectral norm to `sqrt(fan_out/fan_in)` per Yang, Simon and
Bernstein, so `alpha` varies the shape and nothing else. The second sweep is
inconclusive for a different reason: `angle_max` across the arms ranges from
0.55 to 170 radians, so the seven initialisations are in seven different
regimes and comparing them at a single `eta` measures which one happens to suit
`eta = 1`. The same trap `sweep.sbatch` warns about, walked into twice in one
evening. Doing it properly needs a per-initialisation `eta` sweep.

`orthogonal_in` stayed 0/224 even at `alpha = inf`, which was a prediction and
it failed. Two reasons, both fixable:

* `is_identity` uses `atol = 1e-6`, and a flat-spectrum 512x512 weight built
  through fp32 QR gives `max|W^T W - I| = 1.073e-06`. **We miss the cheap basis
  path by seven percent.** The tolerance is an absolute constant where the error
  grows with dimension.
* For non-square layers a flat spectrum gives `W^T W = (fan_out/fan_in) I` --
  proportional to the identity, not equal to it. That case is trivial to
  support: `C = c I` makes `F(X) = 2c(A X + X A)`, so the scalar just multiplies
  the operator.

Together those would put *every* layer on `basis_identity_anchor` under a flat
initialisation, and Cayley would keep it there for the whole run: no
`B^{-1/2}`, no ill-conditioned `P`, and one place where `eps` acts instead of
three.

### What the literature says, and where it leaves us

* **Scale.** `||W||_* = Theta(sqrt(fan_out/fan_in))` for feature learning
  (arXiv:2310.17813), contrasted explicitly with Frobenius or entrywise
  scalings. Our `normal(0, 0.02)` sits at 0.91 of it for square layers, 1.96 for
  the down-projections and 0.51 for the embedding and head.
* **Shape at initialisation: flat.** Dynamical isometry wants every singular
  value of the Jacobian near 1, achieved by orthogonal initialisation and not
  achievable by Gaussian (arXiv:1711.04735).
* **Shape when trained: heavy-tailed.** HT-SR, with the power-law exponent in
  roughly (2, 2.5) for good generalisation.
* A normal optimizer has no conflict here: start flat, let the tail emerge.
  **A spectrum-preserving optimizer has to choose one**, and that bind is not
  addressed anywhere in that literature because it does not consider such
  optimizers.

**Prior art we have to answer.** HTMuon (arXiv:2603.10067) argues that Muon's
orthogonalised update "suppresses the emergence of heavy-tailed weight spectra
and over-emphasises training along noise-dominated directions", and gains up to
0.98 perplexity on LLaMA/C4 by inducing heavier tails. That critique applies to
us **more** strongly than to Muon: Muon sets the singular values of the *update*
to one, we freeze the singular values of the *weight* forever. Their remedy --
change the update's spectrum -- is closed to us, because our update is a
rotation. Initialisation and a learned diagonal are the only routes left, which
turns both from decorations into answers to a published objection.

**And the reconciliation.** "Small Singular Values Matter" (arXiv:2410.17770)
finds that large singular directions align with the activation covariance
within about 1000 updates while the smallest gain overlap only late in
training. So HTMuon is right early and the small directions matter late, which
also explains the split in the pruning literature. For us that lands somewhere
specific: our floor gives the degenerate directions a substantial step from
step zero, and 55-69% of the out-side predicted decrease comes from them.

Worth putting in the introduction: the quantity that develops during training is
the **alignment between a weight's singular vectors and the eigenvectors of the
activation covariance**. That is exactly what this method manipulates
deliberately -- it rotates the singular vectors, preconditioned by an operator
built from `A`. Not "converges faster" but "performs explicitly the alignment
ordinary training finds by itself".

### The effective rank of `A`, and a correction to my own story

The suggestion was that the model should be sized to the effective
dimensionality of the data -- the non-kernel part of `A`. Measured from the
long runs' own diagnostics, at `eta = 1.0`:

    step   null_frac   lam_min    cond_A   floor_in  floor_out
       1      0.0000  5.19e-02  1.33e+03      0.086      0.224
    1001      0.0000  1.60e-02  7.63e+03      0.160      0.585
    2001      0.0000  2.06e-02  3.32e+03      0.118      0.556
    3451      0.0000  2.88e-02  2.71e+03      0.107      0.549

**`A` is full rank throughout, and `cond(A)` is falling** -- 7.6e3 down to
2.7e3. The activations get *better* conditioned as training proceeds.

This retracts a story told repeatedly in this journal today: that training
collapses activations onto a low-dimensional manifold, degenerating `A`, and
that this is the root of the damping trouble. **That was measured at
`eta = 3e-3`**, the operating point we abandoned this afternoon. There
`null_frac` was 0.33 with 6253 negative eigenvalues; at `eta >= 0.5` there are
none. The degeneracy looks like a symptom of training too slowly, not a
property of the data.

So on the question as asked: by this measure the model is **not** wider than the
data supports.

**But `floor_share_out` stays at 0.55-0.63 while `A` is healthy**, because the
out-side degeneracy is not about data at all. The pair is `(I, W A W^T)` and
`rank(W A W^T) <= min(m, n)` arithmetically, whatever the data does. A
1376x512 layer has an exact 864-dimensional kernel permanently. The binding
constraint is the **aspect ratio of the layer**, not the dimensionality of the
data.

Which answers the question this journal has been circling all day -- floor the
kernel or suppress it? Neither. **It should not be there.** The out-side
rotation lives in `so(m)` of dimension `m(m-1)/2`, while the operator has rank
`min(m, n)`; the meaningful problem is `so(n)` embedded in `so(m)` and the rest
is not an ill-conditioned region needing damping, it is not part of the problem.
We created it by solving in a space the problem does not occupy.

That also closes the complexity gap from this morning: working in the range of
`W` costs `O(m^2 n)` against the present `O(m^3)`, which is Pion's order. It is
not an optimisation, it is the correct formulation, and the saving comes from
no longer solving in a dimension the problem does not have.

## 2026-08-27 -- the full-length result, and why concurrent runs must share a seed

### The headline comparison, and it is negative

`ngd-pion` at `eta = 1.0`, `T_fac = 25`, AdamW pinned at 1e-3, run to the full
73 242 steps against the six `pion` runs already on disk:

    optimizer   lr       steps    train      val   card  momentum
    ngd-pion    1.0      73241   3.6452   3.6728   rtx   none
    pion        1e-3     73241   3.3516   3.3719   rtx   lie
    pion        1e-3     73241   3.3649   3.3866   rtx   lie

**We lose by 0.30 in validation.** The published gap between Pion's own two
arms is 0.0079, so this is forty times that.

The `eta = 0.5` arm **crashed** at step 41 500 with
`linalg.eigh: the algorithm failed to converge because the input matrix is
ill-conditioned`, inside `basis_congruence` -- a robustness bug in the
reference, in exactly the congruence path a flat initialisation would remove.

**The comparison does not isolate anything.** `pion` carries
`momentum = "lie"`, `scaling = "rms"` and a truncated retraction; NGD-Pion has
none of the three. The isolating baseline is `pion_ablated`, and **no
full-length run of it exists**. At 150 steps `ngd-pion` beat `pion_ablated` by
6 sd; at 73 242 steps we have nothing to compare against. Momentum in
particular is not a confound to argue away, it is a missing feature.

Also: the 150-step optimum was `eta = 3` and the long runs were launched at
1.0 and 0.5.

### The defect the run exposes: no scale calibration between layers

Per-layer rotation angle **within a single step**:

    step        min      median         max     max/min   widest layer
       1   1.55e-04    4.70e-03    5.66e+00     36 496   blocks.0.attn.wo
    9151   4.29e-04    1.07e-02    2.00e+00      4 658   blocks.0.attn.wo
   18301   3.03e-04    9.74e-03    2.13e+00      7 026   blocks.0.attn.wo
   54951   4.86e-05    1.61e-03    7.92e-01     16 305   blocks.2.attn.wo

Four and a half orders of magnitude, sustained for the whole run, with one
layer consistently at the top. The Fisher is block-diagonal per layer: it
equalises curvature *inside* a block and says nothing about scale *between*
blocks, and a single scalar `eta` is applied to all of them. Pion has
`pion_scaling = "rms"` for precisely this. We have nothing.

The gap against `pion` is also **constant** -- 0.28 at step 500 and 0.28 at
25 000 -- which is the shape of a systematic handicap rather than an
accumulating deficit.

### `S = I` as the suspected cause, with an order-of-magnitude argument

    with the true S = W^T E[dd^T] W :   ||X|| ~ 1 / (||W|| ||delta|| ||x||)
    with S = I  (so S = W^T W)      :   ||X|| ~ ||delta|| / (||W|| ||x||)

The two differ by `||delta||^2` per layer. With the true `S` the step is
*inversely* proportional to the backward signal, which is the natural-gradient
normalisation; with `S = I` it is *directly* proportional, so layers with a
strong backward signal are given a wider step as well. A 70x to 200x spread in
`||delta||` across depth squares to 5e3-4e4, which is the range measured.

So `S = I` may not be the mild simplification this journal has treated it as.
`BackwardProbe` (measurement only, in `harness`) records `||delta||` per layer
to test whether the angle spread really is the square of the `||delta||`
spread.

### Concurrent runs must share a seed

Two probe runs differing **only in seed** were launched together on one node and
crawled. The data loader samples a seed-dependent random permutation of corpus
windows, so different seeds are two independent streams of small random reads
that halve the IOPS and evict each other's pages:

    both seeds running     18.66 s/step
    seed 1 cancelled        4.43 s/step

Same seed is the opposite: the jobs request identical windows in identical
order and warm the cache for each other, which is why the two long runs
yesterday held 0.85 s/step side by side.

**A seed sweep must therefore be run sequentially or on separate nodes.** Only
sweeps that hold the seed fixed are safe to fan out.

**And a correction.** The first attempt at this probe was cancelled after step 0
took 42 s against a normal 3.6, and that was attributed here to
`register_full_backward_hook`. Most of it was the second seed. The hook was
rewritten as a forward hook attaching `output.register_hook`, gated to logged
steps only, which is the better implementation regardless -- it delivers the
same `dL/d(output)` without materialising `grad_input` -- but the factor of
twelve was not its doing. That is twice now that a slowdown of mine was really
another job of mine on the same node.

### `S = I` confirmed as the source of the per-layer scale error, with a correction

`BackwardProbe` over 150 steps at `eta = 1.0`:

    step   spread ||d||   spread angle   corr(log angle, log ||d||)
       1         1169          36 496                        0.951
      61         1308           2 236                        0.930
     121         1147             963                        0.920
     150         1205           1 689                        0.923

**Correlation 0.92 to 0.98.** The per-layer step scale tracks the backward
signal almost exactly, which is what `S = I` predicts.

**But the prediction was stated wrongly here yesterday.** The entry above says
the angle spread should be the *square* of the `||delta||` spread. It should
not. Under `S = I` the step goes as `||delta||`, so the observable is the
spread in `||delta||` itself -- 1147 to 1308 against a measured angle spread of
963 to 2962 at settled steps, the same order. The `||delta||^2` is the
difference *between* `S = I` and the true `S`, not something visible in a run
of either.

**And the conclusion needs correcting too.** Restoring `S` does not *equalise*
the layers, it **inverts** the dependence:

    S = I        ->  ||X|| ~  ||delta||
    true S       ->  ||X|| ~ 1/||delta||

Which is correct, and not the same as equal. The natural gradient is not
supposed to give every block the same step; it is supposed to give each the
right one. Curvature goes as `delta^2`, so the right step goes as
`G/F ~ 1/(||delta|| ||x|| ||W||)` -- **larger where the signal is weaker**. We
currently do the opposite, handing the widest step to the layers whose signal
is already strongest. The defect is not that the spread is large but that its
sign is reversed, and the size of the error is `||delta||^2`.

That also explains the constant 0.28 gap against `pion` from step 500 to
25 000: a systematically misallocated step across blocks is a fixed handicap,
neither accumulating nor healing.

**Which layers.** The widest are the output projections -- `attn.wo` and
`ffn.down`; the narrowest are `attn.wq` and `attn.wk`:

    layer                  angle   ||delta||   lam_max(A)
    blocks.0.attn.wo    1.71e+00    3.81e-06         3.58
    blocks.1.ffn.down   1.42e+00    6.16e-07         6.08
    blocks.4.attn.wq    1.35e-03    5.71e-09       189.6
    blocks.6.attn.wk    1.19e-03    3.48e-09       191.5

Two factors, both pulling the same way. `angle ~ ||delta|| / sqrt(lam_max(A))`,
and the output projections have both a larger backward signal and a much
smaller input covariance -- their inputs are the attention output and the
SwiGLU output, where `lam_max` is 4 to 6, against 190 for the normalised
residual stream that feeds `wq` and `wk`. Predicted ratio 4900 against a
measured 1269: same order. So the activation scale contributes as well as
`||delta||`, though `||delta||` dominates at a correlation of 0.92.

### What measuring `S` costs, after six attempts to measure it

**+16% end to end.** Almost all of it is the `E[dd^T]` accumulation.

    per refactorisation        S = I  763 ms      S measured  968 ms
    per step at T_fac = 25
      opt.step()                      243.9 ms                242.5 ms
      A accumulation                  106.2                   106.4
      D accumulation                    0.0                   154.9
      refactorisation, amortised       30.5                    38.7
      total                           380.6 ms                542.5 ms   x1.43

Cross-checked end to end, two runs strictly sequential on one node: **1.200
against 1.397 s/step, x1.16**. The two agree -- the optimizer costs 43% more
and is roughly a third of a step, so the step costs 16% more.

`opt.step()` itself is unchanged (243.9 against 242.5). The extra congruence on
the out-side shows up only in refactorisation, at +205 ms a time, which is
+8 ms/step once amortised. That also reconciles with yesterday's end-to-end
`T_fac` comparison: 25 against 100 measured +28 ms/step there, and this probe
predicts +23.

**The x5 that started this was one line.** `observe_backward` passed
`delta * n` to get the second moment of `n * delta`, which copies a
`tokens x d` tensor to scale a `d x d` result -- 721 MB per wide layer per
step, sixteen times a step. Passing `scale` to the accumulator instead removed
the entire overhead: peak memory went from a measurable gap back to 83.31
against 83.15 GB.

### Six attempts, and why each measured the wrong thing

Worth recording as a method rather than as a story.

1. Two concurrent jobs, cold node: both 1.65 s/step. Both waiting on the same
   pages; the difference sat under an I/O ceiling.
2. Two concurrent, warm node: both 1.26. Same ceiling, lower.
3. Two concurrent again after more warming: 1.263 against 1.260. Same.
4. One job alone on rtx6002: 1.29 and still falling at step 120 -- the cache
   never finished warming inside 150 steps. And the 0.85 s/step baseline this
   was being compared against had been measured on **rtx6004**: 1.7x between
   two nodes with no co-tenant on either.
5. Micro-benchmark at `t_fac = 100`: 243.9 against 242.5, read as
   "refactorisation is cheap".
6. Micro-benchmark at `t_fac = 25`: 244.3 against 242.9, read as confirmation.

Attempts 5 and 6 measured the same thing, and neither measured refactorisation
at all: the timed window is 3 warmup plus 10 iterations, so at any `t_fac >= 13`
**it never fires**. "244 against 244" did not mean cheap, it meant zero
occurrences. Forcing `t_fac = 1` gave the real figure at last.

Two rules fall out, and both are about the setup rather than the numbers:

* **End-to-end `s/step` is not a portable measure of algorithm cost on this
  cluster.** It is bound by page-cache state more often than by arithmetic, and
  varies 1.7x between nodes doing identical work. Use a micro-benchmark with
  `torch.cuda.synchronize()` for anything that is meant to be a property of the
  method; keep `s/step` for planning wall clock.
* **Check that the event being timed occurs inside the timing window.** Thirteen
  steps at a period of twenty-five is not "cheap", it is "never".
