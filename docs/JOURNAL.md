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
