# Running on Aphrodite

The Cyprus Institute HPC facility. Project **p330**; every submission needs
`-A p330` or it is rejected. `$DATA_p330` points at the project's data
directory, and `data_p330` in `$HOME` is a symlink to the same place.

Allocation: **2000 RTX hours and 1000 B200 hours.**

## Where things live

| | |
|---|---|
| repository | `$DATA_p330/Natural-Pion` |
| container | `$DATA_p330/containers/ngd-pion.sif` |
| corpus | `$DATA_p330/c4/c4_{train,val}.bin` -- and `c4/downloads/`, the source files, kept so a rebuild fetches nothing |
| run outputs | `$DATA_p330/runs/<name>/` |
| SLURM logs | `logs/`, **in the directory you submit from**, and not in git |
| retired files | `$DATA_p330/attic/<date>/`, with `attic/register.tsv` saying where each came from |

Nothing here is deleted. `scripts/retire.sh <path>...` moves things to the
attic instead, which within one filesystem is a rename and so costs neither
time nor space, and the register makes a directory of anonymous files
answerable. `rm` is for what the attic itself outgrows.

The submission scripts, in the order `CLUSTER.md` uses them:

| | |
|---|---|
| `scripts/sbatch/preflight.sbatch` | step 0 -- the operations, then resume-on-a-card and spectrum-under-TF32 |
| `scripts/sbatch/data.sbatch` | step 3 -- the corpus, sharded across cores |
| `scripts/sbatch/throughput.sbatch` | step 1 -- tokens/s and which micro-batch fits |
| `scripts/sbatch/resume.sbatch` | what a SIGKILL costs, before spending eighteen hours |
| `scripts/sbatch/train.sbatch` | one run, including the anchor |
| `scripts/sbatch/sweep.sbatch` | step 5 -- learning rates, as a job array |

`scripts/probes/` holds the one-off measurements the documents quote -- the
TF32 arithmetic and what honest fp32 costs -- so a reader can re-run the number
rather than take it.

The repository lives in the project directory rather than `$HOME` because
`$DATA_p330` is group-readable and `$HOME` is not, and because one
`--bind "$DATA_p330:$DATA_p330"` then covers code, data, container and results
at once. The sbatch scripts default `REPO`, `SIF`, `DATA` and `RUNS` to exactly
these paths, so a submission needs nothing in the environment beyond
`$DATA_p330` itself, which `~/.bashrc` exports.

## The partitions

| partition | GPUs/node | nodes | total | walltime |
|---|---|---|---|---|
| `rtx` | 8 | 4 | **32** | 24 h |
| `b200` | 8 | 2 | **16** | 24 h |
| `a100` | — | 6 | — | 24 h |
| `a5000` | 8 | 1 | 8 | 24 h |

Those are the configured totals. What is actually schedulable — one rtx node is
drained and one node of each GPU pool sits under someone else's reservation —
is in *What the cluster reports* below.

Each `rtx` node carries 64 CPUs and 1 TB of memory against its 8 GPUs, so jobs
ask for 8 cores. SLURM's GRES there is untyped — `gpu:8(S:0-1)` names no model,
unlike `a100` and `a5000` — so the card can only be identified from the card:

```bash
srun -p rtx --gres=gpu:1 --time=00:02:00 nvidia-smi --query-gpu=name,memory.total --format=csv
```

It does not block anything: step 1 measures throughput directly, which is the
number that matters. The model is for the paper's setup section.

`rtx` carries **NVIDIA RTX PRO 6000 Blackwell Server Edition**, 96 GB, compute
capability 12.0 — the same architecture generation as the B200, not the
Ada-generation card an earlier estimate here assumed. **That estimate (~37 h for
a full run on rtx) was built on the wrong card and should not be used.**

**Every partition caps at 24 hours, and that is the binding constraint.** Which
pool each run belongs on therefore depends on measured throughput, not on any
figure in this document. The current defaults — long runs to `b200`, sweeps to
`rtx` — are a placeholder to be revisited after step 1. If a full run fits in 24
h on `rtx`, prefer it: 32 cards against 16.

96 GB also means gradient accumulation may be unnecessary. The whole
512-sequence batch is roughly 15-25 GB for a 58M model, so `--micro-batch 512`
is worth trying: it removes accumulation and lets the covariance see all 131k
tokens of a step.

A requeued job **resumes automatically**: the run directory is named by the
configuration hash, so resubmitting the same command continues from the last
checkpoint, with the sampler's position restored so batches are not replayed.
That makes the 24 h cap survivable, though step 1 still decides which pool each
run belongs on.

## What the cluster reports

Measured 2026-08-24. `sbatch --test-only` is the way to re-ask the scheduling
half of this: it answers where and when a job *would* start without queueing
anything.

| node | state |
|---|---|
| `rtx6003`, `rtx6004` | 64 CPUs, 1030 GB, `gpu:8` — the usable rtx pair |
| `rtx6005` | drained |
| `rtx6006` | held by the `migration` reservation |
| `b201` | 128 CPUs, 2015 GB, `gpu:8`, idle — the usable b200 node |
| `b202` | held by the `migration` reservation |

`migration` is an indefinite reservation belonging to another user, running to
2027-12-31 over `b202`, `rtx6006` and four non-GPU nodes. It does **not** block
us: `--test-only` reports an immediate start on both partitions, onto `b201`
and `rtx6004` respectively. It does halve the b200 pool, so if a b200 job ever
queues unexpectedly long, this reservation is the first thing to check.

Scheduler limits are mostly not a constraint -- **with one that is.**
`MaxArraySize=1001`, `MaxJobCount=10000`, and the p330 association sets no
`MaxJobs` or `MaxSubmitJobs`, so step 5's `--array=0-11%8` passes as written.

**But concurrent GPUs cap at 8 per user.** A 15-task array of one GPU each
(job 278661, 2026-08-27) started exactly 8 and left the remaining 7 pending on
`QOSMaxGRESPerUser`, with all three of the p330 users' other jobs idle. Note
that `sacctmgr show qos` reports `gres/gpu=16` for QOS `normal` -- which is the
QOS the job was actually assigned, and the association carries no `GrpTRES` or
`MaxTRES` of its own -- so the configured number and the enforced one disagree
and the enforced one is 8. Plan fan-out in waves of 8; a 15-arm grid is two
waves and costs one extra scheduling round trip, not a failure. Both
partitions cap at `MaxTime=1-00:00:00`. Billing weights are `gres/gpu=1.0`
against `cpu=0.0625`, so the allocation is spent by GPU-hour and the eight
cores each job asks for cost almost nothing.

**`rtx6004` hangs jobs -- avoid it.** On 2026-08-28 two independent submissions
died there the same way: the process starts, logs step 0, then stops. GPU
utilisation reads **0%** with ~89 GB still allocated, no error on stderr, no
progress for tens of minutes. It happened to three `notrust` arms at 14:30 and
to both full-length `ngd-pion-m` arms at 22:50, while identical jobs on
`rtx6002` and `rtx6003` ran normally on the same corpus and code. The node
reports `idle` and accepts work, so `sinfo` will not warn you. **Pin `-w
rtx6002,rtx6003` rather than letting the scheduler choose**, and check
`nvidia-smi --query-gpu=utilization.gpu` through `srun --overlap` before
leaving a long run unattended: a hung job looks exactly like a slow one in the
log.

**Partition access came back the same afternoon.** At 14:47 on 2026-08-28
`sbatch --test-only -A p330` was accepted on `rtx` again and
`sacctmgr show assoc` listed `b200`, `genoa` and `rtx` once more. So the
morning's loss was transient, and this is the **second** time an access change
here reversed itself within hours -- the first was `genoa`, refused at 10:02
and accepted at 10:52 on 2026-08-25. **Re-test before believing any statement
in this file about which partitions are reachable.** Pending jobs move without
being resubmitted: `scontrol update jobid=<id> partition=rtx`.

**The 8-GPU cap is not what it looked like either.** The morning's 15-task
array was held at 8 by `QOSMaxGRESPerUser`, but with the association restored
11 of this user's tasks ran at once across `b200` and `rtx`. Either the cap is
per partition or it moved with the association; it was not re-derived, only
observed, so treat 8 as a floor rather than a rule.

**As of the morning of 2026-08-28 the account reached only `b200`, briefly.** `sbatch --test-only`
under `-A p330` is refused on `rtx` and `genoa` with *"Invalid account or
account/partition combination"* while `b200` is accepted, and
`sacctmgr show assoc where account=p330 user=cy26ms1` lists a single
partition-scoped association, `b200`. Both partitions still carry
`AllowAccounts=ALL`, so the change is on the association side and is
administrative, not a limit we hit. `rtx` nodes were idle at the time.

That matters for planning rather than for correctness: every short experiment in
this repository ran on `rtx`, and they now compete with our own long runs for
the eight usable GPUs of `b201` (`b202` is reserved). Re-test before believing
this paragraph -- the entry below records `genoa` being refused at 10:02 and
accepted at 10:52 the same morning.

**The account previously reached three partitions.** `genoa` was added on 2026-08-25 --
it was refused at 10:02 and accepted at 10:52 the same morning, so re-test
rather than trust this paragraph. `sbatch --test-only` under `-A p330` is
accepted on `rtx`, `b200` and `genoa`, and still refused on `cpu`, `milan`,
`a100`, `gpu`, `a5000` and `virtual` with *"Invalid account or
account/partition combination"*. Those partitions are open --
`AllowAccounts=ALL` on `cpu` -- so what blocks us is a missing association in
the accounting database, not partition policy, and the request to make is
precisely "associate account `p330` with partition `cpu`".

`genoa` is 17 nodes of 192 cores, 500 GB each, 3264 cores in total, 24 h cap,
and mostly idle. It is the right place for work with no GPU in it -- tokenising
the corpus is hours of CPU -- because it does not strand anybody's cards.

**Every CPU partition here bills a core like a GPU** -- this is not a `genoa`
quirk, and the comparison below is between partition *kinds*, not between
`genoa` and a cheaper alternative that does not exist:

    cpu     CPU=1.0,    Mem=0.208G
    milan   CPU=1.0,    Mem=0.5G
    genoa   CPU=1.0,    Mem=0.375G
    virtual CPU=1.0,    Mem=0.0G
    rtx     cpu=0.0625, Mem=0.007767G, gres/gpu=1.0
    b200    CPU=0.0625, Mem=0.003877G, GRES/gpu=1.0

Of the three CPU partitions reachable by anyone, `genoa` is the largest (3264
cores against `cpu`'s 680) and the most generous per core (500 GB across 192
cores). Asking for it rather than `cpu` was the right request.

Under `MAX_TRES` a job bills the largest of its weighted components, so **one
genoa core-hour bills 1.0 -- the same as a whole GPU on `rtx`**, and a whole
genoa node bills 192 where a whole rtx node bills 8. The node TRES confirm it:
`ne49` is `cpu=192,mem=500G,billing=192`, `rtx6004` is
`cpu=64,mem=1030000M,billing=8`. A 96-core tokenisation job costs 96 billing
units an hour on `genoa` against 6 on `rtx`.

No association here carries `GrpTRESMins` or any other hard quota, so nothing
is enforced by SLURM; what this spends is **fairshare**, and it lands in the
same `p330` pool that sets the priority of our GPU jobs. `QOS normal` caps
concurrency at `cpu=1280` per user, which is 6.6 genoa nodes at once.

`genoa` has its **own 5000-hour quota**, separate from the GPU grant. The unit
is undocumented -- neither the HPCF partitions page nor its job-submission page
states one -- but two things point at core-hours: the institute's own
preparatory-access wording is "typically 5,000 to 100,000 core hours", and the
billing weights are set so that a node bills its own natural resource (`ne49`
is `billing=192` for 192 cores, `rtx6004` is `billing=8` for 8 GPUs). Under
either reading the only CPU work in this project's plan -- tokenising the whole
of C4 at roughly 1 core-hour per 1B tokens, so about 156 -- is affordable, so
the ambiguity changes no decision. Settle it with `hpc.support@cyi.ac.cy`, or
by running one core for two minutes and watching `RawUsage`.

**The fairshare cost of doing that is accepted, deliberately.** 156 core-hours
is 561,600 billing-seconds against the 119,844 this project has spent in total,
so it would depress GPU priority for several days at the 7-day half-life. The
judgement is that this rarely bites: both GPU pools have been idle or
near-idle most of the time we have watched them, and fairshare only decides who
wins when two jobs want the same card. It did bite once -- 1.5 h behind another
user's whole-node job on 2026-08-24 -- and a high score is what took `b201`
back the moment it freed. Worth re-checking before a wide sweep, not before a
single run.

**Both the login node and the compute nodes have network.** huggingface.co,
pypi.org and files.pythonhosted.org answer from `rtx6003` as well as from the
login node, and `nvcr.io` issues an anonymous pull token, so the NGC image
needs no API key and `%post` can install packages from a job. Data preparation
does not have to happen on the login node.

**A compute node's root filesystem is tmpfs** -- `/` shows 504 GB of tmpfs on
`rtx6003`. Anything written to `/tmp` there is in RAM and counts against the
job's memory, not against disk.

## Step 0 — preflight

```bash
mkdir -p logs      # SLURM opens the -o/-e paths before the job script runs
srun -p rtx --gres=gpu:1 --cpus-per-task=8 --time=00:10:00 \
    apptainer exec --nv --bind "$PWD:$PWD" --pwd "$PWD" \
    "$DATA_p330/containers/ngd-pion.sif" python scripts/preflight.py
```

Checks the operations this project depends on rather than comparing version
numbers: that `sm_120` is in the torch build, that `linalg.eigh` runs on 512 and
1376 matrices and returns finite eigenvalues, that the Cayley solve comes back
orthogonal, and that the model does a forward and backward. `eigh` on the GPU is
the one to watch — it runs for every layer at every refactor, and it is exactly
the operation that varies between CUDA builds.

Run it on both pools.

What it reported on `rtx6003`, 2026-08-24, with the 26.07 image:

```
  ok    device  NVIDIA RTX PRO 6000 Blackwell Server Edition, sm_120, 102 GB
  ok    fp32 matmul  184 TFLOPS          <- tf32; honest fp32 is 74
  ok    bf16 matmul  374 TFLOPS
  ok    eigh 512  4.9 ms                 <- 15.4 ms on the 25.04 image
  ok    eigh 1376  13.4 ms
  ok    cayley  orthogonality error 3.9e-06
  info  unguarded solve here is 1100x worse (4.3e-03) -- tf32 matmul is on
  ok    model fwd+bwd  58.2M params, peak 2.1 GB at 8 sequences
```

And what step 1 then measured, with `pion` in the anchor's own configuration:
micro-batch **512 fits** at 83.3 GB of 97.9, so there is no gradient
accumulation and the covariance sees all 131072 tokens of a step. The rate is
**268,590 tokens/s**, a step is 0.488 s, and a 9.6B run is **9.9 h** -- inside
the 24 h cap with room to spare.

Two things to carry forward. The `eigh` at 512 is three times faster on 26.07
than on 25.04, which is most of why that image was adopted -- most matrices in
this model are square. And **2.1 GB at 8 sequences** sets the batch arithmetic:
the head materialises `vocab x tokens` logits in fp32, 128 KB per token per
copy, so the whole 512-sequence batch would want something like 70-85 GB
against the card's 102. `--micro-batch 512` is therefore unlikely; 128 or 256
is the range, and the covariance sees a half or a quarter of each step's tokens
accordingly.

Run `scripts/gpu_smoke.py` beside it. It checks what the CPU suite cannot:
that a run resumes from its checkpoint on a card, and that the spectrum of a
weight holds still even with TF32 switched on deliberately.

## Step 1 — measure throughput

Twenty minutes, and it decides how the 3000 hours are spent.

```bash
sbatch -p rtx  scripts/sbatch/train.sbatch --optimizer ngd-pion --max-steps 200
sbatch -p b200 scripts/sbatch/train.sbatch --optimizer ngd-pion --max-steps 200
```

Read `tokens_per_sec` from the last line of each run's `log.jsonl`.

**Expect the B200 to disappoint.** A 60M model's matrices are 512x512 and
512x1376 — far too small to fill a Blackwell card, so utilisation will be poor
and the peak-FLOPS ratio (roughly 25x) will not appear. If the observed ratio
is only 2-3x, the long runs belong on the **RTX** pool, which has twice the
hours, and the B200 hours are better spent on whatever ends up largest.

Decide the allocation from these two numbers, not from specifications.

## Step 2 — container

```bash
export APPTAINER_CACHEDIR=/nvme/scratch/$USER/apptainer/cache   # persistent
export APPTAINER_TMPDIR=/tmp/apptainer-$USER/tmp                # local disk
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

SIF="$DATA_p330/containers/ngd-pion.sif"
LOCAL=/tmp/apptainer-$USER/ngd-pion.sif

# 1. build on local disk, with mksquashfs held under the login cgroup's cap
apptainer build --ignore-fakeroot-command \
    --mksquashfs-args "-mem 3G -processors 4" \
    "$LOCAL" container/ngd-pion.def

# 2. open it. the exit code above proves nothing
apptainer inspect "$LOCAL"
apptainer exec "$LOCAL" python -c "import torch, numpy, scipy, datasets, transformers, pytest"

# 3. stage the copy beside the canonical path, never onto it
cp "$LOCAL" "$SIF.new"
apptainer inspect "$SIF.new"
cmp "$LOCAL" "$SIF.new"

# 4. publish by rename, which is atomic within a filesystem
mv "$SIF.new" "$SIF"
```

A finished image is **9.3 GB**, and takes about 25 minutes with a warm cache,
most of it compression that `-processors 4` deliberately slows down. If it
comes out at a few hundred megabytes, read the next paragraph.

The shape of that recipe is the part worth keeping. Build somewhere private,
prove the artifact opens, stage it beside its destination, prove the copy opens
too, and only then put it where jobs will find it -- by a rename, so that no
moment exists in which the canonical path holds half an image. `$SIF` is the
default in both sbatch scripts and is visible to the whole p330 group, so a
broken image there does not fail cleanly: jobs start, die somewhere unrelated,
and the hour goes into looking in the wrong place. The same argument makes the
checkpoint write in `harness/train.py` a write-and-rename.

Four things about that command, each of which was measured rather than
assumed.

**Bound `mksquashfs`'s memory, or the build fails and says it succeeded.** It
sizes its caches from the machine's *physical* memory -- 94 GB on the login
node, so roughly 23 GB -- while a login session is capped by cgroup at 8 GB.
It grew to 7.8 GB and was OOM-killed. The squashfs superblock is written last,
so what survived was 187 MB of real compressed data with no valid header, and
apptainer wrapped that in a SIF, printed `Build complete` and exited 0. Twice.
`--mksquashfs-args "-mem 3G -processors 4"` keeps it inside the cap. Small
images stay under it on their own, which is exactly why a throwaway alpine
probe gives no warning at all.

**Verify the artifact. Exit 0 is not evidence.** Open the image and run
something inside it, and do it again after copying it anywhere. Both of the
failed builds above reported success.

**`--fakeroot` is not granted and is not needed.** This account has no entries
in `/etc/subuid` or `/etc/subgid`, so the flag in earlier versions of this
document would have failed. Apptainer 1.5 falls back to a root-mapped user
namespace by itself — unprivileged namespaces are enabled here and
`unshare -U -r` succeeds — and `%post` runs in it as uid 0 with working
outbound network. Verified by building a throwaway alpine image whose `%post`
installed a package and downloaded from PyPI. The plan-B of building elsewhere
and copying a `.sif` across is not required.

**`--ignore-fakeroot-command` is required.** Without it apptainer wraps `%post`
in a `fakeroot` binary that the image does not carry, and the build dies at the
first line of `%post` with *"a shared library is likely missing in the image"*.

The two directories want opposite things, which is why they are split. The
cache holds the base image's 61 layers, 12.1 GB compressed, and it is worth
keeping: with a warm cache a rebuild costs minutes rather than another
download, so it lives on scratch, which survives. The temporary directory is
where those layers are unpacked into tens of gigabytes of small files, and that
is the one that must not be on NFS.

**`APPTAINER_TMPDIR` must be on local disk.** Pointed at `/nvme/scratch`, which
is NFS, the alpine probe hung in *"Extracting OCI image"* and had not finished
five minutes later; pointed at `/tmp`, which is local ZFS with 419 GB free, the
identical build finished in seconds. The NGC image unpacks to tens of
gigabytes, so this is the difference between a twenty-minute build and an
afternoon spent hammering a shared filer.

Rebuild as rarely as possible, and never in the middle of a series. The code
is bind-mounted rather than baked in, so editing the optimizer, the harness or
the tests needs no rebuild at all; only a dependency change does, and a new
pure-Python package can usually skip even that
(`pip install --target "$DATA_p330/pylibs"`, then `PYTHONPATH`). The reason to
care is not convenience: a different torch or CUDA build between two arms of a
comparison is a confound, and the whole point of the comparison is that one
thing differs. Build once, then run the throughput measurement, the anchor, the
sweeps and the final table on that single image.

The image records what it was built from. `apptainer inspect` reports
`org.opencontainers.image.base.digest:
sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c` for the
26.07 build of 2026-08-24, alongside the CUDA, cuDNN and NCCL versions it
carries.
The `.def` pins a tag and tags can move; that digest is what makes "rebuild the
same image" a checkable claim.

The base is NGC's PyTorch image because Megatron expects it and it ships a
built TransformerEngine. The lightweight harness needs none of that; sharing
one image just keeps both paths reproducible from one pinned digest. The
`%post` also installs `pytest` and `scipy`, because step 3 runs the test suite
inside this image and neither is guaranteed to be in the base.

## Step 3 — data

```bash
apptainer exec --bind "$DATA_p330:$DATA_p330" "$DATA_p330/containers/ngd-pion.sif" \
    python scripts/prepare_data.py --out "$DATA_p330/c4" --target-tokens 1.0e10
```

C4 is a cleaned Common Crawl dump and the corpus their 60M ablations use, which
is the only reason it is the one here — the comparison is only meaningful on
the same data. The full budget is 9.6B tokens, so 10B gives a single pass with
a margin. Output is two flat `uint16` files; the T5 vocabulary is 32100, so
uint16 is exact.

Run this on a login node if compute nodes have no network.

## Step 4 — the anchor

**Nothing this harness produces means anything until this lands.** See
`AGENTS.md` for why.

```bash
sbatch scripts/sbatch/train.sbatch --anchor bilateral    # defaults to b200
sbatch scripts/sbatch/train.sbatch --anchor alternate
```

Each is a full 9.6B run. If one does not fit inside 24 h, resubmit the same
command after the job ends and it continues from its checkpoint.

Targets are 3.3575 and 3.3654. The verdict is written to `anchor.json` in each
run directory. Reproducing the 0.0079 **gap** matters more than either level.

Two full 9.6B runs. Budget them from step 1's measurement.

## Step 5 — learning rates

Each optimizer has its own optimum, and on the toy problem they spanned 4.6e-3
to 2.2e+1. Comparing at one shared rate measures which is better *at that rate*
— the most common way an optimizer comparison goes wrong, and the first thing a
reviewer asks.

```bash
sbatch --array=0-11%8 scripts/sbatch/sweep.sbatch        # defaults to rtx
```

Twelve runs at 1x Chinchilla (1.2B tokens, 9155 steps), an eighth of the
published budget and enough to rank.

## Step 6 — the comparison

Three arms, best rate from step 5, three seeds each, full 9.6B:

| arm | purpose |
|---|---|
| `pion` | context — the published configuration |
| `pion_ablated` | the fair baseline: no momentum, no scaling, Cayley |
| `ngd` | differs from `pion_ablated` in `F^-1` alone |

## Budget

Per full 9.6B run, `6ND` is about `3.35e18` FLOPs.

| | per 9.6B run | per 1.2B run |
|---|---|---|
| at ~25 TFLOPS effective | ~37 h | ~4.6 h |
| at ~200 TFLOPS effective | ~4.6 h | ~0.6 h |

**These are placeholders, not estimates for either card.** Both pools are
Blackwell, and a 60M model fills neither, so the achieved rate is a question
about utilisation rather than peak throughput. Step 1 replaces this table.

| phase | runs | length | rough cost |
|---|---|---|---|
| throughput | 2 | 200 steps | minutes |
| anchor | 2 | 9.6B | 2 full runs |
| sweeps | 12 | 1.2B | 1.5 full runs |
| ablations | ~10 | 1.2B | 1.3 full runs |
| comparison | 9 | 9.6B | 9 full runs |

Around 14 full-run equivalents against an allocation that affords dozens.
Compute is not the binding constraint; wall-clock and queue throughput are,
which is why every job is one GPU and sweeps go out as arrays.

## Not on the critical path

Megatron. Their repository is built on it, and integrating means parallel
linear layers instead of `nn.Linear`, a fused QKV projection, and an all-reduced
covariance under tensor parallelism. It is worth doing for **one** run, and only
if the anchor fails in our harness in a way the known differences do not
explain.


## rtx6001 packed an eight-task array onto fewer GPUs than it granted (2026-08-29)

Job 299611 submitted `--array=0-8%8 -w rtx6001 --gres=gpu:1`, which should place
one task per card on an eight-GPU node that `sinfo` reported idle and
`scontrol` confirmed was allocated to us alone. It did not. All eight tasks sat
at step 0 for **two hours and forty-nine minutes**, looping on

    expandable_segments: memory mapping failed with OOM on device 0 while
    trying to map 20971520 bytes (free: 5832704, total: 101975851008)

The tell was that the warnings agreed to the microsecond across processes that
were supposed to own separate cards (`11:00:31.360537`, `.360472`, `.360877`),
and that tasks 6 and 7 -- which started 24 minutes after the first six -- hit
OOM in their first second, with the node already full. After cancelling, an
`srun` on rtx6001 showed all eight cards at 0 MiB, so nothing had leaked; the
memory had been ours.

**The identical array ran fine on rtx6002** minutes earlier (job 299598, eight
arms, 3000 steps each) and again afterwards (job 299817, eight distinct GPU
UUIDs). So this is a property of rtx6001 on the day, not of the submission.

Every GPU sbatch script now prints `CUDA_VISIBLE_DEVICES` and the card's UUID
before starting. Distinct UUIDs across tasks is the check; identical ones mean
cancel immediately rather than wait. Three hours were lost to not having it.

## rtx6003 wedges an eight-task array (2026-08-29, evening)

Job 300486, eight `ngd-pion-s` arms, `-w rtx6003`, one GPU each and eight
distinct UUIDs confirmed at startup. Every arm wrote its step-0 row and then
stopped. Attaching with `srun --overlap --jobid=<real JobId>`:

    GPU utilisation   0 %
    memory used       49309 MiB
    load average      8.00 on 64 cores   (one spinning thread per process)

Memory allocated, cards idle, nothing in stderr. That is the same signature
already recorded for rtx6004. The node had run a 20-hour job and two probes
without trouble earlier the same day, so it is eight-way concurrency that
provokes it, not the node as such -- rtx6002 ran an identical eight-arm array
(job 299598) to completion an hour earlier.

Two practical notes. `srun --overlap --jobid=` needs the **real** `JobId` from
`scontrol show job <array>_<task>`, not the `ArrayJobId`; passing the array id
fails with "Job is pending execution". And `log_every` doubles as the monitoring
interval: at 97 there is no way to tell a wedged run from a slow one for the
first 97 steps, which cost most of an hour here. Keep it small on a grid whose
speed is not yet known.
