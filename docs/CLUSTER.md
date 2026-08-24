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
| corpus | `$DATA_p330/c4/c4_{train,val}.bin` |
| run outputs | `$DATA_p330/runs/<name>/` |
| SLURM logs | `logs/`, **in the directory you submit from** |

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

Scheduler limits are not a constraint here. `MaxArraySize=1001`,
`MaxJobCount=10000`, and the p330 association sets no `MaxJobs` or
`MaxSubmitJobs`, so step 5's `--array=0-11%8` passes as written. Both
partitions cap at `MaxTime=1-00:00:00`. Billing weights are `gres/gpu=1.0`
against `cpu=0.0625`, so the allocation is spent by GPU-hour and the eight
cores each job asks for cost almost nothing.

**The account reaches two partitions, not the whole machine.** `sbatch
--test-only` under `-A p330` is accepted on `rtx` and `b200` and refused on
`cpu`, `milan`, `genoa`, `a100`, `gpu` and `a5000` with *"Invalid account or
account/partition combination"*. That matters for work with no GPU in it --
tokenising the corpus is hours of single-threaded CPU -- which currently has to
occupy a GPU node or be run on the login node. Asking for `cpu` or `genoa`
access is worth doing; it is not a blocker, since a job on `rtx` without
`--gres` allocates no GPU and bills only CPU.

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

## Step 1 — measure throughput

Twenty minutes, and it decides how the 3000 hours are spent.

```bash
sbatch -p rtx  scripts/sbatch/train.sbatch --optimizer ngd --max-steps 200
sbatch -p b200 scripts/sbatch/train.sbatch --optimizer ngd --max-steps 200
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

apptainer build --ignore-fakeroot-command \
    --mksquashfs-args "-mem 3G -processors 4" \
    /tmp/apptainer-$USER/ngd-pion.sif container/ngd-pion.def

apptainer inspect /tmp/apptainer-$USER/ngd-pion.sif          # it must open
cp /tmp/apptainer-$USER/ngd-pion.sif "$DATA_p330/containers/"
apptainer inspect "$DATA_p330/containers/ngd-pion.sif"       # and so must the copy
```

A finished image is **11.9 GB**. If it comes out at a few hundred megabytes,
read the next paragraph.

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
sha256:d1eac6220dd98ef5870b1a76673cfb6f84451135a6d8a174cb92258a6bf4576d` for the
build of 2026-08-24, alongside the CUDA, cuDNN and NCCL versions it carries.
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
