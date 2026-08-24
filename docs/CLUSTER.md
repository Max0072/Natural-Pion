# Running on Aphrodite

The Cyprus Institute HPC facility. Project **p330**; every submission needs
`-A p330` or it is rejected. `$DATA_p330` points at the project's data
directory, and `data_p330` in `$HOME` is a symlink to the same place.

Allocation: **2000 RTX hours and 1000 B200 hours.**

## The partitions

| partition | GPUs/node | nodes | total | walltime |
|---|---|---|---|---|
| `rtx` | 8 | 4 | **32** | 24 h |
| `b200` | 8 | 2 | **16** | 24 h |
| `a100` | — | 6 | — | 24 h |
| `a5000` | 8 | 1 | 8 | 24 h |

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

Still to find:

```bash
scontrol show node rtx6003 | grep -Ei "Gres|RealMemory|CPUTot"   # which RTX card
scontrol show config | grep -Ei "MaxArraySize|MaxSubmitJobs"
apptainer --version
df -h "$DATA_p330"
```

And separately — these are different questions, and on many clusters the answer
differs:

```bash
curl -sI -m 5 https://huggingface.co | head -1                       # login node
srun -A p330 --time=00:02:00 --pty bash -c 'curl -sI -m 5 https://huggingface.co | head -1'   # compute node
```

If compute nodes have no network, C4 and the tokenizer must be staged into
`$DATA_p330` from the login node first, which reorders steps 2 and 3 below.

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
apptainer build --fakeroot "$DATA_p330/containers/ngd-pion.sif" container/ngd-pion.def
```

`--fakeroot` is not granted everywhere. If it is refused, build the `.sif` on a
machine where you have root and copy the single file across — that is the whole
advantage of the format.

The base is NGC's PyTorch image because Megatron expects it and it ships a
built TransformerEngine. The lightweight harness needs none of that; sharing
one image just keeps both paths reproducible from one pinned digest.

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
