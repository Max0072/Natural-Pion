# Working in this repository

NGD-Pion: a curvature-preconditioned variant of [Pion](https://arxiv.org/abs/2605.12492).
Pion rotates each weight matrix on both sides, leaving its singular values fixed
for the whole run, and drives the rotation with the raw gradient. This project
preconditions that rotation by the Fisher operator on the bivector tangent
space instead. Target venue is ICLR; the deadline is roughly one month from
2026-08-23.

**This is the working repository, not the published one.** It keeps the
operational record -- what the cluster actually reports, which build flags were
needed and why, the traps that cost time -- because that record is what makes
the work continuable by whoever picks it up, including its author three weeks
later. What ships with the paper is a curated snapshot: the algorithm, the
optimizer, the harness, the tests and a short reproduction note. So do not thin
this repository out to make it presentable, and do not park operational notes
on a side branch to hide them -- in a public repository every branch is public
anyway, and the split that works is between this repository and the snapshot,
not between two branches of it.

## Read before changing anything

1. **[`ALGORITHM.md`](ALGORITHM.md)** — the specification. Every design decision
   with the measurement that produced it. Each module implements one of its
   sections and says so in its docstring.
2. This file — the state of play, and the mistakes already made.

`ngd_pion/reference.py` is a deliberately naive numpy transcription of the
spec. It is the **oracle**: the torch path is correct exactly insofar as it
reproduces it, and `tests/test_optimizer.py` pins that. Never optimise
`reference.py`; if the two disagree, the reference is right until proven
otherwise.

## State of play

**Done.** The mathematics is verified — the Fisher operator against Monte
Carlo, the closed-form solve against an explicit Kronecker system, the descent
lemma, the sign, Cayley's exactness, spectrum preservation. The optimizer, the
Pion baseline with ablation switches, a LLaMA-60M harness in their
configuration, the anchor machinery, SLURM scripts, a container definition.
140 tests, 24 s in the container on four threads. One is skipped by design --
`square W has no kernel` -- and **none of them touches a GPU**, which is a
property of checking the torch path against a numpy oracle rather than an
oversight, but it does mean anything device-specific has to be checked by
running something on a card.

**Not done, and this is the whole risk.** Nothing has been trained at scale.
The only evidence the method helps is a toy least-squares with an exactly
reachable target, where natural gradient wins almost tautologically. That test
was a kill criterion — failing it would have stopped the project — and passing
it means very little.

**Cluster shape.** `rtx` has 4 nodes of 8 GPUs, `b200` has 2 of 8, and **every
partition caps at 24 hours**. Both pools are Blackwell — `rtx` carries RTX PRO
6000 Blackwell cards with 96 GB — so an earlier estimate here that assumed an
Ada-generation card and put a full run at 37 h on `rtx` was wrong and has been
withdrawn. Which pool each run belongs on is a question for the throughput
measurement, not for arithmetic. Resubmitting the same command resumes from the
checkpoint.

Of those, `rtx6005` is drained and `rtx6006` and `b202` are held by another
user's open-ended reservation, leaving two rtx nodes and one b200 node
schedulable as of 2026-08-24. It does not block submission — `sbatch
--test-only` reports an immediate start on both — but it is the first thing to
check if a b200 job queues for longer than it should. `docs/CLUSTER.md` carries
the node-by-node detail and the scheduler limits.

**Blocked on the cluster, in order:**

1. Throughput on both partitions. A 60M model does not fill a B200; the peak-FLOPS
   ratio against the RTX pool will not be the observed ratio, and which pool the
   long runs belong on depends on measuring it. Twenty minutes, and it changes
   the allocation plan.
2. Container, then C4.
3. **The anchor** — see below. No number this harness produces means anything
   until it lands.
4. Learning-rate sweeps, then the comparison.

## The anchor, and why it comes before everything

Every number here is produced inside a harness we wrote, and no measurement
taken inside it can show the harness is equivalent to theirs. So: run *their*
configuration here and check we reach *their* published figure.

Their paper reports exactly one pair of concrete 60M numbers, in section 2.4.3:
final loss **3.3575** bilateral, **3.3654** alternate, both with Lie+Lie
momentum. **That is not what their 60M shell script runs** — the script sets
`alternate` and the ambient momentum. `harness/anchor.py` follows the number,
not the script; following the script would make a miss uninformative.

Run both sides. Reproducing the **0.0079 gap** between them is a sharper check
than reproducing either level, because the gap is insensitive to data order and
initialisation in a way the level is not.

If it misses, `anchor.KNOWN_DIFFERENCES` lists the harness differences to rule
out before calling anything a bug. A miss inside ~0.05 is more likely the C4
subset than a defect -- but the *gap* is not covered by that excuse, because
every entry there acts on both arms alike. When the gap missed, the cause was
in this repository, and reading their optimizer found it in an afternoon.

**The anchor has now been run, and it missed.** Four complete runs, both sides
on both partitions: bilateral 3.3997 against 3.3575, alternate 3.4352/3.4369
against 3.3654, every one `matched: false`. The two partitions agree to 0.002,
so the miss is systematic. The *gap* came out 0.0355 against their 0.0079 --
4.5x, on the quantity chosen because harness differences should not move it.

**Their optimizer has now been read too**, at
`$DATA_p330/reference/pion/megatron-lm/megatron/core/optimizer/pion.py`, and it
named the cause. Three differences were ours, not Megatron's, and all three are
fixed: `_scale_update_matrix_rms` takes an `update_side` and normalises the
side being applied where ours always normalised both (the gap); their default
leaves the second moment off where `beta2=0.95` was hard-wired here, and
neither beta was a `RunConfig` field, so the choice sat outside the hash; and
their `pion_qkv_split_granularity` defaults to per-head Q where this harness
rotated Q whole. Two entries left `KNOWN_DIFFERENCES` as non-differences: their
Pion never decays a rotated weight, and their sample windows cross document
boundaries exactly as ours do -- the 60M script passes none of
`--reset-position-ids`, `--reset-attention-mask`, `--eod-mask-loss`. The
sampler was changed anyway, because theirs permutes a partition of the stream
while ours drew windows with replacement.

**The re-run landed the gap and not the level.** bilateral 3.4021, alternate
3.4161, gap **+0.0140** against their 0.0079 -- down from +0.0355, but the
criterion set before the numbers (`|gap - 0.0079| <= 0.005`) gives 0.0061 and
is not met. Alternate improved by 0.019 and bilateral by 0.0024, which is our
noise floor, so the scaling fix showed exactly the signature it was diagnosed
to have and the other two fixes moved the level by nothing.

**Then the setting was compared rather than the optimizer**, and the ~0.045
common offset had two causes, both on the 32.9M parameters Pion does not own --
56.5% of this model. Their non-matrix parameters go to Megatron's ordinary
Adam, and `build_optimizers` differed from it twice: it passed **no betas**, so
torch's `(0.9, 0.999)` stood where their script sets `--adam-beta2 0.95`; and
it decayed **every** parameter at 0.1, where Megatron's standard override gives
`wd_mult = 0.0` to every 1-D parameter and every bias. The second is the larger:
over 73242 steps of this cosine, `weight_decay=0.1` multiplies an RMSNorm gain
by **0.0248**, a 40x shrink of something that starts at 1.0.

That interacts with Pion specifically. A network normally absorbs decayed norm
gains by growing its linear weights; under a spectrum-preserving optimizer that
route is closed, because every rotated matrix keeps its singular values for the
whole run. The only compensation left is the embedding and the head.

Both act on the two arms alike, so they move the common offset and not the gap
-- which is the part that was unexplained. `adam_beta1`, `adam_beta2`,
`adam_eps` and `decay_norms_and_biases` are `RunConfig` fields now; the last
defaults to `False` and exists so the completed runs stay reproducible.

**The anchor has to be re-run again**; every result above belongs to the code
as it was. The configuration hash changes each time, so they keep their
directories.

**Their script had been read before that, not inferred.** `opt_llama_60M_pion.sh`
confirms every shape and schedule this harness copies, and both defaults the
anchor deliberately departs from. It also says `--bf16`: their runs are mixed
precision under Megatron with fp32 master weights in the optimizer, while this
harness trains in fp32 with TF32 matmuls -- `anchor_config` sets `bf16` for
that reason. What is left of it in `KNOWN_DIFFERENCES` is only where the master
weights sit, which is the least likely entry there to move a number: both clip
one global norm over every parameter before the step.

Left open, and worth an hour before the paper rather than before the anchor:
whether their own rotations hold the spectrum numerically. They retract with a
degree-2 truncated exponential on weights that Megatron keeps in fp32, on cards
where fp32 matmul means TF32 unless disabled. This project measured what that
does to a Cayley step; nobody has measured what it does to theirs.

## Decisions already made — do not reopen without evidence

Each of these was measured. Reversing one is fine; reversing it on intuition is
not.

| decision | why |
|---|---|
| `S = I` | **not** a claim that backward errors are isotropic — the measured `S` is strongly anisotropic and no simple model fits it. Using it is what makes the step worse: `S = I` reduced held-out loss 1.4-1.9x more than the measured `S` on a toy transformer, because a finite-sample covariance is too noisy to invert. Also collapses the out-side to an orthogonal `eigh` and removes the need for a backward hook |
| Cayley, not their truncated exponential | `R^T R = I + A^4/4` exactly, so the truncation always inflates. More importantly, **ablating their RMS scaling makes it diverge within tens of steps**, so an exact retraction is a *precondition* of the ablation, not a preference |
| one spectral floor `max(lam, eps*lam_max)`, not a shift | a floor is the identity above itself; a shift perturbs well-determined directions too. 134x more accurate on a wide layer at `eps = 1e-4` |
| `eps = 1e-4` | **its lower bound is set by the compute dtype, not the problem.** fp32 machine epsilon is 1.2e-7, so `1e-8` is meaningless there — measured 2e-1 error against fp64 |
| fp32 throughout | fp64 buys nothing (end-to-end error 1e-5 to 1e-3 on real spectra), costs 30-60x on consumer GPUs, and does not exist on some backends |
| `A` accumulated in fp32 minimum | bf16-level noise in `A` gives a step wrong by 10^3-10^4. The rule is about **storage**, not about what feeds it, and the difference is 165x: bf16 *activations* perturb `A` by 4.5e-05, because thousands of independent roundings cancel in the average, while bf16 *storage* perturbs it by 7.4e-03 with nothing to cancel -- and of that, only 1.7e-03 is the single unavoidable rounding, the other 4.3x compounding because the accumulator is re-rounded every step against an increment 20x smaller than itself, which biases the top eigenvalue by 1.9%. Against a floor of `eps*lam_max` at 1e-4, input noise sits below the floor and storage noise sits 70x above it, in exactly the eigenvalues the method inverts. So feeding the covariance from a bf16 forward pass is safe; keeping it in bf16 is not |
| non-orthogonal (Gaussian, std 0.02) init | Pion freezes the spectrum permanently, and a non-trivial spectrum is wanted for expressiveness. Costs only 4.3x on the in-side factorisation; accuracy and step norm are unaffected |
| fixed `T_fac`, not adaptive | a data-dependent refactor schedule diverges across hardware and destroys reproducibility |
| `alpha_max = 1` | the trust-region ratio reads out basis staleness; capping at 1 keeps it one-sided, so a stale basis can only shorten the step |
| the same corpus, not a bigger one | C4 with the T5 vocabulary, 9.6B tokens, as theirs. Their 9.6B comes out of roughly 156B, so both runs consume about 6% of C4 and neither sees a "full stream"; the shards are a deterministic partition of an already shuffled crawl, so our first 64 are exchangeable with any other 64. Tokenising the rest would cost ~156 core-hours, 312 GB and days of fairshare for a different draw from the same distribution, not a better one |
| 9.6B tokens, not 4.9B | 73242 steps at 131072. The 37500-step reading of their shell script came from a **defect in their released code**, not from a second configuration; the paper's figure is the right one and `RunConfig.train_steps` follows it. `prepare_data.py` used to state the wrong number and default to a 5B corpus, which would have gone unnoticed: `TokenCorpus` samples with replacement and has no epoch, so an undersized corpus repeats tokens silently instead of failing |
| no momentum, no RMS scaling in NGD-Pion | deliberate, and both are testable hypotheses rather than oversights. The comparison is against **ablated** Pion, so this is not a confound |

## Traps

Things that look right and are not. Every one of these cost real time here.

- **`eps` tuned in fp64 does not transfer to fp32.** The plateau is four orders
  wide in fp64 and its lower edge moves up by six orders in fp32.
- **The floor belongs on every spectrum that reaches a denominator**, including
  the pencil's own output — not only on the ones being inverted. Flooring
  sources alone lets `lam` round to zero in fp32.
- **The dead block is `0/0` only where *both* indices are in the kernel.** The
  `(kernel, range)` block carries a live gradient over a small denominator and
  holds ~88% of the step. Do not describe the method as immune to the damping
  problem; it is not.
- **`cond(W)` and `cond(lam)` look alarming and mean nothing.** Only `cond(A)`
  matters, and only for deciding fp32. Where `W` is small so is `W^T delta`, so
  extreme eigenvalues meet extreme-small numerators.
- **`from __future__ import annotations` makes `field.type` a string.** Argparse
  built from dataclass fields silently parses everything as `str` and fails deep
  inside numpy. `harness/run.py` resolves annotations; keep it that way.
- **The run log is appended** so preemption keeps history, which means a re-run
  of the same configuration shares the file. A start marker separates attempts;
  readers must take the last one.
- **A truncated run always sits far above a converged target.** That is not a
  miss. `anchor.check` refuses to judge an incomplete run.
- **Apptainer's temporary directory must be on local disk.** Pointed at
  `/nvme/scratch`, which is NFS, an image build hangs in "Extracting OCI image"
  and does not finish; on local `/tmp` the identical build takes seconds. Both
  halves were measured twice, the second time after the OOM below was found, so
  the two are separate faults and not one misread.
- **A failed apptainer build looks exactly like a successful one.**
  `mksquashfs` sizes its caches from *physical* memory -- 94 GB on the login
  node -- while a login session is capped by cgroup at 8 GB, so on a large
  image it is OOM-killed. Its superblock is written last, so what remains is a
  SIF holding real compressed data with no valid header, and apptainer prints
  `Build complete` and exits 0 over it. Build with
  `--mksquashfs-args "-mem 3G -processors 4"`, and open the image before
  believing in it. A small probe image stays under the cap and proves nothing
  about a large one.
- **`--fakeroot` is not granted here, and is not needed.** There are no
  `/etc/subuid` entries for this account. Apptainer falls back to a root-mapped
  user namespace on its own, but its `fakeroot` wrapper is missing from the
  image, so builds need `--ignore-fakeroot-command` or they die at the first
  line of `%post` with a message about a missing shared library.
- **Torch does not move this optimizer's state onto the GPU when it loads.**
  `Optimizer.load_state_dict` relocates tensors it finds directly or nested in
  dicts, lists and tuples. `NGDPion` keeps a `CovarianceAccumulator` and two
  `Basis` dataclasses, and torch descends into neither, so without the override
  in `optimizer.py` a resumed run brings `A` and both bases back on the CPU and
  dies on the first `W @ A`. The tests run on CPU and cannot see it; the first
  run to hit the 24 h wall would have.
- **What the GPU calls fp32 is TF32, and it destroys the spectrum.** On
  Ampere and newer, torch performs fp32 matrix operations in TF32 by default --
  ten bits of mantissa. Measured on an RTX PRO 6000 Blackwell: Cayley's
  orthogonality error `4.3e-03` against `3.9e-06`, and the singular values of a
  weight move by a **relative 1.0** over 200 two-sided steps against `2.6e-04`.
  The property the method is built on is gone, silently, and the CPU suite
  cannot see it because TF32 does not exist there. `ngd_pion` now turns it off
  around its own linear algebra -- the retraction, the factorisation, the
  covariance -- and `scripts/gpu_smoke.py` switches TF32 on deliberately and
  requires the spectrum to hold anyway. It is left on for the *model*, where it
  is worth 2.2-2.6x and costs a relative 1e-3 on a gradient that is about half
  sampling noise.
- **Do not measure throughput with the log turned up.** Writing a log line
  costs about 1.5 s against an NFS flush. At `--log-every 4` and a 0.488 s
  step, that is 1.5 s of measuring per 2 s of work, and the throughput probe
  reported 149,697 tokens/s where the anchor runs measure 268,590 by wall
  clock -- a factor of 1.8, entirely instrument. Measure at the `log_every` a
  real run uses, and cross-check against wall clock, which cannot lie about
  itself.
- **Time nothing on a GPU without warming it up.** The first call of a kind
  pays cuBLAS or cuSOLVER's one-off setup, and preflight reported 15 TFLOPS for
  a matmul that does 184, and made one image look ten times slower at `eigh`
  than it is. Every timing in `preflight.py` now warms up first.
- **Measure a residual in a precision the machine will not degrade.** The same
  preflight check reported Cayley's error as `4.0e-04` when the truth was
  `3.9e-06`: the retraction was guarded, but `RᵀR` was formed in TF32, so the
  instrument was reporting its own error. It forms the residual in fp64 now.
- **Frobenius norm is not spectral norm.** The rotation angle is set by the
  spectral norm; conflating them produced a wrong theory here about the step
  scaling as `sqrt(cond(A))`, which real gradients do not support.

## Open questions, and what would answer each

| question | how to settle it |
|---|---|
| does curvature help at all | the comparison. Everything else is preparation |
| does the Fisher supply the step scale, or is RMS load-bearing | 2x2: `{pion, ngd} x {scaling, none}` at 1.2B tokens |
| is momentum needed | the step is ~47% sampling noise between two independent halves of 225k tokens; the Fisher reweights that noise but does not average it. Their own ablation says Lie-algebra momentum beats ambient, slightly. Add it *before* `F^-1` if added |
| what `T_fac` | `alpha` is the free readout — it is identically 1 on a fresh basis and falls as the basis drifts. Log it, then choose |
| do rotation angles stay bounded without RMS | logged per step as `angle`. On a toy transformer they spanned 1e-4 to 3e-2 with a 286x spread across layers |
| square vs wide layers | the wide `ffn down` is the only matrix per block with a kernel on the in-side and the only one needing `A^{-1/2}`. Expect it to behave differently |
| does `datasets` 2.2.1 still stream `allenai/c4` | the image resolved `datasets` to 2.2.1, four years older than the `transformers` beside it, because pip backtracked against the base image's pins. Old versions expect dataset loading scripts that the Hub has since replaced with parquet. Settle it with the 1e7-token probe run of `prepare_data.py` before the real corpus build; if it fails, pin `datasets` in the `.def` and rebuild |
| does `S = I` still beat the measured `S` at scale | it does on a toy transformer with ~80k tokens estimating `S` at `d_out = 256`. A real run's EMA sees orders of magnitude more, where the estimate may become accurate enough to invert. The run testing this was killed before finishing. **Do not put the claim in the paper until it is checked** |

## Running things

```bash
pytest -q                                        # 140 tests, 24 s
python -m harness.run --optimizer ngd --lr 1e-3  # one run
python -m harness.run --anchor bilateral         # the calibration run
```

See [`docs/CLUSTER.md`](docs/CLUSTER.md) for the cluster sequence.

## Working on the cluster

**The login node is not for computing.** Anything that touches a GPU, or runs
for more than a moment, goes through `srun` or `sbatch` — see
[`docs/CLUSTER.md`](docs/CLUSTER.md). `pytest -q` on the login node is fine: it
is seconds and CPU-only. Preparing C4 is not. A training run certainly is not.
Administrators watch for this and will kill the process.

**Work asynchronously.** Jobs live in the SLURM queue and survive anything that
happens to the interactive session — a dropped connection, a closed tmux, a
restarted login node. Submit and stop watching; a full run takes hours and
there is nothing to see while it does.

```bash
squeue -u $USER          # what is queued or running
sacct -j <jobid>         # what happened to one that finished
tail -f logs/<name>.out  # only if you actually need to watch
```

Do not hold a session open waiting for a run. Submit, report the job id, and
let the next session read the result from `$DATA_p330/runs`.

**Results outlive sessions, context does not.** Every run writes
`manifest.json` beside its log with the configuration, its hash and the git
commit, which is what makes a result readable later by someone who was not
there when it was submitted.

## House rules

- **Measure before asserting.** Most of the wrong turns in this project were
  confident interpretations of correct arithmetic. The mathematics held
  throughout; the readings of it did not.
- **Every claim in `ALGORITHM.md` carries its number.** Keep it that way — a
  decision without a measurement beside it will be reopened by the next reader.
- **One variable per run.** The comparison is `pion_ablated` against `ngd`.
  Published Pion runs alongside only as context.
- **Tests pin findings, not just behaviour.** Several tests exist because a
  subtle result would otherwise silently regress; their docstrings say which.
