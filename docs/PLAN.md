# Plan, from 2026-09-22

Read `docs/VALIDATION.md` first: it says what the numbers below are and are not.
Notation (`val`, `rot`, `adamw`, `B`, gap, seed) is defined at the top of that
file and used the same way here.

**Nothing in this plan has been submitted.** Every stage marked `[compute]`
needs an explicit go-ahead before it runs, because the cluster is shared.

## The claim, stated so it can fail

**Main claim.** In the classical setting -- LLaMA-60M, C4, B = 512 sequences
(131 072 tokens per step), each optimizer given its own tuned `rot` and
`adamw` -- NGD-Pion (`ngd-pion-s`) reaches a lower val than published Pion
(`pion`). Expected size: small, a few hundredths.

**Mechanism claim.** The advantage comes from the Fisher preconditioner
`F^-1`, and it grows with batch size, because the step is about 96% sampling
noise (measured) and a more accurate preconditioner only pays when the noise is
low. Same picture as K-FAC in the literature (Shallue et al. 2018; Zhang et al.
2019 -- as I recall them; not re-checked against the papers).

**What is already on disk, and how much it supports** (all seed 0, n = 1):

    B = 512, 393M tokens, both tuned   ngd 3.7954   pion 3.8412   gap +0.046
    B = 512, 786M tokens, adamw pinned ngd 3.7149   pion 3.7386   gap +0.024
    B = 512, 9.6B tokens, adamw pinned ngd 3.3860   pion 3.3719   gap -0.014
    B = 2048, 393M tokens, both tuned  ngd 4.0514   pion 4.1592   gap at most +0.108
    B = 2048, 786M tokens, adamw pinned ngd 3.9421  pion 4.0781   gap +0.136

The three B = 512 rows say the same thing about horizon: **the gap shrinks as
the run gets longer** (the last two rows share one AdamW setting and are the only
comparable pair). The B = 2048 rows say the gap grows with batch. Everything
here rests on one seed per cell.

## Hypotheses, predictions, and what would kill each

Criteria are written before the numbers exist, and are meant to be edited by the
user now rather than bent later.

**H1 -- the short-run advantage is real.**
Seed 0 gives gap +0.046 at B = 512, 3000 steps, each arm at its tuned optimum.
*Test:* three more seeds (1, 2, 3) per arm at those same optima, then the four
paired gaps (same seed = same init and same data order for both arms).
*Holds if:* all four gaps are positive and mean gap / standard error > 2.35
(one-sided 5% with 3 degrees of freedom).
*Dies if:* mean gap <= 0, or the sign flips in two or more seeds. The claim then
becomes "no measurable difference at B = 512", and the paper is about the regime
where there is one (H2).
*Caveat carried forward:* the optima came from a factor-3 `rot` grid on seed 0.
A positive result at those optima is a lower-bound-ish statement for NGD-Pion
only if Pion's true optimum is not much better than 1e-3; that uncertainty is
listed, not removed.

**H2 -- the advantage grows as the noise falls.**
*Prediction:* gap(B) increases with B at fixed tokens; from the 96.4% noise
figure the step signal is small at B = 512 and larger at 2048.
*Have:* +0.046 (512) and at most +0.108 (2048), tuned; +0.024 and +0.136, pinned.
*Cheap extension:* a third point below 512 (B = 128) to see the trend continue
downward. Needs its own two-dimensional grid; **optional**.
*Kills the mechanism claim if:* gap at 2048 is not larger than at 512 once
Pion's AdamW edge is closed (its optimum at 2048 is still moving outward).
*Never claim:* that a larger batch is a reason to use the method. At the same
tokens B = 2048 is 0.26-0.32 worse than B = 512 for both arms.

**H3 -- `F^-1` is what produces the gap.**
Today's gap is NGD-Pion against *published* Pion, which differs in five things
(see V2 of the validation file). *Test:* `pion_ablated` -- no momentum, no
scaling, Cayley, so one variable from `ngd-pion-s` -- at B = 512, 3000 steps,
with its own tuned `rot` and `adamw`.
*Prediction if `F^-1` matters:* val(`pion_ablated`) is clearly worse than
val(`ngd-pion-s`) by more than the seed noise from H1.
*Dies if:* `pion_ablated` matches `ngd-pion-s` within that noise. Then what
beats Pion is not the preconditioner but the removal of RMS scaling and the
exact retraction, and the paper's mechanism claim has to be rewritten around
that. This is the most important stage in the plan and the only one of the
paper's claims with no measurement yet.
*Dose-response, if H3 holds:* the exponent `p` in `F^-p`. `ngd_s_power` now
reaches the live arm (commit `40a7886`) and `scripts/sbatch/powercal.sbatch`
exists but was never run. If gain rises with `p` the mechanism is the
preconditioner and not a side effect. Needs a step-scale calibration first,
which the sbatch file already argues for.

**H4 -- the advantage survives to the full 9.6B tokens.**
Pinned-AdamW evidence points the wrong way (+0.024 at 786M, -0.014 at 9.6B).
*Test:* first a horizon ladder at tuned settings, 3000 -> 15000 steps, to see
the trend cheaply; then, if the trend holds up, the full pair.
*Report as:* one tuned run against one tuned run, labelled n = 1 each. Do not
say "ahead" or "tied" without the seed noise from H1 beside it.
*If it goes negative at full length:* the positive result is a horizon
statement -- "NGD-Pion reaches a given val in fewer steps, and more so at larger
batch" -- and must be written as the speed-up in step-equivalents, not as a loss
difference (loss is compressive; see the 2026-08-28 entry).

## Stages, in order

| stage | what | cost | gate |
|---|---|---|---|
| **0** | documentation and audit: this file, `VALIDATION.md`, `RESUME.md`, journal, stale-number fixes | none, **done 2026-09-22** | -- |
| **1** `[compute]` | **H1: seeds.** Seeds 1, 2, 3 for `ngd-pion-s` (rot 6e-3, adamw 8e-3) and `pion` (rot 1e-3, adamw 8e-3), B = 512, 3000 steps. Six runs | about 42 min per `ngd-pion-s` run and about half that for `pion`, measured from the `a2dhead` timestamps with seven jobs sharing a node. Two runs per seed at a time, three seeds one after another: **about 2.5 h wall** | user go-ahead |
| **2** `[compute]` | **H3: `pion_ablated`** at B = 512, 3000 steps. First a calibration of `rot` (its old sweep was 150 steps at pinned adamw and is void, `runs/ablated/`), then a `rot` x `adamw` grid on seed 0, then the seeds from stage 1 at its optimum | calibration under 1 h; grid about 5-8 runs of ~25 min; seeds 3 runs | stage 1 result |
| **3** `[compute]` | **H4 ladder:** both arms at their optima, 15000 steps (1.97B tokens, 20% of the budget), AdamW neighbours on both sides | about 3.5 h per `ngd-pion-s` run, about 1.7 h per `pion`; four to six runs | stages 1-2 |
| **4** `[compute]` | **H4 full length:** the two arms, 73 242 steps, tuned `adamw` carried from stage 3, one neighbouring `adamw` each | `ngd-pion-s` about 19.7 h (0.967 s/step on rtx, close to the 24 h partition cap; resubmitting resumes from the checkpoint), `pion` about 9.4 h | stage 3 shows the trend is not negative |
| **5** optional `[compute]` | H2 extension (B = 128) and the `F^-p` dose-response | a grid each | after 1-2 |

Stage 1 comes before every other compute stage because it costs about 2.5 h,
it is the only stage that can put an error bar on today's headline number, and
it decides whether stages 3-4 are worth 30 GPU-hours.

### Rules for every stage (the project's own, and why)

* **Concurrent runs on one node must share a seed.** Different seeds read
  different corpus windows and starve each other for I/O. So stage 1 goes as
  three waves, one seed per wave, both arms inside a wave, not as six runs at
  once.
* **Pin the node** (`-w rtx6002`, the one that has been reliable) and **check
  that the step counter advances** a few minutes after start: RUNNING in SLURM
  does not mean computing, and a pinned job can sit `(Resources)` behind
  another user.
* **Each arm gets its own learning rates.** Never compare two arms at a shared
  `rot`, and never compare a grid edge against another arm's optimum.
* **Set `--adamw-lr` explicitly.** `0` silently means "same as `--lr`".
* `log_every` coprime with `t_fac = 25` (23 or 97): a multiple of 25 samples
  only the one step after a refactorisation.
* One variable per comparison. Before interpreting any result, diff the two
  manifests and say what differs; the mistakes already made here were all
  explanations built on an unchecked setup.
* Write the job ids and configurations into `docs/JOURNAL.md` as jobs are
  submitted, and rewrite `docs/RESUME.md` rather than appending to it.

## What each outcome means for the paper

| outcome | what can be written |
|---|---|
| H1 holds, H3 holds, H4 holds | the main claim, with an error bar, mechanism attributed to `F^-1`. The strong version |
| H1 holds, H3 holds, H4 negative | "NGD-Pion is more sample-efficient at the classical batch and the advantage grows with batch, but is absorbed at long horizons": a regime paper, stated as a step-equivalent speed-up |
| H1 holds, H3 dies | the advantage is real but not the preconditioner's; the contribution is the exact-retraction, unscaled rotation. A different and smaller paper |
| H1 dies | no measurable advantage at the classical batch; what remains is the noise-regime analysis (H2) and the lr-robustness result already on disk (the `rho` controller converges from a hundredfold range of starting rates) |

The user has said the paper should rest on a positive empirical result. The
table above is the reason to run stage 1 first: it is the cheapest way to learn
which row we are in before spending the expensive stages.

## Time

`AGENTS.md` puts the venue deadline at roughly one month after 2026-08-23,
which is about now; today is 2026-09-22. I do not know the actual date, and the
plan is tiered so that it survives either answer:

* **a day or less:** stage 1 only (about 2.5 h) and write what it shows, with
  V1's correction in place of the "tie";
* **about a week:** stages 1, 2 and 3;
* **longer:** all of it, stage 4 being about a day of wall-clock for the pair.

Partition access and GPU caps change within hours, so re-test with `sinfo` and
`sbatch --test-only` before relying on `docs/CLUSTER.md`.

## Decisions for the user

1. The date of the deadline, which fixes the tier.
2. Seeds: three extra per arm (four in total) is what H1's threshold assumes.
   Fewer means a weaker test, more costs more waves.
3. Whether stage 2 (`pion_ablated`) may be started in parallel with stage 1 on
   a second node, or waits for it.
4. Whether to rewrite `ALGORITHM.md` to carry the reversals (`S` measured,
   momentum as an arm, `t_fac` 25); it is the specification and currently says
   the opposite in three tables.
