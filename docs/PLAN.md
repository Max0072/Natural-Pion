# Plan, from 2026-09-22

Read `docs/VALIDATION.md` first: it says what the numbers below are and are not.
Notation (`val`, `rot`, `adamw`, `B`, gap, seed) is defined at the top of that
file and used the same way here.

**Status, 2026-09-22 night:** stage 1 is done (job 332045): **H1 holds**, mean gap
+0.038 (seeds 1-3 alone +0.035), all four seeds positive, mean/SE 13.6. **Stage 3
(the horizon ladder) is running, job 332064.** Nothing else has been submitted. Every other stage marked `[compute]` needs an
explicit go-ahead before it runs, because the cluster is shared.

## The claim, stated so it can fail

**Main claim.** In the classical setting -- LLaMA-60M, C4, B = 512 sequences
(131 072 tokens per step), each optimizer given its own tuned `rot` and
`adamw` -- NGD-Pion (`ngd-pion-s`) reaches a lower val than published Pion
(`pion`). Expected size: small, a few hundredths.

**Supporting observation, not a claim the paper depends on.** The advantage
should grow with batch size, because the step is about 96% sampling noise
(measured) and a more accurate preconditioner only pays when the noise is low.
Same picture as K-FAC in the literature (Shallue et al. 2018; Zhang et al.
2019 -- as I recall them; not re-checked against the papers).

**The baseline is published Pion, and only that.** Decision of the user,
2026-09-22. NGD-Pion is compared to Pion as a whole, the way a method is
compared to a method. There is no ablated-Pion arm in this plan: an optimizer
built by switching off Pion's momentum and RMS scaling and forcing a different
retraction is not Pion, and no result on it says anything about Pion. The plan
therefore makes no claim about which component of NGD-Pion is responsible for a
gain, only whether there is one.

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

**H5 result, and what it changes.** `alpha` is not an emphasis heuristic that
merely damps the step; removing it makes NGD-Pion clearly worse than Pion, at the
best rate found. It varies by both time (near 0 for ~100 steps, 1 by step 2500)
and by layer (0.20 on `wv`, 0.90 on `ffn.gate`) at once, which a single scalar
learning rate cannot reproduce -- so "alpha is an implicit warmup" is weakened,
not confirmed, and a derived replacement (option C, still to think about) has to
preserve both axes, not just the time one.

**H3 -- withdrawn.** It was "the preconditioner `F^-1` is what produces the gap",
to be tested against `pion_ablated`. Withdrawn on 2026-09-22 at the user's
decision: the ablated arm is not a baseline (see above), and the comparison that
matters is against published Pion. The hypothesis number is kept so that the
journal's references to it stay readable. What the audit found about that arm,
so that it does not get rebuilt by someone who has not read this: at 150 steps
it is 0.40 *worse than freezing the matrices* (val 6.109 against 5.7071), its
Cayley retraction is forced on it rather than chosen (their truncated
exponential diverges within tens of steps once the RMS scaling is off), its
manifests record the un-ablated settings, and no run of it exists beyond 150
steps.

**H4 -- the advantage survives to the full 9.6B tokens.**
Pinned-AdamW evidence points the wrong way (+0.024 at 786M, -0.014 at 9.6B).
*Test:* first a horizon ladder at tuned settings, 3000 -> 15000 steps, to see
the trend cheaply; then, if the trend holds up, the full pair.
*Ladder design and rule (job 332064, fixed before the numbers):* both arms, a
cross of five cells each around the 3000-step optimum (rot at half and double,
adamw at half and double), seed 0, 15 000 steps. With `gap15` = val(best `pion`
cell) - val(best `ngd-pion-s` cell): `>= +0.020` the advantage persists and the
full-length pair is justified; `+0.010` to `+0.020` decaying but present, still
worth running; `< +0.010` inside noise or gone, and the full length is not spent
as the headline. A best cell on the edge of the cross means the grid is extended
and the number is not quoted.
*Report as:* one tuned run against one tuned run, labelled n = 1 each. Do not
say "ahead" or "tied" without the seed noise from H1 beside it.
*If it goes negative at full length:* the positive result is a horizon
statement -- "NGD-Pion reaches a given val in fewer steps, and more so at larger
batch" -- and must be written as the speed-up in step-equivalents, not as a loss
difference (loss is compressive; see the 2026-08-28 entry).

**H5 -- how much of the advantage lives in `alpha`.** (Added 2026-09-22, after the
trust-region audit; journal same date.) `alpha = quad/curv` is not a trust region
and is the largest per-layer step multiplier in the method (median 0.64 at the
tuned rate). *Test, ablation B:* `--ngd-trust none`, own `rot` grid, then adamw
bracket, then seeds 1-3 paired with the with-alpha and Pion runs. *Rule, fixed in
the header of `scripts/sbatch/notrust_head.sbatch`:* with `R = (3.8412 - V)/0.0458`,
the share of the seed-0 gap to Pion that survives without alpha: `R >= 0.8` alpha
is not needed; `0.2 <= R < 0.8` part of the advantage lives in it; `R < 0.2` the
advantage over Pion is the step-length damping. This ablates a part of the method
under test, not a Pion baseline (ADR 0008 stands).
*Untested hypothesis it makes askable, not a finding:* `alpha` behaves like an
implicit learning-rate warmup (median 0.001 in the first 100 steps, about 0.15 to
step 500, 0.4 at 1000-1500, 0.93 at 2000-2500, 1.000 in the last 500, seed 1), and
the runs have `warmup_steps = 0` for both arms as in Pion's own script. The gap to
Pion (mean of four seeds) is +0.179 at step 500, +0.076 at 1000, +0.029 at 1500,
+0.026 at 2000, +0.038 at 3000: mostly made early. If an explicit warmup in place
of `alpha` on NGD-Pion reproduces what `alpha` does, then the early advantage is
a step-length effect and the horizon decay has an explanation. Cheap (3000 steps),
not yet run, and to be made after B says whether removing `alpha` costs loss.
**Not planned, by decision (ADR 0013): a warmup for Pion.** The baseline keeps the
published recipe; Pion's step is normalised from step 0 and there is nothing in it
to warm up.

## Stages, in order

| stage | what | cost | gate |
|---|---|---|---|
| **0** | documentation and audit: this file, `VALIDATION.md`, `RESUME.md`, journal, stale-number fixes | none, **done 2026-09-22** | -- |
| **1** `[compute]` | **H1: seeds.** Seeds 1, 2, 3 for `ngd-pion-s` (rot 6e-3, adamw 8e-3) and `pion` (rot 1e-3, adamw 8e-3), B = 512, 3000 steps. Six runs | about 42 min per `ngd-pion-s` run and about half that for `pion`, measured from the `a2dhead` timestamps with seven jobs sharing a node. Two runs per seed at a time, three seeds one after another: **about 2.5 h wall** | user go-ahead |
| **2** | ~~H3: `pion_ablated`~~ **withdrawn 2026-09-22** by the user's decision: the baseline is published Pion | -- | -- |
| **3** `[compute]` | **H4 ladder.** Wave 1 (job 332064) **done**: gap15 = -0.0048, but ngd's best cell (3.5302, rot 3e-3, adamw 8e-3) sits on the edge of its cross on both axes, so the number is not quoted. pion's optimum (3.5254, rot 1e-3, adamw 8e-3) is interior. **Wave 2 submitted as job 332970**, a 3x3 grid around the corner both axes pointed at | wave 1: 30 GPU-h. Wave 2: six runs, ~3.5 h each, ~3.5-4 h wall (rtx6002 idle, all six run at once) |
| **4** `[compute]` | **H4 full length:** the two arms, 73 242 steps, tuned `adamw` carried from stage 3, one neighbouring `adamw` each | `ngd-pion-s` about 19.7 h (0.967 s/step on rtx, close to the 24 h partition cap; resubmitting resumes from the checkpoint), `pion` about 9.4 h | stage 3 shows the trend is not negative |
| **5** optional `[compute]` | H2 extension (B = 128) | a grid | after stage 1 |
| **6** | **H5, ablation B: done** (job 332073). Best without alpha, rot 2.5e-4 adamw 8e-3, val 4.1969 -- interior, but 0.356 *worse than Pion itself* (R = -7.8, far below the `R < 0.2` band anticipated). Checked against the same rot with alpha: 0.47 apart at rot=2e-3, so alpha is not a rescaled learning rate. Waves 2-3 not needed, the result is not near any threshold | six runs, done |

Stage 1 comes before every other compute stage because it costs about 2.5 h,
it is the only stage that can put an error bar on today's headline number, and
it decides whether stages 3-4 are worth 30 GPU-hours.

### Rules for every stage (the project's own, and why)

* **Concurrent runs on one node should share a seed**, because different seeds
  read different corpus windows and can starve each other for I/O. The user
  waived this for stage 1 ("run them all in parallel") and all six runs share
  one node at once; the step time against the solo figure is the check that the
  rule was not needed.
* **Pin a node** (`-w`), but no specific node is trusted any more -- rtx6002
  wedged twice in one evening on 2026-09-22 at ordinary load, after days of
  being the reliable one. **Check that the step counter advances**, more than
  once: right after start (RUNNING in SLURM does not mean computing, and a
  pinned job can sit `(Resources)` behind another user), and again periodically
  through a long array, since a wedge here has twice developed well after a
  healthy start rather than at step 0.
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
| H1 holds, H4 holds | the main claim, with an error bar: NGD-Pion beats published Pion at the classical setting, both tuned. The strong version |
| H1 holds, H4 negative | "NGD-Pion is more sample-efficient at the classical batch and the advantage grows with batch, but is absorbed at long horizons": a regime paper, stated as a step-equivalent speed-up |
| H1 dies | no measurable advantage at the classical batch; what remains is the noise-regime analysis (H2) and the lr-robustness result already on disk (the `rho` controller converges from a hundredfold range of starting rates) |

None of these attributes a gain to one component of the method. NGD-Pion is a
bundle and is measured as one against Pion.

The user has said the paper should rest on a positive empirical result. The
table above is the reason to run stage 1 first: it is the cheapest way to learn
which row we are in before spending the expensive stages.

## Time

`AGENTS.md` puts the venue deadline at roughly one month after 2026-08-23,
which is about now; today is 2026-09-22. I do not know the actual date, and the
plan is tiered so that it survives either answer:

* **a day or less:** stage 1 only (about 2.5 h) and write what it shows, with
  V1's correction in place of the "tie";
* **about a week:** stages 1 and 3;
* **longer:** all of it, stage 4 being about a day of wall-clock for the pair.

Partition access and GPU caps change within hours, so re-test with `sinfo` and
`sbatch --test-only` before relying on `docs/CLUSTER.md`.

## Decisions taken, and still open

Taken, 2026-09-22: no deadline pressure ("just do it"); stage 1 may run and was
submitted as job 332045, all three seeds concurrent; everything goes to `main`;
**the baseline is published Pion and there is no ablated arm** (H3, stage 2).

Recorded in `docs/decisions/` (ADR 0008-0012): the baseline is published Pion,
`pion_ablated` is deleted from the code (0011), and the run records are archived
in `runs-record/` (0012).

Open:

1. Whether to rewrite `ALGORITHM.md`, which still carries the ablated design in
   "Дизайн сравнения" and, in three tables, the reversed `S = I`, no-momentum and
   `t_fac = 100`.
