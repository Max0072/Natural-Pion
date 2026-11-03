# Positioning and the RLVR experiment

Written 2026-11-03 from a discussion of [`FINAL.md`](FINAL.md). It fixes how the paper is positioned, records
a new number (steps to Pion's loss), and lays out the RLVR experiment that the positioning now rests on.
Nothing here has been run on the cluster yet.

---

## 1. Positioning: follow Pion's own framing

**How Pion positions itself** (arXiv:2605.12492, read 2026-11-03). It does not claim to be faster than Muon.
It is "a stable and competitive alternative" to AdamW and Muon, a different class (multiplicative, not
additive), motivated by:

* *algorithmic spectrum control*: the weight spectral norm is fixed by the dynamics, no explicit
  normalisation, muP-compatible, linked to generalisation;
* *minimum hyperspherical energy*: preserved from the Gaussian initialisation.

Its evidence: at 1.3B / 54B tokens it is 0.0125 behind Muon in val loss (2.7350 vs 2.7225) and calls that
"comparable"; it wins on downstream benchmarks (47.69 vs 46.34), on stability indicators (max attention
logit, SwiGLU activation norm), and on RLVR (Table 3). All its ablations are at 60M, like ours.

**Our positioning.** Pion fixes the constraint (the iso-spectral manifold); we show which direction to move
inside it. Everything Pion has is inherited, and more exactly (Cayley is exactly orthogonal, the truncated
exponential is not), and the natural direction reaches the same loss in fewer steps. Two claims:

1. **Same guarantees as Pion, fewer steps** -- measured, §2.
2. **Fewer steps matter most in RL**, where a step costs minutes of generation and our per-step overhead
   (~0.13 s) is invisible. And the natural gradient is RL's own lineage: natural policy gradient (Kakade
   2001), TRPO, ACKTR (K-FAC in RL). This claim needs the experiment in §3.

The current one-sentence claim in `FINAL.md` ("1.29x fewer steps or less") is too narrow and its number is
an artefact (see §2); rewrite it along these lines once §2 has seeds.

**What reviewers will ask, and the answer we need:**

* *"Muon is better per step than us in pretraining (3.6505 vs 3.7216 at 3000 steps). Why not Muon?"* --
  In Pion's RLVR Muon is last (32.08 vs Pion 36.12 on Qwen3-1.7B). The answer holds only if Muon is in
  our RLVR run too.
* *"One model size."* -- A second, larger pretraining size (e.g. 350M) is the next priority after RLVR.
* *Stability plots in Pion's style* (attention logits, activation norms, spectrum drift) are cheap and
  should be added: we inherit those properties.

## 2. Steps to Pion's loss (new number)

`FINAL.md` §5.3 says "at least 1.28-1.29x fewer steps". That bound is the design of the equal-wall-time
comparison restated: step counts were chosen by wall time, and 1.28 is the per-step cost ratio. The real
ratio, from the val curves in `runs-record/` (`scripts/analysis/steps_ratio.py`, seed 0, Algorithm 2 at
the best angle per horizon vs Pion lr 1e-3):

| Pion steps | Pion val | ours, from finals | ratio | ours, mid-run crossing | ratio |
|---|---|---|---|---|---|
| 3 840 | 3.7804 | < 3 000 | > 1.28 | 2 398 | 1.60 |
| 4 500 | 3.7404 | < 3 000 | > 1.50 | 2 783 | 1.62 |
| 19 200 | 3.4960 | 12 868 | 1.49 | 12 907 | 1.49 |
| **73 242** | **3.3706** | **44 172** | **1.66** | 50 449 | 1.45 |

* *From finals*: our final val of each full-schedule run (3000, 15 000, 56 900, 73 242 steps),
  interpolated linearly in log T. The curve is convex in log T, so the chord overestimates steps: the ratio
  is if anything understated. Between 15 000 and 56 900 there is no point, so 1.66 is a coarse
  interpolation.
* *Mid-run*: first step at which one of our runs drops below Pion's final val. Pessimistic, because the
  cosine has not decayed there.

So **at least ~1.45x fewer steps, and the advantage does not shrink with length**. Divided by the 1.28 cost
ratio this is also a 1.13-1.3x wall-clock speed-up in pretraining.

**To do:**

* Seeds: add the seed 1-3 runs at 15 000 (ours) and 19 200 / 3 840 (Pion) to the script for a spread.
* One run of ours near 40 000 steps (angle 0.05) to turn the 1.66 into a measurement (~7 h RTX).
* Then update `FINAL.md` §5.3 and the one-sentence claim.

## 3. The RLVR experiment

### 3.1 Pion's setup (Appendix D.4) -- reproduce it exactly

| | |
|---|---|
| models | Qwen3-1.7B, 400 steps; DeepSeek-R1-Distill-Qwen-1.5B, 781 steps |
| data | DeepMath |
| algorithm | GRPO, 12 rollouts per prompt; global and on-policy mini-batch 128 prompts (64 for R1-Distill) |
| stack | VeRL + vLLM, H200; optimizer updates in fp32 |
| lr | **1e-6 for every method, not tuned** |
| generation | max 4096 tokens (8192 for R1-Distill) |
| their Pion | no second moment, alternating update, attention heads and FFN gate/up split |
| evaluation | POLARIS: AIME24 and AIME25 avg@32, AMC23 avg@8, Minerva avg@4, OlympiadBench avg@8; temperature 1.0, top-k 20, top-p 0.8 |

Their averages, Qwen3-1.7B: Base 16.87, Muon 32.08, AdamW 34.82, **Pion 36.12**. R1-Distill-1.5B: Base
30.20, AdamW 35.97, Muon 37.32, **Pion 38.32**. Their Figure 13 shows accuracy against steps, with Pion
fastest: that is the axis we want to win on.

**Primary metric for us: steps to reach a target accuracy** (e.g. Pion's final average), from accuracy-vs-step
curves. The Table-3-style final table is secondary.

### 3.2 Stage 0: infrastructure (do first, cheap)

**Findings, 2026-11-03 (read, not run).** Pion's code is public: https://github.com/Sphere-AI-Lab/pion (MIT;
linked from https://spherelab.ai/pion/; README still says "coming soon"). It holds a Megatron-LM fork and a VeRL
fork (`0.8.0.dev`) with `verl/custom_optimizer/{pion,muon}.py`, the scripts
`examples/grpo_trainer/run_qwen3_1.7b_{pion,adamw,muon}_deepmath.sh` and the DeepMath preprocessing
(`zwhe99/DeepMath-103K`, 1% held out). As published it will not run without fixes, both checked in the clone:

* `verl/custom_optimizer/__init__.py` imports `pion_ambient` and `pion_ambient_v2`, which are not in the repo;
* `fsdp_workers.py` lines 629 and 637 use `use_orig_for_fsdp`, never defined.

**Whole matrices are unexplained.** The scripts run FSDP1 with `use_orig_params=True` on 8 GPUs, a 1-D mesh,
so FULL_SHARD; the optimizer has no gather (`summon_full_params`, `all_gather`, DTensor: none), and under
FULL_SHARD the original parameters are views into the local shard. Their own guard raises if no 2-D parameter
is visible. How their runs got whole matrices is not in the code we can see. Upstream VeRL has no
gather-for-optimizer path for FSDP (Muon issue #3246 closed as not planned); Muon exists only on the
Megatron backend (PR #7120).

Other details from their script: train and mini-batch 128, prompt 2048 / response 4096, vLLM TP 2, n = 12,
KL loss 0.001 (`low_var_kl`), entropy 0, `total_epochs=1` (about 800 steps; the paper's 400-step stop is not in
the script). Their AdamW script also changes the micro-batch (8 vs 4): not a one-variable comparison.
Evaluation: POLARIS (https://github.com/ChenxinAn-fdu/POLARIS) ships AIME24/25 only; AMC23, Minerva and
OlympiadBench as run by Pion are not found. Qwen3-1.7B: hidden 2048, FFN 6144, 28 layers, 16 heads, 8 KV
heads, head_dim 128, vocab 151 936, tied embeddings.

**Decision proposed:** start from their VeRL fork (it has the data, reward and scripts), fix the two defects,
and give the actor FSDP `NO_SHARD`: 1.7B in fp32 is ~7 GB of weights and as much of gradients, which fits a
B200, and every rank then holds whole matrices. The optimizer step can run replicated at first, then split by
matrix across ranks with a broadcast. The same change makes their Pion runnable, so all methods share it.
Base image: `docker://verlai/verl:vllm017.latest`, their pin.

1. **Is Pion's code with the VeRL integration public** (project page linked from the arXiv abstract)? If
   yes, plug our optimizer into it instead of writing the integration.
2. **FSDP.** VeRL trains with FSDP, which gives each rank a flat shard. Our method, like Pion, needs the
   whole matrix `W` (`W^T G`, the rotation). Options: (a) no parameter sharding -- 1.7B in fp32 is ~7 GB,
   fits on a B200 -- and split the matrices across ranks for the optimizer step, then broadcast; (b) gather
   per matrix, as distributed Muon does. Start with (a).
3. **Memory of the factors.** Qwen3-1.7B: hidden 2048, FFN 6144, 28 layers, GQA. One 6144x6144 fp32 factor is
   ~150 MB; with both sides, bases and Lie momentum it is tens of GB in total. Measure before the first run.
4. **Container.** Checked 2026-11-03: `ngd-pion.sif` (NGC PyTorch 26.07, torch 2.13, transformers 5.15,
   flash-attn 2.7.4) has no vLLM, VeRL, Ray, DeepSpeed or Accelerate. RLVR needs a new image built on VeRL's
   own supported image (versions pinned by VeRL, not by us); building it is the first stage-0 job.
5. **Time one refactor** (one set of 6144-wide `eigh`), separately from the step; see 3.3.
6. **Time one step** (20 steps of GRPO on 8 B200) to fix the budget in §3.6.

### 3.3 Decisions specific to RL

* **Which Fisher.** (Review note: in pretraining the true and the empirical Fisher did not differ, 3.9091 vs
  3.9102 at 3000 steps; the extra backward below is a cost. Start with the empirical `D` and keep the
  `log pi` Fisher as the ablation, unless stage 0 shows the zero-advantage dropout empties `D`.)
  In pretraining `D` is the covariance of loss gradients (empirical Fisher,
  ADR 0006). The natural policy gradient's Fisher is `E[grad log pi grad log pi^T]` over sampled tokens,
  with no advantage weighting. In GRPO the groups that are all right or all wrong have zero advantage and
  would drop out of the empirical `D`. Proposal: collect `D` from a separate backward of `log pi` on a
  subsample of tokens, only on the statistics steps. Then the method is literally NPG/TRPO on the
  iso-spectral manifold -- the strongest framing for the paper. Record it as an ADR.
* **Refactor often -- but measure what a refactor costs first.** 400 steps at T_fac 25 is 16 refactors, so a
  shorter cadence is attractive. It is not free at this size: Qwen3-1.7B's FFN is 6144 wide, so the gate/up
  out-side and the down in-side are 6144x6144 pencils, each a congruence path of three fp32 `eigh` (B^-1/2,
  C, the pencil), 3 such matrices x 28 layers per refactor. fp32 `eigh` at 6144 is seconds per call on a GPU,
  so a refactor can be minutes, comparable to a generation step (correction 2026-11-03, review of this plan).
  Stage 0 times one refactor; then choose T_fac, and spread the `eigh` calls over the 8 GPUs if needed.
* **The angle.** RL moves weights very little. Centre the theta grid on the angle Pion actually turns per
  step at lr 1e-6 (log the largest rotation-plane angle in a Pion run), 3-4 values around it. RL usually
  runs a constant lr: test constant and cosine (in pretraining constant was worse, 6.3 of `FINAL.md`).
* **The rest of the model** (embeddings, head, norms): AdamW at 1e-6, the same for every method.

### 3.4 Stage 2: tuning, with equal budget for all

Short runs (100 steps): theta grid for ours, an lr grid of the same size for Pion and Muon. Pion tuned no
method; tuning only ours will be called unfair (the same reasoning as ADR 0013). Choose by the
accuracy-vs-step curve.

### 3.5 Stage 3: the main run

* **Qwen3-1.7B, 400 steps: AdamW, Muon, Pion, ours.**
* **First check that our Pion reproduces their 36.12** within noise. Without that nothing else convinces.
* **Cheap evaluation every 25 steps** (e.g. AIME24/25 avg@8 and AMC23) for the step curves; the full POLARIS
  protocol at the end.
* **Seeds:** at least 2-3 for ours and Pion. avg@32 removes evaluation noise, not training noise.
* Then, if budget allows: DeepSeek-R1-Distill-1.5B, 781 steps.

### 3.6 Budget (rough estimate, to be replaced by the measurement in stage 0)

One step is 128 x 12 = 1536 rollouts, a few million generated tokens; expect 1-3 min per step on 8 B200,
so 7-20 h per 400-step run, **~55-160 B200-hours per run**. The allocation is 1000 B200-hours
(`CLUSTER.md`): enough for the main comparison (4 methods, 1 seed) and probably seeds for ours and Pion;
tuning and the second model are in question.

Checked 2026-11-03: the `b200` partition is two nodes, `b201` and `b202`, 8 B200 each, 24 h limit per job
(so a 400-step run needs checkpoint and resume). Both were busy: four 20-hour jobs of another user on `b202`,
and on `b201` one job holding 1 GPU but all 2015 GB of memory, which blocks the rest of the node. A full
node may queue for a long time. `sreport`: p330 has used ~811 GPU-hours since 2026-08-01 (RTX and B200
together); SLURM sets no hard cap.

### 3.7 Initialisation does not transfer

The alpha-3 tail result (`FINAL.md` 6.5) is a pretraining result. In RLVR the model is pretrained: the frozen
spectrum is Qwen's own, for us and for Pion. That is a positioning point rather than a loss: a
spectrum-preserving fine-tune cannot change the pretrained model's singular values, a built-in regulariser
(say it in the paper).

### 3.8 Risks

* RL gradients are very noisy; `F^-1` can amplify noise along low-curvature directions. Momentum and the
  floor `eps` are the levers; the floor may need raising.
* If our Pion does not reproduce theirs, the comparison is ours-vs-our-Pion only, as in pretraining
  (decision 0008), and must say so.
