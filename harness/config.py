"""Run configuration, and the identity of a run.

Every field that can change a result lives here, and the hash of this object
is the run's name. Two runs with the same hash saw the same configuration; two
runs with the same hash and the same seed should see the same batches. That is
what makes a sweep auditable after the fact rather than a folder of guesses.

Defaults follow `opt_llama_60M_pion.sh`. Their script derives the step count
rather than stating it -- `TOKEN=9.6`, `GLOBAL_BATCH=512`, and
`TRAIN_ITER = TOTAL_TOKENS / GLOBAL_BATCH / 256` -- which gives 73242 steps at
131072 tokens each. 9.6B tokens is 8x the Chinchilla-optimal budget for a 60M
model, and their comment line `# 1: 1.2; 2: 2.4; 4: 4.8; 8: 9.6;` is that
multiplier table.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field

from .model import ModelConfig

__all__ = ["RunConfig", "git_commit", "machine", "untracked_modules"]


def git_commit() -> str:
    """The working tree's commit, or a marker when it is not a repository.

    `-dirty` means a **tracked** file differs from the commit -- the code that
    ran is not the code the hash names, which is the only thing this marker is
    for. Untracked files are deliberately excluded. They used to count, and the
    consequence was that a notes file created beside the repository stamped two
    anchor runs `-dirty` while the code they executed was clean; a marker that
    fires on things which cannot change a result stops meaning anything exactly
    when it needs to.

    The exception is an untracked *module*, which a run can import and which
    would therefore be invisible here. `untracked_modules` reports those
    separately rather than folding them into a flag that says nothing about
    which file it means.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return out.stdout.strip()[:12] + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def untracked_modules() -> list[str]:
    """Untracked `.py` files, which `git_commit` does not count as dirty.

    An untracked module is the one kind of untracked file a run can actually
    execute, so it belongs in the manifest even though it does not belong in
    `-dirty`. Normally empty; a non-empty list is worth reading before trusting
    the commit hash beside it.
    """
    try:
        out = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True, check=True,
        ).stdout.split()
        return sorted(f for f in out if f.endswith(".py"))
    except Exception:
        return []


def machine(device: str | None = None) -> dict:
    """Which card, which host, which torch. Recorded, never hashed."""
    import platform

    import torch

    info = {
        "host": platform.node(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device) if device is not None else None,
    }
    if device is not None and str(device).startswith("cuda") and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        info["arch"] = f"sm_{cap[0]}{cap[1]}"
    return info


@dataclass(frozen=True)
class RunConfig:
    # what is being compared
    optimizer: str = "ngd-pion"     # ngd-pion | ngd-pion-ref | pion | pion_ablated | adamw
    lr: float = 1e-3
    # AdamW's learning rate. `0` means "follow `lr`", which is what their
    # published configuration does -- one rate for both optimizers, 1e-3 --
    # so the anchor and every configuration written before this field existed
    # are unaffected.
    #
    # Setting it is what any statement about the *rotational* optimizer's
    # learning rate requires. Tied together, raising `lr` also raises AdamW's
    # rate on the embedding, the output head, the norm gains and the biases,
    # and AdamW detonates there long before the rotation does anything at all.
    # The `eta` sweep recorded in docs/JOURNAL.md before 2026-08-26 varied both
    # at once and measured AdamW.
    adamw_lr: float = 0.0
    seed: int = 0

    # the model, as their script configures it
    model: ModelConfig = field(default_factory=ModelConfig)

    # schedule, as their script configures it
    batch_sequences: int = 512      # global batch; 512 * 256 = 131072 tokens/step
    train_steps: int = 73242        # TOTAL_TOKENS / GLOBAL_BATCH / 256 = 9.6e9/512/256
    lr_min: float = 1e-5
    warmup_steps: int = 0
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Arithmetic for the *model*. On an Ampere-or-newer card torch runs fp32
    # matrix operations in TF32 -- ten bits of mantissa -- and on an RTX PRO
    # 6000 Blackwell that is worth 2.2x to 2.6x on exactly the shapes this
    # model multiplies, measured. The gradient it produces is then wrong by a
    # relative 1e-3, against a step that is about 47% sampling noise, and
    # Megatron runs their own experiments in bf16, which is coarser still. So
    # it is on.
    #
    # It is emphatically **not** on inside the optimizer. TF32 there moves the
    # singular values of a weight by a relative 1.0 over 200 two-sided steps
    # (2.6e-04 without), which would silently destroy the one property the
    # method is built on, and it corrupts the covariance the method inverts.
    # `ngd_pion` turns it off around its own linear algebra and does not
    # depend on this field being anything in particular.
    #
    # A scientific field regardless, not plumbing: it changes results, so it
    # is in the hash.
    tf32: bool = True

    # Autocast for the forward pass and the loss. `bf16` is what their own runs
    # use -- `--bf16` in opt_llama_60M_pion.sh -- so it is the setting that
    # makes the anchor a fair test rather than a faster one.
    #
    # It defaults to bf16 because everything that actually runs is bf16:
    # `anchor_config` sets it, and every throughput measurement on record
    # passed `--precision bf16` explicitly. The default said `fp32` for a while
    # after precision stopped being a known difference from their setup, and
    # the mismatch was invisible until jobs 246662/246663 -- a plain
    # `train.sbatch` invocation with no `--precision` -- died allocating a
    # 15.67 GiB logit tensor, which is 512x256x32100 in fp32 against 8.4 GiB in
    # bf16. Nothing had launched a full run through that path before, so no
    # measurement caught it.
    #
    # Weights and the optimizer stay fp32 regardless: autocast changes what the
    # operations compute in, not what the parameters are, which is the same
    # arrangement as Megatron's fp32 master weights. The covariance is fed from
    # activations that are then bf16, and that is measured to be harmless --
    # thousands of independent roundings cancel in the average, perturbing `A`
    # by 4.5e-05, well under the 1e-4 spectral floor. Storing `A` in bf16 would
    # not be, and `CovarianceAccumulator` refuses to.
    precision: str = "bf16"          # fp32 | bf16

    # NGD-Pion
    ngd_eps: float = 1e-4
    ngd_beta: float = 0.95
    ngd_t_fac: int = 100
    # EMA rate for `D = E[delta delta^T]`, used only by `ngd-pion-s`. Shorter
    # than `ngd_beta` on purpose: the backward signal depends on the current
    # loss surface and on everything downstream of a layer, so it moves faster
    # than the activation statistics do.
    ngd_beta_backward: float = 0.5
    # Exponent on the operator's eigenvalues for `ngd-pion-pow`: 1.0 is the
    # natural gradient, 0.5 is what Adam does to its second moment.
    ngd_power: float = 0.5
    # Draw the labels for `E[dd^T]` from the model instead of from the data,
    # every this many steps; `0` keeps the empirical Fisher. Only `ngd-pion-s`
    # reads it.
    #
    # This is the difference between a curvature matrix and a noise covariance.
    # `E[gg^T]` equals `E[grad^2 l]` only when the labels come from the model's
    # own predictive distribution; with the data's labels it is the *empirical*
    # Fisher, whose scale is set by how much per-sample gradients disagree
    # rather than by how curved the loss is. Measured here, the per-component
    # signal-to-noise `|E[g]| / sqrt(E[g^2])` peaks at 0.035, so `E[g^2]`
    # exceeds `E[g]^2` by nearly three orders and `F^-1 G` is inflated by the
    # same amount -- which is why `eta*` sits three orders below the 2 the
    # theory gives for the true Fisher.
    #
    # `D` is an EMA, so the extra pass need not run every step.
    ngd_fisher_mc_every: int = 0
    # Diagnostic only, and only for `ngd-pion-exact`: how often to measure the
    # curvature the step is *not* preconditioned by, and on how many tokens.
    # `alpha = quad/curv` is identically 1 on a fresh basis because `curv` is
    # formed with the same operator that built `X`; this measures the same
    # quadratic form against the per-token truth instead. 0 disables it.
    ngd_exact_every: int = 30
    ngd_exact_tokens: int = 4096
    # 0 disables. A cap on the rotation applied per step, in radians, which is
    # the bound `alpha` structurally cannot provide: `alpha` is identically 1
    # on a fresh basis, so the step after each refactorisation is unbounded.
    ngd_angle_max: float = 0.0
    ngd_alpha_max: float = 1.0
    # Additive Tikhonov damping for `ngd-pion-damped`, as a multiple of a
    # reference scale frozen at the first refactorisation (the median
    # `d_max` over layers). Absolute, not per-layer: a per-layer
    # `lam = c * d_max` would be blind to a uniformly flat layer in exactly
    # the way the relative floor `max(d, eps * d_max)` is, and that
    # blindness is what the measurements pin the huge steps on.
    ngd_lam: float = 0.0
    # Move `lam` by the Levenberg-Marquardt rule on the reduction ratio,
    # which the harness already measures every logged step. With `lam`
    # carrying the trust region, `lr` is meant to stay at 1.
    ngd_lam_adapt: bool = True
    # First-moment averaging for `ngd-pion-m`. `A` and `D` are already EMAs, but
    # they are the *denominator*: `powered.py` shows the operator's eigenvalues
    # are exactly the second moments of the generator components, so this method
    # has Adam's `v` and has never had Adam's `m`. `"lie"` buffers the two
    # generators separately -- the variant Pion's published 60M numbers use --
    # `"ambient"` buffers the raw gradient first, `"none"` is off. Pion's second
    # moment is deliberately not taken: `F` already is one.
    ngd_momentum: str = "none"       # none | lie | ambient
    ngd_beta1: float = 0.9

    # Shampoo on so(n) -- `shampoo-pion`. Read by nothing else; see
    # `ngd_pion/shampoo.py` for why the preconditioner is built from the
    # generators themselves rather than from network statistics.
    #
    # Exponent per side. `0.25` is the original Shampoo, so the two sides
    # compose to Adagrad's `-1/2`. `0.0` disables preconditioning and leaves
    # the raw generator, which is ablated Pion and therefore the control arm
    # that costs nothing to run in the same harness.
    shampoo_power: float = 0.25
    # `0` accumulates a plain sum, as the original does. That carries an
    # implicit `t^-1/2` decay which compounds with the cosine schedule over
    # 73242 steps -- a real interaction to watch on a long run, not an
    # artefact. A positive value makes the accumulator an EMA instead.
    shampoo_beta: float = 0.0
    # `floor` is the relative floor `max(lam, eps * lam_max)` this package uses
    # everywhere; `shift` is the original's `eps I`. The choice is not cosmetic:
    # the shift is not homogeneous, so it forfeits the scale invariance that is
    # the reason for adopting Shampoo here at all. `test_shampoo.py` pins both
    # the invariance and its loss.
    shampoo_damping: str = "floor"   # floor | shift
    shampoo_eps: float = 1e-4
    # Record the per-layer plane-angle spread every this many steps; `0` is off
    # and is the default, because it costs an `svdvals` per side per layer and
    # 24 of this model's 56 weights have a 1376-dimensional side. Turn it on
    # for a diagnostic run, not for a training run. `angle` is free and always
    # recorded.
    shampoo_plane_every: int = 0

    # vanilla Pion
    pion_scaling: str = "rms"
    pion_rms: float = 0.2
    pion_momentum: str = "lie"      # the variant their published 60M numbers use
    pion_retraction: str = "trunc"
    # Bilateral, which is their better published number for this model -- 3.3575
    # against 3.3654 for alternate -- and the one `pion_ablated` is forced to.
    # This default said `True` for a while with nothing defending it, which
    # would have run the context arm in a variant the measurement arm does not
    # use, putting a second difference between them. The context arm exists so
    # the ablated baseline cannot be called a straw man, and that argument only
    # works if the context is Pion at its best.
    #
    # `anchor_config` sets this from `update_side` and does not read the
    # default, so the anchors were never affected either way.
    pion_alternate: bool = False
    pion_beta1: float = 0.9
    pion_beta2: float = 0.95
    # Their `--pion-use-second-momentum` is `store_true` with `default=None`,
    # and `_use_second_momentum` falls through to `False` when it is absent.
    # `opt_llama_60M_pion.sh` passes it only when `USE_SECOND_MOMENTUM` is set
    # in the environment, so their published runs normalise the Lie momentum by
    # its first moment alone.
    #
    # This harness divided by `sqrt(v) + 1e-8` regardless, because
    # `Pion.__init__` defaults `beta2=0.95` and `build_optimizers` never passed
    # a value. Worse, neither beta was a field here, so the choice was outside
    # the hash: two runs differing in it produced the same name and shared a
    # directory. Both are fields now, and the second moment is off by default
    # because that is what their number comes from.
    pion_second_moment: bool = False
    # Their `pion_qkv_split_granularity`, which defaults to `"head"` and which
    # the 60M script does not override: Q is rotated in per-head blocks, K and
    # V whole. This harness keeps Q, K and V as separate matrices already, so
    # only the granularity of Q was ever different.
    pion_split_q_per_head: bool = True

    # AdamW, for everything Pion does not own -- the embedding, the output head
    # and the norm gains, which are 32.9M of this model's 58.2M parameters.
    # `build_optimizers` used to construct `torch.optim.AdamW` without betas and
    # take torch's `(0.9, 0.999)`, while their script sets `--adam-beta2 0.95`
    # (and `--pion-beta2 0.95` beside it, so 0.95 either way). Like the Pion
    # betas before them, these were not fields, so the choice sat outside the
    # configuration hash and no manifest recorded it.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    # Megatron's standard override gives `wd_mult = 0.0` to every parameter with
    # `len(shape) == 1` or a name ending in `.bias`, so their RMSNorm gains are
    # never decayed. This harness applied `weight_decay` to all of `rest`, and
    # at this schedule that multiplies a gain by 0.0248 over the run -- a
    # **40x shrink** of a parameter that starts at 1.0.
    #
    # It matters more here than it would elsewhere. A network normally absorbs
    # decayed norm gains by growing the linear weights, and under Pion that
    # route is closed: the spectrum of every rotated matrix is frozen for the
    # whole run. The only compensation left is the embedding and the head.
    decay_norms_and_biases: bool = False

    # plumbing, deliberately excluded from the hash
    data_path: str = "data/c4_train.bin"
    val_path: str = "data/c4_val.bin"
    eval_every: int = 500
    eval_batches: int = 20
    log_every: int = 50
    # Gradient accumulation chunk. Excluded from the hash: it changes speed and
    # memory, not the result.
    #
    # 512 removes accumulation entirely -- one micro-batch is the whole step --
    # and lets the covariance see all 131k tokens of a step rather than a slice.
    # Measured on an RTX PRO 6000 Blackwell (96 GB) in job 246613, on warm page
    # cache: pion 0.460 s/step and ngd-pion 0.722 at 512, against 0.869 and
    # 0.673 at 256. So 512 is the right default for Pion by a wide margin, and
    # for NGD-Pion the two are close enough that seeing the full step's tokens
    # decides it.
    #
    # It fits only with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, which
    # every GPU sbatch script in scripts/sbatch now sets. Without that flag
    # ngd-pion OOMs at 512 -- not for lack of room (385 MB of optimizer state
    # against 14.6 GB of headroom) but because the unbatched `_apply` leaves the
    # allocator unable to serve 7-30 MB contiguous blocks. Run outside those
    # scripts and you must set it yourself.
    micro_batch: int = 512
    out_dir: str = "runs"

    _EXCLUDED = ("data_path", "val_path", "eval_every", "eval_batches",
                 "log_every", "micro_batch", "out_dir")

    @property
    def tokens_per_step(self) -> int:
        return self.batch_sequences * self.model.seq_len

    @property
    def total_tokens(self) -> int:
        return self.tokens_per_step * self.train_steps

    def scientific_fields(self) -> dict:
        """Everything that can change the result, and nothing that cannot."""
        d = asdict(self)
        for key in self._EXCLUDED:
            d.pop(key, None)
        return d

    @property
    def hash(self) -> str:
        blob = json.dumps(self.scientific_fields(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:10]

    @property
    def name(self) -> str:
        return f"{self.optimizer}-lr{self.lr:g}-s{self.seed}-{self.hash}"

    def manifest(self, device: str | None = None) -> dict:
        """What gets written beside the results so a run can be reconstructed.

        The machine is part of that. Two cards do not agree bitwise -- cuBLAS
        picks different algorithms per architecture, floating-point addition is
        not associative, and some backward kernels accumulate atomically in no
        fixed order -- so a comparison whose arms ran on different hardware has
        a variable in it that nothing in the configuration records. Writing the
        device down does not prevent that; it makes it answerable afterwards.
        """
        return {
            "name": self.name,
            "hash": self.hash,
            "git_commit": git_commit(),
            "untracked_modules": untracked_modules(),
            "tokens_per_step": self.tokens_per_step,
            "total_tokens": self.total_tokens,
            "machine": machine(device),
            "config": asdict(self),
        }
