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

__all__ = ["RunConfig", "git_commit", "machine"]


def git_commit() -> str:
    """The working tree's commit, or a marker when it is not a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return out.stdout.strip()[:12] + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


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
    optimizer: str = "ngd"          # ngd | pion | pion_ablated | adamw
    lr: float = 1e-3
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

    # Autocast for the forward pass and the loss. `bf16` is what their own
    # runs use -- `--bf16` in opt_llama_60M_pion.sh -- so it is the setting
    # that makes the anchor a fair test rather than a faster one, and it is
    # first in anchor.KNOWN_DIFFERENCES while this is `fp32`.
    #
    # Weights and the optimizer stay fp32 regardless: autocast changes what the
    # operations compute in, not what the parameters are, which is the same
    # arrangement as Megatron's fp32 master weights. The covariance is fed from
    # activations that are then bf16, and that is measured to be harmless --
    # thousands of independent roundings cancel in the average, perturbing `A`
    # by 4.5e-05, well under the 1e-4 spectral floor. Storing `A` in bf16 would
    # not be, and `CovarianceAccumulator` refuses to.
    precision: str = "fp32"          # fp32 | bf16

    # NGD-Pion
    ngd_eps: float = 1e-4
    ngd_beta: float = 0.95
    ngd_t_fac: int = 100
    ngd_alpha_max: float = 1.0

    # vanilla Pion
    pion_scaling: str = "rms"
    pion_rms: float = 0.2
    pion_momentum: str = "lie"      # the variant their published 60M numbers use
    pion_retraction: str = "trunc"
    pion_alternate: bool = True

    # plumbing, deliberately excluded from the hash
    data_path: str = "data/c4_train.bin"
    val_path: str = "data/c4_val.bin"
    eval_every: int = 500
    eval_batches: int = 20
    log_every: int = 50
    # Gradient accumulation chunk. Excluded from the hash: it changes speed and
    # memory, not the result. The rtx nodes carry RTX PRO 6000 Blackwell cards
    # with 96 GB, where the whole 512-sequence batch plausibly fits in one go --
    # try `--micro-batch 512` there, which removes accumulation entirely and
    # lets the covariance see all 131k tokens of a step rather than a slice.
    micro_batch: int = 128
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
            "tokens_per_step": self.tokens_per_step,
            "total_tokens": self.total_tokens,
            "machine": machine(device),
            "config": asdict(self),
        }
