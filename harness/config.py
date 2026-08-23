"""Run configuration, and the identity of a run.

Every field that can change a result lives here, and the hash of this object
is the run's name. Two runs with the same hash saw the same configuration; two
runs with the same hash and the same seed should see the same batches. That is
what makes a sweep auditable after the fact rather than a folder of guesses.

Defaults follow `opt_llama_60M_pion.sh`, with one unresolved conflict recorded
in `train_steps`: their script's 37500 steps at 131072 tokens gives 4.9B
tokens, while the paper says 9.6B for the same experiments.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field

from .model import ModelConfig

__all__ = ["RunConfig", "git_commit"]


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
    train_steps: int = 37500        # their script; the paper implies twice this
    lr_min: float = 1e-5
    warmup_steps: int = 0
    weight_decay: float = 0.1
    grad_clip: float = 1.0

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
    micro_batch: int = 32           # gradient accumulation chunk
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

    def manifest(self) -> dict:
        """What gets written beside the results so a run can be reconstructed."""
        return {
            "name": self.name,
            "hash": self.hash,
            "git_commit": git_commit(),
            "tokens_per_step": self.tokens_per_step,
            "total_tokens": self.total_tokens,
            "config": asdict(self),
        }
