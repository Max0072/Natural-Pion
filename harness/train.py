"""The training loop.

One process, one GPU, one run. There is no distributed anything here on
purpose: a 60M model fits comfortably on a single card, so N GPUs are worth
far more as N concurrent experiments than as one faster experiment, and the
sweeps this project needs are wide rather than deep.
"""

from __future__ import annotations

import json
import math
import os
import platform
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn

from ngd_pion.fast import FastNGDPion
from ngd_pion.hooks import attach
from ngd_pion.optimizer import NGDPion
from ngd_pion.pion_baseline import Pion

from .config import RunConfig
from .data import TokenCorpus
from .instrument import BackwardProbe, layer_diagnostics, summarise
from .model import Transformer

__all__ = ["build_optimizers", "lr_at", "train"]


def lr_at(step: int, cfg: RunConfig, base: float | None = None) -> float:
    """Cosine decay to `lr_min`. Their script warms up over zero steps.

    `base` overrides the peak so a second optimizer can share the schedule's
    *shape* without sharing its scale. The floor is scaled by the same factor,
    which keeps `lr_min / lr` -- the depth of the decay -- identical between
    them; a fixed absolute floor would otherwise mean something quite different
    to a rate three hundred times smaller.
    """
    peak = cfg.lr if base is None else base
    floor = cfg.lr_min * (peak / cfg.lr) if cfg.lr else cfg.lr_min
    if step < cfg.warmup_steps:
        return peak * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.train_steps - cfg.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


# Two names for NGD-Pion, and the split is about provenance rather than taste.
#
# `ngd_pion/optimizer.py` is the reference: a transcription of ALGORITHM.md,
# left unoptimised so there is something to check against. `fast.py` is what
# actually runs. Today they are bit-identical -- `tests/test_fast.py` asserts
# equal weight trajectories -- but the next items on the cost list (a
# Newton-Schulz retraction, a batched step) change rounding, and from then on
# the two produce different numbers.
#
# `RunConfig.name` is built from this string, so giving each implementation its
# own name is what keeps a run directory an honest record of the code that
# produced it. A silent alias would defeat exactly that, which is why the old
# bare "ngd" is not accepted: failing loudly costs one puzzled minute, and
# guessing wrong costs a run.
#
# The fast one gets the short name because it is the one the paper runs, and
# because the opposite arrangement fails badly: forgetting a suffix would buy
# 62 days of wall-clock instead of 14 hours, and nothing would say so until the
# job had been queued.
NGD_IMPLEMENTATIONS = {"ngd-pion": FastNGDPion, "ngd-pion-ref": NGDPion}


def build_optimizers(model: Transformer, cfg: RunConfig):
    """Split the parameters the way Pion does, and give each half an optimizer.

    Returns `(rotational_opt, adamw_opt, recorder)`. `rotational_opt` is
    `None` for the plain AdamW baseline, in which case AdamW owns everything,
    and `recorder` is `None` for every optimizer that needs no activations.
    """
    linears, rest = model.parameter_split()
    weights = [m.weight for m in linears]

    if cfg.optimizer == "adamw":
        return None, _adamw(model.parameters(), cfg), None

    if cfg.optimizer in NGD_IMPLEMENTATIONS:
        kw = {}
        if cfg.optimizer != "ngd-pion-ref":
            # the reference takes no diagnostic options, on purpose
            kw["diag_every"] = cfg.log_every
            kw["angle_max"] = cfg.ngd_angle_max
        rot = NGD_IMPLEMENTATIONS[cfg.optimizer](
            weights, lr=cfg.lr, beta=cfg.ngd_beta, eps=cfg.ngd_eps,
            alpha_max=cfg.ngd_alpha_max, t_fac=cfg.ngd_t_fac, **kw,
        )
        recorder = attach(linears, rot)
    elif cfg.optimizer in ("pion", "pion_ablated"):
        ablated = cfg.optimizer == "pion_ablated"
        rot = Pion(
            weights, lr=cfg.lr,
            scaling="none" if ablated else cfg.pion_scaling,
            rms=cfg.pion_rms,
            momentum="none" if ablated else cfg.pion_momentum,
            # ablating the scaling forces an exact retraction: without it their
            # truncated exponential inflates and diverges within tens of steps
            retraction="cayley" if ablated else cfg.pion_retraction,
            alternate=False if ablated else cfg.pion_alternate,
            beta1=cfg.pion_beta1,
            beta2=cfg.pion_beta2 if cfg.pion_second_moment else None,
            # Per-head Q blocks are their published configuration, so the
            # anchor needs them -- but only the anchor. `pion_ablated` is the
            # arm `ngd` is measured against, `NGDPion` rotates whole matrices,
            # and giving one arm a finer granularity than the other would put a
            # second variable in a comparison whose whole claim is that it has
            # one. Lifting this means teaching `NGDPion` about blocks, not
            # turning it on here.
            row_blocks=(
                {id(m.weight): cfg.model.heads for m in model.query_projections()}
                if cfg.pion_split_q_per_head and not ablated
                else None
            ),
        )
        recorder = None
    else:
        raise ValueError(
            f"unknown optimizer {cfg.optimizer!r}; expected one of "
            + ", ".join(repr(k) for k in (*NGD_IMPLEMENTATIONS, "pion", "pion_ablated", "adamw"))
        )

    adamw = _adamw(rest, cfg)
    return rot, adamw, recorder


def _adamw(params, cfg: RunConfig) -> torch.optim.AdamW:
    """AdamW for the parameters Pion does not own, in their configuration.

    Two groups, because Megatron's standard override exempts every 1-D
    parameter and every bias from weight decay and this harness did not. See
    `RunConfig.decay_norms_and_biases` for why that is worse under a
    spectrum-preserving optimizer than under an ordinary one.
    """
    betas = (cfg.adam_beta1, cfg.adam_beta2)
    if cfg.decay_norms_and_biases:
        groups = [{"params": list(params), "weight_decay": cfg.weight_decay}]
    else:
        params = list(params)
        flat = [p for p in params if p.dim() <= 1]
        rest = [p for p in params if p.dim() > 1]
        groups = [
            {"params": rest, "weight_decay": cfg.weight_decay},
            {"params": flat, "weight_decay": 0.0},
        ]
    return torch.optim.AdamW(groups, lr=cfg.adamw_lr or cfg.lr, betas=betas, eps=cfg.adam_eps)


def _autocast(cfg: RunConfig, device: str):
    """Autocast for the forward pass, or a no-op context.

    Only the forward pass and the loss. The optimizer step runs outside it,
    under `no_grad` and with its own exact-fp32 guard, so the retraction and
    the covariance keep the precision they need whatever this is set to.
    """
    if cfg.precision == "fp32":
        return nullcontext()
    if cfg.precision != "bf16":
        raise ValueError(f"precision must be 'fp32' or 'bf16', got {cfg.precision!r}")
    return torch.autocast("cuda" if str(device).startswith("cuda") else "cpu",
                          dtype=torch.bfloat16)


@torch.no_grad()
def evaluate(model: Transformer, batches, cfg: RunConfig, device: str) -> float:
    model.eval()
    total = 0.0
    for x, y in batches:
        with _autocast(cfg, device):
            total += float(model(x, y)[1])
    model.train()
    return total / max(1, len(batches))


class RunLock:
    """One writer per run directory, with a heartbeat so a resume is not blocked.

    A run's directory is named by the configuration hash, so the same
    configuration on two machines is the same directory -- and two trainers
    appending to one log and overwriting one checkpoint produce a run that
    looks finished and is not. Nothing downstream could detect it.

    The lock is refreshed at every checkpoint, roughly every four minutes at
    this scale, so a lock older than the grace period belonged to something
    that died. That is deliberately taken over rather than refused: the cluster
    caps jobs at 24 hours and resubmitting to resume is the intended workflow,
    which a lock left behind by a SIGKILL would otherwise break.
    """

    def __init__(self, path: Path, grace: float = 900.0) -> None:
        self.path = path
        self.grace = grace
        self.owner = (f"{platform.node()} pid {os.getpid()} "
                      f"slurm {os.environ.get('SLURM_JOB_ID', '-')}")

    def take(self, force: bool = False) -> None:
        if self.path.exists() and not force:
            age = time.time() - self.path.stat().st_mtime
            if age < self.grace:
                raise SystemExit(
                    f"{self.path} is held by {self.path.read_text().strip()} "
                    f"({age:.0f}s ago). Two trainers in one run directory "
                    f"interleave their logs and overwrite each other's "
                    f"checkpoints. Use a different --out-dir, or --force if "
                    f"that process is gone."
                )
            print(f"taking over a lock {age/60:.0f} min stale, left by "
                  f"{self.path.read_text().strip()}", flush=True)
        self.path.write_text(self.owner + "\n")

    def touch(self) -> None:
        try:
            self.path.write_text(self.owner + "\n")
        except OSError:
            pass

    def release(self) -> None:
        self.path.unlink(missing_ok=True)


def _save(path: Path, step: int, model, rot, adamw, data) -> None:
    """Write the checkpoint, atomically.

    A job killed while `torch.save` is writing -- the 24 h wall, a preemption,
    a node failure -- leaves a truncated file where the resume path expects a
    checkpoint, and the run then has to start over from nothing. Writing beside
    it and renaming avoids that: `os.replace` is atomic within a filesystem, so
    whatever checkpoint exists is a complete one.
    """
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "rot": rot.state_dict() if rot is not None else None,
            "adamw": adamw.state_dict(),
            "data_rng": data.rng_state,
            "torch_rng": torch.get_rng_state(),
        },
        tmp,
    )
    os.replace(tmp, path)


def _restore(path: Path, model, rot, adamw, data) -> int:
    """Reload a checkpoint, returning the step to continue from."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if rot is not None and state["rot"] is not None:
        rot.load_state_dict(state["rot"])
    adamw.load_state_dict(state["adamw"])
    data.rng_state = state["data_rng"]
    torch.set_rng_state(state["torch_rng"])
    return state["step"] + 1


def train(
    cfg: RunConfig, device: str = "cpu", max_steps: int | None = None, resume: bool = True,
    force: bool = False,
) -> Path:
    """Train one run, picking up from a checkpoint if one is already there.

    Resuming by default is what the cluster wants: every partition caps at 24
    hours, a full 9.6B run may not fit inside that, and a requeued job should
    continue rather than start over. The sampler's position is restored too, so
    a resumed run does not replay the batches it already saw.
    """
    torch.manual_seed(cfg.seed)
    # Applies to the model's own matmuls as well as the optimizer's, which
    # guards itself in any case. Off by default: see RunConfig.tf32.
    torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
    torch.backends.cudnn.allow_tf32 = cfg.tf32
    torch.set_float32_matmul_precision("high" if cfg.tf32 else "highest")
    out = Path(cfg.out_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    lock = RunLock(out / ".run.lock")
    lock.take(force)
    (out / "manifest.json").write_text(json.dumps(cfg.manifest(device), indent=2))
    # Appending rather than truncating keeps a preempted run's history, but it
    # also means a re-run with the same configuration writes into the same
    # file. A start marker separates them so readers can take the last attempt.
    log = (out / "log.jsonl").open("a")
    log.write(json.dumps({"event": "start", "steps": max_steps or cfg.train_steps,
                          "time": time.time()}) + "\n")
    log.flush()

    model = Transformer(cfg.model).to(device)
    rot, adamw, recorder = build_optimizers(model, cfg)
    # Per-layer rows, alongside the summary that goes on the training log.
    # `summarise` reduces to a min and a max, and questions that motivated the
    # instrument in the first place -- does the required step size depend on
    # depth, does the antisymmetric part of the gradient survive bf16 -- cannot
    # be answered from two order statistics over 56 weights.
    diagnostics = (out / "diagnostics.jsonl").open("a") if isinstance(rot, NGDPion) else None
    # Measurement only, attached alongside the diagnostics file and nowhere else.
    probe = BackwardProbe(model.parameter_split()[0]) if diagnostics is not None else None
    names = {id(m.weight): n for n, m in model.named_modules() if isinstance(m, nn.Linear)}

    train_data = TokenCorpus(cfg.data_path, cfg.model.seq_len, seed=cfg.seed)
    val_data = TokenCorpus(cfg.val_path, cfg.model.seq_len, seed=cfg.seed + 1)
    val_batches = val_data.fixed_batches(cfg.micro_batch, cfg.eval_batches, seed=1234, device=device)

    steps = max_steps or cfg.train_steps
    accum = max(1, cfg.batch_sequences // cfg.micro_batch)
    started = time.time()

    # A corpus smaller than the budget is not an error here and never will be:
    # windows are sampled with replacement and there is no epoch, so the run
    # would quietly show the same tokens twice. Log the ratio instead, where a
    # reader can see it.
    passes = train_data.epochs_for(steps * cfg.tokens_per_step)
    log.write(json.dumps({"event": "corpus", "tokens": len(train_data),
                          "passes": round(passes, 3)}) + "\n")
    log.flush()

    first = 0
    checkpoint = out / "checkpoint.pt"
    if resume and checkpoint.exists():
        first = _restore(checkpoint, model, rot, adamw, train_data)
        log.write(json.dumps({"event": "resume", "from_step": first}) + "\n")
        log.flush()
        if first >= steps:
            log.close()
            if diagnostics is not None:
                diagnostics.close()
            lock.release()
            if recorder is not None:
                recorder.remove()
            if probe is not None:
                probe.remove()
            return out

    window_time, window_step = time.time(), first
    for step in range(first, steps):
        lr = lr_at(step, cfg)
        # Two schedules when `adamw_lr` is set, one when it is not. Sharing the
        # rate is their published design and stays the default; decoupling is
        # what makes a sweep over `lr` a sweep over the rotational optimizer
        # rather than over AdamW.
        adamw_lr = lr if not cfg.adamw_lr else lr_at(step, cfg, cfg.adamw_lr)
        if rot is not None:
            for group in rot.param_groups:
                group["lr"] = lr
        for group in adamw.param_groups:
            group["lr"] = adamw_lr

        model.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for micro in range(accum):
            # one micro-batch per step feeds the covariance; see ActivationRecorder
            if recorder is not None:
                recorder.enabled = micro == 0
            if probe is not None:
                # only on the steps that get logged, so it costs nothing on the rest
                probe.enabled = micro == 0 and (
                    step % cfg.log_every == 0 or step == steps - 1
                )
            x, y = train_data.batch(cfg.micro_batch, device)
            with _autocast(cfg, device):
                _, loss = model(x, y)
            (loss / accum).backward()
            loss_sum += loss.detach().item() / accum

        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for opt in (rot, adamw):
            if opt is not None:
                opt.step()

        if step % cfg.log_every == 0 or step == steps - 1:
            now = time.time()
            elapsed = now - started
            # Throughput decides which partition a run belongs on: a 60M model
            # does not fill a B200, so the peak-FLOPS ratio against an RTX card
            # is not the ratio that matters. Two numbers, because the average
            # over a whole attempt carries its startup and warm-up with it and
            # reads low on a short run -- and step 1 is a 200-step run whose
            # answer allocates three thousand GPU-hours. `window` is the rate
            # since the previous logged row; take that one.
            done_here = step - first + 1
            span, moved = now - window_time, step - window_step
            row = {"step": step, "lr": lr, "train_loss": loss_sum,
                   "wall": round(elapsed, 1),
                   "tokens_per_sec": round(cfg.tokens_per_step * done_here / max(elapsed, 1e-9)),
                   "tokens_per_sec_window": round(
                       cfg.tokens_per_step * moved / max(span, 1e-9)) if moved else None}
            if str(device).startswith("cuda"):
                # Peak since the last log line, not since the run began: what
                # decides `micro_batch` is what a step needs, and the head
                # materialises vocab-by-tokens logits, so the answer is not
                # obvious from the parameter count.
                row["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                torch.cuda.reset_peak_memory_stats()
            window_time, window_step = now, step
            if isinstance(rot, NGDPion):
                rows = layer_diagnostics(rot, names, probe)
                row.update(summarise(rows))
                if diagnostics is not None:
                    for r in rows:
                        diagnostics.write(json.dumps(r) + "\n")
                    diagnostics.flush()
            log.write(json.dumps(row) + "\n")
            log.flush()

        if (step + 1) % cfg.eval_every == 0 or step == steps - 1:
            row = {"step": step, "val_loss": evaluate(model, val_batches, cfg, device)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            _save(checkpoint, step, model, rot, adamw, train_data)
            lock.touch()

    log.close()
    if diagnostics is not None:
        diagnostics.close()
    lock.release()
    if recorder is not None:
        recorder.remove()
    if probe is not None:
        probe.remove()
    return out
