"""The training loop.

One process, one GPU, one run. There is no distributed anything here on
purpose: a 60M model fits comfortably on a single card, so N GPUs are worth
far more as N concurrent experiments than as one faster experiment, and the
sweeps this project needs are wide rather than deep.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn

from ngd_pion.hooks import attach
from ngd_pion.optimizer import NGDPion
from ngd_pion.pion_baseline import Pion

from .config import RunConfig
from .data import TokenCorpus
from .instrument import layer_diagnostics, summarise
from .model import Transformer

__all__ = ["build_optimizers", "lr_at", "train"]


def lr_at(step: int, cfg: RunConfig) -> float:
    """Cosine decay to `lr_min`. Their script warms up over zero steps."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.train_steps - cfg.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1.0 + math.cos(math.pi * progress))


def build_optimizers(model: Transformer, cfg: RunConfig):
    """Split the parameters the way Pion does, and give each half an optimizer.

    Returns `(rotational_opt, adamw_opt, recorder)`. `rotational_opt` is
    `None` for the plain AdamW baseline, in which case AdamW owns everything,
    and `recorder` is `None` for every optimizer that needs no activations.
    """
    linears, rest = model.parameter_split()
    weights = [m.weight for m in linears]

    if cfg.optimizer == "adamw":
        return None, torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay), None

    if cfg.optimizer == "ngd":
        rot = NGDPion(
            weights, lr=cfg.lr, beta=cfg.ngd_beta, eps=cfg.ngd_eps,
            alpha_max=cfg.ngd_alpha_max, t_fac=cfg.ngd_t_fac,
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
        )
        recorder = None
    else:
        raise ValueError(f"unknown optimizer {cfg.optimizer!r}")

    adamw = torch.optim.AdamW(rest, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return rot, adamw, recorder


@torch.no_grad()
def evaluate(model: Transformer, batches) -> float:
    model.eval()
    total = 0.0
    for x, y in batches:
        total += float(model(x, y)[1])
    model.train()
    return total / max(1, len(batches))


def train(cfg: RunConfig, device: str = "cpu", max_steps: int | None = None) -> Path:
    torch.manual_seed(cfg.seed)
    out = Path(cfg.out_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(cfg.manifest(), indent=2))
    # Appending rather than truncating keeps a preempted run's history, but it
    # also means a re-run with the same configuration writes into the same
    # file. A start marker separates them so readers can take the last attempt.
    log = (out / "log.jsonl").open("a")
    log.write(json.dumps({"event": "start", "steps": max_steps or cfg.train_steps,
                          "time": time.time()}) + "\n")
    log.flush()

    model = Transformer(cfg.model).to(device)
    rot, adamw, recorder = build_optimizers(model, cfg)
    names = {id(m.weight): n for n, m in model.named_modules() if isinstance(m, nn.Linear)}

    train_data = TokenCorpus(cfg.data_path, cfg.model.seq_len, seed=cfg.seed)
    val_data = TokenCorpus(cfg.val_path, cfg.model.seq_len, seed=cfg.seed + 1)
    val_batches = val_data.fixed_batches(cfg.micro_batch, cfg.eval_batches, seed=1234, device=device)

    steps = max_steps or cfg.train_steps
    accum = max(1, cfg.batch_sequences // cfg.micro_batch)
    started = time.time()

    for step in range(steps):
        lr = lr_at(step, cfg)
        for opt in (rot, adamw):
            if opt is None:
                continue
            for group in opt.param_groups:
                group["lr"] = lr

        model.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for micro in range(accum):
            # one micro-batch per step feeds the covariance; see ActivationRecorder
            if recorder is not None:
                recorder.enabled = micro == 0
            x, y = train_data.batch(cfg.micro_batch, device)
            _, loss = model(x, y)
            (loss / accum).backward()
            loss_sum += loss.detach().item() / accum

        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for opt in (rot, adamw):
            if opt is not None:
                opt.step()

        if step % cfg.log_every == 0 or step == steps - 1:
            row = {"step": step, "lr": lr, "train_loss": loss_sum,
                   "wall": round(time.time() - started, 1)}
            if isinstance(rot, NGDPion):
                row.update(summarise(layer_diagnostics(rot, names)))
            log.write(json.dumps(row) + "\n")
            log.flush()

        if (step + 1) % cfg.eval_every == 0 or step == steps - 1:
            row = {"step": step, "val_loss": evaluate(model, val_batches)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            torch.save({"model": model.state_dict(), "step": step}, out / "checkpoint.pt")

    log.close()
    if recorder is not None:
        recorder.remove()
    return out
