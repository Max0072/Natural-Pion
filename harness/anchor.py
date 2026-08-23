"""The anchor run: reproduce a published number before trusting our own.

Everything this project measures happens inside a harness we wrote. A reviewer
is entitled to ask how we know the harness is equivalent to theirs -- perhaps
our Pion baseline looks weak because of a bug, a different data order or a
subtly different architecture, and no measurement taken inside the harness can
answer that.

The anchor answers it from outside: run *their* configuration in *our* harness
and check we land on *their* published number. Once that holds, differences
measured inside are attributable to what we changed.

Their paper reports exactly one pair of concrete figures for the 60M setting,
in section 2.4.3: final loss **3.3575** for the bilateral update and **3.3654**
for alternate, both with Lie+Lie momentum. Note this is not the configuration
their 60M shell script runs -- that script sets `alternate` and
`transported_ambient_ambient`. The anchor has to follow the number, not the
script, or a miss says nothing.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .config import RunConfig

__all__ = ["anchor_config", "TARGETS", "TOLERANCE", "KNOWN_DIFFERENCES", "check"]

#: Final training loss reported in section 2.4.3, by update side.
TARGETS = {"bilateral": 3.3575, "alternate": 3.3654}

#: What counts as a match. Their own bilateral-to-alternate gap is 0.0079, so a
#: reproduction has to be tighter than that to be saying anything; 0.02 leaves
#: room for data-order and harness differences without admitting a real defect.
TOLERANCE = 0.02

#: Reasons the number could miss even with correct code. A miss should be
#: diagnosed against this list before anything is called a bug.
KNOWN_DIFFERENCES = (
    "flat token stream: windows may span documents, where Megatron's indexed "
    "dataset respects document boundaries",
    "a C4 subset in our own order, not their full stream",
    "separate Q, K, V projections rather than Megatron's fused QKV matrix",
    "Megatron's optimizer wrapper: where gradient clipping and fp32 master "
    "weights sit relative to the step",
    "which parameters weight decay reaches",
)


def anchor_config(update_side: str = "bilateral", **overrides) -> RunConfig:
    """Their published-number configuration, not their shell script's.

    `bilateral` targets 3.3575, `alternate` targets 3.3654. Running both is
    worth the second run: reproducing the 0.0079 *gap* between them is a
    sharper check on the harness than reproducing either level, because the
    gap is insensitive to data and initialisation in a way the level is not.
    """
    if update_side not in TARGETS:
        raise ValueError(f"update_side must be one of {sorted(TARGETS)}, got {update_side!r}")
    return replace(
        RunConfig(),
        optimizer="pion",
        lr=1e-3,
        lr_min=1e-5,
        warmup_steps=0,
        train_steps=73242,          # 9.6e9 / 512 / 256, as their script derives it
        batch_sequences=512,
        weight_decay=0.1,
        grad_clip=1.0,
        pion_scaling="rms",
        pion_rms=0.2,
        pion_momentum="lie",        # Lie+Lie: the variant the numbers come from
        pion_retraction="trunc",    # their degree-2 truncated exponential
        pion_alternate=update_side == "alternate",
        **overrides,
    )


def _last_attempt(log_path: str | Path) -> list[dict]:
    """Rows belonging to the most recent run in an appended log.

    The log is appended so a preempted run keeps its history, which means a
    re-run of the same configuration lands in the same file. Reading everything
    would mix attempts and report the wrong step count.
    """
    rows = [json.loads(line) for line in Path(log_path).read_text().splitlines()]
    starts = [i for i, r in enumerate(rows) if r.get("event") == "start"]
    return rows[starts[-1] + 1 :] if starts else rows


def final_loss(log_path: str | Path, window: int = 10) -> float:
    """Training loss at the end of a run, averaged over the last logged points.

    A single logged value is one batch and is noisy; their figure is read off a
    curve. Averaging a short window is the closest honest equivalent.
    """
    rows = _last_attempt(log_path)
    losses = [r["train_loss"] for r in rows if "train_loss" in r]
    if not losses:
        raise ValueError(f"{log_path} contains no training loss")
    tail = losses[-window:]
    return sum(tail) / len(tail)


def last_step(log_path: str | Path) -> int:
    """Highest step of the most recent attempt, not of the whole file."""
    return max((r["step"] for r in _last_attempt(log_path) if "step" in r), default=-1)


def check(
    log_path: str | Path, update_side: str = "bilateral", expected_steps: int | None = None
) -> dict:
    """Compare a finished run against the published figure.

    Refuses to judge a run that did not finish. A truncated run always sits far
    above the target simply because it stopped early, and reporting that as a
    miss would put a meaningless verdict in the log for someone to trip over
    later. `expected_steps` is what makes the difference detectable.
    """
    target = TARGETS[update_side]
    got = final_loss(log_path)
    delta = got - target
    reached = last_step(log_path) + 1
    complete = expected_steps is None or reached >= expected_steps
    return {
        "update_side": update_side,
        "target": target,
        "measured": round(got, 4),
        "delta": round(delta, 4),
        "relative": round(delta / target, 5),
        "steps_run": reached,
        "steps_expected": expected_steps,
        "status": "complete" if complete else "incomplete",
        "matched": bool(complete and abs(delta) <= TOLERANCE),
        "tolerance": TOLERANCE,
    }
