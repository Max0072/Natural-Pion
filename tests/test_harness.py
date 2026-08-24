"""The training harness: model shapes, data, schedule, and one end-to-end run."""

import json
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from harness.config import RunConfig
from harness.data import TokenCorpus
from harness.model import ModelConfig, Transformer
from harness.train import build_optimizers, lr_at, train
from ngd_pion.optimizer import NGDPion
from ngd_pion.pion_baseline import Pion

SMALL = ModelConfig(vocab_size=256, hidden=64, layers=2, heads=2, ffn_hidden=176, seq_len=32)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("corpus")
    rng = np.random.default_rng(0)
    for name in ("train", "val"):
        rng.integers(0, 256, size=200_000, dtype=np.uint16).tofile(d / f"{name}.bin")
    return d


def test_published_configuration_gives_the_published_size():
    """Their script says 60M; the configuration in model.py has to land there."""
    model = Transformer(ModelConfig())
    total = sum(p.numel() for p in model.parameters())
    assert 55e6 < total < 62e6, f"{total/1e6:.1f}M is not the 60M their script describes"


def test_parameter_split_follows_their_rule():
    """2-D weights that are neither the embedding nor the output head."""
    model = Transformer(ModelConfig())
    linears, rest = model.parameter_split()
    owned = {id(m.weight) for m in linears}
    assert id(model.embed.weight) not in owned
    assert id(model.head.weight) not in owned
    assert all(p.dim() == 2 for p in (m.weight for m in linears))
    assert len(linears) == 7 * ModelConfig().layers
    assert sum(p.numel() for p in rest) + sum(m.weight.numel() for m in linears) == sum(
        p.numel() for p in model.parameters()
    )


def test_block_matrix_shapes_are_the_three_the_spec_predicts():
    """Four square, two tall, one wide -- which decides each layer's basis path."""
    model = Transformer(ModelConfig())
    linears, _ = model.parameter_split()
    shapes = Counter(tuple(m.weight.shape) for m in linears[:7])
    assert shapes == Counter({(512, 512): 4, (1376, 512): 2, (512, 1376): 1})


def test_forward_starts_near_uniform_entropy():
    cfg = ModelConfig()
    model = Transformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    _, loss = model(x, x)
    assert abs(float(loss) - np.log(cfg.vocab_size)) < 0.5


def test_lr_schedule_is_cosine_without_warmup():
    cfg = RunConfig(lr=1e-3, lr_min=1e-5, train_steps=1000, warmup_steps=0)
    assert lr_at(0, cfg) == pytest.approx(1e-3)
    assert lr_at(999, cfg) == pytest.approx(1e-5, rel=1e-3)
    mid = lr_at(500, cfg)
    assert lr_at(750, cfg) < mid < lr_at(250, cfg)


def test_config_hash_covers_science_and_ignores_plumbing():
    cfg = RunConfig()
    assert cfg.hash == RunConfig().hash
    for changed in (dict(lr=2e-3), dict(seed=1), dict(optimizer="pion"), dict(ngd_eps=1e-3)):
        assert replace(cfg, **changed).hash != cfg.hash
    for ignored in (dict(out_dir="x"), dict(eval_every=7), dict(micro_batch=8)):
        assert replace(cfg, **ignored).hash == cfg.hash


def test_token_budget_matches_their_script():
    cfg = RunConfig()
    assert cfg.tokens_per_step == 512 * 256
    assert cfg.total_tokens == 73242 * 131072  # 9.6B, i.e. 8x Chinchilla for 60M
    assert abs(cfg.total_tokens - 9.6e9) / 9.6e9 < 1e-4


def test_corpus_targets_are_inputs_shifted_by_one(corpus):
    data = TokenCorpus(corpus / "train.bin", seq_len=16, seed=0)
    x, y = data.batch(4)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_corpus_is_deterministic_given_a_seed(corpus):
    a = TokenCorpus(corpus / "train.bin", 16, seed=3).batch(4)[0]
    b = TokenCorpus(corpus / "train.bin", 16, seed=3).batch(4)[0]
    assert torch.equal(a, b)


def test_held_out_batches_do_not_move(corpus):
    data = TokenCorpus(corpus / "val.bin", 16, seed=0)
    first = data.fixed_batches(2, 3, seed=99)
    second = data.fixed_batches(2, 3, seed=99)
    assert all(torch.equal(a[0], b[0]) for a, b in zip(first, second))


def test_missing_corpus_says_what_to_run():
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        TokenCorpus("nowhere.bin", 16)


@pytest.mark.parametrize(
    "optimizer,expected",
    [("ngd", NGDPion), ("pion", Pion), ("pion_ablated", Pion), ("adamw", None)],
)
def test_optimizer_wiring(optimizer, expected):
    model = Transformer(SMALL)
    rot, adamw, recorder = build_optimizers(model, RunConfig(optimizer=optimizer, model=SMALL))
    assert rot is None if expected is None else isinstance(rot, expected)
    assert isinstance(adamw, torch.optim.AdamW)
    assert (recorder is not None) == (optimizer == "ngd")
    if recorder is not None:
        recorder.remove()


def test_ablated_pion_is_wired_with_an_exact_retraction():
    """Ablating the scaling forces Cayley -- their truncation diverges without it."""
    model = Transformer(SMALL)
    rot, _, _ = build_optimizers(model, RunConfig(optimizer="pion_ablated", model=SMALL))
    group = rot.param_groups[0]
    assert group["scaling"] == "none"
    assert group["momentum"] == "none"
    assert group["retraction"] == "cayley"


@pytest.mark.parametrize("optimizer", ["ngd", "pion", "pion_ablated", "adamw"])
def test_run_end_to_end_and_reduce_loss(optimizer, corpus, tmp_path):
    cfg = RunConfig(
        optimizer=optimizer, model=SMALL, batch_sequences=8, micro_batch=4,
        train_steps=20, ngd_t_fac=5, eval_every=10, eval_batches=2, log_every=5,
        data_path=str(corpus / "train.bin"), val_path=str(corpus / "val.bin"),
        out_dir=str(tmp_path),
    )
    out = train(cfg, max_steps=20)
    rows = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    train_rows = [r for r in rows if "train_loss" in r]
    assert train_rows[-1]["train_loss"] < train_rows[0]["train_loss"]
    assert any("val_loss" in r for r in rows)
    assert (out / "manifest.json").exists()
    assert (out / "checkpoint.pt").exists()
    if optimizer == "ngd":
        assert "angle_max" in train_rows[-1], "NGD runs must log the diagnostics"
        assert train_rows[-1]["alpha_max"] <= 1.0


def test_anchor_follows_the_published_number_not_the_shell_script():
    """Their 60M script runs a different configuration than their reported figures."""
    from harness.anchor import TARGETS, anchor_config

    cfg = anchor_config("bilateral")
    assert cfg.optimizer == "pion"
    assert cfg.pion_momentum == "lie", "the published figures use Lie+Lie"
    assert cfg.pion_alternate is False, "3.3575 is the bilateral number"
    assert cfg.pion_scaling == "rms" and cfg.pion_rms == 0.2
    assert cfg.pion_retraction == "trunc", "their degree-2 truncated exponential"
    assert cfg.lr == 1e-3 and cfg.lr_min == 1e-5 and cfg.warmup_steps == 0
    assert cfg.total_tokens == 73242 * 131072
    assert TARGETS["bilateral"] == 3.3575

    other = anchor_config("alternate")
    assert other.pion_alternate is True
    assert TARGETS["alternate"] == 3.3654
    assert cfg.hash != other.hash


def test_anchor_tolerance_is_tighter_than_the_gap_it_must_resolve():
    """A tolerance looser than their own bilateral-alternate gap would say nothing."""
    from harness.anchor import TARGETS, TOLERANCE

    gap = TARGETS["alternate"] - TARGETS["bilateral"]
    assert 0 < TOLERANCE < 3 * gap


def test_anchor_check_reads_a_log(tmp_path):
    from harness.anchor import check

    log = tmp_path / "log.jsonl"
    log.write_text("\n".join(json.dumps({"step": i, "train_loss": 3.36}) for i in range(20)))
    result = check(log, "bilateral")
    assert result["matched"] and result["status"] == "complete"
    assert result["measured"] == pytest.approx(3.36)

    log.write_text(json.dumps({"step": 0, "train_loss": 4.0}))
    assert not check(log, "bilateral")["matched"]


def test_anchor_refuses_to_judge_a_truncated_run(tmp_path):
    """A short run always sits above a converged target; that is not a miss."""
    from harness.anchor import check

    log = tmp_path / "log.jsonl"
    log.write_text("\n".join(json.dumps({"step": i, "train_loss": 8.0}) for i in range(6)))
    result = check(log, "bilateral", expected_steps=73242)
    assert result["status"] == "incomplete"
    assert result["matched"] is False
    assert result["steps_run"] == 6 and result["steps_expected"] == 73242

    on_target = tmp_path / "full.jsonl"
    on_target.write_text("\n".join(json.dumps({"step": i, "train_loss": 3.357}) for i in range(10)))
    assert check(on_target, "bilateral", expected_steps=10)["status"] == "complete"


def test_anchor_rejects_an_unknown_update_side():
    from harness.anchor import anchor_config

    with pytest.raises(ValueError, match="update_side"):
        anchor_config("sideways")


def test_cli_flags_parse_to_their_declared_types(monkeypatch):
    """`from __future__ import annotations` turns field.type into a string.

    Resolving the annotations is not cosmetic: without it every flag parses as
    str, and the failure surfaces deep inside numpy rather than at the parse.
    """
    import sys as _sys
    from typing import get_type_hints

    from harness import run as run_module

    captured = {}
    monkeypatch.setattr(run_module, "train", lambda cfg, **kw: captured.setdefault("cfg", cfg) or Path("."))
    monkeypatch.setattr(
        _sys, "argv",
        ["run.py", "--optimizer", "pion", "--lr", "3e-4", "--micro-batch", "8",
         "--train-steps", "5", "--pion-alternate", "false"],
    )
    run_module.main()
    cfg = captured["cfg"]
    hints = get_type_hints(RunConfig)
    for name in ("lr", "micro_batch", "train_steps", "pion_alternate", "seed", "grad_clip"):
        assert isinstance(getattr(cfg, name), hints[name]), f"{name} parsed as the wrong type"
    assert cfg.lr == 3e-4 and cfg.micro_batch == 8 and cfg.pion_alternate is False


def test_anchor_reads_only_the_most_recent_attempt(tmp_path):
    """The log is appended, so a re-run of the same config shares the file.

    Without a start marker the check reports the previous attempt's step count,
    which would let a short re-run be judged against a long one's rows.
    """
    from harness.anchor import check, final_loss, last_step

    log = tmp_path / "log.jsonl"
    rows = [{"event": "start", "steps": 100}]
    rows += [{"step": i, "train_loss": 9.0} for i in range(50)]
    rows += [{"event": "start", "steps": 100}]
    rows += [{"step": i, "train_loss": 3.357} for i in range(3)]
    log.write_text("\n".join(json.dumps(r) for r in rows))

    assert last_step(log) == 2, "must not see the earlier attempt's steps"
    assert final_loss(log) == pytest.approx(3.357)
    assert check(log, "bilateral", expected_steps=100)["steps_run"] == 3


def test_training_writes_a_start_marker(corpus, tmp_path):
    cfg = RunConfig(
        optimizer="adamw", model=SMALL, batch_sequences=4, micro_batch=4,
        train_steps=3, eval_every=100, eval_batches=1, log_every=1,
        data_path=str(corpus / "train.bin"), val_path=str(corpus / "val.bin"),
        out_dir=str(tmp_path),
    )
    out = train(cfg, max_steps=3)
    train(cfg, max_steps=2)
    rows = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert sum(r.get("event") == "start" for r in rows) == 2


def test_a_run_resumes_from_its_checkpoint(corpus, tmp_path):
    """Every partition caps at 24h and a full run may not fit, so a requeued
    job has to continue rather than start over."""
    cfg = RunConfig(
        optimizer="ngd", model=SMALL, batch_sequences=4, micro_batch=4,
        train_steps=8, ngd_t_fac=2, eval_every=2, eval_batches=1, log_every=1,
        data_path=str(corpus / "train.bin"), val_path=str(corpus / "val.bin"),
        out_dir=str(tmp_path),
    )
    out = train(cfg, max_steps=4)
    first = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert first["step"] == 3
    assert first["rot"] is not None, "the rotational optimizer's state must be kept"

    train(cfg, max_steps=8)
    second = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert second["step"] == 7

    rows = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert any(r.get("event") == "resume" and r["from_step"] == 4 for r in rows)
    steps = [r["step"] for r in rows if "train_loss" in r]
    assert max(steps) == 7 and 0 in steps


def test_resume_is_a_no_op_when_already_finished(corpus, tmp_path):
    cfg = RunConfig(
        optimizer="adamw", model=SMALL, batch_sequences=4, micro_batch=4,
        train_steps=3, eval_every=1, eval_batches=1, log_every=1,
        data_path=str(corpus / "train.bin"), val_path=str(corpus / "val.bin"),
        out_dir=str(tmp_path),
    )
    out = train(cfg, max_steps=3)
    before = (out / "log.jsonl").read_text().count("train_loss")
    train(cfg, max_steps=3)
    assert (out / "log.jsonl").read_text().count("train_loss") == before


def test_no_resume_starts_over(corpus, tmp_path):
    cfg = RunConfig(
        optimizer="adamw", model=SMALL, batch_sequences=4, micro_batch=4,
        train_steps=4, eval_every=2, eval_batches=1, log_every=1,
        data_path=str(corpus / "train.bin"), val_path=str(corpus / "val.bin"),
        out_dir=str(tmp_path),
    )
    out = train(cfg, max_steps=2)
    train(cfg, max_steps=2, resume=False)
    rows = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert not any(r.get("event") == "resume" for r in rows)


def test_bf16_autocast_trains_and_leaves_the_weights_in_fp32(corpus, tmp_path):
    """Their runs are --bf16, so ours must be able to be.

    Autocast covers the forward pass and the loss; the parameters stay fp32,
    which is the same arrangement as Megatron's fp32 master weights, and the
    optimizer keeps its own exact arithmetic underneath. This runs on CPU, so
    it pins the wiring rather than the numerics.
    """
    cfg = RunConfig(
        optimizer="ngd", model=SMALL, batch_sequences=4, micro_batch=4,
        train_steps=4, ngd_t_fac=2, eval_every=2, eval_batches=1, log_every=1,
        precision="bf16",
        data_path=str(corpus / "train.bin"), val_path=str(corpus / "val.bin"),
        out_dir=str(tmp_path),
    )
    out = train(cfg, max_steps=4)
    rows = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    losses = [r["train_loss"] for r in rows if "train_loss" in r]
    assert losses and all(math.isfinite(v) for v in losses)
    assert any("val_loss" in r for r in rows)

    state = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    dtypes = {v.dtype for v in state["model"].values() if torch.is_floating_point(v)}
    assert dtypes == {torch.float32}, f"parameters must stay fp32, got {dtypes}"


def test_precision_is_part_of_a_run_s_identity():
    """It changes results, so two precisions are two runs, not one."""
    assert RunConfig(precision="fp32").hash != RunConfig(precision="bf16").hash


def test_an_unknown_precision_is_refused(corpus, tmp_path):
    cfg = RunConfig(
        optimizer="adamw", model=SMALL, batch_sequences=4, micro_batch=4,
        train_steps=1, eval_every=99, log_every=1, precision="fp8",
        data_path=str(corpus / "train.bin"), val_path=str(corpus / "val.bin"),
        out_dir=str(tmp_path),
    )
    with pytest.raises(ValueError, match="precision"):
        train(cfg, max_steps=1)
