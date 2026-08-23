"""The training harness: model shapes, data, schedule, and one end-to-end run."""

import json
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
    assert cfg.total_tokens == 37500 * 131072  # 4.9B; the paper says 9.6B, unresolved


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
