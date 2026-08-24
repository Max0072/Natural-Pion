#!/usr/bin/env python
"""Turn C4 into the flat token file the harness reads.

C4 is a cleaned Common Crawl dump -- roughly 156B tokens of English web text,
and the corpus Pion's 60M ablations use, which is the only reason it is the one
here: the comparison is only meaningful on the same data.

Output is one little-endian uint16 array per split. The T5 vocabulary is 32100
entries, so uint16 is exact and halves the file against uint32.

The budget is **9.6B tokens** -- 73242 steps at 131072 -- which is what the
paper reports and what `RunConfig` runs. An earlier reading of their shell
script gave 37500 steps, i.e. 4.9B; that came from a defect in their released
code, not from a second configuration, and it should not reappear here. The
default target is 10B, which gives a single pass with a margin.

Size the corpus to the budget deliberately: `TokenCorpus` samples random
windows with replacement and has no notion of an epoch, so a corpus smaller
than the budget does not fail or warn -- it silently shows every token more
than once, which is exactly the difference the anchor run is meant to detect.

**Speed.** Measured on one core: streaming C4 costs 467k tokens/s and tokenising
921k, so two thirds of the time is spent getting the bytes rather than turning
them into numbers, and threading the tokeniser buys almost nothing. 10B tokens
is nine to seventeen hours that way. `--shards N` instead cuts the corpus by
**file** -- C4 ships 1024 of them -- and runs N processes over disjoint subsets,
which scales nearly linearly because the bottleneck is per-process CPU. Give it
one core per shard and no more shards than cores; the parts are concatenated
when they finish. Order across shards does not matter: the corpus is a flat
stream sampled at random windows.

This takes hours and is a single stream per shard, so it resumes. Progress is
recorded beside each `.bin` in a small JSON marker, and re-running the same
command continues from it. What resuming saves is the tokenisation, not the
download.

Requires `datasets` and `transformers`, which the container installs and this
repository does not depend on -- nothing else here needs them.

    python scripts/prepare_data.py --out data --target-tokens 1e10 --shards 8
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def _mark(fh, progress: Path, records: int, tokens: int) -> None:
    """Make what is written durable, then record how far it goes.

    The order is the whole point. The marker must never claim more than the
    file holds, or a resume would trust bytes that were never written; the
    reverse costs nothing, because resuming truncates the file back to the
    marker. The marker itself is replaced atomically for the same reason.
    """
    fh.flush()
    os.fsync(fh.fileno())
    tmp = progress.with_name(progress.name + ".tmp")
    tmp.write_text(json.dumps({"records": records, "tokens": tokens}))
    os.replace(tmp, progress)


def repo_files(dataset: str, subset: str, split: str) -> list[str]:
    """The dataset's own file list, which is what shards are cut along.

    Asked of the Hub rather than assembled from a guessed pattern: the layout
    is the dataset's business and has changed before.
    """
    from huggingface_hub import HfApi

    prefix = f"{subset}/c4-{split}"
    files = sorted(
        f for f in HfApi().list_repo_files(dataset, repo_type="dataset")
        if f.startswith(prefix) and f.endswith(".json.gz")
    )
    if not files:
        raise RuntimeError(f"no files matching {prefix}*.json.gz in {dataset}")
    return files


def write_split(path: Path, files: list[str], budget: float, args, label: str) -> int:
    """Stream `files`, tokenise, append to `path` until `budget` tokens.

    Resumable: returns the number of tokens the file holds when it stops.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    progress = path.with_name(path.name + ".progress.json")
    records, written = 0, 0

    if args.no_resume:
        path.unlink(missing_ok=True)
        progress.unlink(missing_ok=True)
    elif path.exists() and progress.exists():
        state = json.loads(progress.read_text())
        records, written = int(state["records"]), int(state["tokens"])
        # uint16 on disk: two bytes a token. Anything past the marker was
        # written but never confirmed, so it goes.
        with path.open("r+b") as fh:
            fh.truncate(written * 2)
        print(f"[{label}] resuming at {written/1e6:.1f}M tokens, {records} records in", flush=True)

    if written >= budget:
        print(f"[{label}] already holds {written/1e9:.3f}B tokens", flush=True)
        return written

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tok.eos_token_id
    assert tok.vocab_size < 65536, "uint16 storage assumes a vocabulary under 65536"

    stream = load_dataset(args.dataset, data_files={"train": files}, split="train",
                          streaming=True)
    if records:
        stream = stream.skip(records)

    with path.open("ab") as fh:
        buffer, texts, since_sync = [], [], 0
        for record in stream:
            texts.append(record["text"])
            if len(texts) < args.batch:
                continue
            for ids in tok(texts, add_special_tokens=False)["input_ids"]:
                buffer.extend(ids)
                buffer.append(eos)
            np.asarray(buffer, dtype=np.uint16).tofile(fh)
            written += len(buffer)
            records += len(texts)
            buffer, texts = [], []
            since_sync += 1
            if since_sync >= args.sync_every:
                _mark(fh, progress, records, written)
                since_sync = 0
                print(f"[{label}] {written/1e6:.1f}M tokens", flush=True)
            if written >= budget:
                break
        _mark(fh, progress, records, written)
    print(f"[{label}] {path}: {written/1e9:.3f}B tokens", flush=True)
    return written


def concatenate(parts: list[Path], target: Path) -> int:
    """Join the shards into the single file the harness opens.

    Written beside the target and renamed, so an interrupted join leaves no
    half corpus wearing the real name. Each part is dropped as it is consumed,
    which keeps the peak at one part above the final size rather than double.
    """
    tmp = target.with_name(target.name + ".joining")
    total = 0
    with tmp.open("wb") as out:
        for part in parts:
            with part.open("rb") as fh:
                while chunk := fh.read(64 << 20):
                    out.write(chunk)
                    total += len(chunk)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, target)
    for part in parts:
        part.unlink(missing_ok=True)
        part.with_name(part.name + ".progress.json").unlink(missing_ok=True)
    return total // 2


def run_shards(args, files: list[str]) -> None:
    """One process per shard, each over its own slice of the file list."""
    out = args.out
    parts = [out / f"c4_train.part{k}.bin" for k in range(args.shards)]
    env = dict(os.environ)
    # One thread per process. N shards times M threads on N cores is thrash,
    # and the tokeniser is only a third of the work in any case.
    env.update(TOKENIZERS_PARALLELISM="false", RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1")

    children = []
    for k in range(args.shards):
        cmd = [sys.executable, os.path.abspath(__file__),
               "--out", str(out), "--target-tokens", repr(args.target_tokens),
               "--val-tokens", repr(args.val_tokens), "--tokenizer", args.tokenizer,
               "--dataset", args.dataset, "--subset", args.subset,
               "--batch", str(args.batch), "--sync-every", str(args.sync_every),
               "--shards", str(args.shards), "--shard-index", str(k)]
        if args.no_resume:
            cmd.append("--no-resume")
        children.append(subprocess.Popen(cmd, env=env))

    failed = [k for k, c in enumerate(children) if c.wait() != 0]
    if failed:
        raise RuntimeError(
            f"shards {failed} failed; the rest kept their progress markers, so "
            "re-running the same command resumes instead of starting over"
        )

    tokens = concatenate(parts, out / "c4_train.bin")
    print(f"{out / 'c4_train.bin'}: {tokens/1e9:.3f}B tokens from {args.shards} shards")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--target-tokens", type=float, default=1e10)
    ap.add_argument("--val-tokens", type=float, default=1e7)
    ap.add_argument("--tokenizer", default="t5-base")
    ap.add_argument("--dataset", default="allenai/c4")
    ap.add_argument("--subset", default="en")
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--sync-every", type=int, default=32,
                    help="batches between durable checkpoints; a crash costs at most this much")
    ap.add_argument("--no-resume", action="store_true",
                    help="rebuild from scratch, discarding any partial corpus")
    ap.add_argument("--shards", type=int, default=1,
                    help="parallel worker processes over disjoint C4 files; one core each")
    ap.add_argument("--shard-index", type=int, default=None,
                    help="internal: which slice this process owns")
    args = ap.parse_args()
    if args.shards < 1:
        ap.error("--shards must be at least 1")
    args.out.mkdir(parents=True, exist_ok=True)

    train_files = repo_files(args.dataset, args.subset, "train")

    if args.shard_index is not None:
        # A child. Its slice is every Nth file, so shards are the same size
        # whatever order the list arrives in, and its budget is its share.
        k, n = args.shard_index, args.shards
        share = args.target_tokens / n
        write_split(args.out / f"c4_train.part{k}.bin", train_files[k::n], share, args,
                    label=f"train {k}/{n}")
        return

    val_files = repo_files(args.dataset, args.subset, "validation")
    write_split(args.out / "c4_val.bin", val_files, args.val_tokens, args, label="val")

    if args.shards == 1:
        write_split(args.out / "c4_train.bin", train_files, args.target_tokens, args,
                    label="train")
    else:
        run_shards(args, train_files)


if __name__ == "__main__":
    main()
