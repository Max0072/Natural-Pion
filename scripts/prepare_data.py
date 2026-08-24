#!/usr/bin/env python
"""Turn C4 into the flat token file the harness reads.

C4 is a cleaned Common Crawl dump -- roughly 156B tokens of English web text,
and the corpus Pion's 60M ablations use, which is the only reason it is the
one here: the comparison is only meaningful on the same data.

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

This takes hours and is a single stream, so it resumes. Progress is recorded
beside each `.bin` in a small JSON marker, and re-running the same command
continues from it instead of starting the corpus again. What resuming saves is
the tokenisation, not the download: a streamed dataset has to be read through
to reach the point it left off.

Requires `datasets` and `transformers`, which the container installs and this
repository does not depend on -- nothing else here needs them.

    python scripts/prepare_data.py --out data --target-tokens 1e10
"""

from __future__ import annotations

import argparse
import json
import os
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
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tok.eos_token_id
    assert tok.vocab_size < 65536, "uint16 storage assumes a vocabulary under 65536"
    args.out.mkdir(parents=True, exist_ok=True)

    for split, budget in (("validation", args.val_tokens), ("train", args.target_tokens)):
        path = args.out / f"c4_{'val' if split == 'validation' else 'train'}.bin"
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
            print(f"{path}: resuming at {written/1e9:.3f}B tokens, {records} records in",
                  flush=True)

        if written >= budget:
            print(f"{path}: already holds {written/1e9:.3f}B tokens")
            continue

        stream = load_dataset(args.dataset, args.subset, split=split, streaming=True)
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
                    print(f"  {split}: {written/1e6:.1f}M tokens", flush=True)
                if written >= budget:
                    break
            _mark(fh, progress, records, written)
        print(f"{path}: {written/1e9:.3f}B tokens")


if __name__ == "__main__":
    main()
