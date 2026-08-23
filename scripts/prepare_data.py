#!/usr/bin/env python
"""Turn C4 into the flat token file the harness reads.

C4 is a cleaned Common Crawl dump -- roughly 156B tokens of English web text,
and the corpus Pion's 60M ablations use, which is the only reason it is the
one here: the comparison is only meaningful on the same data.

Output is one little-endian uint16 array per split. The T5 vocabulary is 32100
entries, so uint16 is exact and halves the file against uint32.

Only a slice is needed. Their schedule is 37500 steps at 131072 tokens, i.e.
4.9B tokens, so `--target-tokens 5e9` is enough for a single pass; the paper
says 9.6B for the same experiments and that conflict is unresolved.

Requires `datasets` and `transformers`, which the container installs and this
repository does not depend on -- nothing else here needs them.

    python scripts/prepare_data.py --out data --target-tokens 5e9
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--target-tokens", type=float, default=5e9)
    ap.add_argument("--val-tokens", type=float, default=1e7)
    ap.add_argument("--tokenizer", default="t5-base")
    ap.add_argument("--dataset", default="allenai/c4")
    ap.add_argument("--subset", default="en")
    ap.add_argument("--batch", type=int, default=1000)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tok.eos_token_id
    assert tok.vocab_size < 65536, "uint16 storage assumes a vocabulary under 65536"
    args.out.mkdir(parents=True, exist_ok=True)

    for split, budget in (("validation", args.val_tokens), ("train", args.target_tokens)):
        path = args.out / f"c4_{'val' if split == 'validation' else 'train'}.bin"
        stream = load_dataset(args.dataset, args.subset, split=split, streaming=True)
        written = 0
        with path.open("wb") as fh:
            buffer, texts = [], []
            for record in stream:
                texts.append(record["text"])
                if len(texts) < args.batch:
                    continue
                for ids in tok(texts, add_special_tokens=False)["input_ids"]:
                    buffer.extend(ids)
                    buffer.append(eos)
                np.asarray(buffer, dtype=np.uint16).tofile(fh)
                written += len(buffer)
                buffer, texts = [], []
                if written >= budget:
                    break
                if written % (100 * args.batch) < args.batch:
                    print(f"  {split}: {written/1e6:.1f}M tokens", flush=True)
        print(f"{path}: {written/1e9:.3f}B tokens")


if __name__ == "__main__":
    main()
