"""Regenerate `docs/JOURNAL_INDEX.md` from the journal's own headings.

`docs/JOURNAL.md` is append-only and past four thousand lines, so finding the
entry that settled a question means grepping for a phrase you have to remember
first. This turns its `##` headings into a dated index.

A generated file rather than a hand-kept one, because a hand-kept table of
contents in an append-only document is wrong by the second append. Run it after
adding an entry:

    python scripts/journal_index.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "JOURNAL.md"
DST = ROOT / "docs" / "JOURNAL_INDEX.md"

HEADING = re.compile(r"^## (.+)$")
DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})(.*)$")


def main() -> None:
    rows = []
    for n, line in enumerate(SRC.read_text().splitlines(), 1):
        m = HEADING.match(line)
        if not m:
            continue
        text = m.group(1).strip()
        d = DATE.match(text)
        if d:
            date = d.group(1)
            rest = d.group(2).lstrip(" ,-—").strip() or "(untitled)"
        else:
            date, rest = "", text
        rows.append((date, rest, n))

    out = [
        "# Journal index",
        "",
        f"Generated from `docs/JOURNAL.md` by `scripts/journal_index.py` --"
        f" {len(rows)} entries, {len(SRC.read_text().splitlines())} lines."
        " Do not edit by hand; re-run the script.",
        "",
        "Line numbers are for `sed -n 'Np,+40p' docs/JOURNAL.md`.",
        "",
        "| date | entry | line |",
        "|---|---|---|",
    ]
    for date, text, n in rows:
        out.append(f"| {date} | {text} | {n} |")
    DST.write_text("\n".join(out) + "\n")
    print(f"{DST.relative_to(ROOT)}: {len(rows)} entries")


if __name__ == "__main__":
    main()
