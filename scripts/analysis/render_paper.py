"""Fill the {{key:format}} placeholders of docs/paper/preprint.md from docs/paper/numbers.json.

    python scripts/analysis/paper_numbers.py && python scripts/analysis/render_paper.py

Writes docs/paper/preprint.filled.md. A key without a value renders as **[key?]**, so a gap is visible and
cannot pass for a number. Format is a Python format spec; `,` thousands separators become thin spaces.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "docs/paper"
nums = json.load(open(P / "numbers.json"))
src = (P / "preprint.md").read_text()
gaps, unknown = [], []


def fill(m):
    key, fmt = m.group(1), m.group(2) or ""
    if key not in nums:
        unknown.append(key)
        return f"**[{key}: unknown key]**"
    v = nums[key]
    if v is None:
        gaps.append(key)
        return f"**[{key}?]**"
    if isinstance(v, bool):
        return "yes" if v else "no"
    out = format(v, fmt) if fmt else (f"{v:.4g}" if isinstance(v, float) else str(v))
    return out.replace(",", " ")


out = re.sub(r"\{\{([A-Za-z0-9_.\-]+)(?::([^}]*))?\}\}", fill, src)
(P / "preprint.filled.md").write_text(out)
print(f"filled; {len(gaps)} gaps {sorted(set(gaps))}; {len(unknown)} unknown keys {sorted(set(unknown))}")
