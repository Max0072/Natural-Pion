"""Steps NGD-Pion needs to reach Pion's final val, at each Pion horizon (seed 0). Usage: python scripts/analysis/steps_ratio.py [runs-record dir]"""
import json, glob, math, sys
R = sys.argv[1] if len(sys.argv) > 1 else "runs-record"
def val(path):
    rows = []
    for l in open(path):
        try: d = json.loads(l)
        except: continue
        if d.get('event') == 'start': rows = []
        if 'val_loss' in d: rows.append((d['step'] + 1, d['val_loss']))
    return rows
def one(pat):
    g = glob.glob(f'{R}/{pat}/log.jsonl'); assert len(g) == 1, (pat, g); return val(g[0])

ours = {  # seed 0, Algorithm 2, best angle per horizon
    3000: one('stepnormcombo/batch/ngd-pion-m-lr0.2-s0-*'),
    15000: one('stepnormfinal/tf25-15k-0.05/*-s0-*'),
    56900: one('stepnormlong/ours56900-lr0.05/*'),
    73242: one('stepnormlong/ours73242-lr0.05/*'),
}
pion = {
    3840: one('stepnormfinal/pion3840/*'),
    4500: one('stepnormfinal/pion4500/*'),
    19200: one('stepnormfinal/pion19200/*'),
    73242: one('stepnormlong/pion73242-lr1e-3/*'),
}
finals = sorted((T, c[-1][1]) for T, c in ours.items())
print('ours finals', finals)

def by_finals(target):
    # ours' final val vs log T, piecewise linear; each point is a full schedule
    for (t0, v0), (t1, v1) in zip(finals, finals[1:]):
        if v1 <= target <= v0:
            f = (v0 - target) / (v0 - v1)
            return math.exp(math.log(t0) + f * (math.log(t1) - math.log(t0)))
    return None

def by_curve(target):
    # first crossing inside one of our runs (mid-schedule, so pessimistic); best over runs
    best = None
    for T, c in ours.items():
        for (s0, v0), (s1, v1) in zip(c, c[1:]):
            if v1 <= target < v0:
                s = s0 + (v0 - target) / (v0 - v1) * (s1 - s0)
                best = s if best is None else min(best, s); break
    return best

print(f"{'Pion T':>7} {'Pion val':>9} | {'ours T (finals)':>15} {'ratio':>6} | {'ours T (mid-run)':>16} {'ratio':>6}")
for T, c in sorted(pion.items()):
    v = c[-1][1]; a = by_finals(v); b = by_curve(v)
    fa = f"{a:15.0f} {T/a:6.2f}" if a else f"{'< 3000':>15} {'> '+format(T/3000,'.2f'):>6}"
    fb = f"{b:16.0f} {T/b:6.2f}" if b else f"{'-':>16} {'-':>6}"
    print(f"{T:7d} {v:9.4f} | {fa} | {fb}")
