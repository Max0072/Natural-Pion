import json, glob, os, math

RUN = '/onyx/data/p330/runs/mcfisher2/ngd-pion-s-lr0.01-s0-1974d5fe08'
ETA = 0.01

rows = []
for line in open(os.path.join(RUN, 'diagnostics.jsonl')):
    try:
        r = json.loads(line)
    except Exception:
        continue
    if r.get('angle') == r.get('angle') and r.get('cond_A') == r.get('cond_A'):
        rows.append(r)

by_step = {}
for r in rows:
    by_step.setdefault(r['step'], []).append(r)


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return float('nan')
    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        rk = [0.0] * n
        for pos, i in enumerate(order):
            rk[i] = pos
        return rk
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else float('nan')


print(f"run: eta={ETA}, angles below are AS LOGGED (multiply by 100 for eta=1)\n")
print(f"  {'step':>5} {'n':>4} {'rho(angle,condA)':>18} {'rho(angle,nullfrac)':>21} "
      f"{'rho(angle,floored)':>20}")
for step in sorted(by_step):
    rs = by_step[step]
    ang = [r['angle'] for r in rs]
    print(f"  {step:>5} {len(rs):>4} "
          f"{spearman([r['cond_A'] for r in rs], ang):>18.3f} "
          f"{spearman([r['null_frac'] for r in rs], ang):>21.3f} "
          f"{spearman([r['floored_frac_in'] for r in rs], ang):>20.3f}")

step0 = by_step[min(by_step)]
step0.sort(key=lambda r: r['cond_A'])
print(f"\n=== step {min(by_step)}, layers sorted by cond_A "
      f"(angle rescaled to eta=1) ===")
print(f"  {'layer':<26} {'cond_A':>10} {'sqrt':>8} {'null':>7} {'flr_in':>7} "
      f"{'angle@eta1':>11} {'angle/sqrt':>11}")
show = step0[:6] + [None] + step0[-6:]
for r in show:
    if r is None:
        print("  ...")
        continue
    sq = math.sqrt(r['cond_A'])
    a = r['angle'] / ETA
    print(f"  {r['name']:<26} {r['cond_A']:>10.1f} {sq:>8.1f} "
          f"{r['null_frac']:>7.3f} {r['floored_frac_in']:>7.3f} "
          f"{a:>11.1f} {a / sq:>11.1f}")

allr = step0
sq = [math.sqrt(r['cond_A']) for r in allr]
a1 = [r['angle'] / ETA for r in allr]
ratio = sorted(x / y for x, y in zip(a1, sq))
print(f"\n  angle/sqrt(cond_A) across {len(allr)} layers: "
      f"min {ratio[0]:.1f}  median {ratio[len(ratio)//2]:.1f}  max {ratio[-1]:.1f}")
print("  a pure-degeneracy account needs this ratio to be O(1) and flat.")

clean = [r for r in allr if r['null_frac'] == 0 and r['n_below_floor'] == 0]
print(f"\n  layers with NO degeneracy at all (null_frac=0, nothing below floor): "
      f"{len(clean)}/{len(allr)}")
if clean:
    ca = sorted(r['angle'] / ETA for r in clean)
    cc = sorted(r['cond_A'] for r in clean)
    print(f"    their angle@eta1: min {ca[0]:.1f} median {ca[len(ca)//2]:.1f} max {ca[-1]:.1f}")
    print(f"    their cond_A:     min {cc[0]:.1f} median {cc[len(cc)//2]:.1f} max {cc[-1]:.1f}")
