import json, os, math

RUN = '/onyx/data/p330/runs/mcfisher2/ngd-pion-s-lr0.01-s0-1974d5fe08'
ETA = 0.01

rows = [json.loads(l) for l in open(os.path.join(RUN, 'diagnostics.jsonl'))]
s0 = min(r['step'] for r in rows)
step0 = {r['name']: r for r in rows if r['step'] == s0}

print(f"=== step {s0}: wq / wk / wv share one input, so one and the same A ===")
print(f"  {'layer':<22} {'cond_A':>9} {'lam_max_A':>10} {'angle@eta1':>11} "
      f"{'delta_rms':>11} {'quad':>10} {'a*drms':>10}")
for b in range(8):
    for p in ('wq', 'wk', 'wv', 'wo'):
        r = step0.get(f'blocks.{b}.attn.{p}')
        if not r:
            continue
        a = r['angle'] / ETA
        print(f"  blocks.{b}.attn.{p:<10} {r['cond_A']:>9.1f} {r['lam_max_A']:>10.3f} "
              f"{a:>11.1f} {r['delta_rms']:>11.3e} {r['quad']:>10.3e} "
              f"{a*r['delta_rms']:>10.3e}")
    print()

print("=== does angle * delta_rms hold still across ALL 56 layers? ===")
vals = sorted((r['angle'] / ETA) * r['delta_rms'] for r in step0.values()
              if r['delta_rms'] == r['delta_rms'])
raw = sorted(r['angle'] / ETA for r in step0.values())
print(f"  angle alone      : min {raw[0]:.1f}  median {raw[len(raw)//2]:.1f}  "
      f"max {raw[-1]:.1f}   spread x{raw[-1]/raw[0]:.0f}")
print(f"  angle*delta_rms  : min {vals[0]:.3e}  median {vals[len(vals)//2]:.3e}  "
      f"max {vals[-1]:.3e}   spread x{vals[-1]/vals[0]:.0f}")
print("  if the 1/delta law were exact the second spread would be ~1.")
