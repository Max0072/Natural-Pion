"""Collect val curves and the rotation angle schedule eta(t) for every t_fac 25 + momentum run (one optimizer family)."""
import glob, json, math, numpy as np
R = '/onyx/data/p330/runs/'
FLOOR = 1e-5
def cosine(p, T): t = np.arange(T); return FLOOR + 0.5*(p-FLOOR)*(1+np.cos(np.pi*t/T))
def linear(p, T): t = np.arange(T); return FLOOR + (p-FLOOR)*(1-t/T)
def wsd(p, T, d):
    t = np.arange(T); u = t/T; s = 1-d
    return np.where(u < s, p, FLOOR + (p-FLOOR)*(1-u)/d)
def val(d):
    f = glob.glob(R+d+'/log.jsonl')
    assert len(f) == 1, (d, f)
    return [(r['step'], r['val_loss']) for r in map(json.loads, open(f[0])) if 'val_loss' in r]
def rprop(name, T, base):
    m = np.ones(T); pts = [json.loads(l) for l in open(R+f'stepnormrprop/{name}/rprop.jsonl')]
    # m is logged every 23 steps; hold it between logs (an approximation: it moves every step)
    k = 0; cur = 1.0
    for t in range(T):
        while k < len(pts) and pts[k]['step'] <= t: cur = pts[k]['m']; k += 1
        m[t] = cur
    return base*m
runs = {}
def add(name, d, eta, kind):
    runs[name] = dict(val=val(d), eta=eta, kind=kind)
for p in ['0.1', '0.2', '0.4', '0.8']:
    add(f'cos{p}', f'stepnormtfacmom/ngd-pion-m-lr{p}-s0-*', cosine(float(p), 3000), 'cos3k')
for s in [1, 2, 3]:
    add(f'cos0.2-s{s}', f'stepnormfinal/tf25/ngd-pion-m-lr0.2-s{s}-*', cosine(0.2, 3000), 'cos3k')
add('lin0.2', 'stepnormcombo/lin0.2/*', linear(0.2, 3000), 'lin')
add('lin0.1', 'stepnormcombo/lin0.1/*', linear(0.1, 3000), 'lin')
add('wsd0.2', 'stepnormwsd/decay0.2/*', wsd(0.2, 3000, 0.2), 'wsd')
add('wsd0.1', 'stepnormwsd/decay0.1/*', wsd(0.2, 3000, 0.1), 'wsd')
add('rp-cos', 'stepnormrprop/cosine/*', rprop('cosine', 3000, cosine(0.2, 3000)), 'rprop')
add('rp-const', 'stepnormrprop/constant/*', rprop('constant', 3000, np.full(3000, 0.2)), 'rprop')
for p in ['0.05', '0.1', '0.2']:
    add(f'15k-cos{p}', f'stepnormfinal/tf25-15k-{p}/*', cosine(float(p), 15000), 'cos15k')
add('15k-cos0.1-unb', 'stepnormclock/ngd-pion-m-lr0.1-s0-*', cosine(0.1, 15000), 'cos15k')
if __name__ == '__main__':
    for n, r in runs.items(): print(n, r['kind'], len(r['eta']), r['val'][-1])
