"""Second try: late points only, and AdamW's own cosine (56.5% of the parameters) as a second progress term."""
import numpy as np
from scipy.optimize import least_squares
from data import runs, cosine
from fit import ema, by, ALL
MIN_STEP = 1499
def model(th, eta):
    L0, lA, al, lB, lec, lg, lD, be = th
    T = len(eta); adam = np.cumsum(cosine(1.0, T))       # AdamW follows cosine over the run's length in every run
    g = eta/(1 + eta/np.exp(lec))
    return L0 + np.exp(lA)*np.cumsum(g)**(-al) + np.exp(lB)*ema(eta, 1-np.exp(lg)) + np.exp(lD)*adam**(-be)
x0 = [3.0, 0.0, 0.5, 0.0, np.log(0.3), np.log(0.01), 0.0, 0.5]
b = ([0, -10, 0.01, -15, -8, -12, -15, 0.01], [5, 10, 3, 10, 5, -0.5, 10, 3])
def resid(th, names):
    return np.array([model(th, runs[n]['eta'])[s]-v for n in names for s, v in runs[n]['val'] if s >= MIN_STEP])
def fit(names):
    best = None
    for k in range(8):
        x = np.clip(np.array(x0) + (0 if k == 0 else np.random.default_rng(k).normal(0, 0.7, 8)), np.array(b[0])+1e-6, np.array(b[1])-1e-6)
        r = least_squares(resid, x, bounds=b, args=(names,))
        if best is None or r.cost < best.cost: best = r
    return best.x
for title, train, test in [('all in', ALL, ALL), ('cosine in; others out', by('cos3k', 'cos15k'), by('lin', 'wsd', 'rprop')),
                           ('3000 in; 15000 out', by('cos3k', 'lin', 'wsd', 'rprop'), by('cos15k'))]:
    th = fit(train); print(f'=== {title}: train rms {np.sqrt(np.mean(resid(th, train)**2)):.4f}')
    for n in test:
        s, v = runs[n]['val'][-1]; p = model(th, runs[n]['eta'])[s]; print(f'    {n:16s} {v:.4f}  pred {p:.4f}  err {p-v:+.4f}')
