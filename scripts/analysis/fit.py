"""Fit loss-vs-schedule laws on the t_fac 25 + momentum family and test them on runs they did not see."""
import numpy as np, sys
from scipy.optimize import least_squares
from data import runs

from scipy.signal import lfilter
def ema(eta, lam):
    return lfilter([1-lam], [1, -lam], eta, zi=[lam*eta[0]])[0]

# Model T (Tissue et al. 2024): L = L0 + A S1^-alpha - C S2, S2 = sum of momentum of the decreases.
def tissue(th, eta):
    L0, lA, al, lC, lam = th
    S1 = np.cumsum(eta)
    d = np.concatenate([[0.0], eta[:-1]-eta[1:]])
    m = np.zeros_like(eta); acc = 0.0
    for i, x in enumerate(d): acc = lam*acc + x; m[i] = acc
    S2 = np.cumsum(m)
    return L0 + np.exp(lA)*S1**(-al) - np.exp(lC)*S2

# Model N (ours): saturating progress plus a lagged noise floor.
#   L = L0 + A (sum g(eta))^-alpha + B ema_lam(eta),  g(eta) = eta / (1 + eta/eta_c)
def noise(th, eta):
    L0, lA, al, lB, lec, lg = th
    lam = 1 - np.exp(lg)
    g = eta/(1 + eta/np.exp(lec))
    return L0 + np.exp(lA)*np.cumsum(g)**(-al) + np.exp(lB)*ema(eta, lam)

MODELS = {'T': (tissue, [3.0, 0.0, 0.5, -3.0, 0.999], ([0, -10, 0.01, -15, 0.9], [5, 10, 3, 5, 0.99999])),
          'N': (noise, [3.0, 0.0, 0.5, 0.0, np.log(0.3), np.log(0.01)], ([0, -10, 0.01, -15, -8, -12], [5, 10, 3, 10, 5, -0.5]))}

def resid(th, f, names):
    out = []
    for n in names:
        r = runs[n]; pred = f(th, r['eta'])
        for s, v in r['val']:
            if s >= 499: out.append(pred[s] - v)
    return np.array(out)

def fit(model, names):
    f, x0, b = MODELS[model]
    best = None
    for trial in range(6):
        x = np.array(x0) + (0 if trial == 0 else np.random.default_rng(trial).normal(0, 0.5, len(x0)))
        x = np.clip(x, np.array(b[0])+1e-6, np.array(b[1])-1e-6)
        r = least_squares(resid, x, bounds=b, args=(f, names))
        if best is None or r.cost < best.cost: best = r
    return best.x

def report(model, train, test):
    f = MODELS[model][0]; th = fit(model, train)
    rms = np.sqrt(np.mean(resid(th, f, train)**2))
    print(f'model {model}  train rms {rms:.4f}  params {np.round(th, 4)}')
    for n in test:
        r = runs[n]; s, v = r['val'][-1]
        print(f'    {n:16s} final {v:.4f}  predicted {f(th, r["eta"])[s]:.4f}  error {f(th, r["eta"])[s]-v:+.4f}')
    return th

ALL = list(runs)
by = lambda *k: [n for n in ALL if runs[n]['kind'] in k]
if __name__ == '__main__':
    for m in ['T', 'N']:
        print('=== all runs in, finals:'); report(m, ALL, ALL)
        print('=== cosine only (3000 and 15000) in; linear, WSD, RProp out:'); report(m, by('cos3k', 'cos15k'), by('lin', 'wsd', 'rprop'))
        print('=== 3000 only in; 15000 out:'); report(m, by('cos3k', 'lin', 'wsd', 'rprop'), by('cos15k'))
