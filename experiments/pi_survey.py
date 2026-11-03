"""Pi(1) for the channels the survey actually estimates with estimator (a).

The corner condition of Theorem 2 treats the shape inflation Pi as a constant, which folds
it into K. Pi depends on c, and carrying that dependence through replaces the terminal slope
of eta_T by that of eta_T/Pi, whose value at the corner is Upsilon*Pi(1). Pi(1) >= 1 always,
so the published condition is conservative and the sharpening can only move a verdict toward
the corner.

That makes the size of Pi(1) a question about the shapes the survey fits, not about the
power-law special case, and it has to be computed on the same fits that produced the
reported Upsilon. Pi(1) is the ratio of known-shape information about T at c=1 to the
profile information, both from the Fisher matrix in (T, a, b) over the full window.

Scope: estimator (a) only, which is where D(1)=1 is the failure event. The DeltaQ(V) channel
is excluded by that restriction, exactly as Section VII-A excludes it.
"""
import os
import sys

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from upsilon_survey import ROOT, measure  # same fit that produced the reported Upsilon

HERE = os.path.dirname(os.path.abspath(__file__))
H = 1e-5


def pi_one(a, b):
    """Known-shape over profile information about T at c=1, for D(tau;a,b)."""
    def D(t, aa, bb):
        return (1 - np.exp(-aa*np.clip(t, 1e-12, None)**bb))/(1 - np.exp(-aa))

    dp = lambda t: a*b*np.clip(t, 1e-12, None)**(b - 1)*np.exp(-a*t**b)/(1 - np.exp(-a))
    dT = lambda t: t*dp(t)
    dth = [lambda t: (D(t, a + H, b) - D(t, a - H, b))/(2*H),
           lambda t: (D(t, a, b + H) - D(t, a, b - H))/(2*H)]

    JTT = quad(lambda t: dT(t)**2, 0, 1, limit=300)[0]
    JTs = np.array([quad(lambda t: dT(t)*g(t), 0, 1, limit=300)[0] for g in dth])
    Jss = np.array([[quad(lambda t: g(t)*h(t), 0, 1, limit=300)[0] for h in dth]
                    for g in dth])
    prof = JTT - JTs @ np.linalg.solve(Jss + 1e-16*np.trace(Jss)*np.eye(2), JTs)
    return JTT/max(prof, 1e-300)


def summarise(dom, ind, fits):
    v = [f for f in fits if f is not None and np.isfinite(f[0])]
    if not v:
        print(f'{dom:9s} {ind:20s}   (no usable units)')
        return None
    U = np.array([f[0] for f in v])
    P = np.array([pi_one(f[2], f[3]) for f in v])
    P = P[np.isfinite(P) & (P > 0)]
    print(f'{dom:9s} {ind:20s} {len(v):4d} {np.median(U):8.2f} {np.median(P):8.2f} '
          f'[{np.percentile(P, 25):5.2f},{np.percentile(P, 75):6.2f}] '
          f'{np.median(U)*np.median(P):9.2f}')
    return np.median(P)


def cmapss_fits():
    d = np.loadtxt(os.path.join(ROOT, 'train_FD001.txt'))
    uid = d[:, 0].astype(int)
    S = {f's{j-4}': j for j in [6, 7, 8, 11, 12, 13, 15, 16, 17, 18, 19, 21, 24, 25]}
    units = [d[uid == u] for u in np.unique(uid)]
    cols = list(S.values())
    X = d[:, cols]
    X = (X - X.mean(0))/(X.std(0) + 1e-12)
    w = np.linalg.svd(X - X.mean(0), full_matrices=False)[2][0]
    out = {nm: [measure(u[:, j]) for u in units] for nm, j in list(S.items())[:6]}
    out['PC1 (fused)'] = [measure(((u[:, cols] - d[:, cols].mean(0)) /
                                   (d[:, cols].std(0) + 1e-12)) @ w) for u in units]
    return out


def xjtu_fits():
    d = np.load(os.path.join(HERE, 'xjtu_features.npz'), allow_pickle=True)
    keys = ['RMS', 'peak', 'kurtosis', 'HF 2-5kHz', 'HF 5-10kHz', 'spec centroid']
    out = {k: [] for k in keys}
    for bid in d.files:
        rec = d[bid].item()
        for k in keys:
            out[k].append(measure(rec[k]))
    return out


def capacity_fits():
    p = os.path.join(HERE, 'severson_traj.npz')
    if not os.path.exists(p):
        return {}
    d = np.load(p, allow_pickle=True)
    keys = [k for k in d.files if k != 'meta']
    return {'bulk capacity': [measure(np.asarray(d[k], float).ravel()) for k in keys]}


if __name__ == '__main__':
    print(f'{"domain":9s} {"indicator":20s} {"n":>4s} {"Ups":>8s} {"Pi(1)":>8s} '
          f'{"[IQR]":>13s} {"Ups*Pi(1)":>9s}')
    print('-'*76)
    allpi = []
    for nm, res in cmapss_fits().items():
        r = summarise('C-MAPSS', nm, res)
        if r:
            allpi.append(r)
    print()
    for nm, res in xjtu_fits().items():
        r = summarise('bearing', nm, res)
        if r:
            allpi.append(r)
    print()
    for nm, res in capacity_fits().items():
        r = summarise('battery', nm, res)
        if r:
            allpi.append(r)
    a = np.array(allpi)
    print()
    print(f'Pi(1) across {len(a)} channel medians: min {a.min():.2f}, '
          f'max {a.max():.2f}, median {np.median(a):.2f}')
    print(f'the power-law limit of the same family is 2 exactly '
          f'(a -> 0: check {pi_one(1e-3, 4.48):.3f})')
