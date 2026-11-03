"""Apply the direction check to the three domains surveyed before it existed.

The check was introduced on the Zhang cells, where ohmic resistance is stable and early yet
the fleet splits on whether it rises or falls, and it was then applied to the milling signals.
The batteries, turbofans and bearings were surveyed before that, so Table 5 currently carries
a dash for twelve of its twenty-one rows. Nothing prevents the check being run on them: the
trajectories are the same ones the survey used.

The rule is milling_test.py's, unchanged: a unit votes by the sign of its late median minus its
early median, and the score is the share agreeing with the majority. The paper strikes a
channel below 75 per cent.
"""
import os

import numpy as np

from upsilon_survey import ROOT

HERE = os.path.dirname(os.path.abspath(__file__))
BAR = 0.75


def agreement(units):
    """Share of units agreeing with the majority direction."""
    signs = []
    for y in units:
        y = np.asarray(y, float)
        y = y[np.isfinite(y)]
        if len(y) < 8:
            continue
        signs.append(int(np.sign(np.median(y[-max(2, len(y)//20):])
                                 - np.median(y[:max(2, len(y)//20)]))))
    signs = [s for s in signs if s != 0]
    if not signs:
        return float('nan'), 0
    return max(signs.count(1), signs.count(-1))/len(signs), len(signs)


def bearings():
    d = np.load(os.path.join(HERE, 'xjtu_features.npz'), allow_pickle=True)
    keys = ['RMS', 'peak', 'kurtosis', 'HF 2-5kHz', 'HF 5-10kHz', 'spec centroid']
    out = {}
    for k in keys:
        # the fleet and decimation of every other bearing number: every second snapshot, units with at least 60 left (12 of 13)
        out[k] = agreement([d[b].item()[k][::2] for b in d.files if len(d[b].item()['RMS'][::2]) >= 60])
    return out


def turbofans():
    d = np.loadtxt(os.path.join(ROOT, 'train_FD001.txt'))
    uid = d[:, 0].astype(int)
    S = {f'sensor {j-4}': j for j in [6, 7, 8, 11, 12, 13]}
    keep = [j for j in range(5, d.shape[1]) if len(np.unique(d[:, j])) > 2]
    Z = (d[:, keep] - d[:, keep].mean(0))/(d[:, keep].std(0) + 1e-12)
    w = np.linalg.svd(Z - Z.mean(0), full_matrices=False)[2][0]
    pc = Z @ w
    units = [uid == u for u in np.unique(uid)]
    out = {nm: agreement([d[m, j] for m in units]) for nm, j in S.items()}
    out['PC1 of the fourteen'] = agreement([pc[m] for m in units])
    return out


def cells():
    p = os.path.join(HERE, '..', 'data', 'severson', 'dq_traj.npz')
    d = np.load(p, allow_pickle=True)
    DQ, QD = d['DQ'], d['QD']
    cap, dq = [], []
    for i in range(len(DQ)):
        g = np.asarray(DQ[i], float)
        q = np.asarray(QD[i], float).ravel()
        q = q[np.isfinite(q) & (q > 0.2) & (q < 1.5)]
        if len(q) > 20:
            cap.append(q)
        v = g[1][np.isfinite(g[1])]
        if len(v) > 20:
            dq.append(v)
    return {'bulk capacity': agreement(cap), '$\\Delta Q(V)$ signature': agreement(dq)}


if __name__ == '__main__':
    print(f'{"domain":10s} {"indicator":24s} {"units":>6s} {"agree":>7s}  verdict')
    print('-'*62)
    for dom, fn in (('battery', cells), ('turbofan', turbofans), ('bearing', bearings)):
        for k, (a, n) in fn().items():
            tag = 'passes' if a >= BAR else f'FAILS the {BAR:.0%} bar'
            print(f'{dom:10s} {k:24s} {n:6d} {a*100:6.0f}%  {tag}')
        print()
    print(f'the milling set, where the check was introduced, gives 100, 100, 64, 93, 93 and 57 '
          f'per cent')
