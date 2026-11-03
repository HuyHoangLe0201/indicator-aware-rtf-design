"""Terminal concentration of the sensors the escape screen flags, on the screen's own channels.

Section VII-G says the two sensors carrying most information beyond the fused component are
also the latest-concentrated of the fourteen the screen covers, and gives three numbers. The
screen itself prints only partial correlations, so this recomputes the shapes on exactly the
channels the screen builds: every continuously varying sensor, oriented to increase with
damage, and the leading principal component of that same set.
"""
import os

import numpy as np

from upsilon_survey import ROOT, measure

STRONGEST = ('s14', 's15', 's9', 's17')


def screen_channels():
    d = np.loadtxt(os.path.join(ROOT, 'train_FD001.txt'))
    uid = d[:, 0].astype(int)
    keep = [j for j in range(5, d.shape[1]) if len(np.unique(d[:, j])) > 2]
    Z = (d[:, keep] - d[:, keep].mean(0))/d[:, keep].std(0)
    w = np.linalg.svd(Z - Z.mean(0), full_matrices=False)[2][0]
    pc = Z @ w
    if np.corrcoef(pc, d[:, 1])[0, 1] < 0:
        w, pc = -w, -pc
    out = {f's{c-4}': [] for c in keep}
    out['PC1 (fused)'] = []
    for u in np.unique(uid):
        m = uid == u
        for j, col in enumerate(keep):
            s = d[m, col]
            if np.corrcoef(s, np.arange(len(s)))[0, 1] < 0:
                s = -s
            out[f's{col-4}'].append(measure(s))
        out['PC1 (fused)'].append(measure(pc[m]))
    return out, len(keep)


if __name__ == '__main__':
    ch, nk = screen_channels()
    print(f'the screen covers {nk} continuously varying sensors plus the fused component')
    print(f'{"channel":14s} {"n":>4s} {"Ups par":>8s} {"Ups np":>8s}')
    print('-'*40)
    rows = []
    for nm, res in ch.items():
        v = [x for x in res if x is not None and np.isfinite(x[0])]
        if not v:
            continue
        up = float(np.median([x[0] for x in v]))
        un = float(np.median([x[1] for x in v]))
        rows.append((nm, len(v), up, un))
    for nm, n, up, un in rows:
        mark = '  <- flagged by the screen' if nm in STRONGEST else ''
        mark += '  <- fused' if nm.startswith('PC1') else ''
        print(f'{nm:14s} {n:4d} {up:8.2f} {un:8.2f}{mark}')
    sen = [r for r in rows if not r[0].startswith('PC1')]
    sen.sort(key=lambda r: -r[2])
    print()
    print('latest-concentrated sensors on the parametric estimator: '
          + ', '.join(f'{r[0]} {r[2]:.1f}' for r in sen[:4]))
    print('paper: the two strongest are 8.7 and 8.4, against 7.0 for the fused component')
