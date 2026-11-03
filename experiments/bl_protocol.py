"""Leave-protocol-out: does a predictor trained on some cycling protocols (or manufacturing groups) transfer to the others?

Within a set the cells are grouped by the protocol or condition encoded in the cell name (GROUP below). The paper's four-feature Huber
map predicts the life of the held-out cells from the early window (0.15 median lives for Delta Q(V), 0.55 for capacity, as in
bl_campaign.py). Two errors are compared on the same cells: random 5-fold cross-validation (repeated 20 times) and
leave-one-group-out. Each error is divided by that of the training-fold mean, so a value
above 1 means the map is worse than predicting the training mean. Groups with fewer than 3 cells are used for training only; a set needs at least two groups of 3 cells.

Usage: python bl_protocol.py SET [SET ...]
"""
import sys, re, warnings
import numpy as np
import cost_real as cr
import bl_common as bc
warnings.filterwarnings('ignore')


def group(name, key):
    k = key[:-4]
    if name == 'HUST':
        return k.split('_')[1].split('-')[0]
    if name == 'Tongji':
        return k.split('_')[1]
    if name == 'MICH':
        return '_'.join(k.split('_')[2:-1])
    if name == 'Stanford':
        return 'Ref' if 'Ref' in k else k.split('_')[2]
    if name == 'XJTU':
        return k.split('_')[1]
    if name in ('HNEI', 'SNL'):
        return '_'.join(k.split('_')[1:-1])
    if name == 'CALCE':
        return k.split('_')[1]
    return 'all'


def rel(X, y, te, tr):
    b = cr.huber(X[tr], y[tr])
    return np.sum((y[te] - X[te] @ b)**2), np.sum((y[te] - y[tr].mean())**2)


def run(name):
    QD, DQ, life, f, keys = bc.load_set(name)
    n = len(life); med = float(np.median(life))
    grp = np.array([group(name, k) for k in keys])
    print(f'\n=== {name}: {n} cells, groups {dict((str(u), int(c)) for u, c in zip(*np.unique(grp, return_counts=True)))}')
    for ch in ('dq', 'capacity'):
        dmg = [bc.damage_scaled(q, f) for q in QD]
        dc = [np.arange(1, len(d) + 1, dtype=float) for d in dmg]
        if ch == 'capacity':
            cyc, ser = dc, dmg; M = int(round(0.55*med))
        else:
            cyc = [d[0].astype(float) for d in DQ]; ser = [d[1].astype(float) for d in DQ]; M = int(round(0.15*med))
        ok = np.array([i for i in range(n) if life[i] >= M and cyc[i][-1] >= M and len(dmg[i]) >= M])
        X = np.array([cr.feats(ser[i], cyc[i], M) for i in ok]); y = life[ok]; g = grp[ok]
        e1 = e0 = 0.0
        rng = np.random.default_rng(3)
        for _ in range(20):
            for fold in np.array_split(rng.permutation(len(ok)), 5):
                tr = np.setdiff1d(np.arange(len(ok)), fold)
                a, b = rel(X, y, fold, tr); e1 += a; e0 += b
        rand = np.sqrt(e1/e0)
        ug = [u for u in np.unique(g) if (g == u).sum() >= 3]
        if len(ug) < 2:
            print(f'  {ch:9s} m={M:4d} {len(ok):3d} cells: random CV {rand:.2f}; fewer than 2 groups of at least 3 cells'); continue
        s1 = s0 = 0.0
        for u in ug:
            te = np.where(g == u)[0]; tr = np.where(g != u)[0]
            a, b = rel(X, y, te, tr); s1 += a; s0 += b
        print(f'  {ch:9s} m={M:4d} {len(ok):3d} cells, {len(ug)} groups: error / training-mean error: random 5-fold {rand:.2f}, leave-group-out {np.sqrt(s1/s0):.2f}', flush=True)


if __name__ == '__main__':
    for s in sys.argv[1:]:
        run(s)
