"""Out-of-sample test of Proposition A.1 and of the role of recruitment, on the further public cell sets (Section S3.6, Table S19b).

Protocol of bl_campaign.py (window m = 0.15 / 0.55 median lives for DeltaQ(V) / capacity, a quarter of the cells held out,
a budget of N1 complete units, labels imputed from the exponent fitted on the training pool), with a least-squares learner.
For every stop fraction c the script reports, as ratios to the error of the training-pool mean (mean over splits):
    predicted  sqrt(Bias^2 + sigma_res^2 p / (n - p - 1)),  Bias^2 and sigma_res^2 from the population fit on all eligible cells;
    imputed    Monte-Carlo campaign with imputed labels;
    true       Monte-Carlo campaign on the same cells with their true lives (recruitment effect only).
It then prints the stop fraction that minimizes each of the three columns.
Usage: CH=dq|capacity python risk_decomposition_sets.py SET N1
"""
import os
import sys
import warnings

import numpy as np

import bl_common as bc
import cost_real as cr

warnings.filterwarnings('ignore')
CH = os.environ.get('CH', 'dq')
NSPLIT = int(os.environ.get('NSPLIT', '10'))
NDRAW = int(os.environ.get('NDRAW', '60'))
GRID = (0.35, 0.50, 0.65, 0.80, 1.00)
P = 4

name, N1 = sys.argv[1], int(sys.argv[2])
QD, DQ, life, f, keys = bc.load_set(name)
n_all = len(life)
dmg = [bc.damage_scaled(QD[i], f) for i in range(n_all)]
dc = [np.arange(1, len(d) + 1, dtype=float) for d in dmg]
if CH == 'capacity':
    cyc = [d.copy() for d in dc]
    series = [np.asarray(d, float) for d in dmg]
else:
    cyc = [DQ[i][0].astype(float) for i in range(n_all)]
    series = [DQ[i][1].astype(float) for i in range(n_all)]
med = float(np.median(life))
M = int(round((0.15 if CH == 'dq' else 0.55) * med))
idx_ok = np.array([i for i in range(n_all) if life[i] >= M and cyc[i][-1] >= M and len(dmg[i]) >= M])
NT = int(round(0.25 * len(idx_ok)))
C = N1 * (1 + cr.KAP0)
X_all = {i: cr.feats(series[i], cyc[i], M) for i in idx_ok}

rows = {c: [] for c in GRID}
for seed in range(NSPLIT):
    rng = np.random.default_rng(seed)
    test = rng.choice(idx_ok, NT, replace=False)
    pool = np.array([i for i in idx_ok if i not in set(test)])
    Xte = np.array([X_all[i] for i in test])
    yte = life[test]
    base = float(np.sqrt(np.mean((yte - life[pool].mean())**2)))
    B0 = cr.pool_beta(pool, dmg, dc, life)
    for c in GRID:
        elig = np.array([i for i in pool if c * life[i] >= M])
        n = min(max(3, int(round(C / (cr.KAP0 + c)))), len(elig))
        if n < P + 3:
            continue
        trajs = [(dc[i][dc[i] <= c * life[i]], dmg[i][dc[i] <= c * life[i]]) for i in elig]
        yl = life[elig] if c >= 1 else cr.impute(trajs, beta0=B0)
        X = np.array([X_all[i] for i in elig])
        b = np.linalg.lstsq(X, yl, rcond=None)[0]
        s2 = float(np.sum((yl - X @ b)**2) / max(len(elig) - P, 1))
        bias2 = float(np.mean((Xte @ b - yte)**2))
        pred = np.sqrt(bias2 + s2 * P / (n - P - 1)) / base
        e_imp, e_true = [], []
        for _ in range(NDRAW):
            take = rng.choice(len(elig), n, replace=False)
            bi = np.linalg.lstsq(X[take], yl[take], rcond=None)[0]
            bt = np.linalg.lstsq(X[take], life[elig][take], rcond=None)[0]
            e_imp.append(np.mean((Xte @ bi - yte)**2))
            e_true.append(np.mean((Xte @ bt - yte)**2))
        rows[c].append((pred, np.sqrt(np.mean(e_imp)) / base, np.sqrt(np.mean(e_true)) / base, n, len(elig)))

print('%s %s window %d, %d usable cells, N1 = %d' % (name, CH, M, len(idx_ok), N1))
print('%5s %5s %6s | %9s %8s %8s' % ('c', 'n', 'elig', 'predicted', 'imputed', 'true'))
best = {}
for c in GRID:
    if len(rows[c]) < 3:
        continue
    a = np.array(rows[c])
    print('%5.2f %5.0f %6.0f | %9.2f %8.2f %8.2f' % (c, a[:, 3].mean(), a[:, 4].mean(), a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()))
    best[c] = a[:, :3].mean(axis=0)
if best:
    print('argmin c: predicted %.2f | imputed MC %.2f | true-label MC %.2f' % tuple(min(best, key=lambda c: best[c][k]) for k in range(3)))
