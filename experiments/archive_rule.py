"""Archive-calibrated choice of the stop fraction c (prospective, uses only units already run to failure).

An archive of n1 completed cells is available (cold start).  For every candidate c the archive trajectories are truncated
at c, labels are imputed exactly as in the campaign, and the OLS fit on imputed labels is compared with the OLS fit on
true labels, in the archive features:
    Bhat(c) = mean over archive of ( x'(b_imputed - b_true) )^2          label-induced bias
    pred(c) = Bhat(c) + p * s2(c) / (n(c) - p - 1)                         plus variance with n(c) = C/(kappa0+c) units
The rule picks the c with the smallest pred.  Its realised test RMSE (Monte-Carlo OLS campaign on the other cells,
imputed labels) is compared with the corner c=1, with the fixed c=0.65 and with the best fixed c in hindsight."""
import sys

import os

import numpy as np

import cost_real as CR

CR.DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'severson', 'dq_traj.npz')
ch, m, n1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
GRID = (0.35, 0.50, 0.65, 0.80, 1.00)
C, p = 21.0, 4
QD, DQ, life = CR.load()
N = len(life)
dmg, dmg_cyc, series, cycles = [], [], [], []
for i in range(N):
    d = CR.damage(QD[i])
    dc = np.arange(1, len(d) + 1, dtype=float)
    dmg.append(d)
    dmg_cyc.append(dc)
    if ch == 'capacity':
        s_, c_ = d, dc
    else:
        c_, s_ = DQ[i][0].astype(float), DQ[i][1].astype(float)
    series.append(np.asarray(s_, float))
    cycles.append(c_)
ok = np.array([life[i] >= m and cycles[i][-1] >= m and np.all(np.isfinite(series[i][np.isfinite(series[i])])) for i in range(N)])
idx = np.where(ok)[0]
X_all = {i: CR.feats(series[i], cycles[i], m) for i in idx}


def impute_set(units, c, b0):
    trajs = []
    for i in units:
        sel = dmg_cyc[i] <= c * life[i]
        trajs.append((dmg_cyc[i][sel], dmg[i][sel]))
    return life[units] if c >= 1.0 else CR.impute(trajs, b0)


rows = []
for seed in range(8):
    rng = np.random.default_rng(100 + seed)
    test = rng.choice(idx, CR.NTEST, replace=False)
    rest = np.array([i for i in idx if i not in set(test)])
    arch = rng.choice(rest, n1, replace=False)
    pool = np.array([i for i in rest if i not in set(arch)])
    Xte = np.array([X_all[i] for i in test])
    yte = life[test]
    base = float(np.sqrt(np.mean((yte - life[rest].mean())**2)))
    b0 = CR.pool_beta(arch, dmg, dmg_cyc, life)
    Xa = np.array([X_all[i] for i in arch])
    bt = np.linalg.lstsq(Xa, life[arch], rcond=None)[0]
    pred, emp = {}, {}
    for c in GRID:
        elig_a = np.array([i for i in arch if c * life[i] >= m])
        n = int(round(C / (CR.KAP0 + c)))
        if len(elig_a) < 8:
            pred[c] = np.inf
        else:
            ya = impute_set(elig_a, c, b0)
            Xe = np.array([X_all[i] for i in elig_a])
            bi = np.linalg.lstsq(Xe, ya, rcond=None)[0]
            bhat = float(np.mean((Xe @ (bi - bt))**2))
            s2 = float(np.sum((ya - Xe @ bi)**2) / max(len(elig_a) - p, 1))
            nn = min(n, len(pool))
            pred[c] = bhat + p * s2 / max(nn - p - 1, 1)
        elig = np.array([i for i in pool if c * life[i] >= m])
        nn = min(n, len(elig))
        if nn < 8:
            emp[c] = np.inf
            continue
        yl = impute_set(elig, c, b0)
        X = np.array([X_all[i] for i in elig])
        mc = []
        for _ in range(150):
            take = rng.choice(len(elig), nn, replace=False)
            bb = np.linalg.lstsq(X[take], yl[take], rcond=None)[0]
            mc.append(np.mean((Xte @ bb - yte)**2))
        emp[c] = np.sqrt(np.mean(mc)) / base
    pick = min(GRID, key=lambda c: pred[c])
    best = min(GRID, key=lambda c: emp[c])
    rows.append((pick, emp[pick], emp[1.0], emp[0.65], emp[best], best))
    print('seed %d | picked c=%.2f (rmse %.2f) | corner %.2f | c=0.65 %.2f | best hindsight c=%.2f (%.2f)' % (seed, pick, emp[pick], emp[1.0], emp[0.65], best, emp[best]), flush=True)
a = np.array(rows)
print('MEAN over seeds: picked %.2f | corner %.2f | fixed 0.65 %.2f | hindsight best %.2f' % (a[:, 1].mean(), a[:, 2].mean(), a[:, 3].mean(), a[:, 4].mean()))
print('picked c distribution:', {float(c): int((a[:, 0] == c).sum()) for c in GRID})
