"""Archive-calibrated choice of the stop fraction on the further public cell sets (Section S3.6; companion of archive_rule.py).

Same rule and protocol as archive_rule.py (prospective: the archive of complete cells chooses c, the choice is then scored on held-out
cells with imputed labels), on the BatteryLife sets of bl_common.py, with window 0.15 / 0.55 median lives for DeltaQ(V) / capacity.
The archive holds a third of the usable cells outside the test quarter and the budget buys N1 complete units.
Usage: CH=dq|capacity python archive_rule_sets.py SET N1
"""
import os
import sys
import warnings

import numpy as np

import bl_common as bc
import cost_real as CR

warnings.filterwarnings('ignore')
ch = os.environ.get('CH', 'dq')
name, n1b = sys.argv[1], int(sys.argv[2])
GRID = (0.35, 0.50, 0.65, 0.80, 1.00)
C, p = n1b * (1 + CR.KAP0), 4
QD, DQ, life, f, keys = bc.load_set(name)
N = len(life)
dmg = [bc.damage_scaled(QD[i], f) for i in range(N)]
dmg_cyc = [np.arange(1, len(d) + 1, dtype=float) for d in dmg]
if ch == 'capacity':
    cycles = [d.copy() for d in dmg_cyc]
    series = [np.asarray(d, float) for d in dmg]
else:
    cycles = [DQ[i][0].astype(float) for i in range(N)]
    series = [DQ[i][1].astype(float) for i in range(N)]
m = int(round((0.15 if ch == 'dq' else 0.55) * float(np.median(life))))
idx = np.array([i for i in range(N) if life[i] >= m and cycles[i][-1] >= m and len(dmg[i]) >= m])
NTEST = int(round(0.25 * len(idx)))
X_all = {i: CR.feats(series[i], cycles[i], m) for i in idx}


def impute_set(units, c, b0):
    trajs = []
    for i in units:
        sel = dmg_cyc[i] <= c * life[i]
        trajs.append((dmg_cyc[i][sel], dmg[i][sel]))
    return life[units] if c >= 1.0 else CR.impute(trajs, b0)


rows = []
for seed in range(10):
    rng = np.random.default_rng(100 + seed)
    test = rng.choice(idx, NTEST, replace=False)
    rest = np.array([i for i in idx if i not in set(test)])
    arch = rng.choice(rest, max(12, len(rest) // 3), replace=False)
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
    print(name, ch, 'seed %d | picked c=%.2f (rmse %.2f) | corner %.2f | c=0.65 %.2f | best hindsight c=%.2f (%.2f)' % (seed, pick, emp[pick], emp[1.0], emp[0.65], best, emp[best]), flush=True)
a = np.array(rows)
print('MEAN over seeds: picked %.2f | corner %.2f | fixed 0.65 %.2f | hindsight best %.2f' % (a[:, 1].mean(), a[:, 2].mean(), a[:, 3].mean(), a[:, 4].mean()))
print('picked c distribution:', {float(c): int((a[:, 0] == c).sum()) for c in GRID})
