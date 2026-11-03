"""Risk of a campaign with imputed labels for an OLS learner:
    E[test MSE] ~= Bias2(c) + sigma2_res(c) * p/(n-p-1)
Bias2(c)   = E_test[(x'b_c - y)^2], b_c = population OLS limit on the eligible pool with the imputed labels
sigma2_res = residual variance of that regression
No fitted constants: everything is computed from the pool.  Compared with the Monte-Carlo campaign (OLS, n units
drawn from the eligible pool, same imputation), channel and window as given on the command line."""
import sys

import os

import numpy as np

import cost_real as CR

CR.DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'severson', 'dq_traj.npz')
ch, m = sys.argv[1], int(sys.argv[2])
GRID = (0.35, 0.50, 0.65, 0.80, 1.00)
C = 21.0
QD, DQ, life = CR.load()
n_all = len(life)
dmg, dmg_cyc, series, cycles = [], [], [], []
for i in range(n_all):
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
ok = np.array([life[i] >= m and cycles[i][-1] >= m and np.all(np.isfinite(series[i][np.isfinite(series[i])])) for i in range(n_all)])
idx = np.where(ok)[0]
X_all = {i: CR.feats(series[i], cycles[i], m) for i in idx}
p = 4

res = {c: [] for c in GRID}
for seed in (0, 1, 2):
    rng = np.random.default_rng(seed)
    test = rng.choice(idx, CR.NTEST, replace=False)
    pool = np.array([i for i in idx if i not in set(test)])
    Xte = np.array([X_all[i] for i in test])
    yte = life[test]
    base = float(np.sqrt(np.mean((yte - life[pool].mean())**2)))
    b0 = CR.pool_beta(pool, dmg, dmg_cyc, life)
    for c in GRID:
        elig = np.array([i for i in pool if c * life[i] >= m])
        n = min(max(4, int(round(C / (CR.KAP0 + c)))), len(elig))
        if n < 8:
            continue
        trajs = []
        for i in elig:
            sel = dmg_cyc[i] <= c * life[i]
            trajs.append((dmg_cyc[i][sel], dmg[i][sel]))
        yl = life[elig] if c >= 1.0 else CR.impute(trajs, b0)
        X = np.array([X_all[i] for i in elig])
        b = np.linalg.lstsq(X, yl, rcond=None)[0]
        resid = yl - X @ b
        s2 = float(np.sum(resid**2) / max(len(elig) - p, 1))
        bias2 = float(np.mean((Xte @ b - yte)**2))
        pred = np.sqrt(bias2 + s2 * p / (n - p - 1)) / base
        # Monte Carlo campaign with OLS
        mc = []
        for _ in range(300):
            take = rng.choice(len(elig), n, replace=False)
            bb = np.linalg.lstsq(X[take], yl[take], rcond=None)[0]
            mc.append(np.mean((Xte @ bb - yte)**2))
        emp = np.sqrt(np.mean(mc)) / base
        res[c].append((pred, emp, n, len(elig), np.sqrt(bias2) / base, np.sqrt(s2 * p / (n - p - 1)) / base))

print('channel %s, window %d' % (ch, m))
print('%5s %6s %6s | %8s %8s | %10s %10s' % ('c', 'n', 'elig', 'predicted', 'MC OLS', 'bias part', 'var part'))
for c in GRID:
    if res[c]:
        a = np.array(res[c])
        print('%5.2f %6.0f %6.0f | %8.2f %8.2f | %10.2f %10.2f' % (c, a[:, 2].mean(), a[:, 3].mean(), a[:, 0].mean(), a[:, 1].mean(), a[:, 4].mean(), a[:, 5].mean()))
