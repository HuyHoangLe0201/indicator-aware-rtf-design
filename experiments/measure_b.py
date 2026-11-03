"""Measure the c-dependent bias floor B(c) of a truncated campaign (Proposition 0'').
For each c: vary the number of training units n at fixed c, fit  MSE(n) = a/n + B  by least squares.
B includes (i) imputation bias and (ii) selection bias of the eligible pool; the oracle arm (true lives
on the same units) isolates (ii). MSE is test MSE as a multiple of base^2 (base = predict pool mean)."""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
import numpy as np
warnings.filterwarnings('ignore')
which = sys.argv[1]
NS = (6, 10, 16, 24, 36)
REPS = 20; SEEDS = (0, 1)

def fit(ns, y):
    ns = np.array(ns, float); A = np.c_[1/ns, np.ones(len(ns))]
    a, B = np.linalg.lstsq(A, np.array(y), rcond=None)[0]
    res = np.array(y) - A @ np.array([a, B]); return a, B, float(np.sqrt(np.mean(res**2)))

def report(title, rows):
    print(f'\n=== {title} ===')
    print(f'{"c":>5s} {"elig":>5s} {"n used":>14s} | {"imputed: a":>10s} {"B":>7s} {"rms res":>7s} | {"oracle: a":>9s} {"B":>7s} | {"B_imp-B_orc":>11s}')
    for c, ne, ns, fi, fo in rows:
        print(f'{c:5.2f} {ne:5d} {str(list(ns)):>14s} | {fi[0]:10.2f} {fi[1]:7.3f} {fi[2]:7.3f} | {fo[0]:9.2f} {fo[1]:7.3f} | {fi[1]-fo[1]:11.3f}')

if which == 'cmapss':
    import cost_cmapss as cc
    m = 100; rows = []
    for c in (0.50, 0.65, 0.80, 1.00):
        acc = {n: ([], []) for n in NS}; ne_all = []
        for seed in SEEDS:
            units, life = cc.load(); rng = np.random.default_rng(seed)
            order = rng.permutation(len(life)); test, cal, pool = order[:cc.N_TEST], order[cc.N_TEST:cc.N_TEST+cc.N_CAL], order[cc.N_TEST+cc.N_CAL:]
            idx = cc.calibrate(units, cal); D = [idx(u) for u in units]
            Xte = np.array([cc.feats(D[i], m) for i in test]); yte = life[test]; base2 = np.mean((yte - life[pool].mean())**2)
            elig = np.array([i for i in pool if c*life[i] >= m]); ne_all.append(len(elig))
            for n in NS:
                if n > len(elig): continue
                for _ in range(REPS):
                    take = rng.choice(elig, n, replace=False); X = np.array([cc.feats(D[i], m) for i in take])
                    yl = life[take] if c >= 1 else cc.impute([(np.arange(1, int(c*life[i])+1, dtype=float), D[i][:int(c*life[i])]) for i in take], cc.cal_beta(D, life, cal))
                    acc[n][0].append(np.mean((cc.fit_huber(X, yl, Xte) - yte)**2)/base2)
                    acc[n][1].append(np.mean((cc.fit_huber(X, life[take], Xte) - yte)**2)/base2)
        ns = [n for n in NS if acc[n][0]]
        if len(ns) < 4: print(f'c={c}: too few n'); continue
        rows.append((c, int(np.mean(ne_all)), ns, fit(ns, [np.mean(acc[n][0]) for n in ns]), fit(ns, [np.mean(acc[n][1]) for n in ns])))
    report('C-MAPSS FD001, PC1, window m=100', rows)
else:
    import cost_real as cr
    cr.DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'severson', 'dq_traj.npz')
    QD, DQ, life = cr.load(); n_all = len(life)
    dmg = [cr.damage(QD[i]) for i in range(n_all)]; dc = [np.arange(1, len(d)+1, dtype=float) for d in dmg]
    for channel, m, grid in (('dq', 100, (0.35, 0.50, 0.65, 0.80, 1.00)), ('capacity', 400, (0.65, 0.80, 1.00))):
        if channel == 'capacity': series, cyc = dmg, dc
        else: cyc = [DQ[i][0].astype(float) for i in range(n_all)]; series = [DQ[i][1].astype(float) for i in range(n_all)]
        ok = np.array([life[i] >= m and cyc[i][-1] >= m for i in range(n_all)]); idx = np.where(ok)[0]; rows = []
        for c in grid:
            acc = {n: ([], []) for n in NS}; ne_all = []
            for seed in SEEDS:
                rng = np.random.default_rng(seed); test = rng.choice(idx, cr.NTEST, replace=False)
                pool = np.array([i for i in idx if i not in set(test)])
                Xte = np.array([cr.feats(series[i], cyc[i], m) for i in test]); yte = life[test]; base2 = np.mean((yte - life[pool].mean())**2)
                elig = np.array([i for i in pool if c*life[i] >= m]); ne_all.append(len(elig))
                for n in NS:
                    if n > len(elig): continue
                    for _ in range(REPS):
                        take = rng.choice(elig, n, replace=False); X = np.array([cr.feats(series[i], cyc[i], m) for i in take])
                        yl = life[take] if c >= 1 else cr.impute([(dc[i][dc[i] <= c*life[i]], dmg[i][dc[i] <= c*life[i]]) for i in take], cr.pool_beta(pool, dmg, dc, life))
                        acc[n][0].append(np.mean((X_ := Xte) @ cr.huber(X, yl) - yte)**2 if False else np.mean((Xte @ cr.huber(X, yl) - yte)**2)/base2)
                        acc[n][1].append(np.mean((Xte @ cr.huber(X, life[take]) - yte)**2)/base2)
            ns = [n for n in NS if acc[n][0]]
            if len(ns) < 4: print(f'{channel} c={c}: eligible {int(np.mean(ne_all))}, too few n points'); continue
            rows.append((c, int(np.mean(ne_all)), ns, fit(ns, [np.mean(acc[n][0]) for n in ns]), fit(ns, [np.mean(acc[n][1]) for n in ns])))
        report(f'Severson cells, channel {channel}, window m={m}', rows)
