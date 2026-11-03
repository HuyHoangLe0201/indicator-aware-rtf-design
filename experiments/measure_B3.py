"""Low-c campaigns (c < 0.5). The a/n+B fit is unreliable here (heavy-tailed imputation), so compare
imputed vs oracle labels directly at fixed n in {16,36}; at larger n the a/n term shrinks, so the gap
approximates the bias floor. 8 splits x 12 draws; CI = bootstrap over splits; also the median label error.
MSE in units of base^2 (predict pool mean)."""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
import numpy as np
warnings.filterwarnings('ignore')
which = sys.argv[1]; SEEDS = tuple(range(8)); REPS = 12; NFIX = (16, 36)
def table(title, rows):
    print(f'\n=== {title} ===')
    print(f'{"c":>5s} {"elig":>5s} {"lab err":>8s} | ' + ' | '.join(f'n={n}: {"imp":>7s} {"orc":>6s} {"gap":>7s} {"95% CI":>17s}' for n in NFIX))
    rng = np.random.default_rng(0)
    for c, ne, lab, per in rows:     # per[n] = (imp[splits], orc[splits])
        cells = []
        for n in NFIX:
            if n not in per or len(per[n][0]) < 6: cells.append(f'n={n}: {"--":>7s} {"--":>6s} {"--":>7s} {"(too few units)":>17s}'); continue
            I, O = np.array(per[n][0]), np.array(per[n][1]); g = [np.mean(I[k]) - np.mean(O[k]) for k in (rng.integers(0, len(I), len(I)) for _ in range(2000))]
            lo, hi = np.percentile(g, [2.5, 97.5]); cells.append(f'n={n}: {I.mean():7.2f} {O.mean():6.2f} {I.mean()-O.mean():7.2f} [{lo:7.2f},{hi:7.2f}]')
        print(f'{c:5.2f} {ne:5d} {lab:8.3f} | ' + ' | '.join(cells))
def one(split, c):
    per = {n: ([], []) for n in NFIX}; ne = []; labs = []
    for seed in SEEDS:
        r = split(seed, c)
        if r is None: continue
        ne.append(r[0]); labs.append(r[2])
        for n, (a, o) in r[1].items(): per[n][0].append(a); per[n][1].append(o)
    return c, int(np.mean(ne)) if ne else 0, float(np.mean(labs)) if labs else float('nan'), per
if which == 'cmapss':
    import cost_cmapss as cc; m = 60
    def split(seed, c):
        units, life = cc.load(); rng = np.random.default_rng(seed)
        order = rng.permutation(len(life)); test, cal, pool = order[:cc.N_TEST], order[cc.N_TEST:cc.N_TEST+cc.N_CAL], order[cc.N_TEST+cc.N_CAL:]
        idx = cc.calibrate(units, cal); D = [idx(u) for u in units]
        Xte = np.array([cc.feats(D[i], m) for i in test]); yte = life[test]; b2 = np.mean((yte - life[pool].mean())**2)
        elig = np.array([i for i in pool if c*life[i] >= m])
        if len(elig) < 12: return None
        res = {}; labe = []
        for n in NFIX:
            nn = min(n, len(elig))
            if nn < n: continue
            a, o = [], []
            for _ in range(REPS):
                take = rng.choice(elig, n, replace=False); X = np.array([cc.feats(D[i], m) for i in take])
                yl = life[take] if c >= 1 else cc.impute([(np.arange(1, int(c*life[i])+1, dtype=float), D[i][:int(c*life[i])]) for i in take], cc.cal_beta(D, life, cal))
                labe.append(np.median(np.abs(yl - life[take])/life[take]))
                a.append(np.mean((cc.fit_huber(X, yl, Xte) - yte)**2)/b2); o.append(np.mean((cc.fit_huber(X, life[take], Xte) - yte)**2)/b2)
            res[n] = (np.mean(a), np.mean(o))
        return len(elig), res, float(np.mean(labe)) if labe else float('nan')
    table('C-MAPSS FD001 PC1, m=60 (units that reach the window: lives >= m/c)', [one(split, c) for c in (0.2, 0.35, 0.5, 1.0)])
else:
    import cost_real as cr
    cr.DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'severson', 'dq_traj.npz')
    QD, DQ, life = cr.load(); n_all = len(life)
    dmg = [cr.damage(QD[i]) for i in range(n_all)]; dc = [np.arange(1, len(d)+1, dtype=float) for d in dmg]
    cyc = [DQ[i][0].astype(float) for i in range(n_all)]; series = [DQ[i][1].astype(float) for i in range(n_all)]
    for m, grid in ((50, (0.1, 0.2, 0.35, 0.5, 1.0)), (100, (0.2, 0.35, 0.5, 1.0))):
        idx_ok = np.where(np.array([life[i] >= m and cyc[i][-1] >= m for i in range(n_all)]))[0]
        def split(seed, c):
            rng = np.random.default_rng(seed); test = rng.choice(idx_ok, cr.NTEST, replace=False)
            pool = np.array([i for i in idx_ok if i not in set(test)])
            Xte = np.array([cr.feats(series[i], cyc[i], m) for i in test]); yte = life[test]; b2 = np.mean((yte - life[pool].mean())**2)
            elig = np.array([i for i in pool if c*life[i] >= m])
            if len(elig) < 20: return None
            res = {}; labe = []
            for n in NFIX:
                if n > len(elig): continue
                a, o = [], []
                for _ in range(REPS):
                    take = rng.choice(elig, n, replace=False); X = np.array([cr.feats(series[i], cyc[i], m) for i in take])
                    yl = life[take] if c >= 1 else cr.impute([(dc[i][dc[i] <= c*life[i]], dmg[i][dc[i] <= c*life[i]]) for i in take], cr.pool_beta(pool, dmg, dc, life))
                    labe.append(np.median(np.abs(yl - life[take])/life[take]))
                    a.append(np.mean((Xte @ cr.huber(X, yl) - yte)**2)/b2); o.append(np.mean((Xte @ cr.huber(X, life[take]) - yte)**2)/b2)
                res[n] = (np.mean(a), np.mean(o))
            return len(elig), res, float(np.mean(labe)) if labe else float('nan')
        table(f'Severson cells, DeltaQ(V) channel, m={m}', [one(split, c) for c in grid])
