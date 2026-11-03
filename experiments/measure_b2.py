"""B(c) with many splits and a bootstrap over splits. MSE(n)=a/n+B fitted on the split-mean curve.
Reports DeltaB = B_imputed - B_oracle with 95% CI from resampling splits (2000 draws)."""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
import numpy as np
warnings.filterwarnings('ignore')
which = sys.argv[1]
NS = (6, 10, 16, 24, 36, 48); REPS = 12; SEEDS = tuple(range(8))
def fit(ns, y):
    ns = np.array(ns, float); A = np.c_[1/ns, np.ones(len(ns))]
    co = np.linalg.lstsq(A, np.asarray(y), rcond=None)[0]; res = np.asarray(y) - A @ co
    return co[0], co[1], float(np.sqrt(np.mean(res**2)))
def summarise(title, per_c):
    print(f'\n=== {title}  ({len(SEEDS)} splits x {REPS} draws) ===')
    print(f'{"c":>5s} {"elig":>5s} {"n pts":>5s} | {"B_imp":>7s} {"B_orc":>7s} {"dB":>7s} {"95% CI over splits":>22s} {"a_imp":>7s} {"res":>6s}')
    rng = np.random.default_rng(0)
    for c, ne, ns, Si, So in per_c:        # Si, So: arrays [n_splits, len(ns)]
        f = lambda idx: (fit(ns, Si[idx].mean(0)), fit(ns, So[idx].mean(0)))
        (ai, Bi, ri), (ao, Bo, ro) = f(np.arange(len(Si)))
        d = []
        for _ in range(2000):
            idx = rng.integers(0, len(Si), len(Si)); (_, b1, _), (_, b2, _) = f(idx); d.append(b1 - b2)
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f'{c:5.2f} {ne:5d} {len(ns):5d} | {Bi:7.3f} {Bo:7.3f} {Bi-Bo:7.3f} [{lo:8.3f},{hi:8.3f}] {ai:7.2f} {ri:6.3f}')
def collect(get_split, c):
    Si = {n: [] for n in NS}; So = {n: [] for n in NS}; ne = []
    for seed in SEEDS:
        out = get_split(seed, c)
        if out is None: continue
        ne.append(out[0])
        for n, (mi, mo) in out[1].items(): Si[n].append(mi); So[n].append(mo)
    ns = [n for n in NS if len(Si[n]) == len(ne) and len(ne) >= 6]
    if len(ns) < 4: return None
    return c, int(np.mean(ne)), ns, np.array([[Si[n][s] for n in ns] for s in range(len(ne))]), np.array([[So[n][s] for n in ns] for s in range(len(ne))])
if which == 'cmapss':
    import cost_cmapss as cc
    m = 100
    def split(seed, c):
        units, life = cc.load(); rng = np.random.default_rng(seed)
        order = rng.permutation(len(life)); test, cal, pool = order[:cc.N_TEST], order[cc.N_TEST:cc.N_TEST+cc.N_CAL], order[cc.N_TEST+cc.N_CAL:]
        idx = cc.calibrate(units, cal); D = [idx(u) for u in units]
        Xte = np.array([cc.feats(D[i], m) for i in test]); yte = life[test]; b2 = np.mean((yte - life[pool].mean())**2)
        elig = np.array([i for i in pool if c*life[i] >= m]); res = {}
        for n in NS:
            if n > len(elig): continue
            a, o = [], []
            for _ in range(REPS):
                take = rng.choice(elig, n, replace=False); X = np.array([cc.feats(D[i], m) for i in take])
                yl = life[take] if c >= 1 else cc.impute([(np.arange(1, int(c*life[i])+1, dtype=float), D[i][:int(c*life[i])]) for i in take], cc.cal_beta(D, life, cal))
                a.append(np.mean((cc.fit_huber(X, yl, Xte) - yte)**2)/b2); o.append(np.mean((cc.fit_huber(X, life[take], Xte) - yte)**2)/b2)
            res[n] = (np.mean(a), np.mean(o))
        return len(elig), res
    summarise('C-MAPSS FD001 PC1, m=100', [r for r in (collect(split, c) for c in (0.5, 0.65, 0.8, 1.0)) if r])
else:
    import cost_real as cr
    cr.DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'severson', 'dq_traj.npz')
    QD, DQ, life = cr.load(); n_all = len(life)
    dmg = [cr.damage(QD[i]) for i in range(n_all)]; dc = [np.arange(1, len(d)+1, dtype=float) for d in dmg]
    for channel, m, grid in (('dq', 100, (0.35, 0.5, 0.65, 0.8, 1.0)), ('capacity', 400, (0.65, 0.8, 1.0))):
        if channel == 'capacity': series, cyc = dmg, dc
        else: cyc = [DQ[i][0].astype(float) for i in range(n_all)]; series = [DQ[i][1].astype(float) for i in range(n_all)]
        idx_ok = np.where(np.array([life[i] >= m and cyc[i][-1] >= m for i in range(n_all)]))[0]
        def split(seed, c):
            rng = np.random.default_rng(seed); test = rng.choice(idx_ok, cr.NTEST, replace=False)
            pool = np.array([i for i in idx_ok if i not in set(test)])
            Xte = np.array([cr.feats(series[i], cyc[i], m) for i in test]); yte = life[test]; b2 = np.mean((yte - life[pool].mean())**2)
            elig = np.array([i for i in pool if c*life[i] >= m]); res = {}
            for n in NS:
                if n > len(elig): continue
                a, o = [], []
                for _ in range(REPS):
                    take = rng.choice(elig, n, replace=False); X = np.array([cr.feats(series[i], cyc[i], m) for i in take])
                    yl = life[take] if c >= 1 else cr.impute([(dc[i][dc[i] <= c*life[i]], dmg[i][dc[i] <= c*life[i]]) for i in take], cr.pool_beta(pool, dmg, dc, life))
                    a.append(np.mean((Xte @ cr.huber(X, yl) - yte)**2)/b2); o.append(np.mean((Xte @ cr.huber(X, life[take]) - yte)**2)/b2)
                res[n] = (np.mean(a), np.mean(o))
            return len(elig), res
        summarise(f'Severson cells, channel {channel}, m={m}', [r for r in (collect(split, c) for c in grid) if r])
