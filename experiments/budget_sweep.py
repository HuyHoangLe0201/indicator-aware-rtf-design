"""Budget sweep on the DeltaQ(V) channel: realized gain of stopping early as a function of the budget.
Budget is expressed as the number of complete units it buys, N1 = C/(1+kappa0). A campaign at c buys n = round(C/(kappa0+c)).
Same protocol as measure_B4.py (imputed labels; oracle = true lives), 16 splits x 12 draws; paired differences with the
corner are taken split by split."""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
import numpy as np
import cost_real as cr
from scipy.optimize import least_squares
warnings.filterwarnings('ignore')
cr.DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'severson', 'dq_traj.npz')
QD, DQ, life = cr.load(); n_all = len(life)
dmg = [cr.damage(QD[i]) for i in range(n_all)]; dc = [np.arange(1, len(d)+1, dtype=float) for d in dmg]
CH = os.environ.get('CH', 'dq')
if CH == 'capacity':
    cyc = [d.copy() for d in dc]; series = [np.asarray(d, float) for d in dmg]
else:
    cyc = [DQ[i][0].astype(float) for i in range(n_all)]; series = [DQ[i][1].astype(float) for i in range(n_all)]
M = int(os.environ.get('WIN', '100'))
NSPLIT = int(os.environ.get('NSPLIT', '16'))
NDRAW = 12
BMODE = os.environ.get('BMODE', 'pool')   # pool: exponent fitted on complete trajectories of the training pool only; leak: 4.483; free: start 2.0
def pool_beta(pool):
    bs = []
    for i in pool:
        t = dc[i]/life[i]; D = dmg[i]; sel = t <= 1.0
        f = lambda p: np.clip(t[sel], 1e-9, 3.)**p[0] - D[sel]
        try: bs.append(float(least_squares(f, [4.], bounds=([0.4], [15.])).x[0]))
        except Exception: pass
    return float(np.median(bs))
GRID = (0.35, 0.50, 0.65, 0.80, 1.00)
N1S = [int(x) for x in os.environ.get('N1S', '6,8,10,12,16,20').split(',')]
idx_ok = np.where(np.array([life[i] >= M and cyc[i][-1] >= M for i in range(n_all)]))[0]
rng0 = np.random.default_rng(0)
print(f'channel {CH} window m={M}, {len(idx_ok)} cells, {NSPLIT} splits x {NDRAW} draws; entries RMSE/base (imputed | oracle)')
for N1 in N1S:
    C = N1 * (1 + cr.KAP0)
    res = {}
    for c in GRID:
        R, O, L, NU = [], [], [], []
        for seed in range(NSPLIT):
            rng = np.random.default_rng(seed); test = rng.choice(idx_ok, cr.NTEST, replace=False)
            pool = np.array([i for i in idx_ok if i not in set(test)])
            Xte = np.array([cr.feats(series[i], cyc[i], M) for i in test]); yte = life[test]
            base = np.sqrt(np.mean((yte - life[pool].mean())**2))
            B0 = 4.483 if BMODE=='leak' else (2.0 if BMODE=='free' else pool_beta(pool))
            elig = np.array([i for i in pool if c*life[i] >= M]); n_ask = max(3, int(round(C/(cr.KAP0 + c)))); n = min(n_ask, len(elig))
            if n < 3: continue
            r, o, l = [], [], []
            for _ in range(NDRAW):
                take = rng.choice(elig, n, replace=False); X = np.array([cr.feats(series[i], cyc[i], M) for i in take])
                yl = life[take] if c >= 1 else cr.impute([(dc[i][dc[i] <= c*life[i]], dmg[i][dc[i] <= c*life[i]]) for i in take], beta0=B0)
                r.append(np.sqrt(np.mean((Xte @ cr.huber(X, yl) - yte)**2))/base)
                o.append(np.sqrt(np.mean((Xte @ cr.huber(X, life[take]) - yte)**2))/base)
                l.append(np.median(np.abs(yl - life[take])/life[take]))
            R.append(np.mean(r)); O.append(np.mean(o)); L.append(np.mean(l)); NU.append(n)
        res[c] = (np.array(R), np.array(O), np.mean(L), np.mean(NU))
    Rc = res[1.0][0]
    print(f'\nN1={N1} (budget C={C:.1f}); corner buys {N1} units')
    print(f'{"c":>5s} {"n":>4s} {"lab err":>8s} | {"imputed":>8s} {"oracle":>7s} | {"paired gain vs corner (imputed)":>32s}')
    for c in GRID:
        R, O, L, NU = res[c]
        d = (Rc - R) / Rc  # positive = early stop better
        bs = [np.mean(d[rng0.integers(0, len(d), len(d))]) for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f'{c:5.2f} {NU:4.0f} {L:8.3f} | {R.mean():8.3f} {O.mean():7.3f} | {100*d.mean():8.1f}% [{100*lo:6.1f},{100*hi:6.1f}]')
