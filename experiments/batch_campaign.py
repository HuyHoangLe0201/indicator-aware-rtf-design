"""Leave-one-batch-out campaigns on the Delta-Q(V) channel (and bulk capacity).

The test cells are all eligible cells of one Severson batch; the campaign draws its units from the other two batches
only, so the predictor is scored on cells manufactured and cycled in a different batch.  Same protocol as
budget_sweep.py (imputed labels from a shape exponent fitted on the pool, huber regression on the four-feature map,
RMSE as a multiple of predicting the pool mean); the paired gain over the corner is taken draw by draw.
Usage: CH=dq WIN=100 python batch_campaign.py     (CH=capacity WIN=400 for bulk capacity)
"""
import os
import sys
import warnings
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost_real as cr

warnings.filterwarnings('ignore')
cr.DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'severson', 'dq_traj.npz')
QD, DQ, life = cr.load()
n_all = len(life)
names = np.load(cr.DATA, allow_pickle=True)['name']
batch = np.array([str(s)[:2] for s in names])
dmg = [cr.damage(QD[i]) for i in range(n_all)]
dc = [np.arange(1, len(d) + 1, dtype=float) for d in dmg]
CH = os.environ.get('CH', 'dq')
if CH == 'capacity':
    cyc = [d.copy() for d in dc]
    series = [np.asarray(d, float) for d in dmg]
else:
    cyc = [DQ[i][0].astype(float) for i in range(n_all)]
    series = [DQ[i][1].astype(float) for i in range(n_all)]
M = int(os.environ.get('WIN', '100'))
NDRAW = int(os.environ.get('NDRAW', '60'))
GRID = (0.50, 0.65, 0.80, 1.00)
N1S = [int(x) for x in os.environ.get('N1S', '6,8,10,20').split(',')]


def pool_beta(pool):
    bs = []
    for i in pool:
        t = dc[i] / life[i]
        D = dmg[i]
        sel = t <= 1.0
        f = lambda p: np.clip(t[sel], 1e-9, 3.) ** p[0] - D[sel]
        try:
            bs.append(float(least_squares(f, [4.], bounds=([0.4], [15.])).x[0]))
        except Exception:
            pass
    return float(np.median(bs))


idx_ok = np.array([i for i in range(n_all) if life[i] >= M and cyc[i][-1] >= M])
print(f'channel {CH}, window {M}: {len(idx_ok)} cells; {NDRAW} draws per cell of the table; entries: paired gain over the corner, %')
for hold in ('b1', 'b2', 'b3'):
    test = np.array([i for i in idx_ok if batch[i] == hold])
    pool = np.array([i for i in idx_ok if batch[i] != hold])
    Xte = np.array([cr.feats(series[i], cyc[i], M) for i in test])
    yte = life[test]
    base = np.sqrt(np.mean((yte - life[pool].mean()) ** 2))
    B0 = pool_beta(pool)
    print(f'\nheld-out batch {hold}: {len(test)} test cells (median life {np.median(yte):.0f}), pool {len(pool)} cells; '
          f'pool-mean RMSE {base:.0f}')
    for N1 in N1S:
        C = N1 * (1 + cr.KAP0)
        rng = np.random.default_rng(1000 + N1)
        res = {}
        for c in GRID:
            elig = np.array([i for i in pool if c * life[i] >= M])
            n = min(max(3, int(round(C / (cr.KAP0 + c)))), len(elig))
            r = []
            for k in range(NDRAW):
                rr = np.random.default_rng(10 * k + N1)
                take = rr.choice(elig, n, replace=False)
                X = np.array([cr.feats(series[i], cyc[i], M) for i in take])
                yl = life[take] if c >= 1 else cr.impute([(dc[i][dc[i] <= c * life[i]], dmg[i][dc[i] <= c * life[i]]) for i in take], beta0=B0)
                r.append(np.sqrt(np.mean((Xte @ cr.huber(X, yl) - yte) ** 2)) / base)
            res[c] = (np.array(r), n)
        Rc = res[1.0][0]
        out = [f'N1={N1:2d} corner {Rc.mean():.2f}']
        for c in GRID[:-1]:
            R, n = res[c]
            d = (Rc - R) / Rc
            se = d.std(ddof=1) / np.sqrt(len(d))
            out.append(f'c={c:.2f} (n={n}): {100 * d.mean():6.1f} +-{196 * se:4.1f}')
        print('  ' + ' | '.join(out))
