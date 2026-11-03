"""Budget table with intervals that allow for overlapping splits (Nadeau and Bengio, 2003).

budget_sweep.py reports a percentile interval over 16 random splits of the same 119 cells.  The splits share cells, so
the spread of the 16 paired gains understates the sampling uncertainty of the fleet.  The corrected resampled
t-interval inflates the variance of the mean gain from s^2/J to s^2 (1/J + n_test/n_train), with J splits.
Same protocol and seeds as budget_sweep.py.  Usage: CH=dq WIN=100 python budget_nb.py
"""
import os
import sys
import warnings
import numpy as np
from scipy import stats
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost_real as cr

warnings.filterwarnings('ignore')
cr.DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'severson', 'dq_traj.npz')
QD, DQ, life = cr.load()
n_all = len(life)
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
NSPLIT = int(os.environ.get('NSPLIT', '16'))
NDRAW = 12
GRID = (0.50, 0.65, 1.00)
N1S = [int(x) for x in os.environ.get('N1S', '6,8,10,12,16,20').split(',')]


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


idx_ok = np.where(np.array([life[i] >= M and cyc[i][-1] >= M for i in range(n_all)]))[0]
print(f'channel {CH} window m={M}: {len(idx_ok)} cells, {NSPLIT} splits x {NDRAW} draws; n_test={cr.NTEST}')
print('gain (%) over the corner: mean, percentile interval over splits (as in the paper), Nadeau-Bengio 95% interval')
for N1 in N1S:
    C = N1 * (1 + cr.KAP0)
    res = {}
    for c in GRID:
        R = []
        for seed in range(NSPLIT):
            rng = np.random.default_rng(seed)
            test = rng.choice(idx_ok, cr.NTEST, replace=False)
            pool = np.array([i for i in idx_ok if i not in set(test)])
            Xte = np.array([cr.feats(series[i], cyc[i], M) for i in test])
            yte = life[test]
            base = np.sqrt(np.mean((yte - life[pool].mean()) ** 2))
            B0 = pool_beta(pool)
            elig = np.array([i for i in pool if c * life[i] >= M])
            n = min(max(3, int(round(C / (cr.KAP0 + c)))), len(elig))
            r = []
            for _ in range(NDRAW):
                take = rng.choice(elig, n, replace=False)
                X = np.array([cr.feats(series[i], cyc[i], M) for i in take])
                yl = life[take] if c >= 1 else cr.impute([(dc[i][dc[i] <= c * life[i]], dmg[i][dc[i] <= c * life[i]]) for i in take], beta0=B0)
                r.append(np.sqrt(np.mean((Xte @ cr.huber(X, yl) - yte) ** 2)) / base)
            R.append(np.mean(r))
        res[c] = np.array(R)
    Rc = res[1.0]
    ntr = len(idx_ok) - cr.NTEST
    fac = 1.0 / NSPLIT + cr.NTEST / ntr
    out = []
    for c in GRID[:-1]:
        d = (Rc - res[c]) / Rc
        m = d.mean()
        se_nb = np.sqrt(fac * d.var(ddof=1))
        tcrit = stats.t.ppf(0.975, NSPLIT - 1)
        rng0 = np.random.default_rng(0)
        bs = [np.mean(d[rng0.integers(0, len(d), len(d))]) for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out.append(f'c={c:.2f}: {100 * m:6.1f} [{100 * lo:6.1f},{100 * hi:6.1f}] NB [{100 * (m - tcrit * se_nb):6.1f},{100 * (m + tcrit * se_nb):6.1f}]')
    print(f'N1={N1:2d}  ' + '  |  '.join(out), flush=True)
