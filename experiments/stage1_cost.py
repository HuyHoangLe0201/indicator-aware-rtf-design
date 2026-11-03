"""Two-stage campaigns on the DeltaQ(V) cells with the cost of the stage-one archive charged to the budget.

budget_sweep.py takes the starting exponent of the imputation from the complete cells of the training pool
without charging their cost. Here a campaign of budget C = N1*(1+kappa0) pays for n1 cells run to failure
(stage one), which supply exact labels and the median shape exponent, and spends the remainder on stage-two
cells stopped at a fraction c whose lives are imputed from that exponent. Three arms share each split:

  corner      all N1 cells run to failure
  free        stage two as in budget_sweep.py: exponent from the whole training pool, no stage-one cost
  n1 = ...    stage one charged, exponent from the n1 stage-one cells only

Entries are the paired gain over the corner on the same split and draws (positive favours stopping early),
with a 95 percent bootstrap interval over splits.
"""
import sys, os, warnings
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost_real as cr
from scipy.optimize import least_squares

warnings.filterwarnings('ignore')
cr.DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'severson', 'dq_traj.npz')
QD, DQ, life = cr.load()
n_all = len(life)
dmg = [cr.damage(QD[i]) for i in range(n_all)]
dc = [np.arange(1, len(d) + 1, dtype=float) for d in dmg]
cyc = [DQ[i][0].astype(float) for i in range(n_all)]
series = [DQ[i][1].astype(float) for i in range(n_all)]
M = int(os.environ.get('WIN', '100'))
NSPLIT = int(os.environ.get('NSPLIT', '12'))
NDRAW = int(os.environ.get('NDRAW', '8'))
CS = [float(x) for x in os.environ.get('CS', '0.50,0.65,0.80').split(',')]
N1S = [int(x) for x in os.environ.get('N1S', '8,12,20').split(',')]
STAGE1 = [int(x) for x in os.environ.get('STAGE1', '3,4,6').split(',')]
K0 = cr.KAP0


def beta_of(cells):
    bs = []
    for i in cells:
        t = dc[i] / life[i]
        sel = t <= 1.0
        f = lambda p: np.clip(t[sel], 1e-9, 3.) ** p[0] - dmg[i][sel]
        try:
            bs.append(float(least_squares(f, [4.], bounds=([0.4], [15.])).x[0]))
        except Exception:
            pass
    return float(np.median(bs))


idx_ok = np.where(np.array([life[i] >= M and cyc[i][-1] >= M for i in range(n_all)]))[0]
rng0 = np.random.default_rng(0)
print(f'DeltaQ(V), window m={M}, {len(idx_ok)} cells, {NSPLIT} splits x {NDRAW} draws; entries: RMSE/base')
for N1 in N1S:
    C = N1 * (1 + K0)
    arms = {}
    for seed in range(NSPLIT):
        rng = np.random.default_rng(seed)
        test = rng.choice(idx_ok, cr.NTEST, replace=False)
        pool = np.array([i for i in idx_ok if i not in set(test)])
        Xte = np.array([cr.feats(series[i], cyc[i], M) for i in test])
        yte = life[test]
        base = np.sqrt(np.mean((yte - life[pool].mean()) ** 2))
        b_pool = beta_of(pool)
        acc = {}
        for _ in range(NDRAW):
            def rmse(take_idx, labels):
                X = np.array([cr.feats(series[i], cyc[i], M) for i in take_idx])
                return np.sqrt(np.mean((Xte @ cr.huber(X, labels) - yte) ** 2)) / base
            take = rng.choice(pool, N1, replace=False)
            acc.setdefault('corner', []).append(rmse(take, life[take]))
            for c in CS:
                elig = np.array([i for i in pool if c * life[i] >= M])
                n_free = min(int(round(C / (K0 + c))), len(elig))
                t2 = rng.choice(elig, n_free, replace=False)
                tr = [(dc[i][dc[i] <= c * life[i]], dmg[i][dc[i] <= c * life[i]]) for i in t2]
                acc.setdefault(('free', c), []).append(rmse(t2, cr.impute(tr, beta0=b_pool)))
                for n1 in STAGE1:
                    if n1 >= N1:
                        continue
                    s1 = rng.choice(pool, n1, replace=False)
                    left = C - n1 * (1 + K0)
                    rest = np.array([i for i in elig if i not in set(s1)])
                    n2 = min(int(round(left / (K0 + c))), len(rest))
                    if n2 < 1:
                        continue
                    t2 = rng.choice(rest, n2, replace=False)
                    tr = [(dc[i][dc[i] <= c * life[i]], dmg[i][dc[i] <= c * life[i]]) for i in t2]
                    lab = np.concatenate([life[s1], cr.impute(tr, beta0=beta_of(s1))])
                    acc.setdefault((n1, c), []).append(rmse(np.concatenate([s1, t2]), lab))
        for k, v in acc.items():
            arms.setdefault(k, []).append(np.mean(v))
    corner = np.array(arms['corner'])
    print(f'\nN1={N1} (budget C={C:.2f}); corner RMSE/base {corner.mean():.3f}')
    print(f'{"arm":>10s} {"c":>5s} | {"RMSE/base":>9s} | paired gain vs corner')
    for c in CS:
        for key in [('free', c)] + [(n1, c) for n1 in STAGE1 if n1 < N1]:
            if key not in arms or len(arms[key]) != NSPLIT:
                continue
            R = np.array(arms[key])
            d = (corner - R) / corner
            bs = [np.mean(d[rng0.integers(0, len(d), len(d))]) for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            name = 'free' if key[0] == 'free' else f'n1={key[0]}'
            print(f'{name:>10s} {c:5.2f} | {R.mean():9.3f} | {100 * d.mean():6.1f}% [{100 * lo:6.1f},{100 * hi:6.1f}]')
