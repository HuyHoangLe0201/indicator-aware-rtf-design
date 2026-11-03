"""Confirmation run on the grid the paper actually tabulates, with enough draws to trust.

Job A used twelve draws and five learners; this uses forty draws, three seeds and the three
learners that bracket the question: the paper's Huber fit, a standardized ridge (linear, but not
the paper's loss), and a random forest (bounded predictions, cannot extrapolate). What is
reported is not only the level but the argmin over c, because the argmin is what the design
criterion predicts and the level is not.
"""
import warnings

import numpy as np

import cost_real as CR
from learner_robust import fit_boost, fit_forest, fit_huber, fit_knn, fit_ridge

warnings.filterwarnings('ignore')
GRID = (0.50, 0.65, 0.80, 1.00)
SEEDS = (0, 1, 2)
REPS, C_BUDGET = 120, 21.0
ARMS = (('Huber (paper)', fit_huber), ('ridge', fit_ridge), ('random forest', fit_forest),
        ('grad. boosting', fit_boost), ('5-NN', fit_knn))


def campaign(channel, m, seed):
    QD, DQ, life = CR.load()
    rng = np.random.default_rng(seed)
    n_all = len(life)
    series, cycles, dmg, dmg_cyc = [], [], [], []
    for i in range(n_all):
        d = CR.damage(QD[i])
        dc = np.arange(1, len(d) + 1, dtype=float)
        dmg.append(d)
        dmg_cyc.append(dc)
        if channel == 'capacity':
            s, cyc = d, dc
        else:
            cyc, s = DQ[i][0].astype(float), DQ[i][1].astype(float)
        series.append(np.asarray(s, float))
        cycles.append(cyc)
    ok = np.array([life[i] >= m and cycles[i][-1] >= m and
                   np.all(np.isfinite(series[i][np.isfinite(series[i])]))
                   for i in range(n_all)])
    idx = np.where(ok)[0]
    test = rng.choice(idx, CR.NTEST, replace=False)
    pool = np.array([i for i in idx if i not in set(test)])
    Xte = np.array([CR.feats(series[i], cycles[i], m) for i in test])
    yte = life[test]
    base = float(np.sqrt(np.mean((yte - life[pool].mean())**2)))
    out = {n: [] for n, _ in ARMS}
    for c in GRID:
        elig = np.array([i for i in pool if c*life[i] >= m])
        n = min(max(4, int(round(C_BUDGET/(CR.KAP0 + c)))), len(elig))
        acc = {nm: [] for nm, _ in ARMS}
        for _ in range(REPS):
            take = rng.choice(elig, n, replace=False)
            trajs = [(dmg_cyc[i][dmg_cyc[i] <= c*life[i]],
                      dmg[i][dmg_cyc[i] <= c*life[i]]) for i in take]
            yl = life[take] if c >= 1.0 else CR.impute(trajs, CR.pool_beta(pool, dmg, dmg_cyc, life))
            X = np.array([CR.feats(series[i], cycles[i], m) for i in take])
            for nm, fn in ARMS:
                acc[nm].append(float(np.sqrt(np.mean((fn(X, yl, Xte) - yte)**2)))/base)
        for nm, _ in ARMS:
            out[nm].append(float(np.mean(acc[nm])))
    return out


if __name__ == '__main__':
    print(f'{REPS} draws, {len(SEEDS)} splits. Error as a multiple of predicting the '
          f'fleet mean.\n')
    for channel, m, label, ups in (('dq', 100, 'DeltaQ(V)', '~0'),
                                   ('capacity', 400, 'bulk capacity', '~10')):
        print(f'--- {label}, Upsilon {ups} ---')
        print(f'{"learner":16s} ' + ' '.join(f'{c:>7.2f}' for c in GRID) + '   argmin')
        runs = [campaign(channel, m, s) for s in SEEDS]
        for nm, _ in ARMS:
            v = np.mean(np.array([r[nm] for r in runs], float), 0)
            print(f'{nm:16s} ' + ' '.join(f'{x:7.2f}' for x in v)
                  + f'   c = {GRID[int(np.argmin(v))]:.2f}')
        print()
    print('criterion predicts: DeltaQ(V) interior (stop early), capacity at c = 1.00.')
