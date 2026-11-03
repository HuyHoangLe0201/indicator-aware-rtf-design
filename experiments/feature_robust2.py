"""Section 9's conjecture about richer feature sets, measured instead of conjectured.

The limitations section says both battery channels are reduced to a single feature and that
"richer feature sets will raise both curves, plausibly without changing their shapes". The
first run at this (feature_robust.py) compared a hand-rolled standardised ridge against the
paper's Huber fit, which confounds the map with the learner: at these sample sizes that ridge
is close to unregularised and behaves quite differently from a centred sklearn ridge. So the
comparison is redone the only way that isolates the map -- the SAME learner on both maps,
everything else held fixed.

Two learners bracket the question: a centred ridge, which extrapolates, and a random forest,
which cannot. The paper's Huber fit on the paper's map is carried as the reference line.
"""
import warnings

import numpy as np

import cost_real as CR
from learner_robust import fit_forest, fit_huber, fit_ridge

warnings.filterwarnings('ignore')

GRID = (0.50, 0.65, 0.80, 1.00)
SEEDS = (0, 1, 2)
REPS, C_BUDGET = 40, 21.0
CH = (('dq', 100, 'DeltaQ(V)'), ('capacity', 400, 'bulk capacity'))


def feats_rich(series, cycles, m):
    """Twelve numbers off the same window: level, shape, curvature, roughness."""
    grid = np.linspace(m/12., m, 12)
    y = np.interp(grid, cycles, series)
    d = np.diff(y)
    dd = np.diff(d)
    sc = float(np.std(y) + 1e-12)
    return np.array([
        1.0,
        y[-1], y[0], y[-1] - y[0], y[-1] - y[5], y[5] - y[0],
        float(np.mean(y)), float(np.std(y)),
        float(np.mean(d)), float(np.mean(dd)),
        np.log(float(np.var(d)) + 1e-12),
        float(np.mean(np.abs(dd))/sc),
    ])


ARMS = (
    ('paper map, Huber', CR.feats, fit_huber),
    ('paper map, ridge', CR.feats, fit_ridge),
    ('rich map,  ridge', feats_rich, fit_ridge),
    ('paper map, forest', CR.feats, fit_forest),
    ('rich map,  forest', feats_rich, fit_forest),
)


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
    yte = life[test]
    base = float(np.sqrt(np.mean((yte - life[pool].mean())**2)))
    XTE = {}
    for name, fmap, _ in ARMS:
        XTE[name] = np.array([fmap(series[i], cycles[i], m) for i in test])

    out = {name: [] for name, _, _ in ARMS}
    for c in GRID:
        elig = np.array([i for i in pool if c*life[i] >= m])
        n = min(max(4, int(round(C_BUDGET/(CR.KAP0 + c)))), len(elig))
        acc = {name: [] for name, _, _ in ARMS}
        for _ in range(REPS):
            take = rng.choice(elig, n, replace=False)
            trajs = [(dmg_cyc[i][dmg_cyc[i] <= c*life[i]],
                      dmg[i][dmg_cyc[i] <= c*life[i]]) for i in take]
            yl = life[take] if c >= 1.0 else CR.impute(trajs, CR.pool_beta(pool, dmg, dmg_cyc, life))
            for name, fmap, fn in ARMS:
                X = np.array([fmap(series[i], cycles[i], m) for i in take])
                p = fn(X, yl, XTE[name])
                acc[name].append(float(np.sqrt(np.mean((p - yte)**2)))/base)
        for name, _, _ in ARMS:
            out[name].append(float(np.nanmean(acc[name])))
    return out


if __name__ == '__main__':
    print(f'{REPS} draws, {len(SEEDS)} splits. Error as a multiple of predicting the fleet '
          f'mean.\nThe map is the only thing that changes between an arm and the arm above '
          f'it with the same learner.\n')
    for channel, m, label in CH:
        print(f'--- {label} (feature window {m} cycles) ---')
        print(f'{"arm":20s} ' + ' '.join(f'{c:>7.2f}' for c in GRID) + '   argmin')
        runs = [campaign(channel, m, s) for s in SEEDS]
        for name, _, _ in ARMS:
            v = np.mean(np.array([r[name] for r in runs], float), 0)
            print(f'{name:20s} ' + ' '.join(f'{x:7.2f}' for x in v)
                  + f'   c = {GRID[int(np.argmin(v))]:.2f}')
        print()
