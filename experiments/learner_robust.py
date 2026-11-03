"""Is Table 3 an artefact of the linear predictor?

cost_real.py trains one learner: Huber-weighted least squares on the four-feature map. The
table's whole claim is an ORDERING -- the DeltaQ(V) channel is usable at every campaign
length, bulk capacity only when run to failure -- and a referee will ask whether that
ordering is a property of the campaign or of the regression. So the same campaign is rerun
with learners that share nothing but the feature map: a plain ridge (different loss), a
random forest and gradient boosting (nonlinear, no additivity), and k-nearest neighbours
(nonparametric, no fitted coefficients at all).

Everything else is held fixed: same splits, same seeds, same imputed labels, same held-out
cells, same base of predicting the training-pool mean. Only the map from (X, y) to a
prediction changes.
"""
import contextlib
import io as _io
import warnings

import numpy as np

import cost_real as CR

warnings.filterwarnings('ignore')

GRID = (0.35, 0.50, 0.65, 0.80, 1.00)
SEEDS = (0, 1, 2)
C_BUDGET = 21.0
REPS = 12                      # imputation dominates; five learners share each draw
CH = (('dq', 100, 'DeltaQ(V)'), ('capacity', 400, 'bulk capacity'))


# --------------------------------------------------------------- the learners
def fit_huber(X, y, Xte):
    return Xte @ CR.huber(X, y)


def fit_ridge(X, y, Xte):
    # features are standardized on the training campaign, so the penalty acts on comparable scales
    # (the log-variance feature is about 10x larger than the others)
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[:, 1:], y)
    return m.predict(Xte[:, 1:])


def fit_forest(X, y, Xte):
    from sklearn.ensemble import RandomForestRegressor
    m = RandomForestRegressor(n_estimators=150, min_samples_leaf=2,
                              random_state=0).fit(X[:, 1:], y)
    return m.predict(Xte[:, 1:])


def fit_boost(X, y, Xte):
    from sklearn.ensemble import GradientBoostingRegressor
    m = GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.08,
                                  random_state=0).fit(X[:, 1:], y)
    return m.predict(Xte[:, 1:])


def fit_knn(X, y, Xte):
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    k = int(min(5, max(1, len(y) - 1)))
    m = make_pipeline(StandardScaler(),
                      KNeighborsRegressor(n_neighbors=k)).fit(X[:, 1:], y)
    return m.predict(Xte[:, 1:])


LEARNERS = (('Huber (paper)', fit_huber), ('ridge', fit_ridge), ('random forest', fit_forest),
            ('grad. boosting', fit_boost), ('5-NN', fit_knn))


# --------------------------------------------------------------- the campaign
def campaign(channel, m, seed):
    """cost_real.run's setup, verbatim, returned instead of printed."""
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

    out = {name: [] for name, _ in LEARNERS}
    for c in GRID:
        elig = np.array([i for i in pool if c*life[i] >= m])
        n = min(max(4, int(round(C_BUDGET/(CR.KAP0 + c)))), len(elig))
        if n < 4:
            for name, _ in LEARNERS:
                out[name].append(np.nan)
            continue
        acc = {name: [] for name, _ in LEARNERS}
        for _ in range(REPS):
            take = rng.choice(elig, n, replace=False)
            trajs = []
            for i in take:
                sel = dmg_cyc[i] <= c*life[i]
                trajs.append((dmg_cyc[i][sel], dmg[i][sel]))
            yl = life[take] if c >= 1.0 else CR.impute(trajs, CR.pool_beta(pool, dmg, dmg_cyc, life))
            X = np.array([CR.feats(series[i], cycles[i], m) for i in take])
            for name, fn in LEARNERS:
                try:
                    p = fn(X, yl, Xte)
                    acc[name].append(np.sqrt(np.mean((p - yte)**2)))
                except Exception:
                    acc[name].append(np.nan)
        for name, _ in LEARNERS:
            out[name].append(float(np.nanmean(acc[name]))/base)
    return out


if __name__ == '__main__':
    print('Prediction error as a multiple of predicting the fleet mean; below 1 is useful.')
    print('Mean of three held-out splits, same seeds and splits for every learner.\n')
    table = {}
    for channel, m, label in CH:
        print(f'--- {label} (feature window {m} cycles) ---')
        print(f'{"learner":16s} ' + ' '.join(f'{c:>7.2f}' for c in GRID))
        runs = []
        for sd in SEEDS:
            buf = _io.StringIO()
            with contextlib.redirect_stdout(buf):
                runs.append(campaign(channel, m, sd))
        for name, _ in LEARNERS:
            a = np.array([r[name] for r in runs], float)
            v = np.nanmean(a, 0)
            table[(label, name)] = v
            print(f'{name:16s} ' + ' '.join(f'{x:7.2f}' for x in v))
        print()

    print('--- the ordering the table claims, learner by learner ---')
    print(f'{"learner":16s} {"dQ usable at every c?":24s} {"capacity usable only at c=1?":30s}')
    for name, _ in LEARNERS:
        d = table[('DeltaQ(V)', name)]
        k = table[('bulk capacity', name)]
        a = bool(np.all(d[np.isfinite(d)] < 1.0))
        b = bool(np.isfinite(k[-1]) and k[-1] < 1.0 and np.all(k[:-2][np.isfinite(k[:-2])] > 1.0))
        print(f'{name:16s} {("yes" if a else "NO"):24s} {("yes" if b else "NO"):30s}')
