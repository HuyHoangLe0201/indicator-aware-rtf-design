"""The campaign experiment of Section 7.9, repeated on a second fleet.

Table 3 runs both plans on the Severson cells and scores the predictor each produces. That
is the paper's only OPERATIONAL validation: everywhere else Upsilon is measured but the two
campaigns are never actually run. One fleet is thin support for the claim that the criterion
decides a campaign, so the same experiment is repeated here on C-MAPSS FD001, whose fused
PC1 channel has Upsilon = 7.54 against a threshold of 1.11. The criterion therefore makes a
falsifiable prediction on this fleet: the best campaign is the one that runs to failure.

Construction of the health index follows upsilon_survey.py exactly -- PC1 of the fourteen
informative sensors, z-scored on pooled statistics -- with one change that the design forces.
A campaign that never observes a failure cannot know what sensor level constitutes one, so
the loadings, the z-scoring, the early baseline and the failure level all come from a small
stage-one set of units run to failure, disjoint from both the campaign draws and the test
set. That is not a convenience: Theorem 5 says stage one must run to failure whatever the
criterion says, so the calibration is part of the plan rather than an exemption from it. Its
cost is the same at every c and so cannot bias the comparison across c.

Two feature windows are run. The long one reproduces the selection the battery capacity
channel suffers, where a campaign stopping at c contributes nothing from a unit with
c*T_i < m and so drops the short-lived units first; the short one nearly removes that
selection. If the verdict holds under both, it is not an artefact of the selection.

Three learners are carried from the start, because a single-learner result is what made the
battery table's magnitude fragile.
"""
import os
import warnings

import numpy as np
from scipy.optimize import least_squares

warnings.filterwarnings('ignore')

ROOT = r'D:\bách khoa Đà Nẵng\nghiencuukhoahoc\baotri\data'
COLS = [6, 7, 8, 11, 12, 13, 15, 16, 17, 18, 19, 21, 24, 25]   # the fourteen, as in the paper
KAP0 = 0.05
N_TEST, N_CAL = 30, 5
C_BUDGET = 21.0
GRID = (0.50, 0.65, 0.80, 1.00)


def load():
    d = np.loadtxt(os.path.join(ROOT, 'train_FD001.txt'))
    uid = d[:, 0].astype(int)
    units = [d[uid == u][:, COLS] for u in np.unique(uid)]
    life = np.array([len(u) for u in units], float)
    return units, life


def calibrate(units, cal):
    """Loadings, scaling, baseline and failure level, from run-to-failure units only."""
    X = np.vstack([units[i] for i in cal])
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = (X - mu)/sd
    w = np.linalg.svd(Z - Z.mean(0), full_matrices=False)[2][0]

    def raw(u):
        return ((u - mu)/sd) @ w

    sgn = np.sign(np.mean([np.median(raw(units[i])[-10:]) - np.median(raw(units[i])[:10])
                           for i in cal]))
    sgn = sgn if sgn != 0 else 1.0
    b0 = float(np.median([np.median(sgn*raw(units[i])[:max(5, len(units[i])//20)])
                          for i in cal]))
    th = float(np.median([np.median(sgn*raw(units[i])[-5:]) for i in cal]))
    span = th - b0
    assert span > 0, 'failure level does not exceed the baseline'

    def index(u):
        return (sgn*raw(u) - b0)/span

    return index


def feats(D, m):
    """The paper's four-feature map, read at twelve absolute cycles up to m."""
    cyc = np.arange(1, len(D) + 1, dtype=float)
    grid = np.linspace(m/12., m, 12)
    y = np.interp(grid, cyc, D)
    return np.array([1., y[-1], y[-1] - y[5], np.log(np.var(np.diff(y)) + 1e-12)])


def life_of(t, D, b):
    """Solve for T in D = (t/T)^b by least squares over the observed window."""
    good = np.isfinite(D) & (D > 1e-6)
    if good.sum() < 3:
        return float(t[-1])
    t2 = t[good]
    D2 = np.clip(D[good], 1e-6, None)
    T0 = float(np.median(t2/D2**(1.0/max(b, 0.2))))
    T0 = float(np.clip(T0, t[-1], 20*t[-1]))

    def r(p):
        return (t2/max(p[0], 1e-6))**b - D2

    try:
        return float(least_squares(r, [T0], bounds=([t[-1]], [20*t[-1]]), max_nfev=60).x[0])
    except Exception:
        return T0


def cal_beta(D, life, cal):
    """Shape exponent of D = (t/T)^b over the complete trajectories of the stage-one (calibration) units, which are run to failure.
    Median over units; held-out and pool units never enter."""
    bs = []
    for i in cal:
        t = np.arange(1, len(D[i]) + 1, dtype=float)/life[i]
        sel = (t <= 1.0) & np.isfinite(D[i])
        try:
            bs.append(float(least_squares(lambda p: np.clip(t[sel], 1e-9, 3.)**p[0] - D[i][sel], [2.], bounds=([0.4], [15.])).x[0]))
        except Exception:
            pass
    return float(np.median(bs))


def impute(trajs, beta0):
    b = beta0
    for _ in range(3):
        Ts = np.array([life_of(t, D, b) for t, D in trajs])

        def r(p):
            return np.concatenate([np.clip(t/T, 1e-9, 3.)**p[0] - D
                                   for (t, D), T in zip(trajs, Ts)])

        try:
            b = float(least_squares(r, [b], bounds=([0.4], [15.]), max_nfev=120).x[0])
        except Exception:
            break
    return np.array([life_of(t, D, b) for t, D in trajs])


def huber(X, y, iters=12):
    b = np.linalg.solve(X.T @ X + 1e-3*np.eye(X.shape[1]), X.T @ y)
    for _ in range(iters):
        r = y - X @ b
        s = 1.4826*np.median(np.abs(r - np.median(r))) + 1e-9
        u = np.abs(r)/(1.345*s)
        wt = np.where(u <= 1, 1.0, 1.0/np.maximum(u, 1e-9))
        A = (X*wt[:, None]).T @ X + 1e-3*np.eye(X.shape[1])
        b = np.linalg.solve(A, (X*wt[:, None]).T @ y)
    return b


def fit_huber(X, y, Xte):
    return Xte @ huber(X, y)


def fit_ridge(X, y, Xte):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[:, 1:], y).predict(Xte[:, 1:])


def fit_forest(X, y, Xte):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=150, min_samples_leaf=2,
                                 random_state=0).fit(X[:, 1:], y).predict(Xte[:, 1:])


ARMS = (('Huber', fit_huber), ('ridge', fit_ridge), ('random forest', fit_forest))
NAMES = [nm for nm, _ in ARMS] + ['oracle']


def campaign(m, seed, reps=40):
    units, life = load()
    rng = np.random.default_rng(seed)
    n_all = len(life)
    order = rng.permutation(n_all)
    test = order[:N_TEST]
    cal = order[N_TEST:N_TEST + N_CAL]
    pool = order[N_TEST + N_CAL:]
    index = calibrate(units, cal)
    D = [index(u) for u in units]
    b0 = cal_beta(D, life, cal)

    Xte = np.array([feats(D[i], m) for i in test])
    yte = life[test]
    base = float(np.sqrt(np.mean((yte - life[pool].mean())**2)))

    rows, sel = [], []
    for c in GRID:
        elig = np.array([i for i in pool if c*life[i] >= m])
        n = min(max(4, int(round(C_BUDGET/(KAP0 + c)))), len(elig))
        sel.append((c, len(elig), n,
                    float(np.median(life[elig])) if len(elig) else float('nan')))
        if len(elig) < 4:
            rows.append((c, {nm: float('nan') for nm, _ in list(ARMS) + [('oracle', None)]}, float('nan')))
            continue
        acc = {nm: [] for nm, _ in ARMS}
        acc['oracle'] = []
        lab = []
        for _ in range(reps):
            take = rng.choice(elig, n, replace=False)
            if c >= 1.0:
                yl = life[take]
            else:
                trajs = []
                for i in take:
                    k = int(c*life[i])
                    trajs.append((np.arange(1, k + 1, dtype=float), D[i][:k]))
                yl = impute(trajs, b0)
            lab.append(float(np.median(np.abs(yl - life[take])/life[take])))
            X = np.array([feats(D[i], m) for i in take])
            for nm, fn in ARMS:
                try:
                    p = fn(X, yl, Xte)
                    acc[nm].append(float(np.sqrt(np.mean((p - yte)**2)))/base)
                except Exception:
                    acc[nm].append(float('nan'))
            # oracle: the same truncated features, but labelled with the true lives. It
            # separates what suspension costs in labels from what it costs in features and
            # in the units it excludes, which is the control Section 7.9 runs on the cells.
            try:
                p = fit_huber(X, life[take], Xte)
                acc['oracle'].append(float(np.sqrt(np.mean((p - yte)**2)))/base)
            except Exception:
                acc['oracle'].append(float('nan'))
        got = {nm: float(np.nanmean(acc[nm])) for nm, _ in ARMS}
        got['oracle'] = float(np.nanmean(acc['oracle']))
        rows.append((c, got, float(np.mean(lab))))
    return base, rows, sel


if __name__ == '__main__':
    SEEDS = (0, 1, 2)
    for m in (60, 100):
        print(f'=== C-MAPSS FD001, PC1 of the fourteen, feature window {m} cycles ===')
        allrows, base0, sel0 = [], None, None
        for s in SEEDS:
            base, rows, sel = campaign(m, s)
            allrows.append(rows)
            if base0 is None:
                base0, sel0 = base, sel
        print(f'base RMSE (predict the pool mean) {base0:.0f} cycles; '
              f'{N_TEST} held out, {N_CAL} stage-one calibration units')
        print(f'{"c":>5s} {"n elig":>7s} {"n used":>7s} {"elig med life":>14s} '
              f'{"label err":>10s}   ' + ' '.join(f'{nm:>14s}' for nm in NAMES))
        for k, c in enumerate(GRID):
            _, ne, nu, ml = sel0[k]
            lab = np.nanmean([r[k][2] for r in allrows])
            vals = [np.nanmean([r[k][1][nm] for r in allrows]) for nm in NAMES]
            print(f'{c:5.2f} {ne:7d} {nu:7d} {ml:14.0f} {lab:10.3f}   '
                  + ' '.join(f'{v:14.2f}' for v in vals))
        print()
        for nm, _ in ARMS:
            v = np.array([np.nanmean([r[k][1][nm] for r in allrows])
                          for k in range(len(GRID))])
            print(f'   argmin, {nm:14s} c = {GRID[int(np.nanargmin(v))]:.2f}')
        print()
    print('criterion (Upsilon 7.54 vs threshold 1.11) predicts the corner: c = 1.00.')
