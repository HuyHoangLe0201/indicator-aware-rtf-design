"""Does Table 3's ordering survive a richer feature map?

Section 9 concedes that both battery channels are reduced to a single feature and conjectures
that richer sets "will raise both curves, plausibly without changing their shapes". That is a
conjecture in a limitations section, which is the weakest place to leave a testable claim.
This script tests it: the same campaign, the same splits, the same imputed labels, but the
four-feature map replaced by a twelve-dimensional one that reads the whole observed window
(level, increments, curvature, roughness) instead of two points and a variance.

Ridge is used for the rich map because at c=1 the campaign buys only twenty units, so an
unregularised fit of twelve coefficients is not identified; the paper's Huber fit is kept on
the paper's map as the reference line.
"""
import warnings

import numpy as np

import cost_real as CR

warnings.filterwarnings('ignore')

GRID = (0.35, 0.50, 0.65, 0.80, 1.00)
SEEDS = (0, 1, 2)
C_BUDGET = 21.0
REPS = 12
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


def ridge(X, y, lam=1.0):
    """Standardised ridge; the constant column is left unpenalised."""
    Z = X[:, 1:]
    mu, sd = Z.mean(0), Z.std(0) + 1e-12
    Zs = (Z - mu)/sd
    A = Zs.T @ Zs + lam*np.eye(Zs.shape[1])
    b = np.linalg.solve(A, Zs.T @ (y - y.mean()))
    return mu, sd, b, float(y.mean())


def ridge_pred(fit, Xte):
    mu, sd, b, y0 = fit
    return ((Xte[:, 1:] - mu)/sd) @ b + y0


ARMS = ('paper map, Huber', 'paper map, ridge', 'rich map, ridge')


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

    Xte4 = np.array([CR.feats(series[i], cycles[i], m) for i in test])
    XteR = np.array([feats_rich(series[i], cycles[i], m) for i in test])
    yte = life[test]
    base = float(np.sqrt(np.mean((yte - life[pool].mean())**2)))

    out = {a: [] for a in ARMS}
    for c in GRID:
        elig = np.array([i for i in pool if c*life[i] >= m])
        n = min(max(4, int(round(C_BUDGET/(CR.KAP0 + c)))), len(elig))
        if n < 4:
            for a in ARMS:
                out[a].append(np.nan)
            continue
        acc = {a: [] for a in ARMS}
        for _ in range(REPS):
            take = rng.choice(elig, n, replace=False)
            trajs = []
            for i in take:
                sel = dmg_cyc[i] <= c*life[i]
                trajs.append((dmg_cyc[i][sel], dmg[i][sel]))
            yl = life[take] if c >= 1.0 else CR.impute(trajs, CR.pool_beta(pool, dmg, dmg_cyc, life))
            X4 = np.array([CR.feats(series[i], cycles[i], m) for i in take])
            XR = np.array([feats_rich(series[i], cycles[i], m) for i in take])
            rms = lambda p: float(np.sqrt(np.mean((p - yte)**2)))
            acc[ARMS[0]].append(rms(Xte4 @ CR.huber(X4, yl)))
            acc[ARMS[1]].append(rms(ridge_pred(ridge(X4, yl), Xte4)))
            acc[ARMS[2]].append(rms(ridge_pred(ridge(XR, yl), XteR)))
        for a in ARMS:
            out[a].append(float(np.nanmean(acc[a]))/base)
    return out


if __name__ == '__main__':
    print('Prediction error as a multiple of predicting the fleet mean; below 1 is useful.')
    print('Mean of three held-out splits; identical splits, draws and labels across arms.\n')
    tab = {}
    for channel, m, label in CH:
        print(f'--- {label} (feature window {m} cycles) ---')
        print(f'{"arm":20s} ' + ' '.join(f'{c:>7.2f}' for c in GRID))
        runs = [campaign(channel, m, sd) for sd in SEEDS]
        for a in ARMS:
            v = np.nanmean(np.array([r[a] for r in runs], float), 0)
            tab[(label, a)] = v
            print(f'{a:20s} ' + ' '.join(f'{x:7.2f}' for x in v))
        print()
    print('--- the claim of Section 9: richer features raise both curves, shapes unchanged ---')
    for a in ARMS:
        d, k = tab[('DeltaQ(V)', a)], tab[('bulk capacity', a)]
        ok_d = bool(np.all(d[np.isfinite(d)] < 1.0))
        ok_k = bool(k[-1] < 1.0 and np.all(k[:-2][np.isfinite(k[:-2])] > 1.0))
        print(f'{a:20s} dQ usable at every c: {"yes" if ok_d else "NO":4s}'
              f'   capacity only at c=1: {"yes" if ok_k else "NO"}')
