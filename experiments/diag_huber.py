"""Why does bulk capacity look ruinous under the paper's fit and not under the others?

Three candidate explanations, and they have different consequences for the paper:

  (i)  a few exploded draws own the mean, in which case 4.14 is an artefact of averaging
       and the honest number is the median;
  (ii) the fit is ill-conditioned because the four features are never centred or scaled and
       differ by twenty orders of magnitude, in which case the effect is numerical;
  (iii) the linear model genuinely cannot survive the imputed labels while a bounded learner
       can, in which case the paper's claim is real but is a claim about linear predictors.

So: per-draw ratios rather than their mean, the conditioning of the design, and the same
Huber fit run on standardised features.
"""
import warnings

import numpy as np

import cost_real as CR

warnings.filterwarnings('ignore')
C_BUDGET, REPS, SEED, M = 21.0, 60, 0, 400
GRID = (0.50, 0.65, 0.80, 1.00)


def huber_std(X, y):
    """The paper's estimator, but centred and scaled first; predictions are back-mapped."""
    Z = X[:, 1:]
    mu, sd = Z.mean(0), Z.std(0) + 1e-12
    Xs = np.column_stack([np.ones(len(Z)), (Z - mu)/sd])
    b = CR.huber(Xs, y)
    return mu, sd, b


def main():
    QD, DQ, life = CR.load()
    rng = np.random.default_rng(SEED)
    n_all = len(life)
    dmg = [CR.damage(QD[i]) for i in range(n_all)]
    dmg_cyc = [np.arange(1, len(d) + 1, dtype=float) for d in dmg]
    series, cycles = dmg, dmg_cyc

    ok = np.array([life[i] >= M and cycles[i][-1] >= M for i in range(n_all)])
    idx = np.where(ok)[0]
    test = rng.choice(idx, CR.NTEST, replace=False)
    pool = np.array([i for i in idx if i not in set(test)])
    Xte = np.array([CR.feats(series[i], cycles[i], M) for i in test])
    yte = life[test]
    base = float(np.sqrt(np.mean((yte - life[pool].mean())**2)))
    print(f'bulk capacity, window {M} cycles, seed {SEED}, {REPS} draws per c')
    print(f'base RMSE (predict the pool mean) = {base:.0f} cycles\n')

    from sklearn.ensemble import RandomForestRegressor
    print(f'{"c":>5s} {"arm":>16s} {"mean":>7s} {"median":>7s} {"p90":>7s} {"max":>8s} '
          f'{"cond(X)":>10s} {"lab err":>8s}')
    for c in GRID:
        elig = np.array([i for i in pool if c*life[i] >= M])
        n = min(max(4, int(round(C_BUDGET/(CR.KAP0 + c)))), len(elig))
        acc = {k: [] for k in ('Huber raw', 'Huber scaled', 'forest')}
        conds, laberr = [], []
        for _ in range(REPS):
            take = rng.choice(elig, n, replace=False)
            trajs = [(dmg_cyc[i][dmg_cyc[i] <= c*life[i]],
                      dmg[i][dmg_cyc[i] <= c*life[i]]) for i in take]
            yl = life[take] if c >= 1.0 else CR.impute(trajs, CR.pool_beta(pool, dmg, dmg_cyc, life))
            X = np.array([CR.feats(series[i], cycles[i], M) for i in take])
            conds.append(float(np.linalg.cond(X.T @ X)))
            laberr.append(float(np.median(np.abs(yl - life[take])/life[take])))
            rms = lambda p: float(np.sqrt(np.mean((p - yte)**2)))/base
            acc['Huber raw'].append(rms(Xte @ CR.huber(X, yl)))
            mu, sd, b = huber_std(X, yl)
            acc['Huber scaled'].append(
                rms(np.column_stack([np.ones(len(Xte)), (Xte[:, 1:] - mu)/sd]) @ b))
            f = RandomForestRegressor(n_estimators=150, min_samples_leaf=2,
                                      random_state=0).fit(X[:, 1:], yl)
            acc['forest'].append(rms(f.predict(Xte[:, 1:])))
        for k, v in acc.items():
            a = np.array(v)
            print(f'{c:5.2f} {k:>16s} {a.mean():7.2f} {np.median(a):7.2f} '
                  f'{np.percentile(a, 90):7.2f} {a.max():8.1f} '
                  f'{np.median(conds):10.1e} {np.median(laberr):8.3f}')
        print()


if __name__ == '__main__':
    main()
