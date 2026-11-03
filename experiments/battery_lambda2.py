"""Measure Lambda in place on the battery fleet, by ablating the covariates.

The fixed-fraction route does not work here. Lambda would come from regressing the 1/n
coefficient b(c) on 1/etaT(c), but etaT(c) = c^9.97 is so steep that the two regimes needed
never overlap: over c in [0.7,1] the measured b is flat to within noise, because imputation at
those truncations is already accurate, while below c=0.7 the imputation error turns into a
bias that does not shrink with n and the a+b/n form the identification rests on fails outright.

The ablation route avoids the sweep. Both S and K sit in the same 1/n coefficient, so removing
the covariates removes K exactly, leaving a predictor that is the training-fleet mean and a
learning curve whose slope is S alone. Restoring them adds K. The control is not optional and
gates the whole measurement: with no covariate the slope must return the fleet variance of
log life, and any setting where it does not is a setting where the learning curve is not
measuring what the identification assumes.
"""
import numpy as np
from battery_lambda import load, feats

M = 100
NS = np.arange(6, 61, 6)
REPS = 900
TOL = 0.12


def learning_slope(Z, y, lam, ntest, seed):
    """Slope of held-out error against 1/n; the intercept column is never penalised."""
    rng = np.random.default_rng(seed)
    pen = lam*np.eye(Z.shape[1])
    pen[0, 0] = 0.0
    out = []
    for n in NS:
        acc = []
        for _ in range(REPS):
            p = rng.permutation(len(y))
            tr, te = p[:n], p[n:n + ntest]
            w = np.linalg.solve(Z[tr].T @ Z[tr] + pen, Z[tr].T @ y[tr])
            acc.append(np.mean((Z[te] @ w - y[te])**2))
        out.append(np.mean(acc))
    e = np.array(out)
    A = np.column_stack([np.ones(len(NS)), 1.0/NS])
    (a, b), *_ = np.linalg.lstsq(A, e, rcond=None)
    r2 = 1 - np.sum((e - A @ np.array([a, b]))**2)/np.sum((e - e.mean())**2)
    return float(a), float(b), float(r2)


if __name__ == '__main__':
    QD, DQ, life = load()
    y_all = np.log(life)
    ser = [DQ[i][1].astype(float) for i in range(len(life))]
    cyc = [DQ[i][0].astype(float) for i in range(len(life))]
    ok = np.array([life[i] >= M and cyc[i][-1] >= M for i in range(len(life))])
    idx = np.where(ok)[0]
    y = y_all[idx]
    X = np.array([feats(ser[i], cyc[i], M) for i in idx])
    Z = np.column_stack([np.ones(len(idx)),
                         (X[:, 1:] - X[:, 1:].mean(0))/(X[:, 1:].std(0) + 1e-12)])
    one = np.ones((len(idx), 1))

    print(f'{len(idx)} cells; Var(log life) = {y.var():.4f}, which the covariate-free slope '
          f'must return')
    print(f'{"ridge":>7s} {"ntest":>6s} {"S":>8s} {"control":>9s} {"S+K":>8s} {"R2":>6s} '
          f'{"K":>8s} {"Lambda":>8s} {"threshold":>10s}')
    keep = []
    for lam in (0.003, 0.01, 0.03, 0.1, 0.3):
        for ntest in (25, 35):
            S = learning_slope(one, y, lam, ntest, 11)[1]
            a, tot, r2 = learning_slope(Z, y, lam, ntest, 11)
            rel = abs(S - y.var())/y.var()
            K = tot - S
            good = rel <= TOL and K > 0
            L = S/K if K > 0 else np.nan
            print(f'{lam:7.3f} {ntest:6d} {S:8.4f} {("pass" if rel <= TOL else f"{rel*100:.0f}% off"):>9s} '
                  f'{tot:8.4f} {r2:6.3f} {K:8.4f} {L:8.3f} {(1+L)/1.05:10.2f}'
                  + ('' if good else '   rejected'))
            if good:
                keep.append(L)
    k = np.array(keep)
    print()
    print(f'{len(k)} settings pass the control: Lambda from {k.min():.3f} to {k.max():.3f}, '
          f'threshold {(1+k.min())/1.05:.2f} to {(1+k.max())/1.05:.2f}')
    print(f'the inherited value is Lambda = 0.241, threshold 1.18')
