"""Measure Lambda in place on the turbofan fleet, by the ablation used for milling and batteries.

Both halves of Lambda sit in the coefficient of 1/n, so removing the covariates removes K
exactly and leaves a learning curve whose slope is S alone; restoring them adds K. The control
is not optional: with no covariate the predictor is the training-fleet mean, so the slope must
return the fleet variance of log life, and any setting where it does not is a setting where the
curve is not measuring what the identification assumes.

C-MAPSS FD001 is the largest fleet in the paper at 100 units run to failure, which makes it the
best-conditioned of the three this can be done on. Features are read over a fixed early window,
as a deployed predictor would read them, and the target is log life.
"""
import os

import numpy as np

ROOT = r'D:\bách khoa Đà Nẵng\nghiencuukhoahoc\baotri\data'
WINDOW = 120
NS = np.arange(6, 61, 6)
REPS = 900
TOL = 0.12


def load(window=WINDOW):
    """One feature vector and one log life per unit, from the first `window` cycles."""
    d = np.loadtxt(os.path.join(ROOT, 'train_FD001.txt'))
    uid = d[:, 0].astype(int)
    keep = [j for j in range(2, d.shape[1]) if d[:, j].std() > 1e-8]
    X, y = [], []
    for u in np.unique(uid):
        m = uid == u
        Xu = d[m][:, keep]
        if len(Xu) < window:
            continue
        w = Xu[:window]
        # level, drift and roughness of each sensor over the window, the same three summaries
        # the battery and milling runs use
        f = np.concatenate([w[-5:].mean(0), w[-5:].mean(0) - w[:5].mean(0)])
        X.append(f)
        y.append(np.log(len(Xu)))
    return np.array(X), np.array(y)


def slope(Z, y, lam, ntest, seed):
    """Slope of held-out error against 1/n; the intercept column is never penalised."""
    rng = np.random.default_rng(seed)
    pen = lam*np.eye(Z.shape[1])
    pen[0, 0] = 0.0
    err = []
    for n in NS:
        acc = []
        for _ in range(REPS):
            p = rng.permutation(len(y))
            tr, te = p[:n], p[n:n + ntest]
            w = np.linalg.solve(Z[tr].T @ Z[tr] + pen, Z[tr].T @ y[tr])
            acc.append(np.mean((Z[te] @ w - y[te])**2))
        err.append(np.mean(acc))
    e = np.array(err)
    A = np.column_stack([np.ones(len(NS)), 1.0/NS])
    (a, b), *_ = np.linalg.lstsq(A, e, rcond=None)
    r2 = 1 - np.sum((e - A @ np.array([a, b]))**2)/np.sum((e - e.mean())**2)
    return float(a), float(b), float(r2), float(e[-1])


if __name__ == '__main__':
    X, y = load()
    # 34 raw summaries on 100 units overfit whatever the penalty, so the design is reduced to
    # its leading components, which is also how the survey fuses these sensors
    W0 = (X - X.mean(0))/(X.std(0) + 1e-12)
    Vt = np.linalg.svd(W0, full_matrices=False)[2]
    Z = np.column_stack([np.ones(len(y)), W0 @ Vt[:3].T])
    one = np.ones((len(y), 1))
    print(f'{len(y)} turbofan units, {X.shape[1]} features from the first {WINDOW} cycles')
    print(f'Var(log life) = {y.var():.4f}, which the covariate-free slope must return')
    print(f'{"ridge":>7s} {"ntest":>6s} {"S":>8s} {"control":>9s} {"S+K":>8s} {"R2":>6s} '
          f'{"K":>8s} {"Lambda":>8s} {"threshold":>10s} {"err(60)":>9s}')
    keep = []
    for lam in (0.03, 0.1, 0.3, 1.0, 3.0):
        for ntest in (25, 35):
            S = slope(one, y, lam, ntest, 5)[1]
            a, tot, r2, last = slope(Z, y, lam, ntest, 5)
            rel = abs(S - y.var())/y.var()
            K = tot - S
            good = rel <= TOL and K > 0
            L = S/K if K > 0 else np.nan
            print(f'{lam:7.2f} {ntest:6d} {S:8.4f} '
                  f'{("pass" if rel <= TOL else f"{rel*100:.0f}% off"):>9s} {tot:8.4f} '
                  f'{r2:6.3f} {K:8.4f} {L:8.3f} {(1+L)/1.05:10.2f} {last:9.4f}'
                  + ('' if good else '   rejected'))
            if good:
                keep.append((last, L))
    if keep:
        best = min(keep)[1]
        vals = [v for _, v in keep]
        print()
        print(f'{len(keep)} settings pass the control: Lambda {min(vals):.3f} to {max(vals):.3f}, '
              f'threshold {(1+min(vals))/1.05:.2f} to {(1+max(vals))/1.05:.2f}')
        print(f'at the penalty that predicts best: Lambda {best:.3f}, '
              f'threshold {(1+best)/1.05:.2f}')
        print(f'the inherited value is Lambda = 0.165, threshold 1.11')
