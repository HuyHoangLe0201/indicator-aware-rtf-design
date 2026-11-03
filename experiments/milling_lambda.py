"""Measure this domain's own threshold instead of inheriting one.

Lambda is S/K, the ratio of the fleet floor to the estimation term, and the paper takes its
three values from the learning curves of a companion study.  On the milling set both halves
can be measured on the same fleet the indicators come from: fit the error of a life prediction
against the number of training cutters, err(n) = S + K/n, and read S and K off the fit.

The prediction is the one a design would actually buy: from what the first cuts of a fresh
cutter look like, and from the cutting conditions the shop already knows, how long will it
last.  Errors are squared errors in log life, averaged over held-out cutters and over random
training subsets, so the curve is an average learning curve and not one particular split.
"""
import numpy as np
from milling_test import campaign
from milling_load import CH, load

KAPPA0 = 0.05
NREP = 400
RNG = np.random.default_rng(0)


def design(limit=0.45, keep=0.5, nfirst=2):
    """Feature matrix from the opening cuts plus the known cutting conditions, and log life."""
    raw = load(keep)
    units = campaign(limit, keep)
    X, y = [], []
    for c, k, _, tr in units:
        f = [np.mean(tr[ch][:nfirst]) for ch in CH]
        r = raw[c]
        X.append(f)
        y.append(np.log(k))
    return np.array(X, float), np.array(y, float)


def add_conditions(X, limit=0.45, keep=0.5):
    """Depth of cut, feed and material are design variables the shop sets, not observations."""
    import scipy.io as sio
    d = sio.loadmat(r'D:\claudecode\generationbound\data\milling\3. Milling\mill.mat',
                    struct_as_record=False, squeeze_me=True)['mill']
    cond = {}
    for r in d:
        cond.setdefault(int(r.case), (float(r.DOC), float(r.feed), float(r.material)))
    cases = [c for c, _, _, _ in campaign(limit, keep)]
    return np.hstack([X, np.array([cond[c] for c in cases], float)])


def ridge_fit(Ztr, ytr, lam):
    """Ridge on already-scaled features; the scaling is fleet-level and carries no labels."""
    A = Ztr.T @ Ztr + lam*len(ytr)*np.eye(Ztr.shape[1])
    w = np.linalg.solve(A, Ztr.T @ (ytr - ytr.mean()))
    return w, ytr.mean()


def learning_curve(X, y, ns, lam):
    """Mean squared held-out error as a function of the number of training units."""
    Z = (X - X.mean(0))/(X.std(0) + 1e-12)
    m = len(y)
    out = []
    for n in ns:
        errs = []
        for _ in range(NREP):
            i = RNG.integers(m)
            pool = [j for j in range(m) if j != i]
            tr = RNG.choice(pool, size=n, replace=False)
            w, b = ridge_fit(Z[tr], y[tr], lam)
            errs.append((Z[i] @ w + b - y[i])**2)
        out.append(float(np.mean(errs)))
    return np.array(out)


def fit_SK(ns, err):
    """Least squares fit of err(n) = S + K/n."""
    A = np.column_stack([np.ones(len(ns)), 1.0/np.asarray(ns, float)])
    (S, K), *_ = np.linalg.lstsq(A, err, rcond=None)
    pred = A @ np.array([S, K])
    ss = 1 - np.sum((err - pred)**2)/np.sum((err - err.mean())**2)
    return float(S), float(K), float(ss)


if __name__ == '__main__':
    ns = np.arange(3, 13)
    for lab, withcond in (('sensors only', False), ('sensors and cutting conditions', True)):
        X, y = design()
        if withcond:
            X = add_conditions(X)
        print(f'{lab}: {len(y)} cutters, {X.shape[1]} features, '
              f'variance of log life {y.var():.3f} (the error of predicting the fleet mean)')
        for lam in (0.03, 0.1, 0.3, 1.0):
            err = learning_curve(X, y, ns, lam)
            S, K, r2 = fit_SK(ns, err)
            tail = (f'Lambda={S/K:.3f}  threshold={(1 + S/K)/(1 + KAPPA0):.2f}'
                    if S > 0 and K > 0 else 'no split into floor and estimation term')
            print(f'   ridge {lam:4.2f}  err ' + ' '.join(f'{e:5.3f}' for e in err)
                  + f'   S={S:6.3f} K={K:6.3f} R2={r2:5.2f}  {tail}')
        print()
