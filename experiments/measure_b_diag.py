"""
Does the excess risk of the fixed-fraction campaign actually follow a + b/n on these cells?

measure_b.py returned b = -1.77 at c = 0.50 on the DeltaQ(V) channel.  A negative coefficient
of 1/n is impossible -- more training units cannot raise the risk -- so either the estimate is
too noisy to use or the 1/n law does not describe this experiment.  The distinction matters:
the first is fixable with more resampling, the second means Lambda cannot be measured this way
on 119 cells and the paper's inherited values have no cheap replacement.

We print the MSE(n) sequence itself, with a bootstrap spread, and compare a mean-based against
a median-based risk so that a single wild cell can be ruled in or out as the cause.
"""
import numpy as np, os
from sklearn.linear_model import Ridge

z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'relgrid.npz'))
T, A, B, FR = z['T'], z['A'], z['B'], z['FRACS']
Y = np.log(T)
NS = [15, 20, 25, 30, 40, 50, 60, 80]
REPS = 1500
RNG = np.random.default_rng(5)


def feats(M, c):
    j = int(np.argmin(np.abs(FR - c)))
    lo = max(1, j//2)
    X = np.column_stack([M[:, j], M[:, j] - M[:, 0],
                         (M[:, j] - M[:, lo])/max(FR[j] - FR[lo], 1e-9)])
    return (X - X.mean(0))/(X.std(0) + 1e-12)


def curve(M, c):
    X = feats(M, c)
    mean_c, med_c, spread = [], [], []
    for n in NS:
        acc = []
        for _ in range(REPS):
            p = RNG.permutation(len(Y))
            tr, te = p[:n], p[n:n + 35]
            m = Ridge(alpha=1.0).fit(X[tr], Y[tr])
            acc.append(float(np.mean((Y[te] - m.predict(X[te]))**2)))
        acc = np.array(acc)
        mean_c.append(acc.mean()); med_c.append(np.median(acc))
        spread.append(acc.std()/np.sqrt(REPS))
    return np.array(mean_c), np.array(med_c), np.array(spread)


def fit_ab(mse):
    inv = 1.0/np.array(NS, float)
    D = np.column_stack([np.ones(len(inv)), inv])
    coef, *_ = np.linalg.lstsq(D, mse, rcond=None)
    resid = mse - D @ coef
    r2 = 1 - np.var(resid)/np.var(mse)
    return float(coef[0]), float(coef[1]), float(r2)


print(__doc__)
print('=' * 78)
for ch, M, nm in (('B', B, 'DeltaQ(V)'), ('A', A, 'bulk capacity')):
    for c in (0.4, 0.5, 0.6, 1.0):
        mean_c, med_c, se = curve(M, c)
        a1, b1, r1 = fit_ab(mean_c)
        a2, b2, r2 = fit_ab(med_c)
        print()
        print(f'{nm}, c = {c:.2f}')
        print('  n        : ' + ' '.join(f'{n:8d}' for n in NS))
        print('  MSE mean : ' + ' '.join(f'{v:8.4f}' for v in mean_c))
        print('  +- se    : ' + ' '.join(f'{v:8.4f}' for v in se))
        print('  MSE med  : ' + ' '.join(f'{v:8.4f}' for v in med_c))
        print(f'  fit on mean  : a = {a1:+.4f}  b = {b1:+.4f}  R^2 = {r1:.3f}')
        print(f'  fit on median: a = {a2:+.4f}  b = {b2:+.4f}  R^2 = {r2:.3f}')
        mono = bool(np.all(np.diff(mean_c) <= 1e-9))
        print(f'  MSE non-increasing in n (mean): {mono}')
