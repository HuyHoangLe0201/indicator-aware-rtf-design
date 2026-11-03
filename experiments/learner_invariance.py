"""Learner invariance (Proposition A.2, Appendix A; Section S3.9, Table S23).

The excess risk of a learner is taken to depend on the number of units n and the label-noise variance s2 only through
s2 / n. The script tests this for four learners on a smooth regression problem: for pairs (n, s2) with the same ratio the
excess test MSE should agree, and it should rise with the ratio. Labels are f(x) + noise with noise independent of x.
Hyper-parameters are tuned by cross-validation inside each fit (so that the risk follows the minimax form).
Usage: python learner_invariance.py
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

rng = np.random.default_rng(0)
d = 3


def f(x):
    return np.sin(2 * x[:, 0]) + x[:, 1]**2 + 0.5 * x[:, 0] * x[:, 2]


learners = {
    'kernel ridge (CV)': lambda: GridSearchCV(KernelRidge(kernel='rbf'), {'alpha': [1e-3, 1e-2, 1e-1, 1], 'gamma': [0.1, 0.3, 1]}, cv=4),
    'k-nearest (CV)': lambda: GridSearchCV(KNeighborsRegressor(), {'n_neighbors': [3, 5, 9, 15, 25, 40]}, cv=4),
    'random forest': lambda: RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=0),
    'linear in 3 features': None,
}
pairs = [(100, 0.25), (200, 0.5), (400, 1.0), (200, 0.25), (400, 0.5), (800, 1.0), (400, 0.25), (800, 0.5), (1600, 1.0)]
xt = rng.uniform(-1, 1, (3000, d))
ft = f(xt)
print('%-22s' % 'learner' + ''.join('  n=%4d s2=%.2f' % p for p in pairs))
print('%-22s' % 'ratio s2/n x 1e3' + ''.join('  %14.2f' % (1000 * p[1] / p[0]) for p in pairs))
for name, mk in learners.items():
    row = []
    for n, s2 in pairs:
        errs = []
        for rep in range(12):
            x = rng.uniform(-1, 1, (n, d))
            y = f(x) + np.sqrt(s2) * rng.standard_normal(n)
            if mk is None:
                A = np.c_[np.ones(n), x]
                b = np.linalg.lstsq(A, y, rcond=None)[0]
                pred = np.c_[np.ones(len(xt)), xt] @ b
            else:
                pred = mk().fit(x, y).predict(xt)
            errs.append(np.mean((pred - ft)**2))
        row.append(np.mean(errs))
    print('%-22s' % name + ''.join('  %14.4f' % e for e in row))
