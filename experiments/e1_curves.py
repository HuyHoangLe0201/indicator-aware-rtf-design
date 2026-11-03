"""
E1: measured learning curves on the real Severson cells.

Paper 1 calibrates its entire validation on a single constant alpha_0 = 0.126, fitted
from ONE published data point.  Paper 1 section 5 also states that its authors "reran
their elastic net model on random subsamples", obtaining ~19% (n=20), 12% (n=60), 9.8%
(n=100) test error and a fitted convergence exponent gamma-hat = 0.41 (CI [0.28,0.54]).
Both are checked here directly.

Design: fixed held-out test set of 40 cells; training pool of 79; subsample n from the
pool; 30 seeds.  Target log(cycle life); errors reported as MAPE on cycle life so they
are comparable with the published figures.
"""
import numpy as np, os
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = os.path.join(os.path.dirname(__file__), '..', 'data', 'severson')
d = np.load(os.path.join(HERE, 'e1_features.npz'))
X, y = d['X'], d['y']
KEEP = X.std(0) > 1e-6
X = X[:, KEEP]
print(f'cells {len(y)}, usable features {X.shape[1]}')

NS = [10, 15, 20, 30, 40, 60, 79]
SEEDS = 30
N_TEST = 40

def models():
    return {
        'elastic net (p=6)':
            make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[.1, .5, .9, 1.],
                                                         cv=3, max_iter=20000)),
        'ridge (p=6)':
            make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25))),
        'MLP 8 (p_eff~64)':
            make_pipeline(StandardScaler(), MLPRegressor((8,), max_iter=4000,
                                                         alpha=1e-2, random_state=0)),
        'MLP 32x16 (p_eff~700)':
            make_pipeline(StandardScaler(), MLPRegressor((32, 16), max_iter=4000,
                                                         alpha=1e-2, random_state=0)),
    }

def mape(logpred, logtrue):
    return float(np.mean(np.abs(np.exp(logpred)-np.exp(logtrue))/np.exp(logtrue)))

if __name__ == '__main__':
    res = {k: {n: [] for n in NS} for k in models()}
    for s in range(SEEDS):
        rng = np.random.default_rng(1000+s)
        perm = rng.permutation(len(y))
        te, pool = perm[:N_TEST], perm[N_TEST:]
        for n in NS:
            if n > len(pool): continue
            tr = pool[:n]
            for nm, mk in models().items():
                try:
                    m = mk.fit(X[tr], y[tr])
                    res[nm][n].append(mape(m.predict(X[te]), y[te]))
                except Exception:
                    pass
    print(f'\n{"model":24s} ' + " ".join(f"{('n='+str(n)):>8s}" for n in NS)
          + f' {"gamma":>7s} {"CI":>14s}')
    print('-'*95)
    for nm in res:
        mu = np.array([np.mean(res[nm][n]) if res[nm][n] else np.nan for n in NS])
        ok = np.isfinite(mu)
        ln, lm = np.log(np.array(NS, float)[ok]), np.log(mu[ok])
        A = np.vstack([ln, np.ones(ok.sum())]).T
        cf, *_ = np.linalg.lstsq(A, lm, rcond=None)
        resid = lm - A@cf
        se = np.sqrt(np.sum(resid**2)/(ok.sum()-2)/np.sum((ln-ln.mean())**2))
        print(f'{nm:24s} ' + " ".join(f"{100*v:7.1f}%" for v in mu)
              + f' {-cf[0]:7.3f} [{-cf[0]-1.96*se:5.2f},{-cf[0]+1.96*se:5.2f}]')
    print('\nPaper 1 section 5 reports for this experiment:')
    print('   ~19% (n=20), 12% (n=60), 9.8% (n=100),  gamma-hat = 0.41  CI [0.28, 0.54]')
    print('Theory: gamma = 1/(2-kappa); corrected battery class beta=4.48 -> kappa=1 -> gamma=1')
