"""
E1 v2: learning curves with a floor-aware fit.

Three fixes over v1.
  * drop n = 10: with six parameters it is degenerate (MAPE 470-590%).
  * scale the target for the neural models; v1's MLPs diverged to MAPE ~1e90.
  * fit  err(n) = floor + A n^{-gamma}  instead of a pure power law.  The measured
    curve flattens near 15%, so a pure power-law fit reports the distance to the floor
    rather than the intrinsic rate -- which is what makes Paper 1's gamma-hat = 0.41
    and a naive refit disagree.  Paper 1 did the same floor subtraction and obtained
    gamma-hat = 1.2 (CI [0.6, 1.9]).
"""
import numpy as np, os, warnings
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from scipy.optimize import curve_fit
warnings.filterwarnings('ignore')

HERE = os.path.join(os.path.dirname(__file__), '..', 'data', 'severson')
d = np.load(os.path.join(HERE, 'e1_features.npz'))
X, y = d['X'], d['y']
X = X[:, X.std(0) > 1e-6]
NS, SEEDS, N_TEST = [15, 20, 30, 40, 60, 79], 30, 40

def wrap(est):
    return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), est),
                                      transformer=StandardScaler())

MODELS = {
    'elastic net': lambda: wrap(ElasticNetCV(l1_ratio=[.1, .5, .9, 1.], cv=3,
                                             max_iter=20000)),
    'ridge':       lambda: wrap(RidgeCV(alphas=np.logspace(-3, 3, 25))),
    'MLP 8':       lambda: wrap(MLPRegressor((8,), max_iter=8000, alpha=1.0,
                                             learning_rate_init=3e-3, random_state=0)),
    'MLP 32x16':   lambda: wrap(MLPRegressor((32, 16), max_iter=8000, alpha=1.0,
                                             learning_rate_init=3e-3, random_state=0)),
}

def mape(lp, lt):
    return float(np.mean(np.abs(np.exp(lp)-np.exp(lt))/np.exp(lt)))

if __name__ == '__main__':
    res = {k: {n: [] for n in NS} for k in MODELS}
    for s in range(SEEDS):
        rng = np.random.default_rng(2000+s)
        perm = rng.permutation(len(y))
        te, pool = perm[:N_TEST], perm[N_TEST:]
        for n in NS:
            tr = pool[:n]
            for nm, mk in MODELS.items():
                try:
                    res[nm][n].append(mape(mk().fit(X[tr], y[tr]).predict(X[te]), y[te]))
                except Exception:
                    pass
    print(f'{"model":12s} ' + " ".join(f"{('n='+str(n)):>8s}" for n in NS)
          + f' {"floor":>7s} {"gamma":>7s} {"gamma (pure)":>13s}')
    print('-'*88)
    for nm in MODELS:
        mu = np.array([np.mean(res[nm][n]) for n in NS])
        ns = np.array(NS, float)
        f = lambda n, fl, A, g: fl + A*n**(-g)
        try:
            p, _ = curve_fit(f, ns, mu, p0=[mu[-1]*0.8, 1.0, 0.5],
                             bounds=([0, 0, 0], [mu.min(), 1e4, 5]), maxfev=40000)
            fl, gam = p[0], p[2]
        except Exception:
            fl, gam = np.nan, np.nan
        cf = np.polyfit(np.log(ns), np.log(mu), 1)
        print(f'{nm:12s} ' + " ".join(f"{100*v:7.1f}%" for v in mu)
              + f' {100*fl:6.1f}% {gam:7.2f} {-cf[0]:13.3f}')
    print('\nPaper 1 section 5: ~19% (n=20), 12% (n=60), 9.8% (n=100); gamma-hat 0.41,')
    print('   and 1.2 (CI [0.6,1.9]) after subtracting an 8% floor.')
    print('Theory with the corrected class (beta=4.48 -> kappa=1): gamma = 1.')
