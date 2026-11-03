"""
E1, final form: measured learning curves on both real domains.

Two earlier attempts were unstable at small n because the estimators selected their own
regularisation by internal cross-validation -- with n = 15 and 3 folds the alpha choice
is made on five samples and occasionally diverges (MAPE 200-600%).  Here regularisation
is FIXED once on the full pool and held constant across every n, so the curve is
deterministic in n and the small-n end is meaningful.

Complexity ladder: ridge on linear features (p = d+1) and on quadratic features
(p = (d+1)(d+2)/2), giving a clean p_eff contrast without neural nets.

Fit: err(n) = floor + A n^{-gamma}.  A pure power law measures the distance to the floor,
not the intrinsic rate -- which is what makes naive gamma estimates disagree.
"""
import numpy as np, os, warnings
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.optimize import curve_fit
warnings.filterwarnings('ignore')

ROOT = r'D:\bách khoa Đà Nẵng\nghiencuukhoahoc\baotri\data'
HERE = os.path.join(os.path.dirname(__file__), '..', 'data', 'severson')

def mk(deg, alpha):
    steps = [StandardScaler()]
    if deg > 1:
        steps += [PolynomialFeatures(deg, include_bias=False), StandardScaler()]
    return make_pipeline(*steps, Ridge(alpha=alpha))

def fit_curve(ns, mu):
    f = lambda n, fl, A, g: fl + A*n**(-g)
    try:
        p, _ = curve_fit(f, ns, mu, p0=[0.5*mu[-1], mu[0], 0.5],
                         bounds=([0, 0, 0.01], [0.999*mu.min(), 1e6, 4]), maxfev=60000)
        return p[0], p[2]
    except Exception:
        return np.nan, np.nan

# ------------------------------------------------------------------- battery
def battery():
    d = np.load(os.path.join(HERE, 'e1_features.npz'))
    X, y = d['X'], d['y']; X = X[:, X.std(0) > 1e-6]
    NS, SEEDS, NTE = [15, 20, 30, 40, 60, 79], 40, 40
    out = {}
    for lbl, deg, al in [('ridge lin  (p=8)', 1, 10.), ('ridge quad (p=36)', 2, 100.)]:
        acc = {n: [] for n in NS}
        for s in range(SEEDS):
            rng = np.random.default_rng(3000+s); pm = rng.permutation(len(y))
            te, pool = pm[:NTE], pm[NTE:]
            for n in NS:
                m = mk(deg, al).fit(X[pool[:n]], y[pool[:n]])
                p = m.predict(X[te])
                acc[n].append(np.mean(np.abs(np.exp(p)-np.exp(y[te]))/np.exp(y[te])))
        out[lbl] = (NS, np.array([np.mean(acc[n]) for n in NS]))
    return out

# ------------------------------------------------------------------- C-MAPSS
def cmapss():
    dd = np.loadtxt(os.path.join(ROOT, 'train_FD001.txt'))
    uid = dd[:, 0].astype(int)
    keep = [j for j in range(2, dd.shape[1]) if dd[:, j].std() > 1e-8]
    U = {}
    for u in np.unique(uid):
        m = uid == u; T = m.sum()
        U[u] = (dd[m][:, keep], np.minimum(T-np.arange(1, T+1), 125.))
    ids = np.array(sorted(U)); NS, SEEDS, NTE = [5, 10, 15, 20, 30, 50, 70], 15, 30
    out = {}
    for lbl, deg, al, p in [('ridge lin  (p=18)', 1, 10., 18),
                            ('ridge quad (p=171)', 2, 300., 171)]:
        acc = {n: [] for n in NS}
        for s in range(SEEDS):
            rng = np.random.default_rng(4000+s); pm = rng.permutation(ids)
            te, pool = pm[:NTE], pm[NTE:]
            Xte = np.vstack([U[u][0] for u in te]); yte = np.concatenate([U[u][1] for u in te])
            for n in NS:
                tr = pool[:n]
                Xtr = np.vstack([U[u][0] for u in tr]); ytr = np.concatenate([U[u][1] for u in tr])
                m = mk(deg, al).fit(Xtr, ytr)
                acc[n].append(np.sqrt(np.mean((m.predict(Xte)-yte)**2)))
        out[lbl] = (NS, np.array([np.mean(acc[n]) for n in NS]), p)
    return out

if __name__ == '__main__':
    print('=== Battery (Severson, 119 cells): MAPE on cycle life ===')
    print(f'{"model":20s} ' + " ".join(f"{('n='+str(n)):>8s}" for n in
                                       [15, 20, 30, 40, 60, 79])
          + f' {"floor":>7s} {"gamma":>7s}')
    for lbl, (ns, mu) in battery().items():
        fl, g = fit_curve(np.array(ns, float), mu)
        print(f'{lbl:20s} ' + " ".join(f"{100*v:7.1f}%" for v in mu)
              + f' {100*fl:6.1f}% {g:7.2f}')
    print('  Paper 1 sec.5: 19% (n=20), 12% (n=60), 9.8% (n=100); gamma 0.41, 1.2 after')
    print('  subtracting an 8% floor.  Corrected class beta=4.48 -> kappa=1 -> gamma=1.\n')

    print('=== C-MAPSS FD001 (100 units): RUL RMSE in cycles, B=125 ===')
    print(f'{"model":20s} ' + " ".join(f"{('n='+str(n)):>7s}" for n in
                                       [5, 10, 15, 20, 30, 50, 70])
          + f' {"floor":>7s} {"gamma":>6s} {"alpha_0":>8s}')
    for lbl, (ns, mu, p) in cmapss().items():
        fl, g = fit_curve(np.array(ns, float), mu)
        a0 = np.median(mu/(125.*np.sqrt(p/np.array(ns, float))))
        print(f'{lbl:20s} ' + " ".join(f"{v:7.2f}" for v in mu)
              + f' {fl:6.2f}  {g:6.2f} {a0:8.4f}')
    print('  Paper 1 holds alpha_0 = 0.126 fixed for every architecture and domain;')
    print('  predicts gamma = 1/(2-kappa), FD001 kappa~0.05 -> gamma~0.51.')
