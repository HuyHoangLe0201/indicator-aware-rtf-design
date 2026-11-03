"""
E1 on C-MAPSS FD001: measured learning curves, and a direct check of alpha_0.

Paper 1 fixes alpha_0 = 0.126 from a SINGLE published point (Zheng et al., LSTM,
RMSE 16.2 at n=100) and then holds it constant across every architecture and domain.
Here alpha_0 is measured instead: RMSE(n) is fitted to

        RMSE(n) = alpha_0 * B * sqrt(p_eff / n),      B = 125 (the RUL cap),

so that alpha_0 and the convergence exponent gamma (RMSE ~ n^-gamma) come out of data.

Setup: 100 run-to-failure units; fixed held-out test set of 30; training pool of 70;
subsample n units from the pool; RUL capped at 125; all time points used.
"""
import numpy as np, os
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = r'D:\bách khoa Đà Nẵng\nghiencuukhoahoc\baotri\data'
B_CAP, NS, SEEDS, N_TEST = 125.0, [5, 10, 15, 20, 30, 50, 70], 12, 30

def load():
    d = np.loadtxt(os.path.join(ROOT, 'train_FD001.txt'))
    uid = d[:, 0].astype(int)
    keep = [j for j in range(2, d.shape[1]) if d[:, j].std() > 1e-8]
    units = {}
    for u in np.unique(uid):
        m = uid == u
        Xu = d[m][:, keep]
        T = m.sum()
        rul = np.minimum(T - np.arange(1, T+1), B_CAP)
        units[u] = (Xu, rul)
    return units, len(keep)

MODELS = {
    'ridge':        lambda: make_pipeline(StandardScaler(),
                                          RidgeCV(alphas=np.logspace(-3, 4, 30))),
    'MLP 16':       lambda: make_pipeline(StandardScaler(),
                                          MLPRegressor((16,), max_iter=600, alpha=1e-2,
                                                       random_state=0)),
    'MLP 64x32':    lambda: make_pipeline(StandardScaler(),
                                          MLPRegressor((64, 32), max_iter=600, alpha=1e-2,
                                                       random_state=0)),
}

def peff(nm, d):
    return {'ridge': d+1, 'MLP 16': 16*(d+2), 'MLP 64x32': 64*(d+1)+32*65+33}[nm]

if __name__ == '__main__':
    units, d = load()
    ids = np.array(sorted(units))
    print(f'units {len(ids)}, sensors kept {d}')
    res = {k: {n: [] for n in NS} for k in MODELS}
    for s in range(SEEDS):
        rng = np.random.default_rng(500+s)
        perm = rng.permutation(ids)
        te, pool = perm[:N_TEST], perm[N_TEST:]
        Xte = np.vstack([units[u][0] for u in te])
        yte = np.concatenate([units[u][1] for u in te])
        for n in NS:
            tr = pool[:n]
            Xtr = np.vstack([units[u][0] for u in tr])
            ytr = np.concatenate([units[u][1] for u in tr])
            for nm, mk in MODELS.items():
                try:
                    m = mk().fit(Xtr, ytr)
                    res[nm][n].append(float(np.sqrt(np.mean((m.predict(Xte)-yte)**2))))
                except Exception:
                    pass
    print(f'\n{"model":11s} {"p_eff":>7s} ' + " ".join(f"{('n='+str(n)):>7s}" for n in NS)
          + f' {"gamma":>7s} {"alpha_0":>8s}')
    print('-'*92)
    for nm in MODELS:
        mu = np.array([np.mean(res[nm][n]) if res[nm][n] else np.nan for n in NS])
        p = peff(nm, d)
        ok = np.isfinite(mu)
        ln, lm = np.log(np.array(NS, float)[ok]), np.log(mu[ok])
        A = np.vstack([ln, np.ones(ok.sum())]).T
        cf, *_ = np.linalg.lstsq(A, lm, rcond=None)
        a0 = float(np.median(mu[ok]/(B_CAP*np.sqrt(p/np.array(NS, float)[ok]))))
        print(f'{nm:11s} {p:7d} ' + " ".join(f"{v:7.2f}" for v in mu)
              + f' {-cf[0]:7.3f} {a0:8.4f}')
    print('\nPaper 1: alpha_0 = 0.126 held fixed for every architecture and domain.')
    print('Paper 1 predicts gamma = 1/(2-kappa); FD001 kappa ~ 0.05 -> gamma ~ 0.51.')
