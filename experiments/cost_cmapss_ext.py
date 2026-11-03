"""Addresses R2.1/R4.3 (implementable absolute stop time) and R2.4 (censoring-aware baseline).
Same fleet, split, features, learner and budget as cost_cmapss.py. New:
  * fractional design c  (as in the paper)       and      absolute stop time t_s with F(t_s)=q
  * label strategies: 'impute+Huber' (paper), 'Tobit' (right-censored Gaussian AFT on log T,
    no label imputation), 'failed-only' (drop censored units), 'oracle' (true lives).
Output: prediction RMSE as a multiple of predicting the pool mean (below 1 = useful)."""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import cost_cmapss as cc
warnings.filterwarnings('ignore')

def tobit_fit(X, ylog, cens):
    mu, sd = X[:, 1:].mean(0), X[:, 1:].std(0) + 1e-9
    Z = np.c_[X[:, 0], (X[:, 1:] - mu)/sd]
    b0 = np.linalg.lstsq(Z, ylog, rcond=None)[0]
    def nll(p):
        b, ls = p[:-1], max(p[-1], np.log(0.15)); s = np.exp(ls); r = (ylog - Z @ b)/s
        ll = np.where(cens, norm.logsf(r), norm.logpdf(r) - ls)
        return -np.sum(ll) + 0.5*np.sum(b[1:]**2)
    res = minimize(nll, np.r_[b0, np.log(np.std(ylog - Z @ b0) + 1e-3)], method='BFGS')
    return res.x[:-1], mu, sd
def tobit_predict(par, Xte):
    b, mu, sd = par; Z = np.c_[Xte[:, 0], (Xte[:, 1:] - mu)/sd]; return np.exp(Z @ b)

def run(m, seed, designs, reps=40):
    units, life = cc.load(); rng = np.random.default_rng(seed)
    order = rng.permutation(len(life)); test, cal, pool = order[:cc.N_TEST], order[cc.N_TEST:cc.N_TEST+cc.N_CAL], order[cc.N_TEST+cc.N_CAL:]
    index = cc.calibrate(units, cal); D = [index(u) for u in units]
    Xte = np.array([cc.feats(D[i], m) for i in test]); yte = life[test]
    base = float(np.sqrt(np.mean((yte - life[pool].mean())**2)))
    Tbar = life[pool].mean(); out = {}
    for name, kind, par in designs:
        if kind == 'frac':
            obs = np.minimum(life, np.floor(par*life))             # needs T_i: retrospective
            cost = cc.KAP0 + par; fail = np.full(len(life), par >= 1.0)
        else:
            ts = np.inf if par >= 1.0 else np.quantile(life[pool], par)
            obs = np.minimum(life, np.floor(ts)); fail = life <= ts
            cost = cc.KAP0 + np.mean(np.minimum(life[pool], ts))/Tbar
        elig = np.array([i for i in pool if obs[i] >= m]); n = min(max(4, int(round(cc.C_BUDGET/cost))), len(elig))
        if len(elig) < 8: out[name] = None; continue
        acc = {k: [] for k in ('impute+Huber', 'Tobit', 'failed-only', 'oracle')}; share = []
        for _ in range(reps):
            take = rng.choice(elig, n, replace=False); X = np.array([cc.feats(D[i], m) for i in take])
            f = fail[take]; share.append(f.mean())
            yl = life[take].copy()
            if not f.all():
                tr = [(np.arange(1, int(obs[i])+1, dtype=float), D[i][:int(obs[i])]) for i in take[~f]]
                yl[~f] = cc.impute(tr, cc.cal_beta(D, life, cal)) if len(tr) >= 3 else obs[take[~f]]
            r = lambda p: float(np.sqrt(np.mean((p - yte)**2)))/base
            acc['impute+Huber'].append(r(cc.fit_huber(X, yl, Xte)))
            acc['oracle'].append(r(cc.fit_huber(X, life[take], Xte)))
            try: acc['Tobit'].append(r(tobit_predict(tobit_fit(X, np.log(obs[take]), ~f), Xte)))
            except Exception: acc['Tobit'].append(np.nan)
            acc['failed-only'].append(r(cc.fit_huber(X[f], life[take][f], Xte)) if f.sum() >= 6 else np.nan)
        out[name] = ({k: float(np.nanmean(v)) for k, v in acc.items()}, float(np.mean(share)), n, len(elig))
    return out

if __name__ == '__main__':
    designs = [(f'c={c:.2f}', 'frac', c) for c in (0.5, 0.65, 0.8, 1.0)] + \
              [(f'F(ts)={q:.2f}', 'abs', q) for q in (0.1, 0.25, 0.5, 0.75, 1.0)]
    for m in (60, 100):
        res = [run(m, s, designs) for s in (0, 1, 2)]
        print(f'\n=== FD001 PC1, window m={m} cycles; RMSE / pool-mean RMSE, mean of 3 splits x 40 draws ===')
        print(f'{"design":>12s} {"n":>3s} {"elig":>4s} {"share failed":>12s}  {"impute+Huber":>12s} {"Tobit":>7s} {"failed-only":>11s} {"oracle":>7s}')
        for nm, _, _ in designs:
            rr = [r[nm] for r in res if r[nm] is not None]
            if not rr: print(f'{nm:>12s}   (too few eligible units)'); continue
            g = lambda k: np.nanmean([x[0][k] for x in rr])
            print(f'{nm:>12s} {rr[0][2]:3d} {rr[0][3]:4d} {np.mean([x[1] for x in rr]):12.2f}  {g("impute+Huber"):12.2f} {g("Tobit"):7.2f} {g("failed-only"):11.2f} {g("oracle"):7.2f}')
