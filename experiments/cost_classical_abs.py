"""The same absolute-stop campaigns scored on the two objectives (R3.4): the learner and the classical fleet quantities.

cost_batt_abs.py draws campaigns on the 119 Severson cells: every cell is stopped at an absolute cycle t_s (the F-quantile of the training-pool lives),
the number of cells is n = C / (kappa0 + E[min(T, t_s)] / Tbar), and a life predictor is trained on them.  Here the SAME draws (same cells, same stop,
same cost) are also scored on what classical test planning targets: the lognormal life distribution of the fleet, estimated by censored maximum likelihood
from the failures and suspensions of the campaign, and the error in its 10th percentile, its median and its mean.  The truth is the lognormal fitted to
all 119 complete lives, so the error measures the campaign and not the lognormal assumption.

Outcomes, each relative to the same quantity at F = 1 (run to failure) on the same splits:
  learner      RMSE of the life prediction on 30 held-out cells, label strategies impute+Huber and Tobit (as cost_batt_abs.py)
  q10, median, mean   root mean square error of ln q-hat (the classical estimate from the same n cells; 'same units')
  q10 (all)    the classical analyst is not limited to cells that reach the feature window: the same budget buys n = C/cost cells of the whole pool
"""
import os
import sys
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cost_real as cr
import cost_cmapss_ext as ce

warnings.filterwarnings('ignore')
cr.DATA = os.path.join(HERE, '..', 'data', 'severson', 'dq_traj.npz')
K0, C_BUDGET, NTEST = 0.05, 21.0, 30
Z10 = norm.ppf(0.10)


def lognormal_cens(obs, fail):
    """Censored lognormal MLE: (mu, sigma) of ln T from times 'obs' with failure indicator 'fail'."""
    y = np.log(obs)
    mu0 = y.mean()
    s0 = max(y[fail].std() if fail.sum() >= 3 else 0.4, 0.1)

    def nll(p):
        mu, ls = p
        s = np.exp(ls)
        z = (y - mu) / s
        ll = np.where(fail, norm.logpdf(z) - ls, norm.logsf(z))
        return -ll.sum()
    r = minimize(nll, [mu0, np.log(s0)], method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 400})
    return r.x[0], float(np.exp(r.x[1]))


def run(channel, m, qs, seeds):
    QD, DQ, life = cr.load()
    n_all = len(life)
    series, cycles, dmg, dmg_cyc = [], [], [], []
    for i in range(n_all):
        d = cr.damage(QD[i]); dc = np.arange(1, len(d) + 1, dtype=float)
        dmg.append(d); dmg_cyc.append(dc)
        if channel == 'capacity':
            s, cyc = d, dc
        else:
            cyc, s = DQ[i][0].astype(float), DQ[i][1].astype(float)
        series.append(np.asarray(s, float)); cycles.append(cyc)
    ok = np.array([life[i] >= m and cycles[i][-1] >= m and np.all(np.isfinite(series[i][np.isfinite(series[i])])) for i in range(n_all)])
    idx = np.where(ok)[0]
    ly = np.log(life)
    mu_t, s_t = ly.mean(), ly.std(ddof=0)
    truth = {'q10': mu_t + Z10 * s_t, 'median': mu_t, 'mean': mu_t + 0.5 * s_t ** 2}
    keys = ('impute', 'Tobit', 'oracle', 'q10', 'median', 'mean', 'q10_all', 'n', 'n_all', 'elig', 'failshare')
    res = {q: {k: [] for k in keys} for q in qs}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        test = rng.choice(idx, NTEST, replace=False)
        pool = np.array([i for i in idx if i not in set(test)])
        allpool = np.array([i for i in range(n_all) if i not in set(test)])
        Xte = np.array([cr.feats(series[i], cycles[i], m) for i in test]); yte = life[test]
        base = float(np.sqrt(np.mean((yte - life[pool].mean()) ** 2))); Tbar = life[pool].mean()
        for q in qs:
            ts = np.inf if q >= 1 else np.quantile(life[pool], q)
            obs = np.minimum(life, np.floor(ts)); fail = life <= ts
            cost = K0 + np.mean(np.minimum(life[pool], ts)) / Tbar
            elig = np.array([i for i in pool if obs[i] >= m])
            n = min(max(4, int(round(C_BUDGET / cost))), len(elig))
            n_all_units = min(int(round(C_BUDGET / cost)), len(allpool))
            if len(elig) < 8 or n < 6:
                for k in keys:
                    res[q][k].append(np.nan)
                continue
            acc = {k: [] for k in ('impute', 'Tobit', 'oracle', 'q10', 'median', 'mean', 'q10_all')}
            sh = []
            for _ in range(30):
                take = rng.choice(elig, n, replace=False)
                X = np.array([cr.feats(series[i], cycles[i], m) for i in take])
                f = fail[take]; sh.append(f.mean())
                rr = lambda p: float(np.sqrt(np.mean((p - yte) ** 2))) / base
                acc['oracle'].append(rr(Xte @ cr.huber(X, life[take])))
                yl = life[take].copy()
                if not f.all():
                    tr = [(dmg_cyc[i][dmg_cyc[i] <= obs[i]], dmg[i][dmg_cyc[i] <= obs[i]]) for i in take[~f]]
                    try:
                        yl[~f] = cr.impute(tr, cr.pool_beta(pool, dmg, dmg_cyc, life)) if len(tr) >= 3 else obs[take[~f]]
                    except Exception:
                        yl[~f] = obs[take[~f]]
                acc['impute'].append(rr(Xte @ cr.huber(X, yl)))
                try:
                    par = ce.tobit_fit(X, np.log(obs[take]), ~f); acc['Tobit'].append(rr(ce.tobit_predict(par, Xte)))
                except Exception:
                    acc['Tobit'].append(np.nan)
                try:
                    mu, s = lognormal_cens(obs[take].astype(float), f)
                    acc['q10'].append((mu + Z10 * s - truth['q10']) ** 2)
                    acc['median'].append((mu - truth['median']) ** 2)
                    acc['mean'].append((mu + 0.5 * s * s - truth['mean']) ** 2)
                except Exception:
                    for k in ('q10', 'median', 'mean'):
                        acc[k].append(np.nan)
                take2 = rng.choice(allpool, n_all_units, replace=False)
                try:
                    mu, s = lognormal_cens(obs[take2].astype(float), fail[take2])
                    acc['q10_all'].append((mu + Z10 * s - truth['q10']) ** 2)
                except Exception:
                    acc['q10_all'].append(np.nan)
            for k in ('impute', 'Tobit', 'oracle'):
                res[q][k].append(np.nanmean(acc[k]))
            for k in ('q10', 'median', 'mean', 'q10_all'):
                res[q][k].append(np.sqrt(np.nanmean(acc[k])))
            res[q]['n'].append(n); res[q]['n_all'].append(n_all_units); res[q]['elig'].append(len(elig)); res[q]['failshare'].append(np.mean(sh))
    return res


if __name__ == '__main__':
    import json
    qs = (0.2, 0.35, 0.5, 0.75, 1.0)
    seeds = tuple(range(int(sys.argv[1]) if len(sys.argv) > 1 else 8))
    out = {}
    for channel, m in (('dq', 100), ('capacity', 400)):
        res = run(channel, m, qs, seeds)
        out[channel] = res
        print(f'\n=== {channel}, feature window m={m}; {len(seeds)} splits x 30 campaigns; budget C={C_BUDGET}; ratios to F=1 on the same splits ===')
        ref = {k: np.array(res[1.0][k], float) for k in ('impute', 'Tobit', 'q10', 'median', 'mean', 'q10_all')}
        print(f'{"F(t_s)":>7s} {"n":>4s} {"n(all)":>6s} {"fail":>5s} | {"impute":>8s} {"Tobit":>8s} | {"q10":>8s} {"median":>8s} {"mean":>8s} {"q10(all)":>9s}   (absolute: q10 rmse of ln q, learner/base)')
        for q in qs:
            R = res[q]
            if not R['n']:
                print(f'{q:7.2f}   (too few eligible cells)'); continue
            def rel(k):
                a = np.array(R[k], float); b = ref[k]
                k_ = min(len(a), len(b))
                return np.nanmean(a[:k_] / b[:k_]) if k_ else np.nan
            print(f'{q:7.2f} {np.mean(R["n"]):4.0f} {np.mean(R["n_all"]):6.0f} {np.mean(R["failshare"]):5.2f} | {rel("impute"):8.2f} {rel("Tobit"):8.2f} | '
                  f'{rel("q10"):8.2f} {rel("median"):8.2f} {rel("mean"):8.2f} {rel("q10_all"):9.2f}   ({np.nanmean(R["q10"]):.3f}, {np.nanmean(R["impute"]):.2f})')
    json.dump({ch: {str(q): {k: [float(x) for x in v] for k, v in r.items()} for q, r in rr.items()} for ch, rr in out.items()},
              open(os.path.join(HERE, 'cost_classical_abs.json'), 'w'))
