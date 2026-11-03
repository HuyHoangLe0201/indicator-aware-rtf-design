"""Implementable absolute-stop-time campaigns on the 119 Severson cells, with censoring-aware learners (R2.1, R2.4, R4.3, R3.4).
Design: stop every cell at an absolute cycle t_s = quantile_q(training-pool lives); cells that fail earlier release the rig.
Cost per cell kappa0 + E[min(T,t_s)]/Tbar; number of cells n = C/cost, C = 21 (as in the fraction-of-life tables).
Label strategies on the same cells, same features, same budget:
  impute+Huber  (the paper's: complete-pool shape, per-cell life by least squares, Huber regression)
  Tobit         (right-censored Gaussian AFT on ln T, no imputation)
  Cox           (ridge Cox PH, Breslow baseline, median survival)
  GBM-AFT       (gradient-boosted trees on the censored Gaussian AFT likelihood)
  oracle        (true lives)
Outcome: RMSE of life prediction on 30 held-out cells / RMSE of predicting the pool mean."""
import sys, os, warnings
import numpy as np
from scipy.stats import norm
from sklearn.tree import DecisionTreeRegressor
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '.'))
import cost_real as cr
import cost_cmapss_ext as ce
import survival_baseline as sb
warnings.filterwarnings('ignore')
cr.DATA = os.path.join(HERE, '..', 'data', 'severson', 'dq_traj.npz')
K0, C_BUDGET, NTEST = 0.05, 21.0, 30


def gbm_aft_fit(X, ylog, cens, sigma, rounds=120, lr=0.1, depth=2):
    mu0 = np.mean(ylog); F = np.full(len(ylog), mu0); trees = []
    for _ in range(rounds):
        r = (ylog - F) / sigma
        g = np.where(cens, norm.pdf(r) / np.maximum(norm.sf(r), 1e-9) / sigma, r / sigma)
        t = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=3).fit(X[:, 1:], g)
        F = F + lr * sigma**2 * t.predict(X[:, 1:]); trees.append(t)
    return mu0, trees, lr, sigma


def gbm_aft_predict(par, X):
    mu0, trees, lr, sigma = par
    F = np.full(len(X), mu0)
    for t in trees:
        F = F + lr * sigma**2 * t.predict(X[:, 1:])
    return np.exp(F)


def run(channel, m, qs, seeds=(0, 1, 2, 3, 4, 5), reps=30):
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
    res = {q: {k: [] for k in ('impute+Huber', 'Tobit', 'Cox', 'GBM-AFT', 'oracle', 'n', 'elig', 'failshare', 'invalid')} for q in qs}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        test = rng.choice(idx, NTEST, replace=False)
        pool = np.array([i for i in idx if i not in set(test)])
        Xte = np.array([cr.feats(series[i], cycles[i], m) for i in test]); yte = life[test]
        base = float(np.sqrt(np.mean((yte - life[pool].mean())**2))); Tbar = life[pool].mean()
        for q in qs:
            ts = np.inf if q >= 1 else np.quantile(life[pool], q)
            obs = np.minimum(life, np.floor(ts)); fail = life <= ts
            cost = K0 + np.mean(np.minimum(life[pool], ts)) / Tbar
            elig = np.array([i for i in pool if obs[i] >= m])
            n = min(max(4, int(round(C_BUDGET / cost))), len(elig))
            if len(elig) < 8 or n < 6:
                res[q]['invalid'].append(1); continue
            acc = {k: [] for k in ('impute+Huber', 'Tobit', 'Cox', 'GBM-AFT', 'oracle')}; sh = []; inv = 0
            for _ in range(reps):
                take = rng.choice(elig, n, replace=False); X = np.array([cr.feats(series[i], cycles[i], m) for i in take])
                f = fail[take]; sh.append(f.mean())
                r = lambda p: float(np.sqrt(np.mean((p - yte)**2))) / base
                acc['oracle'].append(r(cr.huber(X, life[take]) @ Xte.T) if False else r(Xte @ cr.huber(X, life[take])))
                yl = life[take].copy()
                if not f.all():
                    tr = [(dmg_cyc[i][dmg_cyc[i] <= obs[i]], dmg[i][dmg_cyc[i] <= obs[i]]) for i in take[~f]]
                    try: yl[~f] = cr.impute(tr, cr.pool_beta(pool, dmg, dmg_cyc, life)) if len(tr) >= 3 else obs[take[~f]]
                    except Exception: inv += 1; yl[~f] = obs[take[~f]]
                acc['impute+Huber'].append(r(Xte @ cr.huber(X, yl)))
                ylog = np.log(obs[take]); cens = ~f
                try:
                    par = ce.tobit_fit(X, ylog, cens); acc['Tobit'].append(r(ce.tobit_predict(par, Xte)))
                except Exception: acc['Tobit'].append(np.nan); inv += 1
                try: acc['Cox'].append(r(sb.cox_pred(sb.cox_fit(X, obs[take].astype(float), f), Xte)) if f.sum() >= 3 else np.nan)
                except Exception: acc['Cox'].append(np.nan); inv += 1
                try:
                    sig = max(0.15, np.std(np.log(obs[take][f]) - np.mean(np.log(obs[take][f]))) * 0.7) if f.sum() >= 4 else 0.3
                    acc['GBM-AFT'].append(r(gbm_aft_predict(gbm_aft_fit(X, ylog, cens, sig), Xte)))
                except Exception: acc['GBM-AFT'].append(np.nan); inv += 1
            for k in acc: res[q][k].append(np.nanmean(acc[k]) if not np.all(np.isnan(acc[k])) else np.nan)
            res[q]['n'].append(n); res[q]['elig'].append(len(elig)); res[q]['failshare'].append(np.mean(sh)); res[q]['invalid'].append(inv)
    return res


if __name__ == '__main__':
    import json
    qs = (0.2, 0.35, 0.5, 0.75, 1.0)
    seeds = tuple(range(8))
    allres = {}
    for channel, m in (('dq', 100), ('capacity', 400)):
        res = run(channel, m, qs, seeds=seeds)
        allres[channel] = res
        print(f'\n=== {channel}, window m={m}: RMSE / pool-mean RMSE, mean (s.e. over {len(seeds)} splits) of 30 draws each; budget C={C_BUDGET} ===')
        print(f'{"F(t_s)":>7s} {"n":>4s} {"elig":>5s} {"fail":>5s} {"invalid":>7s} {"impute+Huber":>15s} {"Tobit":>13s} {"Cox":>13s} {"GBM-AFT":>13s} {"oracle":>13s}  {"paired diff impute vs F=1":>26s}')
        ref = np.array(res[1.0]['impute+Huber'], float)
        for q in qs:
            R = res[q]
            if not R['n']: print(f'{q:7.2f}   (too few eligible cells)'); continue
            def g(k):
                v = np.array(R[k], float); v = v[np.isfinite(v)]
                return f'{v.mean():6.2f} ({v.std(ddof=1)/np.sqrt(len(v)):4.2f})'
            d = np.array(R['impute+Huber'], float) - ref
            print(f'{q:7.2f} {np.mean(R["n"]):4.0f} {np.mean(R["elig"]):5.0f} {np.mean(R["failshare"]):5.2f} {int(np.sum(R["invalid"])):7d} '
                  f'{g("impute+Huber"):>15s} {g("Tobit"):>13s} {g("Cox"):>13s} {g("GBM-AFT"):>13s} {g("oracle"):>13s}  {d.mean():+.3f} ({d.std(ddof=1)/np.sqrt(len(d)):.3f})')
    json.dump({ch: {str(q): {k: [float(x) for x in v] for k, v in r.items()} for q, r in rr.items()} for ch, rr in allres.items()}, open('cost_batt_abs.json', 'w'))
