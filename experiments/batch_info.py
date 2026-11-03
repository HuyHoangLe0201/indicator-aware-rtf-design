"""Does the Delta-Q(V) verdict hold within each manufacturing batch of the Severson cells?

The 119 cells come from three batches (b1, b2, b3).  The survey measures the information curve of the Delta-Q(V)
channel on the pooled fleet; here the same corrected-odds estimator (dqv_mar.py) is applied to each batch alone and
to every pair of batches.  For each subset we report the odds at the end of life, the normalized curve, its slope,
and the certificate of Theorem 5(3)(ii)(e): the largest margin eta(c) - eta_*(c) over observed windows with omega >= 1.
Lambda = 0.241, kappa0 = 0.05 as in the survey.  Uses the shipped derived arrays only.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dqv_mar as M

KAPPA0, LAM = 0.05, 0.241


def envelope(c, plateau, lam=LAM, k0=KAPPA0):
    eta0 = 1.0 / plateau
    b = (1 + k0) / (1 + eta0)
    return (k0 + c) / (lam * (1 - c) + b) - eta0


def analyse(X, y, idx, B=1000, seed=1):
    om, eta, c, sl, sat, pl = M.summarize(X, y, idx, M.odds_mar)
    env = envelope(c, pl)
    ok = np.isfinite(om) & (om >= 1.0) & (c < 1.0)
    margin = np.where(ok, eta - env, -np.inf)
    j = int(np.argmax(margin))
    # bootstrap over cells for the certificate and the slope
    rng = np.random.default_rng(seed)
    cert, sls = 0, []
    n = len(idx)
    for _ in range(B):
        sub = idx[rng.integers(0, n, n)]
        o2, e2, c2, s2, _, p2 = M.summarize(X, y, sub, M.odds_mar)
        if not np.isfinite(s2):
            continue
        env2 = envelope(c2, p2)
        ok2 = np.isfinite(o2) & (o2 >= 1.0) & (c2 < 1.0)
        cert += bool(np.any(ok2 & (e2 > env2)))
        sls.append(s2)
    sls = np.array(sls) if sls else np.array([np.nan])
    th = M.theta_b(pl, LAM)
    return dict(n=n, med=float(np.median(np.exp(y[idx]))), plateau=pl, slope=sl, theta=th,
                best_c=float(c[j]), eta=float(eta[j]), env=float(env[j]), margin=float(margin[j]),
                cert=cert / max(len(sls), 1), p_slope_below=float(np.mean(sls < th)),
                slo=float(np.nanpercentile(sls, 2.5)) if np.isfinite(sls).any() else np.nan, shi=float(np.nanpercentile(sls, 97.5)) if np.isfinite(sls).any() else np.nan)


if __name__ == '__main__':
    X, y, life = M.load()
    d = np.load(M.DATA, allow_pickle=True)
    batch = np.array([str(s)[:2] for s in d['name']])
    subsets = [('all', np.arange(len(y)))]
    for b in ('b1', 'b2', 'b3'):
        subsets.append((b, np.where(batch == b)[0]))
    for hold in ('b1', 'b2', 'b3'):
        subsets.append((f'without {hold}', np.where(batch != hold)[0]))
    print(f'Lambda={LAM}, kappa0={KAPPA0}; corrected odds (x on y), bootstrap over cells')
    print(f'{"subset":12s} {"n":>4s} {"med":>5s} {"omega":>6s} {"slope":>6s} {"theta":>6s} {"best c":>6s} {"eta":>5s} {"env":>5s} {"margin":>6s}'
          f' {"P(cert)":>7s} {"P(sl<th)":>8s} {"slope 95%":>14s}')
    for name, idx in subsets:
        r = analyse(X, y, idx)
        print(f'{name:12s} {r["n"]:4d} {r["med"]:5.0f} {r["plateau"]:6.1f} {r["slope"]:6.2f} {r["theta"]:6.2f} {r["best_c"]:6.2f} '
              f'{r["eta"]:5.2f} {r["env"]:5.2f} {r["margin"]:6.2f} {100*r["cert"]:6.1f}% {100*r["p_slope_below"]:7.1f}% '
              f'[{r["slo"]:5.2f},{r["shi"]:5.2f}]')
