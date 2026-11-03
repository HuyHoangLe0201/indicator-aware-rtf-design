"""Instrumented rerun of the battery campaign of Table 6 (cost_real.run, seeds 0-2, 120 draws, C=21) that reports what a
reader needs to reconstruct it (R1.5): budget, requested and realized training-set size at every c, eligible pool,
number of draws, draws in which the imputation touched a constraint, failed shape refits, non-finite predictions, and the
error ratio so the rerun can be checked against the published table.  The logic is cost_real.run, copied verbatim except for
counters; nothing about the estimator is changed."""
import os, sys, warnings
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '.'))
import cost_real as cr
warnings.filterwarnings('ignore')
cr.DATA = os.path.join(HERE, '..', 'data', 'severson', 'dq_traj.npz')
KAP0, NTEST, C = 0.05, 30, 21.0


class Stat:
    def __init__(s): s.refit_fail = 0; s.at_upper = 0; s.at_lower = 0; s.beta_bound = 0


def impute_counted(trajs, st, beta0):
    def life_of(t, D, b):
        lo, hi = t[-1] * 1.0001, t[-1] * 25.
        T = minimize_scalar(lambda T: np.sum((np.clip(t / T, 1e-9, 3.)**b - D)**2), bounds=(lo, hi), method='bounded',
                            options={'xatol': 1e-2}).x
        return T, lo, hi
    b = beta0
    for _ in range(3):
        Ts = np.array([life_of(t, D, b)[0] for t, D in trajs])
        r = lambda p: np.concatenate([np.clip(t / T, 1e-9, 3.)**p[0] - D for (t, D), T in zip(trajs, Ts)])
        try:
            b = float(least_squares(r, [b], bounds=([0.4], [15.]), max_nfev=120).x[0])
        except Exception:
            st.refit_fail += 1
            break
    if b <= 0.41 or b >= 14.99: st.beta_bound += 1
    out = []
    for t, D in trajs:
        T, lo, hi = life_of(t, D, b)
        if T >= 0.99 * hi: st.at_upper += 1
        if T <= lo * 1.001 + 1e-6 * lo: st.at_lower += 1
        out.append(T)
    return np.array(out), b


def run(channel, m, grid, seed, reps=120):
    QD, DQ, life = cr.load()
    rng = np.random.default_rng(seed)
    n_all = len(life)
    series, cycles, dmg, dmg_cyc = [], [], [], []
    for i in range(n_all):
        d = cr.damage(QD[i]); dc = np.arange(1, len(d) + 1, dtype=float)
        dmg.append(d); dmg_cyc.append(dc)
        if channel == 'capacity': s, cyc = d, dc
        else: cyc, s = DQ[i][0].astype(float), DQ[i][1].astype(float)
        series.append(np.asarray(s, float)); cycles.append(cyc)
    ok = np.array([life[i] >= m and cycles[i][-1] >= m and np.all(np.isfinite(series[i][np.isfinite(series[i])])) for i in range(n_all)])
    idx = np.where(ok)[0]
    test = rng.choice(idx, NTEST, replace=False)
    pool = np.array([i for i in idx if i not in set(test)])
    Xte = np.array([cr.feats(series[i], cycles[i], m) for i in test]); yte = life[test]
    base = float(np.sqrt(np.mean((yte - life[pool].mean())**2)))
    rows = []
    for c in grid:
        elig = np.array([i for i in pool if c * life[i] >= m])
        n_ask = max(4, int(round(C / (KAP0 + c)))); n = min(n_ask, len(elig))
        if n < 4: rows.append(dict(c=c, n_ask=n_ask, elig=len(elig), n=n)); continue
        st = Stat(); rm = []; nonfin = 0; betas = []; draws_bound = 0
        for r in range(reps):
            take = rng.choice(elig, n, replace=False)
            trajs = [(dmg_cyc[i][dmg_cyc[i] <= c * life[i]], dmg[i][dmg_cyc[i] <= c * life[i]]) for i in take]
            before = (st.at_upper, st.at_lower)
            if c >= 1.0: yl = life[take]
            else:
                yl, b = impute_counted(trajs, st, cr.pool_beta(pool, dmg, dmg_cyc, life)); betas.append(b)
            if (st.at_upper, st.at_lower) != before: draws_bound += 1
            X = np.array([cr.feats(series[i], cycles[i], m) for i in take])
            mu = cr.huber(X, yl)
            pr = Xte @ mu
            if not np.all(np.isfinite(pr)): nonfin += 1; continue
            rm.append(np.sqrt(np.mean((pr - yte)**2)))
        rows.append(dict(c=c, n_ask=n_ask, elig=len(elig), n=n, ratio=float(np.mean(rm)) / base, valid=len(rm), reps=reps,
                         draws_bound=draws_bound, refit_fail=st.refit_fail, beta_bound=st.beta_bound,
                         beta_med=float(np.median(betas)) if betas else np.nan))
    return rows, len(idx), len(pool)


if __name__ == '__main__':
    GRID = [0.50, 0.65, 0.80, 1.00]
    for channel, m in (('dq', 100), ('capacity', 400)):
        print(f'\n=== channel {channel}, window m={m}, budget C={C}, test cells {NTEST}, 120 draws per cell, Huber(IRLS) ===')
        acc = {}
        for seed in (0, 1, 2):
            rows, nidx, npool = run(channel, m, GRID, seed)
            print(f' seed {seed}: {nidx} cells reach the window, pool {npool}')
            for r in rows:
                if 'ratio' not in r: print(f'   c={r["c"]:.2f} n_ask={r["n_ask"]} elig={r["elig"]} : no campaign'); continue
                print(f'   c={r["c"]:.2f} n_ask={r["n_ask"]:3d} elig={r["elig"]:3d} n_used={r["n"]:3d} valid={r["valid"]}/{r["reps"]} '
                      f'draws_with_bound_hit={r["draws_bound"]:3d} refit_fail={r["refit_fail"]} beta_at_bound={r["beta_bound"]} beta_med={r["beta_med"]:.2f} ratio={r["ratio"]:.2f}')
                acc.setdefault(r['c'], []).append(r)
        print(' mean ratio over seeds: ' + ', '.join(f'c={c:.2f}: {np.mean([r["ratio"] for r in v]):.2f}' for c, v in acc.items()))
