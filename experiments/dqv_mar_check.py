"""Checks on the survival-corrected information curve of dqv_mar.py.

(1) Selection test on real data.  A window that is read on (almost) all cells gives a reference odds value.  Dropping the
    cells that live less than L cycles and re-estimating from the rest imitates the survivor bias of a long window on
    data whose truth is known.  The corrected estimator should return the reference value; the survivors' R^2 should not.
(2) The same estimator for the capacity channel (QD) so that the two channels are on one footing.
(3) Robustness of the end-of-life slope: three slope definitions, a quadratic x-on-y regression, and the 12-cell fixed set.
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dqv_mar as M

HERE = M.HERE


def load_channel(key):
    """Bulk capacity as in dq_odds.py: median of the last five valid readings up to cycle m."""
    d = np.load(M.DATA, allow_pickle=True)
    QD, life = d[key], d['life'].astype(float)
    X = np.full((len(life), len(M.W)), np.nan)
    for i in range(len(life)):
        q = np.asarray(QD[i], float)
        for j, m in enumerate(M.W):
            if life[i] > m:
                w = q[:m + 1]
                w = w[np.isfinite(w) & (w > .2) & (w < 1.5)]
                if len(w) >= 5:
                    X[i, j] = np.median(w[-5:])
    return X, np.log(life), life


def odds_quad(x, y):
    """x on (y, y^2), population odds = Var(m(y)) / mean(s^2) over all y."""
    ok = np.isfinite(x)
    if ok.sum() < M.MIN_N + 5:
        return np.nan
    xs, ys = x[ok], y[ok]
    yc = y - y.mean()
    A = np.column_stack([np.ones(len(ys)), ys - y.mean(), (ys - y.mean()) ** 2])
    beta, *_ = np.linalg.lstsq(A, xs, rcond=None)
    res = xs - A @ beta
    s2 = res @ res / (len(ys) - 3)
    mfull = beta[0] + beta[1] * yc + beta[2] * yc ** 2
    return float(mfull.var(ddof=1) / s2)


def selection_test(X, y, life, ref_cols, cuts, fn_list, B=300, seed=11):
    rng = np.random.default_rng(seed)
    n = len(y)
    print('  reference = odds on all cells that have the window; rows: window, cut L; columns: reference, naive, corrected (bootstrap spread of the corrected value in brackets)')
    rows = []
    for j in ref_cols:
        ref = M.odds_mar(X[:, j], y)
        for L in cuts:
            keep = life >= L
            # impossible to also need the window: x observed only where the cell is alive at the window
            xk = np.where(keep, X[:, j], np.nan)
            # corrected estimator sees the dropped cells as missing but uses all y for Var(y)
            cor = M.odds_mar(xk, y)
            nai = M.odds_naive(xk, y)
            bs = []
            for _ in range(B):
                idx = rng.integers(0, n, n)
                bs.append(M.odds_mar(xk[idx], y[idx]))
            bs = np.array(bs)
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            rows.append((M.W[j], L, keep.sum(), ref, nai, cor, lo, hi))
            print(f'   m={M.W[j]:4d} L={L:4d} (kept {keep.sum():3d}): reference {ref:6.2f}  naive {nai:6.2f}  corrected {cor:6.2f} [{lo:6.2f},{hi:6.2f}]')
    return rows


if __name__ == '__main__':
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    Xq, y, life = M.load()
    print('=== (1) selection test on the real cells (Delta Q(V)) ===')
    cols = [M.W.index(m) for m in (100, 200, 300, 400)]
    rows = selection_test(Xq, y, life, cols, (500, 600, 700, 800, 900), None)
    refs = np.array([r[3] for r in rows]); nai = np.array([r[4] for r in rows]); cor = np.array([r[5] for r in rows])
    cov = np.mean([(r[6] <= r[3] <= r[7]) for r in rows])
    print(f'  mean |corrected - reference| / reference = {np.mean(np.abs(cor - refs) / refs):.2f}; mean |naive - reference| / reference = {np.mean(np.abs(nai - refs) / refs):.2f}; reference inside the corrected 95% interval in {cov * 100:.0f}% of the rows')

    print('\n=== (2) capacity channel (QD) ===')
    Xc, yc, lc = load_channel('QD')
    print('  survivors', dict(zip(M.W, np.isfinite(Xc).sum(0).tolist())))
    for name, fn in (('naive', M.odds_naive), ('corrected', M.odds_mar)):
        om, eta, c, sl, sat, pl = M.summarize(Xc, yc, np.arange(len(yc)), fn)
        print(f'  {name:9s} odds ' + ' '.join(f'{v:6.2f}' for v in om))
    print('  (window grid: ' + ' '.join(str(m) for m in M.W) + ')')
    nc = len(yc)
    rngc = np.random.default_rng(5)
    OMc = np.array([M.curve(Xc, yc, rngc.integers(0, nc, nc), M.odds_mar) for _ in range(B)])
    loc, hic = np.nanpercentile(OMc, [2.5, 97.5], axis=0)
    omc = M.curve(Xc, yc, np.arange(nc), M.odds_mar)
    print('  corrected odds with 95% interval:')
    for j, m in enumerate(M.W):
        print(f'    m={m:4d}: {omc[j]:6.2f} [{loc[j]:6.2f}, {hic[j]:6.2f}]')
    print('  selection test on the capacity channel at m = 400 (reference = all cells that reach 400):')
    selection_test(Xc, yc, lc, [M.W.index(400)], (500, 600, 700, 800, 900), None)
    omq = M.curve(Xq, y, np.arange(n := len(y)), M.odds_mar)
    np.savez(os.path.join(HERE, 'cap_info_curve.npz'), W=np.array(M.W), odds=omc, lo=loc, hi=hic, median_life=np.median(lc))

    print('\n=== (3) end-of-life slope: variants, corrected estimator, Delta Q(V) ===')
    n = len(y)
    rng = np.random.default_rng(20260804)
    var = {'local line c in [0.7,1.3]': [], 'central difference c=0.83 vs 1.11': [], 'isotonic fit, local line': [], 'quadratic x on y, local line': []}
    plat = []
    from numpy import polyfit
    def iso(v):
        # pool-adjacent-violators, nondecreasing
        v = list(v); w = [1] * len(v); out = []
        for a, b in zip(v, w):
            out.append([a, b])
            while len(out) > 1 and out[-2][0] > out[-1][0]:
                a2, b2 = out.pop(); a1, b1 = out.pop(); out.append([(a1 * b1 + a2 * b2) / (b1 + b2), b1 + b2])
        r = []
        for a, b in out:
            r += [a] * b
        return np.array(r)
    jc = [M.W.index(m) for m in (600, 800)]
    for b in range(B):
        idx = rng.integers(0, n, n) if b else np.arange(n)
        om = M.curve(Xq, y, idx, M.odds_mar)
        oq = M.curve(Xq, y, idx, odds_quad)
        med = np.median(np.exp(y[idx])); c = np.array(M.W) / med
        ok = np.isfinite(om)
        if ok.sum() < 8:
            continue
        pl = np.nanmean(om[(c >= 0.5) & (c <= 1.1) & ok])
        var['local line c in [0.7,1.3]'].append(M.slope_at(om / pl, c))
        var['central difference c=0.83 vs 1.11'].append((om[jc[1]] - om[jc[0]]) / pl / (c[jc[1]] - c[jc[0]]))
        okv = np.isfinite(om)
        e = np.full(len(om), np.nan); e[okv] = iso(om[okv] / pl)
        var['isotonic fit, local line'].append(M.slope_at(e, c))
        okq = np.isfinite(oq)
        plq = np.nanmean(oq[(c >= 0.5) & (c <= 1.1) & okq])
        var['quadratic x on y, local line'].append(M.slope_at(oq / plq, c))
        plat.append(pl)
    for k, v in var.items():
        v = np.array(v); v = v[np.isfinite(v)]
        print(f'  {k:38s} point {v[0]:6.2f}  median {np.median(v):6.2f}  95% [{np.percentile(v, 2.5):6.2f}, {np.percentile(v, 97.5):6.2f}]  P(<1.29)={np.mean(v < 1.29) * 100:.0f}%  P(<=0)={np.mean(v <= 0) * 100:.0f}%')
    om = M.curve(Xq, y, np.arange(n), odds_quad)
    print('  quadratic odds ' + ' '.join(f'{v:6.2f}' for v in om))
