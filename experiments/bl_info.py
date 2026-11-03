"""Information curve of Delta Q(V) and the measured threshold on further public cell sets.

For each set: (i) the survival-corrected odds curve of Delta Q(V) for ln T over windows that are fixed fractions of the median
life (the estimator of dqv_mar.py), its slope at c = 1 with a bootstrap over cells, the plateau, and theta_b of Theorem 5(3)(ii)
at the measured Lambda; (ii) Lambda by the covariate ablation of battery_lambda2.py with the sizes of the learning curve scaled
to the set. Usage: python bl_info.py SET [SET ...]; B (bootstrap) from the environment.
"""
import os, sys
import numpy as np
import dqv_mar as dm
import battery_lambda2 as bl2
from battery_lambda import feats
import bl_common as bc

CFR = (0.07, 0.14, 0.27, 0.40, 0.55, 0.80, 1.00, 1.20)
B = int(os.environ.get('B', '1000'))
TOL = 0.2


def run(s):
    QD, DQ, life, f, keys = bc.load_set(s)
    n = len(life)
    med = float(np.median(life))
    W = tuple(int(round(c*med)) for c in CFR)
    dm.W = W; dm.MIN_N = max(12, n//5)
    X = np.full((n, len(W)), np.nan)
    for i in range(n):
        cyc, val = DQ[i]
        for j, m in enumerate(W):
            if life[i] >= m and cyc[-1] >= m and cyc[0] <= m:
                X[i, j] = np.interp(m, cyc, val)
    y = np.log(life)
    om, eta, c, sl, sat, pl = dm.summarize(X, y, np.arange(n), dm.odds_mar)
    OM, SL, SAT, PL, ET = dm.boot(X, y, dm.odds_mar, B, 20261103)
    surv = np.isfinite(X).sum(0)
    print(f'\n=== {s}: {n} cells, median life {med:.0f}; survivors by window {dict(zip(W, surv.tolist()))}')
    print('  odds   ' + ' '.join(f'{v:6.2f}' for v in om))
    print('  eta    ' + ' '.join(f'{v:6.2f}' for v in eta))
    lo, hi = (np.nanpercentile(SL, [2.5, 97.5]) if len(SL) > 20 else (np.nan, np.nan))
    print(f'  plateau {pl:.2f}; slope at c=1: {sl:.2f} (bootstrap median {np.nanmedian(SL):.2f}, 95% [{lo:.2f}, {hi:.2f}], {len(SL)} usable)')

    # Lambda by covariate ablation, at the window of 0.15 median lives
    m = int(round(0.15*med))
    idx = [i for i in range(n) if life[i] >= m and DQ[i][0][-1] >= m]
    yy = np.log(life[idx])
    Xf = np.array([feats(DQ[i][1].astype(float), DQ[i][0].astype(float), m) for i in idx])
    Z = np.column_stack([np.ones(len(idx)), (Xf[:, 1:] - Xf[:, 1:].mean(0))/(Xf[:, 1:].std(0) + 1e-12)])
    one = np.ones((len(idx), 1))
    ntest = max(8, int(round(0.2*len(idx))))
    bl2.NS = np.unique(np.linspace(5, len(idx) - ntest - 1, 8).astype(int))
    bl2.REPS = 400
    keep = []
    print(f'  Lambda ablation at m={m}, {len(idx)} cells, ntest={ntest}, Var(ln T)={yy.var():.4f}')
    for lam in (0.01, 0.1, 0.3):
        S = bl2.learning_slope(one, yy, lam, ntest, 11)[1]
        a, tot, r2 = bl2.learning_slope(Z, yy, lam, ntest, 11)
        rel = abs(S - yy.var())/yy.var(); K = tot - S
        good = rel <= TOL and K > 0
        L = S/K if K > 0 else np.nan
        print(f'    ridge {lam:5.2f}: S={S:.4f} ({rel*100:.0f}% off) S+K={tot:.4f} K={K:.4f} Lambda={L:.3f}' + ('' if good else '   rejected'))
        if good:
            keep.append(L)
    if keep:
        for L in (min(keep), max(keep)):
            th = dm.theta_b(pl, L)
            print(f'  Lambda={L:.3f}: corner threshold theta_b={th:.2f} (plateau {pl:.2f}); slope {sl:.2f}; P(slope >= theta_b)={np.mean(SL >= th)*100:.0f}%')
    else:
        print('  no setting passes the control; Lambda not measured')


if __name__ == '__main__':
    for s in sys.argv[1:]:
        run(s)
