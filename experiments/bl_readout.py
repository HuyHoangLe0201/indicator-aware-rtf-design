"""Is Delta Q(V) a readout of the capacity trajectory? The test of readout_test2.py on further public cell sets.

For each set the probe cycle k is a fixed fraction of the median life (0.15, 0.20 and 0.30; on the Severson cells, k = 100, 150, 200
is 0.14 to 0.28 of the median life of 719). Controls: a four-feature summary of the capacity curve up to k (level, fade so far, recent
fade rate over the last k/2 cycles, curvature), applied on ranks with a quadratic expansion; the null is enforced by permuting
Delta Q(V) within a k-nearest-neighbour ball of the control space. Reported: partial rank correlation of Delta Q(V) with log T
given the capacity trajectory, its permutation p-value, and the R^2 of log T without and with Delta Q(V).

Usage: python bl_readout.py SET [SET ...]
"""
import sys
import numpy as np
from scipy import stats
import bl_common as bc

RNG = np.random.default_rng(1)
NPERM = 4000


def features(QD, DQ, life, k):
    X, dq, lf = [], [], []
    h = max(k // 2, 10)
    for i in range(len(life)):
        L = float(life[i])
        if L < k*1.2:
            continue
        q = np.asarray(QD[i], float); c, v = DQ[i]
        if k >= len(q) or q[k] <= 0 or k < c.min() or k > c.max():
            continue
        q1 = float(np.median(q[2:12]))
        lvl = float(q[k]); rate = float(q[k] - q[k - h])/h; early = float(q[h] - q1)/h
        X.append([lvl, lvl - q1, rate, rate - early]); dq.append(float(np.interp(k, c, v))); lf.append(L)
    return np.array(X), np.array(dq), np.array(lf)


def rankit(a):
    a = np.asarray(a, float)
    return stats.norm.ppf((stats.rankdata(a) - 0.5)/len(a))


def resid_on(X, y):
    R = np.column_stack([rankit(X[:, j]) for j in range(X.shape[1])])
    cols = [np.ones(len(y))] + [R[:, j] for j in range(R.shape[1])]
    for j in range(R.shape[1]):
        for l in range(j, R.shape[1]):
            cols.append(R[:, j]*R[:, l])
    A = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def knn_perm(X, x, y, kn=12):
    Z = np.column_stack([rankit(X[:, j]) for j in range(X.shape[1])])
    d = ((Z[:, None, :] - Z[None, :, :])**2).sum(-1)
    kn = min(kn, len(x) - 1)
    nb = np.argsort(d, axis=1)[:, :kn]
    ry = resid_on(X, y)
    stat = lambda xx: abs(stats.spearmanr(resid_on(X, xx), ry).statistic)
    obs, cnt = stat(x), 0
    for _ in range(NPERM):
        xp = x[nb[np.arange(len(x)), RNG.integers(0, kn, len(x))]]
        v = stat(xp)
        cnt += (np.isfinite(v) and v >= obs)
    return obs, (cnt + 1)/(NPERM + 1)


if __name__ == '__main__':
    print(f'{"set":9s} {"k":>5s} {"n":>4s} {"R2 cap":>7s} {"R2 +dQ":>7s} {"partial rho":>12s} {"kNN p":>9s}')
    for s in sys.argv[1:]:
        QD, DQ, life, f, keys = bc.load_set(s)
        med = float(np.median(life))
        for frac in (0.15, 0.20, 0.30):
            k = int(round(frac*med))
            X, dq, lf = features(QD, DQ, life, k)
            if len(dq) < 20:
                print(f'{s:9s} {k:5d} {len(dq):4d}  too few cells'); continue
            ll = np.log(lf)
            rho = stats.spearmanr(resid_on(X, dq), resid_on(X, ll))
            obs, p = knn_perm(X, dq, ll)
            A0 = np.column_stack([np.ones(len(ll))] + [rankit(X[:, j]) for j in range(X.shape[1])])
            A1 = np.column_stack([A0, rankit(dq)])
            r0 = ll - A0 @ np.linalg.lstsq(A0, ll, rcond=None)[0]
            r1 = ll - A1 @ np.linalg.lstsq(A1, ll, rcond=None)[0]
            print(f'{s:9s} {k:5d} {len(dq):4d} {1 - np.var(r0)/np.var(ll):7.3f} {1 - np.var(r1)/np.var(ll):7.3f} '
                  f'{rho.statistic:+12.3f} {p:9.1e}', flush=True)
