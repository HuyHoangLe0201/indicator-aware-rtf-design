"""
Hardening the readout test.

readout_test.py refutes 'DeltaQ(V) = g(capacity at the same cycle)'.  A better sceptic asks
for more: the damage state need not be the instantaneous capacity, it could be the whole
capacity trajectory so far -- level, fade rate, curvature.  If DeltaQ(V) is a readout of THAT,
it is still the same process seen through a different transducer, which is exactly what the
paper's physical rule forbids as an escape channel.

So we control for a four-feature summary of the capacity curve up to the probe cycle and ask
whether DeltaQ(V) still predicts the failure time.  Controls are applied on ranks with a
quadratic expansion, which absorbs any smooth monotone map of those features, and the null is
enforced by permutation within a k-nearest-neighbour ball in the control space -- a local
permutation test that assumes no functional form at all.
"""
import numpy as np
from scipy import stats

D = np.load('../data/severson/dq_traj.npz', allow_pickle=True)
RNG = np.random.default_rng(1)


def features(k):
    """capacity-trajectory summary at probe cycle k, plus dQ(V) and life."""
    X, dq, life = [], [], []
    for i in range(len(D['life'])):
        L = float(D['life'][i])
        if L < k * 1.2:
            continue
        q = np.asarray(D['QD'][i], float)
        c, v = np.asarray(D['DQ'][i], float)
        if k >= len(q) or not np.isfinite(q[k]) or q[k] <= 0:
            continue
        if k < c.min() or k > c.max():
            continue
        q1 = float(np.median(q[2:12]))                       # settled initial capacity
        lvl = float(q[k])
        rate = float(q[k] - q[k - 50]) / 50.0                # recent fade rate
        early = float(q[k // 2] - q1) / max(k // 2, 1)       # early fade rate
        curv = rate - early                                  # change in fade rate
        X.append([lvl, lvl - q1, rate, curv])
        dq.append(float(np.interp(k, c, v)))
        life.append(L)
    return np.array(X), np.array(dq), np.array(life)


def rankit(a):
    a = np.asarray(a, float)
    return stats.norm.ppf((stats.rankdata(a) - 0.5) / len(a))


def resid_on(X, y):
    """Residual of y on a quadratic expansion of the rank-transformed controls."""
    R = np.column_stack([rankit(X[:, j]) for j in range(X.shape[1])])
    cols = [np.ones(len(y))] + [R[:, j] for j in range(R.shape[1])]
    for j in range(R.shape[1]):
        for l in range(j, R.shape[1]):
            cols.append(R[:, j] * R[:, l])
    A = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def knn_perm(X, x, y, kn=12, nperm=20000):
    """Permute x among each cell's kn nearest neighbours in standardized control space."""
    Z = np.column_stack([rankit(X[:, j]) for j in range(X.shape[1])])
    d = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    nb = np.argsort(d, axis=1)[:, :kn]
    stat = lambda xx: abs(stats.spearmanr(resid_on(X, xx), resid_on(X, y)).statistic)
    obs, cnt = stat(x), 0
    for _ in range(nperm):
        xp = x[nb[np.arange(len(x)), RNG.integers(0, kn, len(x))]]
        v = stat(xp)
        cnt += (np.isfinite(v) and v >= obs)
    return obs, (cnt + 1) / (nperm + 1)


print(__doc__)
print('=' * 78)
for k in (100, 150, 200):
    X, dq, life = features(k)
    ll = np.log(life)
    print()
    print(f'--- probe at cycle {k}:  n = {len(dq)} cells,  controls = '
          f'[capacity, fade so far, recent rate, curvature] ---')

    # how much of log T do the capacity features explain on their own?
    r_ll = resid_on(X, ll)
    print(f'    capacity trajectory alone explains R^2 = '
          f'{1 - np.var(r_ll)/np.var(ll):.3f} of var(log T)')

    r_dq = resid_on(X, dq)
    print(f'    capacity trajectory alone explains R^2 = '
          f'{1 - np.var(r_dq)/np.var(dq):.3f} of var(dQ(V))')

    rho = stats.spearmanr(r_dq, r_ll)
    print(f'    dQ(V) vs log T GIVEN the whole capacity trajectory : '
          f'rho = {rho.statistic:+.3f}  p = {rho.pvalue:.2e}')
    obs, p = knn_perm(X, dq, ll)
    print(f'      kNN local permutation (no functional form)      : '
          f'|rho| = {obs:.3f}  p = {p:.2e}')

    # incremental R^2 of adding dQ(V)
    A0 = np.column_stack([np.ones(len(ll))] + [rankit(X[:, j]) for j in range(X.shape[1])])
    A1 = np.column_stack([A0, rankit(dq)])
    r0 = ll - A0 @ np.linalg.lstsq(A0, ll, rcond=None)[0]
    r1 = ll - A1 @ np.linalg.lstsq(A1, ll, rcond=None)[0]
    print(f'    R^2(log T): capacity only {1-np.var(r0)/np.var(ll):.3f}  ->  '
          f'+dQ(V) {1-np.var(r1)/np.var(ll):.3f}')

print()
print('=' * 78)
print('If dQ(V) were any readout of the capacity trajectory, both partials would be null.')
