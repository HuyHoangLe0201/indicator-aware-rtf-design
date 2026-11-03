"""
Is the DeltaQ(V) channel a different physical process, or a saturating readout of capacity?

This is the load-bearing question of the paper.  Upsilon is NOT invariant under a monotone
change of readout -- the bearing section shows a five-fold spread across indicators that all
watch the same damage -- so a sceptic can propose that DeltaQ(V) is simply g(capacity) for
some concave g, that the low Upsilon is an artifact of that g, and that the crossing the
paper rests on is a statement about scales rather than about physics.

The proposal is falsifiable.  If DeltaQ(V) = g(capacity) pointwise for ANY function g, then
DeltaQ(V) is a deterministic transform of capacity and can carry no information about the
failure time beyond what capacity carries:

        DeltaQ(V)  is independent of  T  given  capacity.

So we test that conditional independence at fixed early cycles, controlling for capacity
non-parametrically (stratification into capacity bins, plus a spline-residual partial
correlation) so that no particular shape of g is assumed.  A significant partial association
refutes the readout explanation outright.

We run the mirror test as well -- capacity against T given DeltaQ(V) -- because the
interesting outcome is not merely that the channels differ, but which one is redundant.
"""
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline

D = np.load('../data/severson/dq_traj.npz', allow_pickle=True)
CYCLES = (100, 150, 200)
NBIN = 4
RNG = np.random.default_rng(0)


def at_cycle(k):
    """capacity, log10 Var dQ(V), and life for every cell that reaches cycle k."""
    cap, dq, life = [], [], []
    for i in range(len(D['life'])):
        L = float(D['life'][i])
        if L < k * 1.2:                      # need the cell to live well past the probe
            continue
        q = np.asarray(D['QD'][i], float)
        c, v = np.asarray(D['DQ'][i], float)
        if k >= len(q) or not np.isfinite(q[k]) or q[k] <= 0:
            continue
        if k < c.min() or k > c.max():
            continue
        cap.append(q[k])
        dq.append(float(np.interp(k, c, v)))
        life.append(L)
    return np.array(cap), np.array(dq), np.array(life)


def spline_resid(x, y):
    """Residual of y after removing an arbitrary smooth function of x."""
    o = np.argsort(x)
    xs, ys = x[o], y[o]
    xs = xs + np.arange(len(xs)) * 1e-9          # break ties for the spline
    s = UnivariateSpline(xs, ys, s=len(xs) * np.var(ys) * 0.25, k=3)
    r = np.empty_like(y)
    r[o] = ys - s(xs)
    return r


def stratified_spearman(ctrl, x, y, nbin=NBIN):
    """Spearman of x vs y within bins of the control variable, pooled by Fisher z."""
    edges = np.quantile(ctrl, np.linspace(0, 1, nbin + 1))
    edges[-1] += 1e-12
    zs, ws, ns = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (ctrl >= a) & (ctrl < b)
        if m.sum() < 8:
            continue
        r = stats.spearmanr(x[m], y[m]).statistic
        if not np.isfinite(r):
            continue
        r = np.clip(r, -0.999, 0.999)
        zs.append(np.arctanh(r)); ws.append(m.sum() - 3); ns.append(m.sum())
    if not zs:
        return np.nan, np.nan, []
    zs, ws = np.array(zs), np.array(ws, float)
    zbar = float((zs * ws).sum() / ws.sum())
    se = float(1 / np.sqrt(ws.sum()))
    p = 2 * stats.norm.sf(abs(zbar / se))
    return float(np.tanh(zbar)), p, ns


def perm_test(ctrl, x, y, nperm=20000):
    """Permute x within control bins: the exact null 'x adds nothing given ctrl'."""
    edges = np.quantile(ctrl, np.linspace(0, 1, NBIN + 1)); edges[-1] += 1e-12
    idx = [np.where((ctrl >= a) & (ctrl < b))[0] for a, b in zip(edges[:-1], edges[1:])]
    idx = [j for j in idx if len(j) >= 8]
    obs = abs(stratified_spearman(ctrl, x, y)[0])
    cnt = 0
    for _ in range(nperm):
        xp = x.copy()
        for j in idx:
            xp[j] = x[RNG.permutation(j)]
        v = abs(stratified_spearman(ctrl, xp, y)[0])
        cnt += (np.isfinite(v) and v >= obs)
    return (cnt + 1) / (nperm + 1)


print(__doc__)
print('=' * 78)
for k in CYCLES:
    cap, dq, life = at_cycle(k)
    ll = np.log(life)
    print()
    print(f'--- probe at cycle {k}:  n = {len(cap)} cells ---')
    print(f'    marginal Spearman  capacity vs log T : {stats.spearmanr(cap, ll).statistic:+.3f}')
    print(f'    marginal Spearman  dQ(V)    vs log T : {stats.spearmanr(dq, ll).statistic:+.3f}')
    print(f'    Spearman  dQ(V) vs capacity          : {stats.spearmanr(dq, cap).statistic:+.3f}')

    r, p, ns = stratified_spearman(cap, dq, ll)
    pp = perm_test(cap, dq, ll)
    print(f'    dQ(V) vs log T GIVEN capacity  : rho = {r:+.3f}  p = {p:.2e}  '
          f'perm p = {pp:.2e}   bins {ns}')
    rs = stats.spearmanr(spline_resid(cap, dq), spline_resid(cap, ll))
    print(f'      spline-residual check        : rho = {rs.statistic:+.3f}  p = {rs.pvalue:.2e}')

    r2, p2, ns2 = stratified_spearman(dq, cap, ll)
    pp2 = perm_test(dq, cap, ll)
    print(f'    capacity vs log T GIVEN dQ(V)  : rho = {r2:+.3f}  p = {p2:.2e}  '
          f'perm p = {pp2:.2e}   bins {ns2}')
    rs2 = stats.spearmanr(spline_resid(dq, cap), spline_resid(dq, ll))
    print(f'      spline-residual check        : rho = {rs2.statistic:+.3f}  p = {rs2.pvalue:.2e}')

print()
print('=' * 78)
print('Reading: if dQ(V) were g(capacity), the FIRST partial would be null at every cycle.')
print('A significant first partial refutes the readout explanation.  If in addition the')
print('SECOND partial is null, capacity is the redundant channel, not dQ(V).')
