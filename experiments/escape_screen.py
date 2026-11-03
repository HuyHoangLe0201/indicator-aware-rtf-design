"""
Screening for escape channels by conditional independence, on turbofans and bearings.

On the Severson cells the escape channel announces itself: controlling for the whole capacity
trajectory, DeltaQ(V) still carries rank correlation -0.80 with log T at cycle 100 and raises
the explained variance from 0.53 to 0.84.  That is what an escape channel looks like as a
measurement, and it suggests a screening rule that does not require guessing at physics:

    a candidate is an escape channel only if it carries information about T beyond the
    damage channel, early, when the damage channel is still uninformative.

The paper claims turbofans and bearings have no such channel, and asserts in passing that
'fusion cannot create information that none of the inputs carries early'.  Both are testable.
Here every candidate indicator is given its best shot -- its own four-feature trajectory
summary, not merely its level -- against the damage channel's four-feature summary, and we
report the partial rank correlation, a local-permutation p-value that assumes no functional
form, and the incremental explained variance.  The battery numbers are printed alongside as
the effect size a real escape channel produces.

A null here is informative only in proportion to the power behind it, so the achievable
effect size is reported for each domain rather than left implicit.
"""
import numpy as np, os
from scipy import stats

ROOT = r'D:\bách khoa Đà Nẵng\nghiencuukhoahoc\baotri\data'
RNG = np.random.default_rng(7)


# ----------------------------------------------------------------------------- helpers
def rankit(a):
    a = np.asarray(a, float)
    return stats.norm.ppf((stats.rankdata(a) - 0.5)/len(a))


def design(X, quad=True):
    R = np.column_stack([rankit(X[:, j]) for j in range(X.shape[1])])
    cols = [np.ones(len(R))] + [R[:, j] for j in range(R.shape[1])]
    if quad:
        for j in range(R.shape[1]):
            for l in range(j, R.shape[1]):
                cols.append(R[:, j]*R[:, l])
    return np.column_stack(cols)


def resid(X, y, quad=True):
    A = design(X, quad)
    return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]


def r2(X, y, quad=True):
    return 1 - np.var(resid(X, y, quad))/np.var(y)


def knn_perm(Xc, x, y, kn=10, nperm=4000):
    """Permute the candidate among nearest neighbours in the damage-channel control space."""
    Z = np.column_stack([rankit(Xc[:, j]) for j in range(Xc.shape[1])])
    d = ((Z[:, None, :] - Z[None, :, :])**2).sum(-1)
    nb = np.argsort(d, axis=1)[:, :kn]
    ry = resid(Xc, y)
    stat = lambda xx: abs(stats.spearmanr(resid(Xc, xx), ry).statistic)
    obs, cnt = stat(x), 0
    for _ in range(nperm):
        xp = x[nb[np.arange(len(x)), RNG.integers(0, kn, len(x))]]
        v = stat(xp)
        cnt += (np.isfinite(v) and v >= obs)
    return obs, (cnt + 1)/(nperm + 1)


def summarize(tr, k):
    """Four-feature summary of a trajectory up to index k: level, drift, rate, curvature."""
    base = float(np.median(tr[:max(2, k//10)]))
    lvl = float(np.median(tr[max(0, k-3):k+1]))
    rate = float(tr[k] - tr[max(0, k-20)])/20.0
    early = float(tr[k//2] - base)/max(k//2, 1)
    return [lvl, lvl - base, rate, rate - early]


def screen(tag, units, damage_key, cand_keys, probes, min_life_mult=1.3):
    """units: list of dicts {key -> trajectory array, 'T': life}."""
    print()
    print('=' * 78)
    print(f'{tag}:  damage channel = {damage_key},  {len(cand_keys)} candidates,  '
          f'{len(units)} units')
    for k in probes:
        use = [u for u in units if u['T'] >= k*min_life_mult and len(u[damage_key]) > k]
        if len(use) < 20:
            print(f'  probe {k}: only {len(use)} units, skipped')
            continue
        y = np.log(np.array([u['T'] for u in use], float))
        Xd = np.array([summarize(u[damage_key], k) for u in use])
        base_r2 = r2(Xd, y)
        # smallest |rho| this n could resolve at p=0.05, two-sided
        # the controls are the full quadratic design of the damage summary (all columns but the constant)
        det = np.tanh(1.96/np.sqrt(max(len(use) - (design(Xd).shape[1] - 1) - 3, 1)))
        print()
        print(f'  --- probe at cycle {k}:  n = {len(use)},  damage channel alone '
              f'R^2(log T) = {base_r2:.3f},  resolvable |rho| >= {det:.2f} ---')
        print(f'      {"candidate":22s} {"rho|damage":>11s} {"perm p":>9s} {"R2 with":>9s} '
              f'{"dR2":>7s}')
        rows = []
        for ck in cand_keys:
            x = np.array([np.median(u[ck][max(0, k-3):k+1]) for u in use], float)
            Xc = np.array([summarize(u[ck], k) for u in use])
            rho, p = knn_perm(Xd, x, y)
            joint = r2(np.column_stack([Xd, Xc]), y, quad=False)
            rows.append((abs(rho), ck, rho, p, joint, joint - r2(Xd, y, quad=False)))
        for _, ck, rho, p, joint, d in sorted(rows, reverse=True):
            print(f'      {ck:22s} {rho:11.3f} {p:9.4f} {joint:9.3f} {d:+7.3f}')
        best = max(rows)
        print(f'      strongest candidate: {best[1]}  |rho| = {best[0]:.3f}  '
              f'p = {best[3]:.4f}  (Bonferroni over {len(cand_keys)}: '
              f'p_adj = {min(1.0, best[3]*len(cand_keys)):.3f})')


# ----------------------------------------------------------------------------- C-MAPSS
def cmapss():
    d = np.loadtxt(os.path.join(ROOT, 'train_FD001.txt'))
    uid = d[:, 0].astype(int)
    keep = [j for j in range(5, d.shape[1]) if len(np.unique(d[:, j])) > 2]
    Z = (d[:, keep] - d[:, keep].mean(0))/d[:, keep].std(0)
    # leading principal component of the continuously varying sensors = the fused indicator
    w = np.linalg.svd(Z - Z.mean(0), full_matrices=False)[2][0]
    pc = Z @ w
    if np.corrcoef(pc, d[:, 1])[0, 1] < 0:            # orient to increase with damage
        w, pc = -w, -pc
    units = []
    for u in np.unique(uid):
        m = uid == u
        rec = {'T': float(m.sum()), 'PC1': pc[m]}
        for j, col in enumerate(keep):
            s = d[m, col]
            if np.corrcoef(s, np.arange(len(s)))[0, 1] < 0:
                s = -s
            rec[f's{col-4}'] = s
        units.append(rec)
    cands = [f's{c-4}' for c in keep]
    screen('C-MAPSS FD001', units, 'PC1', cands, probes=(40, 60, 80))


if __name__ == '__main__':
    print(__doc__)
    cmapss()
