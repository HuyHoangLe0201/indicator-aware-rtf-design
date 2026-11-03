"""
Stating the crossing without estimating Upsilon.

direct_cstar.py tried to replace Upsilon by optimising the budgeted risk on the measured
curve, and failed for an instructive reason: at the last measured window only 63 of 119 cells
are still alive, and there is no observation at all between 80% and 100% of median life, which
is exactly where Upsilon lives.  Any fleet with dispersed lives has its sparsest data in the
terminal region, so an empirical curve cannot resolve a terminal derivative.  That is why a
parametric estimator is needed for the convex channel -- not convenience.

The same data support a stronger claim from the other side.  What the paper's crossing really
asserts is that DeltaQ(V) has collected essentially all of its predictive power long before
end of life while bulk capacity has not.  Saturation is an interior property.  It is visible
in the region where nearly every cell is alive, and it needs neither a terminal derivative nor
a choice of parameterization.

We locate, for each channel, the fraction of median life at which its predictive power first
reaches 95% of its own maximum, report how many cells stand behind that point, and bootstrap
the fraction over cells.
"""
import numpy as np
from scipy import stats
import channel_abs
from channel_abs import collect, loo_r2

MGRID = [20, 40, 60, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
channel_abs.MGRID = MGRID          # collect() reads the module global, not our copy
NBOOT = 600
RNG = np.random.default_rng(3)

rows = collect()
T = np.array([r[0] for r in rows], float)
tmed = float(np.median(T))
print(f'cells: {len(rows)},  median life {tmed:.0f} cycles')
print()


def series(idx, ch, use_loo):
    """R^2 against log T at each window, over the given cell subset."""
    out = []
    for m in MGRID:
        xs, ys = [], []
        for i in idx:
            d = rows[i][ch]
            if m in d:
                xs.append(d[m]); ys.append(np.log(rows[i][0]))
        if len(xs) < 25:
            out.append((m, np.nan, len(xs))); continue
        if use_loo:
            v = loo_r2(xs, ys)
        else:
            v = stats.spearmanr(xs, ys).statistic**2
        out.append((m, float(v), len(xs)))
    return out


def sat_fraction(ser, thr=0.95):
    """First window reaching thr of the maximum, as a fraction of median life."""
    v = np.array([x[1] for x in ser], float)
    m = np.array([x[0] for x in ser], float)
    ok = np.isfinite(v)
    if not ok.any() or np.nanmax(v) <= 0:
        return np.nan, np.nan
    peak = np.nanmax(v)
    hit = np.where(ok & (v >= thr*peak))[0]
    if not len(hit):
        return np.nan, np.nan
    j = hit[0]
    return m[j]/tmed, np.array([x[2] for x in ser])[j]


print('LEAVE-ONE-OUT HUBER R^2 against log T, by absolute window')
print(f'{"m":>6s} {"c=m/Tmed":>9s} {"alive":>6s} {"capacity":>9s} {"dQ(V)":>8s}')
sa = series(range(len(rows)), 1, True)
sb = series(range(len(rows)), 2, True)
for (m, va, na), (_, vb, nb) in zip(sa, sb):
    print(f'{m:6d} {m/tmed:9.3f} {na:6d} {va:9.3f} {vb:8.3f}')

print()
for nm, ser in (('bulk capacity', sa), ('DeltaQ(V)', sb)):
    f, n = sat_fraction(ser)
    print(f'  {nm:14s} reaches 95% of its own maximum at c = {f:.3f} '
          f'({f*100:.0f}% of median life), on {int(n)} of {len(rows)} cells')

print()
print(f'bootstrap over cells, {NBOOT} resamples, Spearman^2 as the fast statistic')
res = {1: [], 2: []}
for _ in range(NBOOT):
    idx = RNG.integers(0, len(rows), len(rows))
    for ch in (1, 2):
        f, _n = sat_fraction(series(idx, ch, False))
        if np.isfinite(f):
            res[ch].append(f)
for ch, nm in ((1, 'bulk capacity'), (2, 'DeltaQ(V)')):
    a = np.array(res[ch])
    lo, hi = np.percentile(a, [2.5, 97.5])
    print(f'  {nm:14s} c_sat = {np.median(a):.3f}   95% CI [{lo:.3f}, {hi:.3f}]   '
          f'n_boot = {len(a)}')
a, b = np.array(res[1]), np.array(res[2])
k = min(len(a), len(b))
print()
print(f'  P(dQ(V) saturates earlier than capacity) = '
      f'{float(np.mean(b[:k] < a[:k])):.3f} over paired resamples')
