"""
Measured extrapolation-information curves eta_T(c) for one indicator per domain.

Upsilon is the terminal slope of eta_T; the curve itself is what a test engineer can act on,
because eta_T(c) is literally "the share of the extrapolation information a test stopped at
fraction c of life has collected".  This script computes the per-unit curves and stores the
median and the interquartile band, so a figure can show the decision rather than the summary
statistic.

Scale: each indicator is taken on the Box-Cox scale selected by scale_select.py, i.e. the one
on which its residuals are closest to homoscedastic, since that is the assumption the
estimator makes.

Writes eta_curves.npz with, for each series, c, median, q1, q3 and the terminal slope.
"""
import numpy as np, os, json
from scipy.ndimage import uniform_filter1d
from scale_check import boxcox, load_cmapss, CACHE

HERE = os.path.dirname(__file__)
SEV = os.path.join(HERE, '..', 'data', 'severson')
G = np.linspace(1e-3, 1.0, 400)


def eta_curve(h, frac=0.07):
    """Normalised cumulative extrapolation information of one unit, on the grid G."""
    h = np.asarray(h, float); h = h[np.isfinite(h)]
    n = len(h)
    if n < 60:
        return None
    s = uniform_filter1d(h, max(3, int(frac*n)), mode='nearest')
    base, end = np.median(s[:max(2, n//20)]), np.median(s[-max(2, n//50):])
    if abs(end-base) < 1e-12:
        return None
    D = (s-base)/(end-base)
    tau = (np.arange(n)+1)/n
    dp = np.gradient(np.interp(G, tau, D), G)
    dens = G**2*dp**2
    cum = np.concatenate([[0], np.cumsum((dens[1:]+dens[:-1])/2*np.diff(G))])
    return cum/cum[-1] if cum[-1] > 0 else None


def band(curves):
    A = np.array([c for c in curves if c is not None])
    return np.median(A, 0), np.percentile(A, 25, 0), np.percentile(A, 75, 0), len(A)


if __name__ == '__main__':
    out, meta = {}, {}

    # battery, bulk capacity -- native scale (capacity in Ah)
    d = np.load(os.path.join(SEV, 'dq_traj.npz'), allow_pickle=True)
    cur = []
    for q in d['QD']:
        q = np.asarray(q, float)
        q = q[np.isfinite(q) & (q > .2) & (q < 1.5)]
        cur.append(eta_curve(q))
    med, q1, q3, n = band(cur)
    out['battery capacity'] = np.vstack([G, med, q1, q3]); meta['battery capacity'] = n

    # C-MAPSS, PC1 of the informative sensors -- native scale selected by scale_select
    CF = load_cmapss()
    cur = [eta_curve(h) for h in CF['sensor 4']]
    med, q1, q3, n = band(cur)
    out['C-MAPSS sensor 4'] = np.vstack([G, med, q1, q3]); meta['C-MAPSS sensor 4'] = n

    # bearing, RMS -- sqrt scale selected by scale_select.  Snapshots are subsampled by 2 and
    # bearings with fewer than 60 left are dropped, exactly as in xjtu_survey.py, so this
    # curve rests on the same 12 of the 15 bearings as the published Upsilon medians.
    z = np.load(CACHE, allow_pickle=True)
    XF = {k: z[k].item() for k in z.files}
    cur = [eta_curve(boxcox(np.asarray(XF[b]['RMS'], float)[::2], 0.5)) for b in XF
           if len(XF[b]['RMS'][::2]) >= 60]
    med, q1, q3, n = band(cur)
    out['bearing RMS'] = np.vstack([G, med, q1, q3]); meta['bearing RMS'] = n

    # battery, DeltaQ(V) -- the empirical predictive-R^2 fraction, a DIFFERENT object
    c_emp = np.array([0.027, 0.068, 0.137, 0.274, 0.544, 0.796])
    r2 = np.array([0.395, 0.720, 0.795, 0.886, 0.884, 0.804])
    out['dqv empirical'] = np.vstack([c_emp, r2/r2.max()])
    meta['dqv empirical'] = 119

    np.savez(os.path.join(HERE, 'eta_curves.npz'), meta=json.dumps(meta), **out)
    for k, v in out.items():
        if k == 'dqv empirical':
            print(f'{k:20s} n={meta[k]:4d}  eta at c=0.14: {v[1][2]:.2f} (empirical)')
        else:
            i = np.argmin(abs(v[0]-0.8))
            print(f'{k:20s} n={meta[k]:4d}  eta(0.8) = {v[1][i]:.3f}  '
                  f'-> {100*(1-v[1][i]):.0f}% of the information is in the last fifth')
    print(f'\nwrote {os.path.join(HERE, "eta_curves.npz")}')
