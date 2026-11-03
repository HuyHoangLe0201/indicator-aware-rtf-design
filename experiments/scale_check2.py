"""
Scale sweep, second arm: the NONPARAMETRIC estimator.

scale_check.py sweeps the Box-Cox scale using the parametric stretched-exponential fit.  That
fit collapses to Upsilon = 0 on the three frequency-domain bearing indicators -- the same
saturation artefact already documented for PRONOSTIA -- so for bearings the published numbers
come from the nonparametric estimator instead (xjtu_survey.profile).  This script repeats the
sweep with that estimator, so the robustness claim is tested with the estimator that actually
produced the numbers.

Reads the cached features written by scale_check.py; prints the sweep and writes
scale_check_np.npz.
"""
import numpy as np, os, json
from scipy.ndimage import uniform_filter1d
from scale_check import boxcox, load_cmapss, CACHE, LAM, THRESH

HERE = os.path.dirname(__file__)


def ups_np(h, frac=0.07):
    """Nonparametric Upsilon: normalise, smooth, differentiate, integrate.  Identical to
    xjtu_survey.profile but returning Upsilon only."""
    h = np.asarray(h, float); h = h[np.isfinite(h)]
    n = len(h)
    if n < 60:
        return np.nan
    s = uniform_filter1d(h, max(3, int(frac*n)), mode='nearest')
    base, end = np.median(s[:max(2, n//20)]), np.median(s[-max(2, n//50):])
    if abs(end-base) < 1e-12:
        return np.nan
    D = (s-base)/(end-base)
    tau = (np.arange(n)+1)/n
    g = np.linspace(1e-3, 1.0, 400)
    dp = np.gradient(np.interp(g, tau, D), g)
    dens = g**2*dp**2
    cum = np.concatenate([[0], np.cumsum((dens[1:]+dens[:-1])/2*np.diff(g))])
    return float(dens[-1]/cum[-1]) if cum[-1] > 0 else np.nan


if __name__ == '__main__':
    z = np.load(CACHE, allow_pickle=True)
    XF = {k: z[k].item() for k in z.files}
    print(f'{len(XF)} bearings from cache')
    CF = load_cmapss()

    series = {}
    for ind in ['RMS', 'peak', 'kurtosis', 'HF 2-5kHz', 'HF 5-10kHz', 'spec centroid']:
        series[('bearing', ind)] = [np.asarray(XF[b][ind], float) for b in XF]
    for ind, us in CF.items():
        series[('C-MAPSS', ind)] = us

    hdr = (f'{"domain":9s} {"indicator":15s} ' + ' '.join(f'{n:>9s}' for n, _ in LAM)
           + f' {"span":>9s}  verdict')
    print('\n' + hdr); print('-'*len(hdr))
    store = {}
    for (dom, ind), us in series.items():
        row, per_unit = [], None
        for _, lam in LAM:
            v = np.array([ups_np(boxcox(h, lam)) for h in us])
            v = v[np.isfinite(v)]
            if lam == 1.0:
                per_unit = v
            row.append(np.median(v) if len(v) else np.nan)
        row = np.array(row)
        store[f'{dom}|{ind}'] = row
        store[f'{dom}|{ind}|units'] = per_unit
        ok = np.all(row >= THRESH)
        span = np.nanmax(row)/max(np.nanmin(row), 1e-9)
        print(f'{dom:9s} {ind:15s} ' + ' '.join(f'{x:9.2f}' for x in row)
              + f' {span:8.1f}x  ' + ('corner in all' if ok else 'NOT STABLE'))

    np.savez(os.path.join(HERE, 'scale_check_np.npz'),
             lam=json.dumps([n for n, _ in LAM]), **store)
    print(f'\nthreshold {THRESH}. Compare with scale_check.py (parametric arm).')
