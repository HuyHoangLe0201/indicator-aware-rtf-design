"""
Is the shape estimator of Upsilon stable under a monotone change of the indicator's scale?

Upsilon is defined for the degradation function D of the model X = D(tau) + sigma W.  It is
therefore a property of the scale on which the noise is additive, and a monotone transform
h -> g(h) of the recorded indicator changes it.  For an indicator recorded in physical units
(RMS in g, a turbofan sensor in engineering units) the native scale is the defensible default.
For a DERIVED feature -- log10 Var[Q_m(V) - Q_10(V)] -- there is no native scale, and the
estimator is correspondingly ill-posed.

This script quantifies the sensitivity for every indicator in the survey by sweeping a
Box-Cox family, lambda in {0 (log), 1/2, 1 (native), 2}, and reporting whether the corner
verdict (Upsilon >= 1.18) survives the whole sweep.

Writes scale_check.npz and prints a table.
"""
import numpy as np, os, glob, json
from scipy.ndimage import uniform_filter1d
from upsilon_dist import measure, CMAPSS, XJTU, xjtu_features

HERE = os.path.dirname(__file__)
CACHE = os.path.join(HERE, 'xjtu_features.npz')
THRESH = 1.18
LAM = [('log', 0.0), ('sqrt', 0.5), ('native', 1.0), ('square', 2.0)]


def boxcox(h, lam):
    h = np.asarray(h, float)
    s = h - np.nanmin(h) + 1e-6*(np.nanmax(h)-np.nanmin(h) + 1e-12)   # shift to positive
    if lam == 0.0:
        return np.log(s)
    return (s**lam - 1.0)/lam


def load_xjtu():
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return {k: z[k].item() for k in z.files}
    bds = sorted(glob.glob(os.path.join(XJTU, '*', 'Bearing*')))
    F = {}
    for bd in bds:
        nm = os.path.basename(os.path.dirname(bd)) + '/' + os.path.basename(bd)
        f = xjtu_features(bd)
        if len(f['RMS']) >= 60:
            F[nm] = f
        print(f'  {nm:28s} {len(f["RMS"]):5d} snapshots')
    np.savez(CACHE, **{k: np.array(v, dtype=object) for k, v in F.items()})
    return F


def load_cmapss():
    d = np.loadtxt(os.path.join(CMAPSS, 'train_FD001.txt'))
    uid = d[:, 0].astype(int)
    S = {'sensor 2': 6, 'sensor 3': 7, 'sensor 4': 8, 'sensor 7': 11,
         'sensor 8': 12, 'sensor 9': 13}
    units = [d[uid == u] for u in np.unique(uid)]
    return {nm: [u[:, j] for u in units] for nm, j in S.items()}


if __name__ == '__main__':
    print('loading XJTU bearings ...')
    XF = load_xjtu()
    print(f'{len(XF)} bearings\nloading C-MAPSS ...')
    CF = load_cmapss()

    series = {}
    for ind in ['RMS', 'peak', 'kurtosis', 'HF 2-5kHz', 'HF 5-10kHz', 'spec centroid']:
        series[('bearing', ind)] = [uniform_filter1d(
            np.asarray(XF[b][ind], float), max(3, int(.07*len(XF[b][ind]))), mode='nearest')
            for b in XF]
    for ind, us in CF.items():
        series[('C-MAPSS', ind)] = us

    hdr = f'{"domain":9s} {"indicator":15s} ' + ' '.join(f'{n:>9s}' for n, _ in LAM) \
          + f' {"span":>7s}  verdict'
    print('\n' + hdr); print('-'*len(hdr))
    store = {}
    for (dom, ind), us in series.items():
        row = []
        for _, lam in LAM:
            v = np.array([measure(boxcox(h, lam)) for h in us])
            v = v[np.isfinite(v)]
            row.append(np.median(v) if len(v) else np.nan)
        row = np.array(row)
        store[f'{dom}|{ind}'] = row
        ok = np.all(row >= THRESH)
        span = np.nanmax(row)/max(np.nanmin(row), 1e-9)
        print(f'{dom:9s} {ind:15s} ' + ' '.join(f'{x:9.2f}' for x in row)
              + f' {span:6.1f}x  ' + ('corner in all' if ok else 'NOT STABLE'))

    np.savez(os.path.join(HERE, 'scale_check.npz'),
             lam=json.dumps([n for n, _ in LAM]), **store)
    print(f'\nthreshold {THRESH}; "corner in all" means the run-to-failure verdict does not '
          f'depend on the scale choice.')
