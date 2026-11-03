"""
How far does Upsilon actually move across the Box-Cox family, once the orientation artefact
of scale_check2.py is removed?

scale_check2 swept the family without first orienting each indicator to increase with damage,
so for a falling indicator the family's singular point sat at end of life and the reported
spans were inflated.  This repeats the sweep with orient() applied, using the nonparametric
estimator, and reports the honest span per indicator.  It is the number quoted in the
Limitations section, so it needs to be the corrected one.
"""
import numpy as np, os
from scale_check import boxcox, load_cmapss, CACHE
from scale_check2 import ups_np
from scale_select import orient, LAM, THR

HERE = os.path.dirname(__file__)

if __name__ == '__main__':
    z = np.load(CACHE, allow_pickle=True)
    XF = {k: z[k].item() for k in z.files}
    CF = load_cmapss()

    series = {}
    d = np.load(os.path.join(HERE, '..', 'data', 'severson', 'dq_traj.npz'), allow_pickle=True)
    cap = []
    for q in d['QD']:
        q = np.asarray(q, float)
        q = q[np.isfinite(q) & (q > .2) & (q < 1.5)]
        if len(q) >= 60:
            cap.append(np.median(np.lib.stride_tricks.sliding_window_view(
                np.pad(q, 4, mode='edge'), 9), axis=1))
    series[('battery', 'bulk capacity')] = cap
    for ind in ['RMS', 'peak', 'kurtosis', 'HF 2-5kHz', 'HF 5-10kHz', 'spec centroid']:
        series[('bearing', ind)] = [np.asarray(XF[b][ind], float) for b in XF]
    for ind, us in CF.items():
        series[('C-MAPSS', ind)] = us

    hdr = (f'{"domain":9s} {"indicator":15s} ' + ' '.join(f'{n:>9s}' for n, _ in LAM)
           + f' {"span":>8s} {"thresh":>7s}  verdict across the whole family')
    print(hdr); print('-'*len(hdr))
    spans = []
    for (dom, ind), us in series.items():
        row = []
        for _, lam in LAM:
            v = np.array([ups_np(boxcox(orient(h), lam)) for h in us])
            v = v[np.isfinite(v)]
            row.append(np.median(v) if len(v) else np.nan)
        row = np.array(row)
        th = THR[dom]
        span = np.nanmax(row)/max(np.nanmin(row), 1e-12)
        spans.append(span)
        n_corner = int(np.sum(row >= th))
        verdict = ('corner at all 5' if n_corner == len(row)
                   else 'interior at all 5' if n_corner == 0
                   else f'FLIPS ({n_corner}/5 corner)')
        print(f'{dom:9s} {ind:15s} ' + ' '.join(f'{x:9.2f}' for x in row)
              + f' {span:7.1f}x {th:7.2f}  {verdict}')
    print(f'\nlargest span across the unconstrained family: {np.nanmax(spans):.1f}x')
