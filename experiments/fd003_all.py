"""Every varying sensor of C-MAPSS FD003, swept for stability.

Section VII-H says the terminal concentration across the sixteen varying sensors of FD003 runs
from 2.45 to 10.95, that every value is stable to within 1.5x across the smoothing sweep, and
that the lowest sits at more than twice the turbofan threshold. fd003_stability.py reports a
handful of named channels rather than the whole set, so this sweeps all of them.
"""
import os

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import least_squares

from upsilon_survey import ROOT

FR = (0.03, 0.07, 0.15)
THRESH = 1.11


def ups(tr, smooth):
    """Parametric arm: fit D = tau**beta on the window and return Ups = 2 beta + 1.

    This is fd003_stability's estimator verbatim, so the sweep is comparable with the three
    channels that script names. The nonparametric arm runs lower on C-MAPSS, as Section VII-C
    already reports for FD001.
    """
    y = uniform_filter1d(np.asarray(tr, float), max(3, int(len(tr)*smooth)), mode='nearest')
    b = np.median(y[:max(2, len(y)//10)])
    e = np.median(y[-max(2, len(y)//20):])
    if abs(e - b) < 1e-12:
        return np.nan
    D = (y - b)/(e - b)
    t = (np.arange(len(y)) + 1)/len(y)
    m = (D > -0.4) & (D < 1.6)
    if m.sum() < 40:
        return np.nan
    r = least_squares(lambda p: np.clip(t[m], 1e-9, 1.)**p[0] - D[m], [3.],
                      bounds=([0.05], [15.]), max_nfev=400)
    return 2*float(r.x[0]) + 1


if __name__ == '__main__':
    d = np.loadtxt(os.path.join(ROOT, 'train_FD003.txt'))
    uid = d[:, 0].astype(int)
    keep = [j for j in range(5, d.shape[1]) if len(np.unique(d[:, j])) > 2]
    Z = (d[:, keep] - d[:, keep].mean(0))/(d[:, keep].std(0) + 1e-12)
    w = np.linalg.svd(Z - Z.mean(0), full_matrices=False)[2][0]
    pc = Z @ w
    if np.corrcoef(pc, d[:, 1])[0, 1] < 0:
        w, pc = -w, -pc
    units = [uid == u for u in np.unique(uid)]
    print(f'FD003: {len(keep)} varying sensors, {len(units)} units')
    print(f'{"channel":12s} ' + ' '.join(f'{("Ups "+str(f)):>10s}' for f in FR)
          + f' {"drift":>7s}  verdict')
    print('-'*62)
    meds, drifts = [], []
    for col in keep + ['PC1']:
        vals = []
        for f in FR:
            if col == 'PC1':
                v = [ups(pc[m], f) for m in units]
            else:
                s = d[:, col]
                v = [ups(s[m] if np.corrcoef(s[m], np.arange(m.sum()))[0, 1] >= 0
                         else -s[m], f) for m in units]
            v = np.array(v, float)
            vals.append(np.nanmedian(v))
        vals = np.array(vals)
        drift = np.nanmax(vals)/max(np.nanmin(vals), 1e-9)
        lab = 'PC1' if col == 'PC1' else f's{col-4}'
        med = float(np.nanmedian(vals))
        if lab != 'PC1':
            meds.append(med)
            drifts.append(drift)
        print(f'{lab:12s} ' + ' '.join(f'{v:10.2f}' for v in vals)
              + f' {drift:6.2f}x  ' + ('corner' if med >= THRESH else 'INTERIOR'))
    m = np.array(meds)
    print()
    print(f'across the {len(m)} sensors: Ups from {m.min():.2f} to {m.max():.2f}; '
          f'worst drift {max(drifts):.2f}x')
    print(f'lowest over the turbofan threshold {THRESH}: {m.min()/THRESH:.2f}x')
    print('paper: 2.45 to 10.95, every value stable to within 1.5x, lowest more than twice')
