"""Terminal concentration of the XJTU-SY bearing indicators when the reading-noise variance is proportional to (D + 0.1)^p.

D is the absolute value of the smoothed fleet-mean normalized curve (baseline = median of the first 5% of life, terminal = mean of the last 3 readings,
as in upsilon_eff_real.py).  With weights w = (D + 0.1)^(-p) the information about ln T from [0, c] is  A(c) = int_0^c w (tau D')^2 dtau  and
Upsilon is the secant slope of A(c)/A(1) over c in [0.9, 1].  p = 0 is the white-noise estimate of the main text.
Usage: python noise_exponent_sweep.py   (reads bearing_feats.npz)
"""
import os
import numpy as np
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
G = np.linspace(0.005, 1.0, 200)
dg = G[1] - G[0]


def mean_curve(series_list):
    cur = []
    for h in series_list:
        h = np.asarray(h, float)
        h = h[np.isfinite(h)]
        if len(h) < 30:
            continue
        n = len(h)
        base = np.median(h[:max(3, n // 20)])
        end = np.mean(h[-3:])
        if abs(end - base) < 1e-12:
            continue
        z = (h - base) / (end - base)
        tau = np.arange(1, n + 1) / n
        cur.append(np.interp(G, tau, savgol_filter(z, min(21, n // 3 * 2 + 1), 2)))
    return np.mean(cur, axis=0), len(cur)


def ups_weighted(M, p):
    Ms = savgol_filter(M, 25, 3)
    Dp = savgol_filter(M, 25, 3, deriv=1, delta=dg)
    s = G * Dp
    w = 1.0 / (np.abs(Ms) + 0.1) ** p
    cs = np.linspace(0.9, 1.0, 11)
    J = np.array([np.sum((w * s * s)[G <= c + 1e-12]) * dg for c in cs])
    return np.polyfit(cs, J / J[-1], 1)[0]


f = np.load(os.path.join(HERE, 'bearing_feats.npz'))
print('indicator    units     p=0     p=1     p=2     p=4    (bearing threshold 2.27)')
for ind in ('RMS', 'HF 2-5kHz', 'kurtosis'):
    ser = [f[k] for k in f.files if k.startswith('X:') and k.endswith('|' + ind)]
    M, n = mean_curve(ser)
    print(f'{ind:11s} {n:5d} ' + ' '.join(f'{ups_weighted(M, p):7.2f}' for p in (0, 1, 2, 4)))
