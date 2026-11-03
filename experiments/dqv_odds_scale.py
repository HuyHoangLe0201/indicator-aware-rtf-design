"""DeltaQ(V) terminal concentration and saturation on the R^2 scale (as in dqv_interval.py) and on the information scale (odds)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import dqv_interval as dv
dv.DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'severson', 'dq_traj.npz')
W = dv.WINDOWS


def curve(X, y, idx, scale):
    r2 = np.array([dv.loo_r2(X[idx, j], y[idx]) for j in range(X.shape[1])])
    if not np.all(np.isfinite(r2)) or r2.max() <= 0:
        return None
    r2c = np.clip(r2, 0, 0.999)
    f = r2c if scale == 'r2' else r2c / (1 - r2c)
    eta = f / f.max()
    med = np.median(np.exp(y[idx]))
    sat = W[int(np.argmax(eta >= 0.95))] / med
    slope = (eta[-1] - eta[-2]) / ((W[-1] - W[-2]) / med)
    # also the slope from the running-maximum curve, the monotone information
    return r2, eta, sat, slope


X, y, life = dv.load(); n = len(y)
for scale in ('r2', 'odds'):
    r2, eta, sat, sl = curve(X, y, np.arange(n), scale)
    print(f'--- scale {scale}: eta by window', dict(zip(W, np.round(eta, 3))), f'sat {sat:.3f}  terminal slope {sl:.3f}')
    rng = np.random.default_rng(20260803)
    sats, sls = [], []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        o = curve(X, y, idx, scale)
        if o is None: continue
        sats.append(o[2]); sls.append(o[3])
    sats, sls = np.array(sats), np.array(sls)
    print('   usable', len(sats), ' saturation median %.3f CI [%.3f, %.3f]' % ((np.median(sats),) + tuple(np.percentile(sats, [2.5, 97.5]))))
    print('   terminal slope median %.3f CI [%.3f, %.3f]; share above 1.18: %.2f%%' % ((np.median(sls),) + tuple(np.percentile(sls, [2.5, 97.5])) + (100 * np.mean(sls >= 1.18),)))

# ---- fixed composition: only the cells that live through the longest window, so every window is read on the same units
mask = np.isfinite(X[:, -1])
Xf = np.where(mask[:, None], X, np.nan)
idx0 = np.where(mask)[0]
print('\nfixed set: %d cells alive at %d cycles; life quartiles %s' % (mask.sum(), W[-1], np.percentile(life[mask], [0, 25, 50, 75, 100]).round(0)))
for scale in ('r2', 'odds'):
    r2, eta, sat, sl = curve(Xf, y, idx0, scale)
    print('--- fixed set, scale %s: eta' % scale, dict(zip(W, np.round(eta, 3))), 'slope over last two windows %.3f' % sl)
    rng = np.random.default_rng(7)
    sls = []
    for _ in range(2000):
        idx = rng.choice(idx0, len(idx0))
        o = curve(Xf, y, idx, scale)
        if o is not None: sls.append(o[3])
    sls = np.array(sls)
    print('   slope median %.3f CI [%.3f, %.3f]; share >= 1.18: %.1f%%' % ((np.median(sls),) + tuple(np.percentile(sls, [2.5, 97.5])) + (100 * np.mean(sls >= 1.18),)))
