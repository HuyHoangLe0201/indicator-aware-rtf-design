"""
Per-channel Lambda, with the measurement noise estimated properly.

The first attempt took sigma to be the residual of a shape fit.  That is wrong twice over:
the trajectory was smoothed first, so the residual is autocorrelated by construction, and
whatever survives smoothing is shape misspecification rather than sensor noise.  Both channels
came back with lag-1 autocorrelation above 0.95, which is the signature of exactly that.

Here sigma is estimated from the high-frequency content of the RAW trajectory instead, using
second differences, which annihilate any locally linear trend:

        Var(y_{k+2} - 2y_{k+1} + y_k) = 6 sigma^2   for white noise on a smooth curve,

with a median-absolute-deviation scaling so that a few outlying cycles cannot inflate it.  The
shape parameter and its fleet scatter S_theta still come from the smoothed curve, where they
belong.  Both are expressed on the normalized scale, so sigma^2/S_theta is dimensionless and
comparable across channels.

Reported alongside is the same quantity for the flexible shape family, so the conclusion can
be seen not to depend on assuming a power law for a channel that saturates.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import uniform_filter1d

K0 = 0.05
D = np.load('../data/severson/dq_traj.npz', allow_pickle=True)


def sigma_hf(raw, span):
    """Measurement noise from second differences of the raw series, on the normalized scale."""
    y = np.asarray(raw, float)
    y = y[np.isfinite(y)]
    if len(y) < 20:
        return np.nan
    d2 = y[2:] - 2*y[1:-1] + y[:-2]
    mad = np.median(np.abs(d2 - np.median(d2)))
    return (mad/0.6745)/np.sqrt(6.0)/abs(span)


def shapes(raw, smooth=25):
    y = np.asarray(raw, float)
    y = y[np.isfinite(y)]
    if len(y) < 60:
        return None
    ys = uniform_filter1d(y, max(3, len(y)//smooth), mode='nearest')
    base = np.median(ys[:max(2, len(ys)//20)]); end = np.median(ys[-max(2, len(ys)//50):])
    if abs(end - base) < 1e-12:
        return None
    Dn = (ys - base)/(end - base)
    tau = (np.arange(len(ys)) + 1)/len(ys)
    m = (Dn > -0.4) & (Dn < 1.6)
    if m.sum() < 60:
        return None
    tau, Dn = tau[m], Dn[m]
    rp = least_squares(lambda p: np.clip(tau, 1e-9, 1.)**p[0] - Dn, [3.0],
                       bounds=([0.3], [15.]), max_nfev=400)

    def f(p):
        a, b = np.exp(p[0]), np.exp(p[1])
        t = np.clip(tau, 1e-9, 1.)**b
        return (1 - np.exp(-a*t))/(1 - np.exp(-a)) - Dn
    rf = least_squares(f, [np.log(0.5), np.log(3.0)],
                       bounds=([np.log(1e-3), np.log(0.2)], [np.log(60.), np.log(20.)]),
                       max_nfev=2000)
    return float(rp.x[0]), float(np.exp(rf.x[1])), end - base


def channel(name, traj):
    bp, bf, sg = [], [], []
    for t in traj:
        s = shapes(t)
        if s is None:
            continue
        b1, b2, span = s
        v = sigma_hf(t, span)
        if not np.isfinite(v):
            continue
        bp.append(b1); bf.append(b2); sg.append(v)
    bp, bf, sg = map(np.array, (bp, bf, sg))
    return dict(name=name, n=len(sg), beta=np.median(bp),
                Sp=float(np.var(bp, ddof=1)), Sf=float(np.var(bf, ddof=1)),
                sigma=float(np.median(sg)))


cap = [-np.asarray(q, float)[1:] for q in D['QD']]
dq = [np.asarray(t, float)[1] for t in D['DQ']]
rows = [channel('bulk capacity', cap), channel('DeltaQ(V)', dq)]

print(__doc__)
print('=' * 78)
print(f'{"channel":16s} {"n":>4s} {"beta":>7s} {"sigma_hf":>9s} '
      f'{"S(power)":>9s} {"s^2/S":>9s} {"S(flex)":>9s} {"s^2/S":>9s}')
print('-'*78)
for r in rows:
    r['rp'] = r['sigma']**2/r['Sp']
    r['rf'] = r['sigma']**2/r['Sf']
    print(f'{r["name"]:16s} {r["n"]:4d} {r["beta"]:7.3f} {r["sigma"]:9.5f} '
          f'{r["Sp"]:9.4f} {r["rp"]:9.5f} {r["Sf"]:9.4f} {r["rf"]:9.5f}')

a, b = rows[0], rows[1]
print()
print(f'per-channel ratio, power-law shape : {max(a["rp"],b["rp"])/min(a["rp"],b["rp"]):7.1f}x')
print(f'per-channel ratio, flexible shape  : {max(a["rf"],b["rf"])/min(a["rf"],b["rf"]):7.1f}x')

print()
print('Verdicts with per-channel thresholds, lambda_m calibrated so that bulk capacity keeps')
print('the published Lambda = 0.241.')
for tag, key in (('power-law shape', 'rp'), ('flexible shape', 'rf')):
    lam_m = 0.241/a[key]
    print()
    print(f'  [{tag}]  implied lambda_m = {lam_m:.0f}')
    print(f'  {"channel":16s} {"Ups":>6s} {"Lambda":>9s} {"threshold":>10s} {"verdict":>16s}')
    for r, ups in ((a, 9.97), (b, 1.04)):
        lam = lam_m*r[key]
        thr = (1 + lam)/(1 + K0)
        print(f'  {r["name"]:16s} {ups:6.2f} {lam:9.3f} {thr:10.3f} '
              f'{("run to failure" if ups >= thr else "stop early"):>16s}')
