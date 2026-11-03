"""
Lambda is a per-channel quantity, but the paper thresholds a whole domain with one value.

Lambda = lambda_m sigma^2 / S_theta.  Both sigma, the measurement noise on the normalized
indicator, and S_theta, the fleet scatter of its shape, are properties of the CHANNEL, not of
the asset.  Yet Section VII compares Upsilon for bulk capacity and for DeltaQ(V) against one
battery threshold of 1.18, and the bearing indicators against one threshold of 2.27.  If the
two battery channels have different sigma^2/S_theta they are entitled to different thresholds,
and the comparison as written is not apples to apples.

This matters in two directions.  It could rescue a verdict -- a noisy channel earns a higher
threshold, so it is more easily declared run-to-failure -- or it could break one.  We measure
sigma^2/S_theta per channel and re-derive each verdict against its own threshold.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import uniform_filter1d

K0 = 0.05
D = np.load('../data/severson/dq_traj.npz', allow_pickle=True)


def shape_fit(y, smooth=25):
    """Fit D(tau)=tau^beta to a normalized trajectory; return beta and the residual sd.

    y is oriented so that it INCREASES with damage before it gets here.
    """
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if len(y) < 60:
        return None
    y = uniform_filter1d(y, max(3, len(y)//smooth), mode='nearest')
    base = np.median(y[:max(2, len(y)//20)])
    end = np.median(y[-max(2, len(y)//50):])
    if abs(end - base) < 1e-12:
        return None
    Dn = (y - base)/(end - base)
    tau = (np.arange(len(y)) + 1)/len(y)
    m = (Dn > -0.4) & (Dn < 1.6)
    if m.sum() < 60:
        return None
    tau, Dn = tau[m], Dn[m]
    r = least_squares(lambda p: np.clip(tau, 1e-9, 1.)**p[0] - Dn, [3.0],
                      bounds=([0.3], [15.]), max_nfev=400)
    return float(r.x[0]), float(np.std(r.fun))


def channel(name, traj):
    fits = [f for f in (shape_fit(t) for t in traj) if f]
    b = np.array([f[0] for f in fits]); s = np.array([f[1] for f in fits])
    S, sg = float(np.var(b, ddof=1)), float(np.median(s))
    return dict(name=name, n=len(fits), beta=float(np.median(b)), S=S, sigma=sg, ratio=sg*sg/S)


# capacity: falls with damage -> negate so it rises
cap = [-np.asarray(q, float)[1:] for q in D['QD']]
# DeltaQ(V): log10 Var already rises with damage
dq = [np.asarray(t, float)[1] for t in D['DQ']]

rows = [channel('bulk capacity', cap), channel('DeltaQ(V)', dq)]

print(__doc__)
print('=' * 76)
print(f'{"channel":16s} {"n":>4s} {"beta":>7s} {"S_theta":>9s} {"sigma":>8s} {"sig^2/S":>10s}')
print('-'*76)
for r in rows:
    print(f'{r["name"]:16s} {r["n"]:4d} {r["beta"]:7.3f} {r["S"]:9.4f} {r["sigma"]:8.4f} '
          f'{r["ratio"]:10.5f}')
a, b = rows[0]['ratio'], rows[1]['ratio']
print()
print(f'ratio between the two channels of the SAME fleet: {max(a,b)/min(a,b):.1f}x')

print()
print('Re-deriving the two battery verdicts with per-channel thresholds.')
print('The paper uses one lambda_m for the domain; calibrate it so that bulk capacity keeps')
print('the published Lambda = 0.241, then carry the SAME lambda_m to the other channel.')
lam_m = 0.241 / rows[0]['ratio']
print(f'  implied lambda_m = {lam_m:.1f}')
print()
print(f'{"channel":16s} {"Ups":>7s} {"Lambda":>8s} {"threshold":>10s} {"verdict":>16s}')
print('-'*76)
for r, ups in ((rows[0], 9.97), (rows[1], 1.04)):
    lam = lam_m * r['ratio']
    thr = (1 + lam)/(1 + K0)
    verdict = 'run to failure' if ups >= thr else 'stop early'
    print(f'{r["name"]:16s} {ups:7.2f} {lam:8.3f} {thr:10.3f} {verdict:>16s}')
print()
print('Also report the shared-threshold verdicts for comparison (Lambda = 0.241, thr = 1.181):')
for nm, ups in (('bulk capacity', 9.97), ('DeltaQ(V)', 1.04)):
    print(f'  {nm:16s} Ups {ups:5.2f}  ->  '
          f'{"run to failure" if ups >= 1.181 else "stop early"}')
