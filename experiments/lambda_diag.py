"""
Diagnostic for lambda_perchannel.py: is the 785x split real, or is it misspecification?

sigma in Lambda = lambda_m sigma^2 / S_theta is supposed to be MEASUREMENT noise.  The
estimate used is the residual of a power-law shape fit, so it is only measurement noise if
the power law actually fits.  DeltaQ(V) saturates, and a power law fitted to a saturating
curve leaves structured residual that has nothing to do with the sensor.  If that is what
drives the 785x, the number is an artifact and must not be quoted.

Three checks:
  1. fit quality per channel (R^2, and how much of the residual is autocorrelated structure
     rather than white noise);
  2. how many cells are pinned at the parameter bounds, which would deflate S_theta;
  3. re-estimate both quantities with a flexible two-parameter family that can represent
     both a convex and a saturating shape, and see whether the split survives.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import uniform_filter1d

D = np.load('../data/severson/dq_traj.npz', allow_pickle=True)
BND = (0.3, 15.0)


def prep(y, smooth=25):
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if len(y) < 60:
        return None
    y = uniform_filter1d(y, max(3, len(y)//smooth), mode='nearest')
    base = np.median(y[:max(2, len(y)//20)]); end = np.median(y[-max(2, len(y)//50):])
    if abs(end - base) < 1e-12:
        return None
    Dn = (y - base)/(end - base)
    tau = (np.arange(len(y)) + 1)/len(y)
    m = (Dn > -0.4) & (Dn < 1.6)
    if m.sum() < 60:
        return None
    return tau[m], Dn[m]


def fit_pow(tau, Dn):
    r = least_squares(lambda p: np.clip(tau, 1e-9, 1.)**p[0] - Dn, [3.0],
                      bounds=([BND[0]], [BND[1]]), max_nfev=400)
    return float(r.x[0]), r.fun


def fit_flex(tau, Dn):
    """D = (1-exp(-a tau^b))/(1-exp(-a)): convex for small a, saturating for large a."""
    def f(p):
        a, b = np.exp(p[0]), np.exp(p[1])
        t = np.clip(tau, 1e-9, 1.)**b
        return (1 - np.exp(-a*t))/(1 - np.exp(-a)) - Dn
    r = least_squares(f, [np.log(0.5), np.log(3.0)],
                      bounds=([np.log(1e-3), np.log(0.2)], [np.log(60.), np.log(20.)]),
                      max_nfev=2000)
    return float(np.exp(r.x[0])), float(np.exp(r.x[1])), r.fun


def lag1(e):
    e = e - e.mean()
    d = np.dot(e, e)
    return float(np.dot(e[:-1], e[1:])/d) if d > 0 else np.nan


def run(name, traj):
    B, S, R2, AC, hit = [], [], [], [], 0
    FA, FB, FS, FR2, FAC = [], [], [], [], []
    for t in traj:
        p = prep(t)
        if p is None:
            continue
        tau, Dn = p
        b, res = fit_pow(tau, Dn)
        B.append(b); S.append(np.std(res)); AC.append(lag1(res))
        R2.append(1 - np.var(res)/np.var(Dn))
        hit += (b <= BND[0]*1.001) or (b >= BND[1]*0.999)
        a, bb, fres = fit_flex(tau, Dn)
        FA.append(a); FB.append(bb); FS.append(np.std(fres)); FAC.append(lag1(fres))
        FR2.append(1 - np.var(fres)/np.var(Dn))
    B, S, R2, AC = map(np.array, (B, S, R2, AC))
    FB, FS, FR2, FAC = map(np.array, (FB, FS, FR2, FAC))
    print(f'--- {name}  (n = {len(B)}) ---')
    print(f'  power law : beta {np.median(B):6.3f}  sd(beta) {np.std(B, ddof=1):6.3f}  '
          f'S_theta {np.var(B, ddof=1):8.4f}')
    print(f'              sigma {np.median(S):.4f}   R^2 {np.median(R2):.4f}   '
          f'lag-1 autocorr of residual {np.median(AC):+.3f}   at-bound {hit}')
    print(f'              sigma^2/S_theta = {np.median(S)**2/np.var(B, ddof=1):.5f}')
    print(f'  flexible  : b    {np.median(FB):6.3f}  sd(b)    {np.std(FB, ddof=1):6.3f}  '
          f'S_theta {np.var(FB, ddof=1):8.4f}')
    print(f'              sigma {np.median(FS):.4f}   R^2 {np.median(FR2):.4f}   '
          f'lag-1 autocorr of residual {np.median(FAC):+.3f}')
    print(f'              sigma^2/S_theta = {np.median(FS)**2/np.var(FB, ddof=1):.5f}')
    print()
    return (np.median(S)**2/np.var(B, ddof=1), np.median(FS)**2/np.var(FB, ddof=1))


print(__doc__)
print('=' * 76)
print('A residual that is mostly white has lag-1 autocorrelation near 0 after smoothing;')
print('one dominated by shape misspecification stays close to +1.')
print()
cap = [-np.asarray(q, float)[1:] for q in D['QD']]
dq = [np.asarray(t, float)[1] for t in D['DQ']]
pc, fc = run('bulk capacity', cap)
pd_, fd = run('DeltaQ(V)', dq)
print('=' * 76)
print(f'ratio under the power law      : {max(pc,pd_)/min(pc,pd_):8.1f}x')
print(f'ratio under the flexible family: {max(fc,fd)/min(fc,fd):8.1f}x')
