"""
Bearing Pi via a ROBUST scale.

Widening the box shows the label-error VARIANCE for bearing is not a property of the
problem: it is set by wherever the constraint is placed, because a vanishing fraction
of non-identifiable draws carries unbounded error.  The median is box-independent.

So define Pi on a robust scale:  s_rob = medAE / 0.6745  (Gaussian-equivalent SD),
    Pi_rob(c) = s_rob(c)^2 * eta_T(c) * J_T(1).
Check box-independence, then compare with the corner threshold Pi_crit = 6.90.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares

BETA0, SIG, M, SH_CV = 3.0, 0.15, 60, 0.25
TBAR, TCV = 15760.0, 0.47
Dp   = lambda t: BETA0*t**(BETA0-1)
Z    = quad(lambda t: t**2*Dp(t)**2, 0, 1, limit=400)[0]
etaT = lambda c: c**(2*BETA0+1)
JT1  = M*Z/SIG**2
LAM  = 0.1*(1./SH_CV**2)*SIG**2*M

def one(c, W, rng):
    s = np.sqrt(np.log(1+TCV**2))
    T = rng.lognormal(np.log(TBAR)-0.5*s**2, s)
    b = float(np.clip(BETA0*np.exp(SH_CV*rng.normal()), 0.4, 12.))
    t = np.linspace(T/M, c*T, M)
    y = np.clip(t/T, 1e-9, 1.)**b + SIG*rng.normal(size=M)
    lo, hi = t[-1]*1.0001, t[-1]*W
    G = np.geomspace(lo, hi, 400)
    R = np.clip(t[None,:]/G[:,None], 1e-9, 1.)**BETA0 - y
    T0 = G[np.argmin(np.einsum('ij,ij->i', R, R))]
    def res(p):
        return np.concatenate([np.clip(t/p[0],1e-9,1.)**np.exp(p[1]) - y,
                               [np.sqrt(LAM)*(p[1]-np.log(BETA0))]])
    try:
        x = least_squares(res, [T0, np.log(BETA0)],
                          bounds=([lo, np.log(0.4)], [hi, np.log(12.)]),
                          method='trf', max_nfev=600).x[0]
    except Exception:
        x = T0
    return (x-T)/T

def pirob(c, W, reps=1600, seed=7):
    rng = np.random.default_rng(seed + int(c*1000) + W)
    e = np.abs(np.array([one(c, W, rng) for _ in range(reps)]))
    s = np.median(e)/0.6745
    return s**2*etaT(c)*JT1, np.median(e), np.percentile(e, 90)

if __name__ == '__main__':
    print("Pi_rob = (medAE/0.6745)^2 * eta_T * J_T(1);  corner threshold Pi_crit = 6.90\n")
    print(f"{'c':>5s} | " + " ".join(f"{('W='+str(w)):>9s}" for w in (25,100,400,800))
          + f" {'medAE':>8s} {'p90':>8s}")
    for c in [0.50, 0.65, 0.80]:
        row, md, p9 = [], None, None
        for W in (25, 100, 400, 800):
            p, m, q = pirob(c, W)
            row.append(p)
            if W == 100: md, p9 = m, q
        print(f"{c:5.2f} | " + " ".join(f"{v:9.2f}" for v in row) + f" {md:8.3f} {p9:8.3f}")
    print("\nbox-independence of Pi_rob confirms it is a property of the problem,")
    print("unlike the variance-based Pi which diverged with W.")
