"""
Re-measure Pi for the bearing domain with a widened constraint set.

Diagnosis of the earlier failure: the estimator's box T in [t_max, W*t_max] with W=25
CAPS the error.  At c=0.15 the shape-known CR bound is sigma_T/sqrt(eta_T)
= 0.0171/sqrt(1.7e-6) = 13.1 (a 1310% relative SD), while the box admits at most
~W*c = 3.7.  A capped estimator beats a vacuous bound, so Pi measures below 1.

Test: sweep the box width W.  If Pi converges as W grows, that limit is the answer.
If Pi keeps growing with W, the label problem is unidentifiable at that c and Pi is
simply undefined there.
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
UPS  = Dp(1.0)**2/Z
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
                          method='trf', max_nfev=600).x
        Th, at = x[0], x[0] >= hi*0.999
    except Exception:
        Th, at = T0, True
    return (Th-T)/T, at

def measure(c, W, reps=900, seed=11):
    rng = np.random.default_rng(seed + int(c*1000) + W)
    e, a = zip(*[one(c, W, rng) for _ in range(reps)])
    e = np.array(e)
    return (np.sqrt(np.mean(e**2)), np.median(np.abs(e)), np.mean(a))

if __name__ == '__main__':
    print(f"sigma_T = {1/np.sqrt(JT1):.4f}   Ups = {UPS:.2f}   "
          f"Pi_crit (corner threshold) = 6.90\n")
    print(f"{'c':>5s} {'eta_T':>10s} {'CR bound':>9s} | " +
          " ".join(f"{('W='+str(w)):>16s}" for w in (25, 100, 400)))
    print(f"{'':>5s} {'':>10s} {'':>9s} | " +
          " ".join(f"{'Pi':>7s} {'atBnd':>8s}" for _ in range(3)))
    for c in [0.35, 0.50, 0.65, 0.80]:
        eT = etaT(c); cr = np.sqrt(1/(JT1*eT))
        row = []
        for W in (25, 100, 400):
            rms, med, at = measure(c, W)
            row.append(f"{rms**2*eT*JT1:7.1f} {100*at:7.1f}%")
        print(f"{c:5.2f} {eT:10.2e} {cr:9.3f} | " + " ".join(row))
    print("\nconvergence check at c=0.65 (identifiable window, boundary-free):")
    print(f"{'W':>6s} {'Pi':>8s} {'atBnd':>8s} {'medAE':>8s}")
    for W in (25, 50, 100, 200, 400, 800):
        rms, med, at = measure(0.65, W, reps=1400)
        print(f"{W:6d} {rms**2*etaT(0.65)*JT1:8.1f} {100*at:7.1f}% {med:8.3f}")
