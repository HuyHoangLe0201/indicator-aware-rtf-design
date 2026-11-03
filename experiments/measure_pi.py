"""
Measure the nuisance inflation Pi for each benchmark domain.

Pi is defined by   Var[(That-T)/T] = Pi * sigma_T^2 / eta_T(c),   sigma_T^2 = 1/J_T(1),
J_T(1) = (M/sigma^2) int_0^1 tau^2 D'^2 .  So

    Pi(c) = L(c)^2 * eta_T(c) * J_T(1).

The ORACLE arm is used (shrinkage centre = true fleet-mean shape) so that what is
measured is the cost of co-estimating the per-unit shape, not the misspecification of
a complete-pooling first stage.  Pi is calibrated on c <= 0.65 to avoid the
first-passage channel.

Domains differ in the number of shape nuisance parameters:
  power-law        theta = beta        (p_theta = 1)
  stretched-exp    theta = (alpha_b,b) (p_theta = 2)
so a smaller Pi is expected for the power-law domains.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares, minimize_scalar

SIGMA_OF = {'FD001': 0.10, 'Bearing': 0.15, 'Battery': 0.05}   # Paper 1 Table 2, sigma/B
FLEET    = {'FD001': (192.0, 0.10), 'Bearing': (15760.0, 0.47), 'Battery': (806.0, 0.46)}
M_OBS, LAM_MULT = 60, 0.1

def model(name):
    if name == 'Battery':
        th0 = np.array([3.0, 0.7])
        D   = lambda tau, p: (1-np.exp(-p[0]*np.clip(tau,1e-9,1)**p[1]))/(1-np.exp(-p[0]))
        Dp  = lambda t: th0[0]*th0[1]*t**(th0[1]-1)*np.exp(-th0[0]*t**th0[1])/(1-np.exp(-th0[0]))
        lo, hi = np.array([0.3,0.3]), np.array([15.,1.])
    else:
        b   = 1.1 if name == 'FD001' else 3.0
        th0 = np.array([b])
        D   = lambda tau, p: np.clip(tau,1e-9,1)**p[0]
        Dp  = lambda t, b=b: b*t**(b-1)
        lo, hi = np.array([0.4]), np.array([8.])
    return th0, D, Dp, lo, hi

def run(name, SH_CV=0.25, n=120, reps=5, grid=(0.15,0.25,0.35,0.50,0.65,0.80,1.00)):
    th0, D, Dp, plo, phi = model(name)
    sig = SIGMA_OF[name]; Tbar, cv = FLEET[name]
    Z    = quad(lambda t: t**2*Dp(t)**2, 0, 1, limit=400)[0]
    etaT = lambda c: quad(lambda t: t**2*Dp(t)**2, 0, min(c,1.0), limit=400)[0]/Z
    JT1  = M_OBS*Z/sig**2
    Ups  = Dp(1.0)**2/Z
    lam  = LAM_MULT*(1./SH_CV**2)*sig**2*M_OBS
    k    = len(th0)
    s    = np.sqrt(np.log(1+cv**2)); mu = np.log(Tbar)-0.5*s**2

    out = {}
    for c in grid:
        errs = []
        for r in range(reps):
            rng = np.random.default_rng(int(c*1000)*97 + r + hash(name) % 1000)
            for _ in range(n):
                T  = rng.lognormal(mu, s)
                th = th0*np.exp(SH_CV*rng.normal(size=k))
                th = np.clip(th, plo, phi)
                t  = np.linspace(T/M_OBS, c*T, M_OBS)
                y  = D(t/T, th) + sig*rng.normal(size=M_OBS)
                blo, bhi = t[-1]*1.0001, t[-1]*25.
                # stage: 1-D grid for T at the oracle centre, then joint polish w/ shrinkage
                G  = np.linspace(blo, bhi, 260)
                R  = np.array([D(t/g, th0) for g in G]) - y
                T0 = G[np.argmin(np.einsum('ij,ij->i', R, R))]
                def res(p):
                    return np.concatenate([y - D(t/p[0], np.exp(p[1:])),
                                           np.sqrt(lam)*(p[1:]-np.log(th0))])
                try:
                    Th = least_squares(res, np.concatenate([[T0], np.log(th0)]),
                                       bounds=(np.concatenate([[blo], np.log(plo)]),
                                               np.concatenate([[bhi], np.log(phi)])),
                                       method='trf', max_nfev=400).x[0]
                except Exception:
                    Th = T0
                errs.append((Th-T)/T)
        L = float(np.sqrt(np.mean(np.array(errs)**2)))
        out[c] = (L, etaT(c), L**2*etaT(c)*JT1)
    return out, JT1, Ups, etaT

if __name__ == '__main__':
    print(f"{'domain':9s} {'p_theta':>7s} {'sigma':>6s} {'J_T(1)':>9s} {'Ups':>6s}")
    res = {}
    for nm in ['FD001','Bearing','Battery']:
        o, JT1, Ups, eT = run(nm)
        res[nm] = (o, JT1, Ups)
        k = 2 if nm == 'Battery' else 1
        print(f"{nm:9s} {k:7d} {SIGMA_OF[nm]:6.2f} {JT1:9.1f} {Ups:6.2f}")
    print(f"\n{'domain':9s} " + " ".join(f"{('c='+str(c)):>8s}" for c in
                                         (0.15,0.25,0.35,0.50,0.65,0.80,1.00)) + f" {'Pi fit':>8s}")
    for nm in ['FD001','Bearing','Battery']:
        o, JT1, Ups = res[nm]
        row = " ".join(f"{o[c][2]:8.1f}" for c in (0.15,0.25,0.35,0.50,0.65,0.80,1.00))
        Pi  = float(np.mean([o[c][2] for c in (0.15,0.25,0.35,0.50,0.65)]))
        print(f"{nm:9s} {row} {Pi:8.1f}")
        res[nm] = (o, JT1, Ups, Pi)
    print(f"\n{'domain':9s} {'Ups':>6s} {'Pi':>7s} {'varpi=Pi*Ups':>13s}")
    for nm in ['FD001','Bearing','Battery']:
        o, JT1, Ups, Pi = res[nm]
        print(f"{nm:9s} {Ups:6.2f} {Pi:7.1f} {Pi*Ups:13.1f}")
    np.save('pi_measured.npy',
            {nm: (res[nm][3], res[nm][2]) for nm in res}, allow_pickle=True)
