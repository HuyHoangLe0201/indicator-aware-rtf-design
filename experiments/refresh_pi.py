"""
Regenerate the Pi / oracle / boundary / verdict tables under FIXED-RATE sampling.
M(c) = max(8, round(c*M0)),  M0 = 60.   Penalty Lam = lambda/M0 with
lambda = lam_mult * sigma^2 * M0 / s^2, so Lam is c-free.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares
from scipy.stats import lognorm

M0, SH, LAMM = 60, 0.25, 0.1
DOM = {  # name: (kind, sigma, Tbar, cv, Lambda_design, theta0)
 'FD001'  : ('pow', 0.10,   192.0, 0.10, 0.165, np.array([1.1])),
 'Bearing': ('pow', 0.15, 15760.0, 0.47, 1.382, np.array([3.0])),
 'Battery': ('str', 0.05,   806.0, 0.46, 0.241, np.array([3.0, 0.7])),
}
BOX = dict(pow=(np.array([0.4]), np.array([12.])),
           str=(np.array([0.3, 0.3]), np.array([15., 1.])))

def fD(kind):
    if kind == 'pow':
        return lambda tau, p: np.clip(tau, 1e-9, 1.)**p[0]
    return lambda tau, p: (1-np.exp(-p[0]*np.clip(tau,1e-9,1.)**p[1]))/(1-np.exp(-p[0]))

def dprime(kind, th):
    if kind == 'pow':
        b = th[0]; return lambda t: b*t**(b-1)
    a, b = th
    return lambda t: a*b*t**(b-1)*np.exp(-a*t**b)/(1-np.exp(-a))

def AofC(kind, th, c):
    dp = dprime(kind, th)
    return quad(lambda t: (t*dp(t))**2, 0, c, limit=400)[0]

def oracle_label(name, c, W=25, reps=1200, seed=3):
    kind, sig, Tbar, cv, _, th0 = DOM[name]
    D = fD(kind); plo, phi = BOX[kind]
    lam = LAMM*sig**2*M0/SH**2
    M = max(8, int(round(c*M0)))
    s = np.sqrt(np.log(1+cv**2)); rng = np.random.default_rng(seed+int(c*1000)+W)
    k = len(th0); e = []
    for _ in range(reps):
        T = rng.lognormal(np.log(Tbar)-0.5*s**2, s)
        th = np.clip(th0*np.exp(SH*rng.normal(size=k)), plo, phi)
        t = np.linspace(T/M0, c*T, M)
        y = D(t/T, th) + sig*rng.normal(size=M)
        lo, hi = t[-1]*1.0001, t[-1]*W
        G = np.geomspace(lo, hi, 300)
        R = np.array([D(t/g, th0) for g in G]) - y
        T00 = G[np.argmin(np.einsum('ij,ij->i', R, R))]
        def res(p):
            return np.concatenate([D(t/p[0], np.exp(p[1:])) - y,
                                   np.sqrt(lam)*(p[1:]-np.log(th0))])
        try:
            Th = least_squares(res, np.concatenate([[T00], np.log(th0)]),
                               bounds=(np.concatenate([[lo], np.log(plo)]),
                                       np.concatenate([[hi], np.log(phi)])),
                               method='trf', max_nfev=500).x[0]
        except Exception:
            Th = T00
        e.append((Th-T)/T)
    e = np.abs(np.array(e))
    return np.median(e)/0.6745, np.median(e)

GRID = [0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 1.00]
print("=== Table: Pi_rob(c) per domain, FIXED RATE ===")
print(f"{'domain':9s} {'Ups':>5s} " + " ".join(f"{('c='+str(c)):>8s}" for c in GRID)
      + f" {'Pi fit':>8s}")
PI = {}
for nm, (kind, sig, Tb, cv, Ld, th0) in DOM.items():
    A1 = AofC(kind, th0, 1.0); dp = dprime(kind, th0); Ups = dp(1.0)**2/A1
    row, vals = [], []
    for c in GRID:
        srob, _ = oracle_label(nm, c)
        JT = M0*AofC(kind, th0, c)/sig**2
        p = srob**2*JT; row.append(p)
        if c <= 0.65: vals.append(p)
    PI[nm] = (Ups, float(np.mean(vals)))
    print(f"{nm:9s} {Ups:5.2f} " + " ".join(f"{v:8.2f}" for v in row)
          + f" {PI[nm][1]:8.2f}")

print("\n=== Table: bearing Pi_rob vs box width, FIXED RATE ===")
print(f"{'c':>5s} " + " ".join(f"{('W='+str(w)):>9s}" for w in (25,100,400,800)))
for c in (0.50, 0.65, 0.80):
    JT = M0*AofC('pow', DOM['Bearing'][5], c)/DOM['Bearing'][1]**2
    print(f"{c:5.2f} " + " ".join(f"{oracle_label('Bearing',c,W=W,reps=1000)[0]**2*JT:9.2f}"
                                  for W in (25,100,400,800)))

print("\n=== Table: verdicts, FIXED RATE ===")
def cornerq(nm, varpi, k0=0.05):
    kind, sig, Tbar, cv, Lam, th0 = DOM[nm]
    dp = dprime(kind, th0); Z = AofC(kind, th0, 1.0)
    cg = np.linspace(1e-4, 1., 300)
    ev = np.array([AofC(kind, th0, x)/Z for x in cg])
    s = np.sqrt(np.log(1+cv**2)); d = lognorm(s=s, scale=Tbar*np.exp(-.5*s**2))
    Tg = np.linspace(d.ppf(1e-4), d.ppf(1-1e-4), 400); pg = d.pdf(Tg); pg /= np.trapezoid(pg,Tg)
    hi = d.ppf(0.999); tsg = np.linspace(0.05*Tbar, hi, 200); g = []
    for ts in tsg:
        kb = k0 + np.trapezoid(np.minimum(Tg, ts)*pg, Tg)/Tbar
        eb = (np.trapezoid(np.interp(np.minimum(1., ts/Tg), cg, ev)*pg, Tg)
              + varpi*d.cdf(ts))/(1+varpi)
        g.append(kb*(1./max(eb,1e-12)/Lam + 1.))
    j = int(np.argmin(g))
    return j >= len(tsg)-2, d.cdf(tsg[j])
print(f"{'domain':9s} {'Ups':>5s} {'Pi_crit':>8s} {'Pi meas':>8s} {'varpi':>8s} {'verdict':>22s}")
for nm, (Ups, Pi) in PI.items():
    vc = np.nan
    for v in np.concatenate([[0.], np.geomspace(0.02, 500, 80)]):
        if cornerq(nm, v)[0]: vc = v; break
    cn, F = cornerq(nm, Pi*Ups)
    print(f"{nm:9s} {Ups:5.2f} {vc/Ups:8.2f} {Pi:8.2f} {Pi*Ups:8.1f} "
          f"{('corner' if cn else f'interior F={F:.3f}'):>22s}")
