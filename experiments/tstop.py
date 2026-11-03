"""
Theorem 3 re-derived in the absolute stop time t_s (replaces the fixed fraction c).

Campaign: every unit is put on test at 0 and removed at failure or at t_s, whichever
comes first.  Unit i is COMPLETE iff T_i <= t_s (failure witnessed, boundary channel
active); otherwise it is censored at fraction c_i = t_s/T_i.

  cost per unit      kbar(ts) = kappa0 + E[min(T,ts)]/Tbar          (SATURATES)
  info per unit      etabar(ts) = E[eta_T(min(1,ts/T))] + varpi F(ts)
  varpi = Pi * J_bnd / J_T(1)      (boundary weight relative to inflated path)

  G(ts) = kbar(ts) * [ 1/etatilde(ts) / Lambda + 1 ],   etatilde = etabar/(1+varpi)

Analytic criterion (see notes): with h = f/Fbar the fleet HAZARD,
  Ups_star(ts) = Tbar [ varpi h(ts) + Ups/ts ] / (1+varpi),
  run-everything-to-failure  iff  Ups_star >= (1 + Lambda(1+varpi))/(1+kappa0).
Setting varpi=0 and ts=Tbar recovers the original criterion Ups >= (1+Lambda)/(1+kappa0).
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.stats import lognorm

def make_fleet(Tbar, cv):
    s = np.sqrt(np.log(1+cv**2)); sc = Tbar*np.exp(-0.5*s**2)
    d = lognorm(s=s, scale=sc)
    return d, Tbar

def domain(name):
    if name == 'Battery':
        AB, BB = 3.0, 0.7
        Dp = lambda t: AB*BB*t**(BB-1)*np.exp(-AB*t**BB)/(1-np.exp(-AB))
        return Dp, 806.0, 0.46, 0.241
    if name == 'Bearing':
        Dp = lambda t: 3.0*t**2.0
        return Dp, 15760.0, 0.47, 1.382
    Dp = lambda t: 1.1*t**0.1
    return Dp, 192.0, 0.10, 0.165

def build(Dp):
    Z  = quad(lambda t: t**2*Dp(t)**2, 0, 1, limit=400)[0]
    eT = lambda c: (0.0 if c <= 0 else
                    quad(lambda t: t**2*Dp(t)**2, 0, min(c,1.0), limit=400)[0]/Z)
    Ups = Dp(1.0)**2/Z                       # eta_T'(1), terminal concentration
    jr  = Dp(1.0)**2/Z                       # J_bnd / J_T(1)  (M/sigma^2 cancels)
    return eT, Ups, jr

def solve(name, kappa0, Pi, verbose=False):
    Dp, Tbar, cv, Lam = domain(name)
    eT, Ups, jr = build(Dp)
    d, _ = make_fleet(Tbar, cv)
    varpi = Pi*jr
    Tg = np.linspace(d.ppf(1e-4), d.ppf(1-1e-4), 900)
    pg = d.pdf(Tg); pg = pg/np.trapezoid(pg, Tg)

    def kbar(ts):
        mu = np.trapezoid(np.minimum(Tg, ts)*pg, Tg)
        return kappa0 + mu/Tbar
    eTv = np.vectorize(eT)
    def etabar(ts):
        return float(np.trapezoid(eTv(np.minimum(1.0, ts/Tg))*pg, Tg) + varpi*d.cdf(ts))
    def G(ts):
        et = etabar(ts)/(1+varpi)
        return kbar(ts)*(1.0/max(et,1e-9)/Lam + 1.0)

    hi = d.ppf(0.999)
    r  = minimize_scalar(G, bounds=(0.05*Tbar, hi), method='bounded',
                         options={'xatol': Tbar*1e-4})
    ts = r.x
    corner = G(hi) <= G(ts)*1.0000001
    return dict(ts=ts, rel=ts/Tbar, Fcomp=d.cdf(ts), varpi=varpi,
                Ups=Ups, corner=corner, hi_rel=hi/Tbar)

if __name__ == '__main__':
    print("consistency: varpi=0 must reproduce the fixed-fraction criterion\n")
    print(f"{'domain':9s} {'kappa0':>7s} {'Pi':>5s} {'varpi':>6s} {'Ups':>6s} "
          f"{'ts*/Tbar':>9s} {'F(ts*)':>8s} {'c*_old':>7s}")
    OLD = {'FD001':{0.05:1.000,0.2:1.000,0.5:1.000},
           'Bearing':{0.05:1.000,0.2:1.000,0.5:1.000},
           'Battery':{0.05:0.385,0.2:0.485,0.5:0.604}}
    for name in ['FD001','Bearing','Battery']:
        for k0 in [0.05, 0.2, 0.5]:
            for Pi in [0.0, 27.5]:
                r = solve(name, k0, Pi)
                tag = "  <- corner (all to failure)" if r['corner'] else ""
                print(f"{name:9s} {k0:7.2f} {Pi:5.1f} {r['varpi']:6.2f} {r['Ups']:6.2f} "
                      f"{r['rel']:9.3f} {r['Fcomp']:8.3f} {OLD[name][k0]:7.3f}{tag}")
        print()
