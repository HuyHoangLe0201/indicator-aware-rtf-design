"""Critical boundary weight varpi_crit that flips the design to the corner.

Reported alongside the measured Pi, so the corner/interior verdict does not hinge on
a Pi measurement that is only reliable for one domain.
"""
import numpy as np
from scipy.integrate import quad
from scipy.stats import lognorm

DOM = {'FD001'  : (lambda t: 1.1*t**0.1,  192.0, 0.10, 0.165),
       'Bearing': (lambda t: 3.0*t**2.0, 15760.0, 0.47, 1.382),
       'Battery': (lambda t: 3.0*0.7*t**(-0.3)*np.exp(-3.0*t**0.7)/(1-np.exp(-3.0)),
                   806.0, 0.46, 0.241)}
MEAS = {'FD001': (3.20, 15.4, 'no: 4-25'), 'Bearing': (7.00, 60.4, 'no: 0-199'),
        'Battery': (0.29, 27.7, 'yes: 24-30')}

def setup(Dp, Tbar, cv):
    Z  = quad(lambda t: t**2*Dp(t)**2, 0, 1, limit=400)[0]
    cg = np.linspace(1e-4, 1.0, 400)
    ev = np.array([quad(lambda t: t**2*Dp(t)**2, 0, x, limit=200)[0]/Z for x in cg])
    s  = np.sqrt(np.log(1+cv**2)); d = lognorm(s=s, scale=Tbar*np.exp(-0.5*s**2))
    Tg = np.linspace(d.ppf(1e-4), d.ppf(1-1e-4), 500)
    pg = d.pdf(Tg); pg = pg/np.trapezoid(pg, Tg)
    return cg, ev, d, Tg, pg

def curve(pack, Tbar, Lam, k0, varpi, tsg):
    cg, ev, d, Tg, pg = pack
    out = np.empty(len(tsg))
    for i, ts in enumerate(tsg):
        kb = k0 + np.trapezoid(np.minimum(Tg, ts)*pg, Tg)/Tbar
        eb = (np.trapezoid(np.interp(np.minimum(1.0, ts/Tg), cg, ev)*pg, Tg)
              + varpi*d.cdf(ts))/(1+varpi)
        out[i] = kb*(1.0/max(eb, 1e-12)/Lam + 1.0)
    return out

if __name__ == '__main__':
    print(f"{'domain':9s} {'Ups':>5s} {'Pi meas':>8s} {'Pi stable?':>11s} "
          f"{'varpi_crit':>11s} {'varpi=Pi*Ups':>13s} {'verdict':>20s}")
    for nm, (Dp, Tbar, cv, Lam) in DOM.items():
        Ups, Pi, rel = MEAS[nm]
        pack = setup(Dp, Tbar, cv)
        d = pack[2]; hi = d.ppf(0.999)
        tsg = np.linspace(0.05*Tbar, hi, 220)
        vc = np.nan
        for v in np.concatenate([[0.0], np.geomspace(0.02, 400, 90)]):
            g = curve(pack, Tbar, Lam, 0.05, v, tsg)
            if np.argmin(g) >= len(tsg)-2:
                vc = v; break
        g = curve(pack, Tbar, Lam, 0.05, Pi*Ups, tsg)
        j = int(np.argmin(g)); corner = j >= len(tsg)-2
        verd = 'corner' if corner else f'interior F={d.cdf(tsg[j]):.3f}'
        print(f"{nm:9s} {Ups:5.2f} {Pi:8.1f} {rel:>11s} {vc:11.2f} {Pi*Ups:13.1f} {verd:>20s}")
    print("\nvarpi_crit: smallest boundary weight forcing test-everything-to-failure "
          "(kappa0=0.05)")
