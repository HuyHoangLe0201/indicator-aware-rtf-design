"""E3 diagnostic v2 -- bounded joint fit, robust summaries, failure accounting."""
import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import quad

ALPHA_B, B_STR = 3.0, 0.7
T_MEAN, T_CV, SIGMA, M_OBS, SH_CV = 806.0, 0.46, 0.05, 60, 0.25

def Dg(tau, ab, b):
    tau = np.clip(tau, 1e-9, 1.0)
    return (1 - np.exp(-ab*tau**b))/(1 - np.exp(-ab))

def etaT_ref(c, ab=ALPHA_B, b=B_STR):
    dp = lambda t: ab*b*t**(b-1)*np.exp(-ab*t**b)/(1-np.exp(-ab))
    return (quad(lambda t: t**2*dp(t)**2,0,c,limit=400)[0]
            / quad(lambda t: t**2*dp(t)**2,0,1,limit=400)[0])

def one_cell(c, rng, shape_known):
    s = np.sqrt(np.log(1+T_CV**2)); T = rng.lognormal(np.log(T_MEAN)-0.5*s**2, s)
    ab, b = (ALPHA_B, B_STR) if shape_known else (
        ALPHA_B*np.exp(SH_CV*rng.normal()), float(np.clip(B_STR*np.exp(SH_CV*rng.normal()),.3,1.)))
    t = np.linspace(T/M_OBS, c*T, M_OBS)
    y = Dg(t/T, ab, b) + SIGMA*rng.normal(size=M_OBS)
    Tlo, Thi = t[-1]*1.0001, t[-1]*25.0
    if shape_known:
        Ts = np.linspace(Tlo, Thi, 400)
        Th = Ts[np.argmin([np.sum((y-Dg(t/T_, ALPHA_B, B_STR))**2) for T_ in Ts])]
    else:
        res = lambda p: y - Dg(t/p[0], p[1], p[2])
        best, bl = None, np.inf
        for T0 in (t[-1]*1.2, t[-1]*2.0, T_MEAN):          # multi-start
            try:
                r = least_squares(res, [np.clip(T0,Tlo,Thi), ALPHA_B, B_STR],
                                  bounds=([Tlo,0.3,0.3],[Thi,15.0,1.0]),
                                  method='trf', max_nfev=300)
                if r.cost < bl: bl, best = r.cost, r.x[0]
            except Exception:
                pass
        Th = best if best is not None else t[-1]*1.3
    return T, Th

def sweep(shape_known, reps=300):
    grid = [0.15,0.25,0.35,0.5,0.7,1.0]
    rows = []
    for c in grid:
        rng = np.random.default_rng(7)
        T, Th = zip(*[one_cell(c, rng, shape_known) for _ in range(reps)])
        e = (np.array(Th)-np.array(T))/np.array(T)
        rows.append((c, etaT_ref(c), np.sqrt(np.mean(e**2)),
                     np.median(np.abs(e)), np.mean(np.abs(e) > 1.0)))
    b_rmse = rows[-1][2]; b_mad = rows[-1][3]
    print(f"\n--- shape {'KNOWN' if shape_known else 'UNKNOWN (bounded joint fit)'} ---")
    print(f"{'c':>5s} {'eta_T':>7s} {'relRMSE':>9s} {'ratio':>6s} | "
          f"{'medAE':>7s} {'ratio':>6s} | {'1/sqrt(etaT)':>12s} {'fail%':>6s}")
    for c, et, rm, md, fr in rows:
        print(f"{c:5.2f} {et:7.4f} {rm:9.4f} {rm/b_rmse:6.2f} | "
              f"{md:7.4f} {md/b_mad:6.2f} | {1/np.sqrt(et):12.2f} {100*fr:6.1f}")

if __name__ == '__main__':
    print("theory: label error should scale as 1/sqrt(eta_T(c))")
    sweep(True); sweep(False)
