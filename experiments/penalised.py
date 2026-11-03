"""
Expansion about the PENALISED optimum, plus a fix to the sampling convention.

(A) Sampling.  Earlier simulations used M=60 points on [0,cT] for every c, i.e. a
    sampling RATE proportional to 1/c.  A real campaign measures at a fixed rate, so
    M(c) = c*M0.  With fixed rate, J_T(c) = M0 A(c)/sigma^2 exactly matches
    J_T(1) eta_T(c) of Definition 4.1; with fixed M it did not (off by 1/c).

(B) Expansion.  The penalised estimator does not converge to the truth.  Linearising
    the penalised normal equations at the truth, with K=(C+Lam)^-1, rho^2=BKB'/A:

      Dlog T = [ xi_T - B K (xi_l - Lam d) ] / (A (1-rho^2))

    giving a STOCHASTIC part, reduced by the penalty,
        Var_st = vs^2 [ A(1-rho^2) - B K Lam K B' ] / (A^2 (1-rho^2)^2)
    and a BIAS part that does not vanish with M,
        bias^2 = B K Lam S_th Lam K B' / (A^2 (1-rho^2)^2).
    Reference (shape known) is vs^2/A, so

      Pi = [ A(1-rho^2) - B K Lam K B' + vs^-2 B K Lam S_th Lam K B' ] / (A (1-rho^2)^2)

    The old formula 1/(1-rho^2) is the Lam->0 limit of the stochastic part alone.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares, minimize_scalar

M0, SH, LAMM = 60, 0.25, 0.1

def blocks(Dp, dlg, c):
    A = quad(lambda t: (t*Dp(t))**2, 0, c, limit=400)[0]
    B = np.array([quad(lambda t: -(t*Dp(t))*d(t), 0, c, limit=400)[0] for d in dlg])
    C = np.array([[quad(lambda t: d1(t)*d2(t), 0, c, limit=400)[0] for d2 in dlg]
                  for d1 in dlg])
    return A, B, C

def Pi_pred(Dp, dlg, c, sig, old=False):
    k = len(dlg)
    A, B, C = blocks(Dp, dlg, c)
    Lam = (LAMM*sig**2/SH**2)*np.eye(k)          # fixed-rate: Lam = lambda/M0, c-free
    K   = np.linalg.inv(C + Lam)
    r2  = float(B @ K @ B / A)
    if old:
        return 1.0/(1.0-r2)
    vs2 = sig**2/M0
    Sth = (SH**2)*np.eye(k)
    red = float(B @ K @ Lam @ K @ B)             # variance reduction from the penalty
    bia = float(B @ K @ Lam @ Sth @ Lam @ K @ B) # squared shrinkage bias
    return (A*(1-r2) - red + bia/vs2) / (A*(1-r2)**2)

# ---------------- simulation with FIXED-RATE sampling ------------------------
def simulate(kind, c, sig, reps=1400, seed=5):
    rng = np.random.default_rng(seed + int(c*1000))
    M = max(8, int(round(M0*c)))
    e = []
    for _ in range(reps):
        if kind == 'bat':
            th = np.array([3.0, 0.7])*np.exp(SH*rng.normal(size=2))
            th[1] = np.clip(th[1], .3, 1.)
            f = lambda tau, p: (1-np.exp(-p[0]*np.clip(tau,1e-9,1)**p[1]))/(1-np.exp(-p[0]))
            th0, plo, phi = np.array([3.0,0.7]), np.array([.3,.3]), np.array([15.,1.])
            Tb, cv = 806.0, 0.46
        else:
            th = np.array([3.0*np.exp(SH*rng.normal())])
            f  = lambda tau, p: np.clip(tau,1e-9,1)**p[0]
            th0, plo, phi = np.array([3.0]), np.array([.4]), np.array([12.])
            Tb, cv = 15760.0, 0.47
        s = np.sqrt(np.log(1+cv**2)); T = rng.lognormal(np.log(Tb)-0.5*s**2, s)
        t = np.linspace(T/M0, c*T, M)            # fixed RATE: M(c)=c*M0 points
        y = f(t/T, th) + sig*rng.normal(size=M)
        lam = LAMM*(1./SH**2)*sig**2*M0
        lo, hi = t[-1]*1.0001, t[-1]*25.
        G = np.geomspace(lo, hi, 300)
        R = np.array([f(t/g, th0) for g in G]) - y
        T0 = G[np.argmin(np.einsum('ij,ij->i', R, R))]
        def res(p):
            return np.concatenate([f(t/p[0], np.exp(p[1:])) - y,
                                   np.sqrt(lam)*(p[1:]-np.log(th0))])
        try:
            Th = least_squares(res, np.concatenate([[T0], np.log(th0)]),
                               bounds=(np.concatenate([[lo], np.log(plo)]),
                                       np.concatenate([[hi], np.log(phi)])),
                               method='trf', max_nfev=500).x[0]
        except Exception:
            Th = T0
        e.append((Th-T)/T)
    return np.median(np.abs(np.array(e)))/0.6745

AB, BB = 3.0, 0.7
bat_Dp = lambda t: AB*BB*t**(BB-1)*np.exp(-AB*t**BB)/(1-np.exp(-AB))
def bat_dab(t):
    Mm = 1-np.exp(-AB); Nn = 1-np.exp(-AB*t**BB)
    return AB*(t**BB*np.exp(-AB*t**BB)*Mm - Nn*np.exp(-AB))/Mm**2
bat_db = lambda t: BB*AB*t**BB*np.log(t)*np.exp(-AB*t**BB)/(1-np.exp(-AB))
bea_Dp = lambda t: 3.0*t**2.0
bea_dl = [lambda t: 3.0*t**3.0*np.log(t)]

if __name__ == '__main__':
    for kind, nm, Dp, dlg, sig, grid in [
        ('bat','Battery',bat_Dp,[bat_dab,bat_db],0.05,[0.25,0.35,0.50,0.65,0.80]),
        ('bea','Bearing',bea_Dp,bea_dl,          0.15,[0.50,0.65,0.80])]:
        print(f"\n--- {nm} (fixed-rate sampling, M(c)=c*{M0}) ---")
        print(f"{'c':>5s} {'M(c)':>5s} {'old 1/(1-r2)':>13s} {'new (penalised)':>16s} "
              f"{'measured':>9s} {'old ratio':>10s} {'new ratio':>10s}")
        ro, rn = [], []
        for c in grid:
            A = blocks(Dp, dlg, c)[0]
            JT = M0*A/sig**2
            s  = simulate(kind, c, sig)
            meas = s**2*JT
            po = Pi_pred(Dp, dlg, c, sig, old=True)
            pn = Pi_pred(Dp, dlg, c, sig, old=False)
            ro.append(abs(np.log(meas/po))); rn.append(abs(np.log(meas/pn)))
            print(f"{c:5.2f} {max(8,int(round(M0*c))):5d} {po:13.2f} {pn:16.2f} "
                  f"{meas:9.2f} {meas/po:10.2f} {meas/pn:10.2f}")
        print(f"  mean |log ratio|: old {np.mean(ro):.3f} (x{np.exp(np.mean(ro)):.2f})"
              f"   new {np.mean(rn):.3f} (x{np.exp(np.mean(rn)):.2f})")
