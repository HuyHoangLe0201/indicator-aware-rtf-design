"""E3 pilot v2 -- same budget sweep, but the degradation SHAPE is unknown to the
learner and varies across the fleet, so the failure time must be extrapolated
with (alpha_b, b) as nuisance parameters.  This is the faithful setting."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.integrate import quad

ALPHA_B, B_STR = 3.0, 0.7
T_MEAN, T_CV, SIGMA, M_OBS, SH_CV = 806.0, 0.46, 0.05, 60, 0.25
T_FEAT = 121.0

def Dg(tau, ab, b):
    tau = np.clip(tau, 1e-9, 1.0)
    return (1-np.exp(-ab*tau**b))/(1-np.exp(-ab))

def draw(rng):
    s = np.sqrt(np.log(1+T_CV**2))
    T  = rng.lognormal(np.log(T_MEAN)-0.5*s**2, s)
    ab = ALPHA_B*np.exp(SH_CV*rng.normal())
    b  = float(np.clip(B_STR*np.exp(SH_CV*rng.normal()), .3, 1.))
    return T, ab, b

def impute(T, ab, b, c, rng):
    t = np.linspace(T/M_OBS, c*T, M_OBS)
    y = Dg(t/T, ab, b) + SIGMA*rng.normal(size=M_OBS)
    lo, hi = t[-1]*1.0001, t[-1]*25.0
    res = lambda p: y - Dg(t/p[0], p[1], p[2])
    best, bl = t[-1]*1.3, np.inf
    for T0 in (t[-1]*1.2, T_MEAN):
        try:
            r = least_squares(res, [np.clip(T0,lo,hi), ALPHA_B, B_STR],
                              bounds=([lo,.3,.3],[hi,15.,1.]), method='trf', max_nfev=200)
            if r.cost < bl: bl, best = r.cost, r.x[0]
        except Exception: pass
    return best

def feats(T, ab, b, rng):
    t = np.linspace(T_FEAT/12, T_FEAT, 12)
    y = Dg(t/T, ab, b) + SIGMA*rng.normal(size=12)
    return np.array([1., y[-1], y[-1]-y[5], np.log(np.var(np.diff(y))+1e-12)])

def campaign(c, C, k0, rng, Xte, Tte):
    n = max(5, int(round(C/(k0+c))))
    X = np.empty((n,4)); yl = np.empty(n)
    for i in range(n):
        T, ab, b = draw(rng)
        yl[i] = impute(T, ab, b, c, rng)
        X[i]   = feats(T, ab, b, rng)
    mu = np.linalg.solve(X.T@X + 1e-4*np.eye(4), X.T@yl)
    return float(np.sqrt(np.mean((Xte@mu - Tte)**2))), n

def theory_cstar(k0, Lam):
    dp = lambda t: ALPHA_B*B_STR*t**(B_STR-1)*np.exp(-ALPHA_B*t**B_STR)/(1-np.exp(-ALPHA_B))
    Z  = quad(lambda t: t**2*dp(t)**2,0,1,limit=400)[0]
    eT = lambda c: quad(lambda t: t**2*dp(t)**2,0,c,limit=400)[0]/Z
    return minimize_scalar(lambda c:(k0+c)*(1/eT(c)+Lam), bounds=(.05,1.),
                           method='bounded', options={'xatol':1e-9}).x

if __name__ == '__main__':
    K0, LAM, REPS = 0.05, 0.241, 14
    C = 40*(K0+1.)
    grid = [0.15,0.25,0.35,0.5,0.65,0.8,1.0]
    rng = np.random.default_rng(999)
    tri = [draw(rng) for _ in range(500)]
    Tte = np.array([x[0] for x in tri])
    Xte = np.array([feats(*x, rng) for x in tri])
    print(f"budget C={C:.2f} (=40 cells to failure)  kappa0={K0}  reps={REPS}")
    print(f"theory c* = {theory_cstar(K0,LAM):.3f}   [uses eta_T with shape KNOWN]\n")
    print(f"{'c':>5s} {'n(c)':>5s} {'RMSE':>9s} {'s.e.':>6s} {'rel':>6s}")
    out=[]
    for c in grid:
        v=[]
        for r in range(REPS):
            rr = np.random.default_rng(int(c*1000)*131 + r)
            a,n = campaign(c, C, K0, rr, Xte, Tte); v.append(a)
        v=np.array(v); out.append((c,n,v.mean(),v.std(ddof=1)/np.sqrt(REPS)))
    bm = min(o[2] for o in out)
    for c,n,m,se in out:
        print(f"{c:5.2f} {n:5d} {m:9.1f} {se:6.1f} {m/bm:6.3f}" + ("  <== optimum" if m==bm else ""))
