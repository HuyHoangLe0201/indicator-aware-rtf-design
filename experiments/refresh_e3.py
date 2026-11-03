"""E3 design sweep regenerated under FIXED-RATE sampling, M(c)=max(8,round(c*M0))."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.integrate import quad

AB, BB = 3.0, 0.7
T_MEAN, T_CV, SIGMA, M0, SH = 806.0, 0.46, 0.05, 60, 0.25
T_FEAT, LAM = 121.0, 0.1*(1./0.25**2)*0.05**2*60

def Dg(tau, ab, b):
    tau = np.clip(tau, 1e-9, 1.0)
    return (1-np.exp(-ab*tau**b))/(1-np.exp(-ab))

def draw(rng):
    s = np.sqrt(np.log(1+T_CV**2))
    return (rng.lognormal(np.log(T_MEAN)-0.5*s**2, s),
            AB*np.exp(SH*rng.normal()),
            float(np.clip(BB*np.exp(SH*rng.normal()), .3, 1.)))

def window(T, ab, b, c, rng):
    M = max(8, int(round(c*M0)))                  # FIXED RATE
    t = np.linspace(T/M0, c*T, M)
    return t, Dg(t/T, ab, b) + SIGMA*rng.normal(size=M)

def fT(t, y, a, b_):
    return minimize_scalar(lambda T_: np.sum((y-Dg(t/T_, a, b_))**2),
                           bounds=(t[-1]*1.0001, t[-1]*25.), method='bounded',
                           options={'xatol':1e-3}).x

def impute(win, arm):
    if arm == 'oracle':
        ab, bb = AB, BB
    else:
        ab, bb = AB, BB
        Ts = np.array([fT(t,y,ab,bb) for t,y in win])
        for _ in range(3):
            r = lambda p: np.concatenate([y-Dg(t/T_, p[0], p[1])
                                          for (t,y),T_ in zip(win,Ts)])
            try:
                ab, bb = least_squares(r,[ab,bb],bounds=([.3,.3],[15.,1.]),
                                       method='trf',max_nfev=150).x
            except Exception: pass
            Ts = np.array([fT(t,y,ab,bb) for t,y in win])
    out = np.empty(len(win))
    for i,(t,y) in enumerate(win):
        lo, hi = t[-1]*1.0001, t[-1]*25.
        T0 = fT(t, y, ab, bb)
        def res(p):
            return np.concatenate([y-Dg(t/p[0], np.exp(p[1]), np.clip(np.exp(p[2]),.3,1.)),
                                   np.sqrt(LAM)*np.array([p[1]-np.log(ab), p[2]-np.log(bb)])])
        try:
            out[i] = least_squares(res,[T0,np.log(ab),np.log(bb)],
                                   bounds=([lo,np.log(.3),np.log(.3)],
                                           [hi,np.log(15.),0.]),
                                   method='trf',max_nfev=400).x[0]
        except Exception:
            out[i] = T0
    return out

def feats(T, ab, b, rng):
    t = np.linspace(T_FEAT/12, T_FEAT, 12)
    y = Dg(t/T, ab, b) + SIGMA*rng.normal(size=12)
    return np.array([1., y[-1], y[-1]-y[5], np.log(np.var(np.diff(y))+1e-12)])

def campaign(c, C, k0, rng, Xte, Tte, arm):
    n = max(5, int(round(C/(k0+c))))
    cells = [draw(rng) for _ in range(n)]
    win = [window(T,ab,b,c,rng) for T,ab,b in cells]
    X   = np.array([feats(T,ab,b,rng) for T,ab,b in cells])
    Tt  = np.array([T for T,_,_ in cells])
    yl  = impute(win, arm)
    mu  = np.linalg.solve(X.T@X + 1e-4*np.eye(4), X.T@yl)
    return (float(np.sqrt(np.mean((Xte@mu-Tte)**2))), n,
            float(np.median(np.abs(yl-Tt)/Tt)))

if __name__ == '__main__':
    K0, REPS, C = 0.05, 12, 40*1.05
    grid = [0.15,0.25,0.35,0.5,0.65,0.8,1.0]
    rng = np.random.default_rng(999)
    tri = [draw(rng) for _ in range(500)]
    Tte = np.array([x[0] for x in tri]); Xte = np.array([feats(*x,rng) for x in tri])
    print(f"budget C={C:.1f}, kappa0={K0}, reps={REPS}; trivial baseline RMSE={T_CV*T_MEAN:.0f}")
    for arm in ['pooled', 'oracle']:
        print(f"\n--- arm: {arm} (fixed rate) ---")
        print(f"{'c':>5s} {'M(c)':>5s} {'n':>4s} {'RMSE':>8s} {'s.e.':>6s} {'rel':>6s} {'labMed':>8s}")
        out = []
        for c in grid:
            v, l = [], []
            for r in range(REPS):
                rr = np.random.default_rng(int(c*1000)*131+r)
                a, n, lb = campaign(c, C, K0, rr, Xte, Tte, arm)
                v.append(a); l.append(lb)
            v = np.array(v)
            out.append((c, max(8,int(round(c*M0))), n, v.mean(),
                        v.std(ddof=1)/np.sqrt(REPS), np.mean(l)))
        bm = min(o[3] for o in out)
        for c, M, n, m, se, lb in out:
            print(f"{c:5.2f} {M:5d} {n:4d} {m:8.1f} {se:6.1f} {m/bm:6.3f} {lb:8.3f}"
                  + ("  <== optimum" if m == bm else ""))
