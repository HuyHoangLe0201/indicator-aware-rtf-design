"""
E3 robustness: does the interior optimum survive when EVERY c is given its BEST
shrinkage strength?  Fixed lambda penalises c=1 unfairly (complete pooling is
misspecified there, since a full trajectory identifies the cell's own shape).
We sweep lambda and report the envelope min_lambda RMSE(c, lambda).
"""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar

ALPHA_B, B_STR = 3.0, 0.7
T_MEAN, T_CV, SIGMA, M_OBS, SH_CV = 806.0, 0.46, 0.05, 60, 0.25
T_FEAT, TGRID = 121.0, 320
LAM0 = (1.0/SH_CV**2)*SIGMA**2*M_OBS          # base shrinkage

def Dg(tau, ab, b):
    tau = np.clip(tau, 1e-9, 1.0)
    return (1-np.exp(-ab*tau**b))/(1-np.exp(-ab))

def draw(rng):
    s = np.sqrt(np.log(1+T_CV**2))
    return (rng.lognormal(np.log(T_MEAN)-0.5*s**2, s),
            ALPHA_B*np.exp(SH_CV*rng.normal()),
            float(np.clip(B_STR*np.exp(SH_CV*rng.normal()), .3, 1.)))

def window(T, ab, b, c, rng):
    t = np.linspace(T/M_OBS, c*T, M_OBS)
    return t, Dg(t/T, ab, b) + SIGMA*rng.normal(size=M_OBS)

def fit_T(t, y, ab, bb):
    lo, hi = t[-1]*1.0001, t[-1]*25.
    G = np.linspace(lo, hi, TGRID)
    R = Dg(t[None,:]/G[:,None], ab, bb) - y[None,:]
    T0 = G[np.argmin(np.einsum('ij,ij->i', R, R))]; d = (hi-lo)/TGRID
    return minimize_scalar(lambda T_: np.sum((y-Dg(t/T_, ab, bb))**2),
                           bounds=(max(lo,T0-2*d), min(hi,T0+2*d)),
                           method='bounded', options={'xatol':1e-3}).x

def pooled_shape(win, n_iter=3):
    ab, bb = ALPHA_B, B_STR
    Ts = np.array([fit_T(t,y,ab,bb) for t,y in win])
    for _ in range(n_iter):
        res = lambda p: np.concatenate([y-Dg(t/T_, p[0], p[1])
                                        for (t,y),T_ in zip(win,Ts)])
        try:
            ab, bb = least_squares(res,[ab,bb],bounds=([.3,.3],[15.,1.]),
                                   method='trf',max_nfev=120).x
        except Exception: pass
        Ts = np.array([fit_T(t,y,ab,bb) for t,y in win])
    return ab, bb, Ts

def impute(win, lam_mult):
    ab, bb, Ts = pooled_shape(win)
    if lam_mult == np.inf:                     # complete pooling
        return Ts
    lam = lam_mult*LAM0
    out = np.empty(len(win))
    for i,(t,y) in enumerate(win):
        lo, hi = t[-1]*1.0001, t[-1]*25.
        def res(p):
            T_, la, lb = p
            pen = (np.sqrt(lam)*np.array([la-np.log(ab), lb-np.log(bb)])
                   if lam > 0 else np.zeros(2))
            return np.concatenate([y-Dg(t/T_, np.exp(la), np.clip(np.exp(lb),.3,1.)), pen])
        try:
            out[i] = least_squares(res,[Ts[i],np.log(ab),np.log(bb)],
                                   bounds=([lo,np.log(.3),np.log(.3)],
                                           [hi,np.log(15.),0.]),
                                   method='trf',max_nfev=150).x[0]
        except Exception:
            out[i] = Ts[i]
    return out

def feats(T, ab, b, rng):
    t = np.linspace(T_FEAT/12, T_FEAT, 12)
    y = Dg(t/T, ab, b) + SIGMA*rng.normal(size=12)
    return np.array([1., y[-1], y[-1]-y[5], np.log(np.var(np.diff(y))+1e-12)])

def campaign(c, C, k0, rng, Xte, Tte, lam_mult):
    n = max(5, int(round(C/(k0+c))))
    cells = [draw(rng) for _ in range(n)]
    win   = [window(T,ab,b,c,rng) for T,ab,b in cells]
    X     = np.array([feats(T,ab,b,rng) for T,ab,b in cells])
    Tt    = np.array([T for T,_,_ in cells])
    yl    = impute(win, lam_mult)
    mu    = np.linalg.solve(X.T@X + 1e-4*np.eye(4), X.T@yl)
    return (float(np.sqrt(np.mean((Xte@mu-Tte)**2))),
            float(np.median(np.abs(yl-Tt)/Tt)), n)

if __name__ == '__main__':
    K0, REPS, C = 0.05, 10, 40*1.05
    grid = [0.15,0.25,0.35,0.5,0.65,0.8,1.0]
    lams = [0.0, 0.1, 1.0, 10.0, np.inf]
    rng = np.random.default_rng(999)
    tri = [draw(rng) for _ in range(500)]
    Tte = np.array([x[0] for x in tri]); Xte = np.array([feats(*x,rng) for x in tri])
    print(f"baseline (predict fleet mean) RMSE = {T_CV*T_MEAN:.0f};  theory c* = 0.385")
    print(f"\n{'c':>5s} {'n':>4s} " + " ".join(f"{('lam='+str(l)):>10s}" for l in lams)
          + f" {'ENVELOPE':>9s} {'best lam':>9s} {'labMedAE*':>10s}")
    env = []
    for c in grid:
        row, labs = [], []
        for lm in lams:
            v, lb = [], []
            for r in range(REPS):
                rr = np.random.default_rng(int(c*1000)*131+r)
                a, l, n = campaign(c, C, K0, rr, Xte, Tte, lm)
                v.append(a); lb.append(l)
            row.append(np.mean(v)); labs.append(np.mean(lb))
        j = int(np.argmin(row)); env.append((c, n, row[j], lams[j], labs[j]))
        print(f"{c:5.2f} {n:4d} " + " ".join(f"{x:10.1f}" for x in row)
              + f" {row[j]:9.1f} {str(lams[j]):>9s} {labs[j]:10.3f}")
    bc = min(env, key=lambda z: z[2])
    print(f"\nENVELOPE optimum: c* = {bc[0]:.2f} (n={bc[1]}, RMSE={bc[2]:.1f}, "
          f"lambda={bc[3]});  theory c* = 0.385")
