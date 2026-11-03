"""Decouple n from c.

In the budget sweep n(c)=C/(k0+c), so the U-shaped label error could be either
  (i)  driven by n  (few cells at large c -> poorly determined shrinkage target), or
  (ii) intrinsic to c (misspecification of the complete-pooling first stage).
Fix n and sweep c.  If the rise at large c survives, it is (ii).
"""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar

AB, BB = 3.0, 0.7
T_MEAN, T_CV, SIGMA, M_OBS, SH_CV = 806.0, 0.46, 0.05, 60, 0.25
LAM = 0.1*(1./SH_CV**2)*SIGMA**2*M_OBS

def Dg(tau, ab, b):
    tau = np.clip(tau, 1e-9, 1.0)
    return (1-np.exp(-ab*tau**b))/(1-np.exp(-ab))

def draw(rng):
    s = np.sqrt(np.log(1+T_CV**2))
    return (rng.lognormal(np.log(T_MEAN)-0.5*s**2, s),
            AB*np.exp(SH_CV*rng.normal()),
            float(np.clip(BB*np.exp(SH_CV*rng.normal()), .3, 1.)))

def win_of(T, ab, b, c, rng):
    t = np.linspace(T/M_OBS, c*T, M_OBS)
    return t, Dg(t/T, ab, b) + SIGMA*rng.normal(size=M_OBS)

def fT(t, y, a, b_):
    return minimize_scalar(lambda T_: np.sum((y-Dg(t/T_, a, b_))**2),
                           bounds=(t[-1]*1.0001, t[-1]*25.), method='bounded',
                           options={'xatol':1e-3}).x

def run(win, oracle_shape=None):
    """returns imputed T; if oracle_shape given, skip stage 1 and use it"""
    if oracle_shape is None:
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
    else:
        ab, bb = oracle_shape
        Ts = np.array([fT(t,y,ab,bb) for t,y in win])
    out = np.empty(len(win))
    for i,(t,y) in enumerate(win):
        lo, hi = t[-1]*1.0001, t[-1]*25.
        def res(p):
            T_, la, lb = p
            return np.concatenate([y-Dg(t/T_, np.exp(la), np.clip(np.exp(lb),.3,1.)),
                                   np.sqrt(LAM)*np.array([la-np.log(ab), lb-np.log(bb)])])
        try:
            out[i] = least_squares(res,[Ts[i],np.log(ab),np.log(bb)],
                                   bounds=([lo,np.log(.3),np.log(.3)],
                                           [hi,np.log(15.),0.]),
                                   method='trf',max_nfev=400).x[0]
        except Exception:
            out[i] = Ts[i]
    return out

if __name__ == '__main__':
    grid = [0.15,0.25,0.35,0.5,0.65,0.8,1.0]
    print(f"{'c':>5s} | " + " ".join(f"{('n='+str(N)):>9s}" for N in (40,120,240))
          + f" {'oracle shape':>13s}")
    print(f"{'':>5s} | " + " "*33 + "  (n=120, true fleet-mean shape given)")
    for c in grid:
        row = []
        for N in (40,120,240):
            e = []
            for r in range(6):
                rng = np.random.default_rng(int(c*1000)*7 + r)
                cells = [draw(rng) for _ in range(N)]
                w  = [win_of(T,ab,b,c,rng) for T,ab,b in cells]
                Tt = np.array([T for T,_,_ in cells])
                e.append(np.abs(run(w)-Tt)/Tt)
            row.append(np.sqrt(np.mean(np.concatenate(e)**2)))
        e = []
        for r in range(6):
            rng = np.random.default_rng(int(c*1000)*7 + r)
            cells = [draw(rng) for _ in range(120)]
            w  = [win_of(T,ab,b,c,rng) for T,ab,b in cells]
            Tt = np.array([T for T,_,_ in cells])
            e.append(np.abs(run(w, oracle_shape=(AB,BB))-Tt)/Tt)
        orc = np.sqrt(np.mean(np.concatenate(e)**2))
        print(f"{c:5.2f} | " + " ".join(f"{v:9.3f}" for v in row) + f" {orc:13.3f}")
