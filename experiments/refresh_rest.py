"""Regenerate tab:decouple, tab:boundary and tab:e3indep under FIXED-RATE sampling."""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares, minimize_scalar

AB, BB = 3.0, 0.7
TBAR, TCV, SIG, M0, SH = 806.0, 0.46, 0.05, 60, 0.25
TFEAT, LAM = 121.0, 0.1*(1./0.25**2)*0.05**2*60
GRID = [0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 1.00]
Mof = lambda c: max(8, int(round(c*M0)))

def Dg(tau, ab, b):
    tau = np.clip(tau, 1e-9, 1.0)
    return (1-np.exp(-ab*tau**b))/(1-np.exp(-ab))
dp   = lambda t: AB*BB*t**(BB-1)*np.exp(-AB*t**BB)/(1-np.exp(-AB))
Aof  = lambda c: quad(lambda t: (t*dp(t))**2, 0, c, limit=400)[0]
JT1, JBND = M0*Aof(1.0)/SIG**2, M0*dp(1.0)**2/SIG**2
etaT = lambda c: Aof(c)/Aof(1.0)

def draw(rng):
    s = np.sqrt(np.log(1+TCV**2))
    return (rng.lognormal(np.log(TBAR)-0.5*s**2, s),
            AB*np.exp(SH*rng.normal()),
            float(np.clip(BB*np.exp(SH*rng.normal()), .3, 1.)))

def win_of(T, ab, b, c, rng):
    M = Mof(c); t = np.linspace(T/M0, c*T, M)
    return t, Dg(t/T, ab, b) + SIG*rng.normal(size=M)

def fT(t, y, a, b_):
    return minimize_scalar(lambda T_: np.sum((y-Dg(t/T_, a, b_))**2),
                           bounds=(t[-1]*1.0001, t[-1]*25.), method='bounded',
                           options={'xatol':1e-3}).x

def shrink(t, y, ab, bb, T0):
    lo, hi = t[-1]*1.0001, t[-1]*25.
    def res(p):
        return np.concatenate([Dg(t/p[0], np.exp(p[1]), np.clip(np.exp(p[2]),.3,1.)) - y,
                               np.sqrt(LAM)*np.array([p[1]-np.log(ab), p[2]-np.log(bb)])])
    try:
        return least_squares(res, [T0, np.log(ab), np.log(bb)],
                             bounds=([lo, np.log(.3), np.log(.3)],
                                     [hi, np.log(15.), 0.]),
                             method='trf', max_nfev=400).x[0]
    except Exception:
        return T0

def impute(win, oracle):
    if oracle:
        ab, bb = AB, BB
    else:
        ab, bb = AB, BB
        Ts = np.array([fT(t,y,ab,bb) for t,y in win])
        for _ in range(3):
            r = lambda p: np.concatenate([Dg(t/T_, p[0], p[1])-y
                                          for (t,y),T_ in zip(win,Ts)])
            try:
                ab, bb = least_squares(r,[ab,bb],bounds=([.3,.3],[15.,1.]),
                                       method='trf',max_nfev=150).x
            except Exception: pass
            Ts = np.array([fT(t,y,ab,bb) for t,y in win])
    return np.array([shrink(t, y, ab, bb, fT(t,y,ab,bb)) for t,y in win])

def indep_global(t, y):
    lo, hi = t[-1]*1.0001, t[-1]*25.
    AG = np.exp(np.linspace(np.log(0.6), np.log(12.), 11))
    BG = np.linspace(0.32, 1.0, 11); TG = np.geomspace(lo, hi, 120)
    tau = np.clip(t[None,:]/TG[:,None], 1e-9, 1.)
    Dm = (1-np.exp(-AG[:,None,None,None]*tau[None,None]**BG[None,:,None,None])) \
         /(1-np.exp(-AG[:,None,None,None]))
    S = np.einsum('ijkm,ijkm->ijk', Dm-y, Dm-y)
    i,j,k = np.unravel_index(np.argmin(S), S.shape)
    def res(p): return Dg(t/p[0], p[1], p[2]) - y
    try:
        return least_squares(res, [TG[k], AG[i], BG[j]],
                             bounds=([lo,.3,.3],[hi,15.,1.]),
                             method='trf', max_nfev=1500).x[0]
    except Exception:
        return TG[k]

def feats(T, ab, b, rng):
    t = np.linspace(TFEAT/12, TFEAT, 12)
    y = Dg(t/T, ab, b) + SIG*rng.normal(size=12)
    return np.array([1., y[-1], y[-1]-y[5], np.log(np.var(np.diff(y))+1e-12)])

# ---------------- Table: decouple -------------------------------------------
print("=== tab:decouple (fixed rate): label RMS ===")
print(f"{'arm':24s} " + " ".join(f"{c:>7.2f}" for c in GRID))
rows = []
for lab, n, orc in [('estimated shape, n=40',40,False), ('estimated shape, n=120',120,False),
                    ('estimated shape, n=240',240,False), ('ORACLE shape, n=120',120,True)]:
    r = []
    for c in GRID:
        e = []
        for s in range(4):
            rng = np.random.default_rng(31+int(c*1000)+s)
            cells = [draw(rng) for _ in range(n)]
            w  = [win_of(T,ab,b,c,rng) for T,ab,b in cells]
            Tt = np.array([T for T,_,_ in cells])
            e.append(np.abs(impute(w, orc)-Tt)/Tt)
        r.append(float(np.sqrt(np.mean(np.concatenate(e)**2))))
    rows.append((lab, r)); print(f"{lab:24s} " + " ".join(f"{v:7.3f}" for v in r))
ORC = rows[-1][1]

# ---------------- Table: boundary -------------------------------------------
print("\n=== tab:boundary (fixed rate): oracle arm vs Def 4.1 ===")
eT = np.array([etaT(c) for c in GRID]); orc = np.array(ORC)
fit = np.array(GRID) <= 0.65
Pi = float(np.mean(orc[fit]**2*eT[fit])*JT1)
no_b = np.sqrt(Pi/(JT1*eT))
wi_b = np.sqrt(1./(JT1*eT/Pi + np.where(np.array(GRID) >= 1.0, JBND, 0.)))
print(f"J_T(1)={JT1:.1f}  J_bnd={JBND:.1f}  ratio={JBND/JT1:.2f}  Pi(c<=0.65)={Pi:.1f}")
print(f"{'c':>5s} {'measured':>9s} {'path only':>10s} {'+boundary':>10s} {'r path':>7s} {'r +bnd':>7s}")
for i,c in enumerate(GRID):
    tag = "" if fit[i] else "  <- OUT OF SAMPLE"
    print(f"{c:5.2f} {orc[i]:9.3f} {no_b[i]:10.3f} {wi_b[i]:10.3f} "
          f"{no_b[i]/orc[i]:7.2f} {wi_b[i]/orc[i]:7.2f}{tag}")
oos = ~fit
print(f"out-of-sample mean |log ratio|: path {np.mean(np.abs(np.log(no_b[oos]/orc[oos]))):.3f}"
      f"  +boundary {np.mean(np.abs(np.log(wi_b[oos]/orc[oos]))):.3f}")

# ---------------- Table: e3indep --------------------------------------------
print("\n=== tab:e3indep (fixed rate): converged independent vs partial pooling ===")
K0, C, REPS = 0.05, 42.0, 8
rng = np.random.default_rng(999)
tri = [draw(rng) for _ in range(400)]
Tte = np.array([x[0] for x in tri]); Xte = np.array([feats(*x,rng) for x in tri])
for arm in ['independent', 'partial']:
    print(f"-- {arm} --")
    print(f"{'c':>5s} {'n':>4s} {'RMSE':>8s} {'labMed':>8s} {'labP90':>8s}")
    for c in GRID:
        v, md, p9 = [], [], []
        for r in range(REPS):
            rr = np.random.default_rng(int(c*1000)*17+r)
            n = max(5, int(round(C/(K0+c))))
            cells = [draw(rr) for _ in range(n)]
            w  = [win_of(T,ab,b,c,rr) for T,ab,b in cells]
            X  = np.array([feats(T,ab,b,rr) for T,ab,b in cells])
            Tt = np.array([T for T,_,_ in cells])
            yl = (np.array([indep_global(t,y) for t,y in w]) if arm=='independent'
                  else impute(w, False))
            e  = np.abs(yl-Tt)/Tt
            mu = np.linalg.solve(X.T@X+1e-4*np.eye(4), X.T@yl)
            v.append(np.sqrt(np.mean((Xte@mu-Tte)**2)))
            md.append(np.median(e)); p9.append(np.percentile(e,90))
        print(f"{c:5.2f} {n:4d} {np.mean(v):8.1f} {np.mean(md):8.3f} {np.mean(p9):8.3f}")
