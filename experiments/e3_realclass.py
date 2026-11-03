"""
E3 re-run with the CORRECT battery degradation class (Erratum: convex power law).

  D(tau) = tau^beta,  beta ~ lognormal(median 4.483, log-CV 0.297)   [fitted, 124 cells]
  fleet lifetimes    = the 124 REAL Severson cycle lives
  sampling           = fixed rate, M(c) = max(8, round(c*M0))
  sigma              = 0.05 on the normalised degradation scale (Paper 1 Table 2)

Two feature arms, because a fixed early window may carry no signal under a convex class:
  'early'    features from a fixed 121-cycle window (Severson-style deployment constraint)
  'observed' features from the observed window [0, cT] itself (upper bound on what is learnable)
"""
import numpy as np, os
from scipy.optimize import least_squares, minimize_scalar

BETA0, LOGCV, SIG, M0 = 4.483, 0.297, 0.05, 60
TFEAT, LAM = 121.0, 0.1*(1./0.297**2)*0.05**2*60
HERE = os.path.join(os.path.dirname(__file__), '..', 'data', 'severson')
LIVES = np.load(os.path.join(HERE, 'severson_traj.npz'), allow_pickle=True)['life']
Mof = lambda c: max(8, int(round(c*M0)))
Dg  = lambda tau, b: np.clip(tau, 1e-9, 1.0)**b

def draw(rng):
    return (float(rng.choice(LIVES)), float(BETA0*np.exp(LOGCV*rng.normal())))

def window(T, b, c, rng):
    M = Mof(c); t = np.linspace(T/M0, c*T, M)
    return t, Dg(t/T, b) + SIG*rng.normal(size=M)

def fT(t, y, b):
    return minimize_scalar(lambda T_: np.sum((y-Dg(t/T_, b))**2),
                           bounds=(t[-1]*1.0001, t[-1]*25.), method='bounded',
                           options={'xatol':1e-3}).x

def impute(win):
    """partially pooled: shared beta by complete pooling, then per-unit shrinkage"""
    b = BETA0
    Ts = np.array([fT(t, y, b) for t, y in win])
    for _ in range(3):
        r = lambda p: np.concatenate([Dg(t/T_, p[0])-y for (t, y), T_ in zip(win, Ts)])
        try:
            b = float(least_squares(r, [b], bounds=([.4], [15.]),
                                    method='trf', max_nfev=150).x[0])
        except Exception:
            pass
        Ts = np.array([fT(t, y, b) for t, y in win])
    out = np.empty(len(win))
    for i, (t, y) in enumerate(win):
        lo, hi = t[-1]*1.0001, t[-1]*25.
        def res(p):
            return np.concatenate([Dg(t/p[0], np.exp(p[1]))-y,
                                   [np.sqrt(LAM)*(p[1]-np.log(b))]])
        try:
            out[i] = least_squares(res, [Ts[i], np.log(b)],
                                   bounds=([lo, np.log(.4)], [hi, np.log(15.)]),
                                   method='trf', max_nfev=400).x[0]
        except Exception:
            out[i] = Ts[i]
    return out

def feats(T, b, c, rng, arm):
    hi = TFEAT if arm == 'early' else c*T
    t = np.linspace(hi/12, hi, 12)
    y = Dg(t/T, b) + SIG*rng.normal(size=12)
    return np.array([1., y[-1], y[-1]-y[5], np.log(np.var(np.diff(y))+1e-12)])

def campaign(c, C, k0, rng, arm, Xte, Tte):
    n = max(5, int(round(C/(k0+c))))
    cells = [draw(rng) for _ in range(n)]
    win = [window(T, b, c, rng) for T, b in cells]
    X   = np.array([feats(T, b, c, rng, arm) for T, b in cells])
    Tt  = np.array([T for T, _ in cells])
    yl  = impute(win)
    mu  = np.linalg.solve(X.T@X + 1e-4*np.eye(4), X.T@yl)
    return (float(np.sqrt(np.mean((Xte@mu-Tte)**2))), n,
            float(np.median(np.abs(yl-Tt)/Tt)))

if __name__ == '__main__':
    K0, REPS, C = 0.05, 10, 40*1.05
    grid = [0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 1.00]
    base = LIVES.std(ddof=1)
    print(f"fleet: n={len(LIVES)} real Severson lives, mean {LIVES.mean():.0f}, "
          f"sd {base:.0f}")
    print(f"class: D=tau^beta, beta median {BETA0}, log-CV {LOGCV};  "
          f"eta_T(c)=c^{2*BETA0+1:.2f}")
    print(f"trivial baseline (predict fleet mean) RMSE = {base:.0f}\n")
    for arm in ['early', 'observed']:
        rng = np.random.default_rng(7)
        tri = [draw(rng) for _ in range(500)]
        Tte = np.array([x[0] for x in tri])
        print(f"--- feature arm: {arm} ---")
        print(f"{'c':>5s} {'M(c)':>5s} {'n':>4s} {'RMSE':>8s} {'s.e.':>6s} {'rel':>6s} "
              f"{'labMed':>8s} {'eta_T(c)':>10s}")
        out = []
        for c in grid:
            Xte = np.array([feats(T, b, c, rng, arm) for T, b in tri])
            v, l = [], []
            for r in range(REPS):
                rr = np.random.default_rng(int(c*1000)*29+r)
                a, n, lb = campaign(c, C, K0, rr, arm, Xte, Tte)
                v.append(a); l.append(lb)
            v = np.array(v)
            out.append((c, n, v.mean(), v.std(ddof=1)/np.sqrt(REPS), np.mean(l)))
        bm = min(o[2] for o in out)
        for c, n, m, se, lb in out:
            print(f"{c:5.2f} {Mof(c):5d} {n:4d} {m:8.1f} {se:6.1f} {m/bm:6.3f} "
                  f"{lb:8.3f} {c**(2*BETA0+1):10.2e}"
                  + ("  <== optimum" if m == bm else ""))
        print()
