"""
E3 pilot v3 -- budget sweep with HIERARCHICAL (partially pooled) label imputation.

Motivation: fitting (T_i, alpha_i, b_i) independently per cell is degenerate under
censoring (v2: every design worse than predicting the fleet mean).  A practitioner
shares the degradation SHAPE across the fleet.  That couples the two channels:
more cells (small c) -> better pooled shape -> better labels for every cell.
Neither Thm 1 nor Thm 3 currently models this breadth benefit.

Three label-imputation arms, same budget sweep n(c) = C/(kappa0+c):
  indep    : per-cell joint fit of (T_i, alpha_i, b_i)              [= v2 baseline]
  pooled   : complete pooling of shape, block-coordinate descent
  partial  : pooled shape + per-cell shrinkage (empirical-Bayes ridge)
"""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.integrate import quad

ALPHA_B, B_STR = 3.0, 0.7
T_MEAN, T_CV, SIGMA, M_OBS, SH_CV = 806.0, 0.46, 0.05, 60, 0.25
T_FEAT = 121.0
TGRID  = 320

def Dg(tau, ab, b):
    tau = np.clip(tau, 1e-9, 1.0)
    return (1 - np.exp(-ab*tau**b))/(1 - np.exp(-ab))

def draw(rng):
    s = np.sqrt(np.log(1+T_CV**2))
    T  = rng.lognormal(np.log(T_MEAN)-0.5*s**2, s)
    ab = ALPHA_B*np.exp(SH_CV*rng.normal())
    b  = float(np.clip(B_STR*np.exp(SH_CV*rng.normal()), .3, 1.))
    return T, ab, b

def window(T, ab, b, c, rng):
    t = np.linspace(T/M_OBS, c*T, M_OBS)
    return t, Dg(t/T, ab, b) + SIGMA*rng.normal(size=M_OBS)

# ---------- T given shape: vectorised 1-D grid + golden refine ---------------
def fit_T(t, y, ab, bb):
    lo, hi = t[-1]*1.0001, t[-1]*25.0
    G  = np.linspace(lo, hi, TGRID)
    R  = Dg(t[None, :]/G[:, None], ab, bb) - y[None, :]
    T0 = G[np.argmin(np.einsum('ij,ij->i', R, R))]
    d  = (hi-lo)/TGRID
    return minimize_scalar(lambda T_: np.sum((y-Dg(t/T_, ab, bb))**2),
                           bounds=(max(lo, T0-2*d), min(hi, T0+2*d)),
                           method='bounded', options={'xatol': 1e-3}).x

# ---------- shape given all T: pooled 2-D least squares ---------------------
def fit_shape(win, Ts, p0):
    def res(p):
        ab, bb = p
        return np.concatenate([y - Dg(t/T_, ab, bb) for (t, y), T_ in zip(win, Ts)])
    try:
        r = least_squares(res, p0, bounds=([.3, .3], [15., 1.]),
                          method='trf', max_nfev=120)
        return r.x
    except Exception:
        return np.array(p0)

def impute_pooled(win, n_iter=4, partial=False):
    ab, bb = ALPHA_B, B_STR
    Ts = np.array([fit_T(t, y, ab, bb) for t, y in win])
    for _ in range(n_iter):
        ab, bb = fit_shape(win, Ts, [ab, bb])
        Ts = np.array([fit_T(t, y, ab, bb) for t, y in win])
    if not partial:
        return Ts
    # empirical-Bayes shrinkage: per-cell shape with ridge pull to (ab,bb)
    lam = 1.0/max(SH_CV**2, 1e-3) * SIGMA**2 * M_OBS
    out = np.empty(len(win))
    for i, (t, y) in enumerate(win):
        def res(p):
            T_, la, lb = p
            pen = np.sqrt(lam)*np.array([la-np.log(ab), lb-np.log(bb)])
            return np.concatenate([y - Dg(t/T_, np.exp(la), np.clip(np.exp(lb), .3, 1.)), pen])
        try:
            r = least_squares(res, [Ts[i], np.log(ab), np.log(bb)],
                              bounds=([t[-1]*1.0001, np.log(.3), np.log(.3)],
                                      [t[-1]*25., np.log(15.), 0.0]),
                              method='trf', max_nfev=150)
            out[i] = r.x[0]
        except Exception:
            out[i] = Ts[i]
    return out

def impute_indep(win):
    out = np.empty(len(win))
    for i, (t, y) in enumerate(win):
        lo, hi = t[-1]*1.0001, t[-1]*25.
        best, bl = t[-1]*1.3, np.inf
        for T0 in (t[-1]*1.2, T_MEAN):
            try:
                r = least_squares(lambda p: y - Dg(t/p[0], p[1], p[2]),
                                  [np.clip(T0, lo, hi), ALPHA_B, B_STR],
                                  bounds=([lo, .3, .3], [hi, 15., 1.]),
                                  method='trf', max_nfev=200)
                if r.cost < bl: bl, best = r.cost, r.x[0]
            except Exception: pass
        out[i] = best
    return out

def feats(T, ab, b, rng):
    t = np.linspace(T_FEAT/12, T_FEAT, 12)
    y = Dg(t/T, ab, b) + SIGMA*rng.normal(size=12)
    return np.array([1., y[-1], y[-1]-y[5], np.log(np.var(np.diff(y))+1e-12)])

def campaign(c, C, k0, rng, Xte, Tte, arm):
    n = max(5, int(round(C/(k0+c))))
    cells = [draw(rng) for _ in range(n)]
    win   = [window(T, ab, b, c, rng) for T, ab, b in cells]
    X     = np.array([feats(T, ab, b, rng) for T, ab, b in cells])
    Ttrue = np.array([T for T, _, _ in cells])
    if   arm == 'indep':   yl = impute_indep(win)
    elif arm == 'pooled':  yl = impute_pooled(win, partial=False)
    else:                  yl = impute_pooled(win, partial=True)
    lab = float(np.median(np.abs(yl-Ttrue)/Ttrue))
    mu  = np.linalg.solve(X.T@X + 1e-4*np.eye(4), X.T@yl)
    return float(np.sqrt(np.mean((Xte@mu - Tte)**2))), n, lab

def theory_cstar(k0, Lam):
    dp = lambda t: ALPHA_B*B_STR*t**(B_STR-1)*np.exp(-ALPHA_B*t**B_STR)/(1-np.exp(-ALPHA_B))
    Z  = quad(lambda t: t**2*dp(t)**2, 0, 1, limit=400)[0]
    eT = lambda c: quad(lambda t: t**2*dp(t)**2, 0, c, limit=400)[0]/Z
    return minimize_scalar(lambda c: (k0+c)*(1/eT(c)+Lam), bounds=(.05, 1.),
                           method='bounded', options={'xatol': 1e-9}).x

if __name__ == '__main__':
    K0, LAM, REPS = 0.05, 0.241, 12
    C = 40*(K0+1.)
    grid = [0.15, 0.25, 0.35, 0.5, 0.65, 0.8, 1.0]
    rng  = np.random.default_rng(999)
    tri  = [draw(rng) for _ in range(500)]
    Tte  = np.array([x[0] for x in tri])
    Xte  = np.array([feats(*x, rng) for x in tri])

    print(f"budget C={C:.1f} (=40 cells to failure)  kappa0={K0}  reps={REPS}")
    print(f"trivial baseline (predict fleet mean) RMSE = {T_CV*T_MEAN:.0f}")
    print(f"theory c* = {theory_cstar(K0, LAM):.3f}\n")
    for arm in ['indep', 'pooled', 'partial']:
        print(f"--- arm: {arm} ---")
        print(f"{'c':>5s} {'n':>4s} {'RMSE':>8s} {'s.e.':>6s} {'rel':>6s} {'label medAE':>12s}")
        out = []
        for c in grid:
            v, lb = [], []
            for r in range(REPS):
                rr = np.random.default_rng(int(c*1000)*131 + r)
                a, n, l = campaign(c, C, K0, rr, Xte, Tte, arm)
                v.append(a); lb.append(l)
            v = np.array(v)
            out.append((c, n, v.mean(), v.std(ddof=1)/np.sqrt(REPS), np.mean(lb)))
        bm = min(o[2] for o in out)
        for c, n, m, se, l in out:
            print(f"{c:5.2f} {n:4d} {m:8.1f} {se:6.1f} {m/bm:6.3f} {l:12.3f}"
                  + ("  <== optimum" if m == bm else ""))
        print()
