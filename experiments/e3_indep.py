"""
E3: a PROPERLY CONVERGED independent-fit arm.

The lambda=0 arm of e3_lambda.py was warm-started from the pooled estimate with a
capped iteration budget: its label median was fine but its TAIL was not, and the
tail is what corrupts the downstream OLS.  Here the per-cell fit of (T, alpha_b, b)
is done by exhaustive 3-D grid search followed by local polish, so the global
optimum is found up to grid resolution -- no initialisation dependence.

Reported per design point: test RMSE, and the FULL label-error distribution
(median / 90th pct / max) plus the rate at which the fit sits on a bound.
"""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar

AB, BB = 3.0, 0.7
T_MEAN, T_CV, SIGMA, M_OBS, SH_CV = 806.0, 0.46, 0.05, 60, 0.25
T_FEAT = 121.0

AB_G = np.exp(np.linspace(np.log(0.6), np.log(12.0), 11))
B_G  = np.linspace(0.32, 1.00, 11)
NT   = 140

def Dg(tau, ab, b):
    tau = np.clip(tau, 1e-9, 1.0)
    return (1-np.exp(-ab*tau**b))/(1-np.exp(-ab))

def draw(rng):
    s = np.sqrt(np.log(1+T_CV**2))
    return (rng.lognormal(np.log(T_MEAN)-0.5*s**2, s),
            AB*np.exp(SH_CV*rng.normal()),
            float(np.clip(BB*np.exp(SH_CV*rng.normal()), .3, 1.)))

def window(T, ab, b, c, rng):
    t = np.linspace(T/M_OBS, c*T, M_OBS)
    return t, Dg(t/T, ab, b) + SIGMA*rng.normal(size=M_OBS)

# ---------- exhaustive 3-D grid, fully vectorised, then polish --------------
def fit_global(t, y):
    lo, hi = t[-1]*1.0001, t[-1]*25.0
    TG = np.linspace(lo, hi, NT)
    tau = t[None, :]/TG[:, None]                       # (NT, M)
    tau = np.clip(tau, 1e-9, 1.0)
    # broadcast to (nAB, nB, NT, M)
    ab = AB_G[:, None, None, None]; bb = B_G[None, :, None, None]
    Dm = (1-np.exp(-ab*tau[None, None]**bb))/(1-np.exp(-ab))
    S  = np.einsum('ijkm,ijkm->ijk', Dm-y, Dm-y)
    i, j, k = np.unravel_index(np.argmin(S), S.shape)
    p0 = [TG[k], AB_G[i], B_G[j]]
    try:
        r = least_squares(lambda p: y - Dg(t/p[0], p[1], p[2]), p0,
                          bounds=([lo, .3, .3], [hi, 15., 1.]),
                          method='trf', max_nfev=3000, xtol=1e-12, ftol=1e-12)
        Th, onb = r.x[0], (r.x[0] <= lo*1.001 or r.x[0] >= hi*0.999)
    except Exception:
        Th, onb = p0[0], True
    return Th, onb

def pooled_shape(win, n_iter=3):
    ab, bb = AB, BB
    fT = lambda t, y, a, b_: minimize_scalar(
        lambda T_: np.sum((y-Dg(t/T_, a, b_))**2),
        bounds=(t[-1]*1.0001, t[-1]*25.), method='bounded',
        options={'xatol': 1e-3}).x
    Ts = np.array([fT(t, y, ab, bb) for t, y in win])
    for _ in range(n_iter):
        res = lambda p: np.concatenate([y-Dg(t/T_, p[0], p[1])
                                        for (t, y), T_ in zip(win, Ts)])
        try:
            ab, bb = least_squares(res, [ab, bb], bounds=([.3, .3], [15., 1.]),
                                   method='trf', max_nfev=150).x
        except Exception:
            pass
        Ts = np.array([fT(t, y, ab, bb) for t, y in win])
    return ab, bb, Ts

def impute_partial(win, lam_mult=0.1):
    ab, bb, Ts = pooled_shape(win)
    lam = lam_mult*(1.0/SH_CV**2)*SIGMA**2*M_OBS
    out = np.empty(len(win))
    for i, (t, y) in enumerate(win):
        lo, hi = t[-1]*1.0001, t[-1]*25.
        def res(p):
            T_, la, lb = p
            return np.concatenate([y-Dg(t/T_, np.exp(la), np.clip(np.exp(lb), .3, 1.)),
                                   np.sqrt(lam)*np.array([la-np.log(ab), lb-np.log(bb)])])
        try:
            out[i] = least_squares(res, [Ts[i], np.log(ab), np.log(bb)],
                                   bounds=([lo, np.log(.3), np.log(.3)],
                                           [hi, np.log(15.), 0.]),
                                   method='trf', max_nfev=400).x[0]
        except Exception:
            out[i] = Ts[i]
    return out, 0.0

def impute_indep(win):
    r = [fit_global(t, y) for t, y in win]
    return np.array([x[0] for x in r]), float(np.mean([x[1] for x in r]))

def feats(T, ab, b, rng):
    t = np.linspace(T_FEAT/12, T_FEAT, 12)
    y = Dg(t/T, ab, b) + SIGMA*rng.normal(size=12)
    return np.array([1., y[-1], y[-1]-y[5], np.log(np.var(np.diff(y))+1e-12)])

def campaign(c, C, k0, rng, Xte, Tte, arm):
    n = max(5, int(round(C/(k0+c))))
    cells = [draw(rng) for _ in range(n)]
    win = [window(T, ab, b, c, rng) for T, ab, b in cells]
    X = np.array([feats(T, ab, b, rng) for T, ab, b in cells])
    Tt = np.array([T for T, _, _ in cells])
    yl, onb = impute_indep(win) if arm == 'indep' else impute_partial(win)
    e = np.abs(yl-Tt)/Tt
    mu = np.linalg.solve(X.T@X + 1e-4*np.eye(4), X.T@yl)
    return (float(np.sqrt(np.mean((Xte@mu-Tte)**2))), n,
            float(np.median(e)), float(np.percentile(e, 90)), float(e.max()), onb)

if __name__ == '__main__':
    K0, REPS, C = 0.05, 10, 40*1.05
    grid = [0.15, 0.25, 0.35, 0.5, 0.65, 0.8, 1.0]
    rng = np.random.default_rng(999)
    tri = [draw(rng) for _ in range(500)]
    Tte = np.array([x[0] for x in tri]); Xte = np.array([feats(*x, rng) for x in tri])
    print(f"budget C={C:.1f}  kappa0={K0}  reps={REPS};  trivial baseline RMSE = {T_CV*T_MEAN:.0f}")
    for arm in ['indep', 'partial']:
        lbl = 'independent (global 3-D grid + polish)' if arm == 'indep' else 'partial pooling, lam=0.1'
        print(f"\n--- {lbl} ---")
        print(f"{'c':>5s} {'n':>4s} {'RMSE':>8s} {'s.e.':>6s} {'rel':>6s} | "
              f"{'lab med':>8s} {'lab p90':>8s} {'lab max':>8s} {'onBnd%':>7s}")
        out = []
        for c in grid:
            v, md, p9, mx, ob = [], [], [], [], []
            for r in range(REPS):
                rr = np.random.default_rng(int(c*1000)*131+r)
                a, n, m_, p_, x_, o_ = campaign(c, C, K0, rr, Xte, Tte, arm)
                v.append(a); md.append(m_); p9.append(p_); mx.append(x_); ob.append(o_)
            v = np.array(v)
            out.append((c, n, v.mean(), v.std(ddof=1)/np.sqrt(REPS),
                        np.mean(md), np.mean(p9), np.mean(mx), np.mean(ob)))
        bm = min(o[2] for o in out)
        for c, n, m, se, a1, a2, a3, a4 in out:
            print(f"{c:5.2f} {n:4d} {m:8.1f} {se:6.1f} {m/bm:6.3f} | "
                  f"{a1:8.3f} {a2:8.3f} {a3:8.1f} {100*a4:7.1f}"
                  + ("  <== optimum" if m == bm else ""))
