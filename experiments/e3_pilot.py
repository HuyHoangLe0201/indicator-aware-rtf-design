"""
E3 pilot -- budget-constrained design sweep on a synthetic fleet matched to
Severson battery parameters (Paper 1, Table 2).

Design question: at FIXED rig-time budget C, is it better to run few cells to
failure (depth) or many cells partially (breadth)?   n(c) = C/(kappa0 + c).

Theory (Thm 3):  c* = 1  iff  Upsilon >= (1+Lambda)/(1+kappa0).
Battery: Upsilon = 0.290, Lambda = 0.241  ->  INTERIOR optimum, c* = 0.385 at kappa0=0.05.

Deployment constraint (as in Severson et al.): the *deployed* model must predict
cycle life from an early fixed window t_feat.  The design variable c controls only
how long each cell is CYCLED during the lab campaign, i.e. how accurate the
training LABEL is.  This isolates the label-quality channel from the feature channel.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.integrate import quad

rng_global = np.random.default_rng(0)

# ---------------- fleet / degradation model (Paper 1 eq. 1) ------------------
ALPHA_B, B_STR = 3.0, 0.7          # stretched-exponential shape
T_MEAN, T_CV   = 806.0, 0.46       # Severson cycle life mean / CV
SIGMA          = 0.05              # noise on normalised degradation scale
T_FEAT         = 121.0             # deployment feature window (~0.15 * T_MEAN)
M_OBS          = 60                # measurement points per observed window

def D(tau):
    tau = np.clip(tau, 0.0, 1.0)
    return (1 - np.exp(-ALPHA_B * tau**B_STR)) / (1 - np.exp(-ALPHA_B))

def draw_fleet(n, rng):
    s = np.sqrt(np.log(1 + T_CV**2)); mu = np.log(T_MEAN) - 0.5 * s**2
    return rng.lognormal(mu, s, size=n)

def observe(T, c, rng):
    """noisy degradation samples on [0, c*T]"""
    t = np.linspace(T / M_OBS, c * T, M_OBS)
    return t, D(t / T) + SIGMA * rng.normal(size=M_OBS)

def impute_T(t, y):
    """NLS estimate of failure time from a censored window"""
    f = lambda Th: np.sum((y - D(t / Th))**2)
    lo, hi = t[-1] * 1.001, t[-1] * 40.0
    return minimize_scalar(f, bounds=(lo, hi), method='bounded',
                           options={'xatol': 1e-3}).x

def features(T, rng):
    """deployment features from the FIXED early window [0, T_FEAT] (Severson-style)"""
    t = np.linspace(T_FEAT / 12, T_FEAT, 12)
    y = D(t / T) + SIGMA * rng.normal(size=12)
    return np.array([1.0, y[-1], y[-1] - y[5], np.log(np.var(np.diff(y)) + 1e-12)])

# ---------------- one budget-constrained campaign ---------------------------
def campaign(c, C, kappa0, rng, Xte, Tte):
    n = max(4, int(round(C / (kappa0 + c))))
    T = draw_fleet(n, rng)
    Xtr = np.empty((n, 4)); ytr = np.empty(n)
    for i in range(n):
        t, y = observe(T[i], c, rng)
        ytr[i] = impute_T(t, y)                 # label, degraded by censoring
        Xtr[i] = features(T[i], rng)            # features, window fixed
    A = Xtr.T @ Xtr + 1e-6 * np.eye(4)
    mu = np.linalg.solve(A, Xtr.T @ ytr)
    return float(np.sqrt(np.mean((Xte @ mu - Tte)**2))), n

# ---------------- theory prediction -----------------------------------------
def theory_cstar(kappa0, Lam):
    dp = lambda t: ALPHA_B*B_STR*t**(B_STR-1)*np.exp(-ALPHA_B*t**B_STR)/(1-np.exp(-ALPHA_B))
    Z  = quad(lambda t: t**2*dp(t)**2, 0, 1, limit=400)[0]
    eT = lambda c: quad(lambda t: t**2*dp(t)**2, 0, c, limit=400)[0] / Z
    g  = lambda c: (kappa0 + c) * (1.0 / eT(c) + Lam)
    return minimize_scalar(g, bounds=(0.05, 1.0), method='bounded',
                           options={'xatol': 1e-9}).x

# ---------------- sweep ------------------------------------------------------
if __name__ == '__main__':
    KAPPA0, LAM, N_REP = 0.05, 0.241, 24
    C_BUDGET = 60 * (KAPPA0 + 1.0)          # budget = 60 cells run to failure
    grid = np.array([0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9,1.0])

    rng = np.random.default_rng(12345)
    Tte = draw_fleet(600, rng)
    Xte = np.array([features(t, rng) for t in Tte])

    print(f"budget C={C_BUDGET:.2f}  kappa0={KAPPA0}  Lambda={LAM}  reps={N_REP}")
    print(f"theory  c* = {theory_cstar(KAPPA0, LAM):.3f}\n")
    print(f"{'c':>5s} {'n(c)':>5s} {'RMSE mean':>10s} {'s.e.':>7s} {'rel.':>6s}")

    out = []
    for c in grid:
        vals = []
        for r in range(N_REP):
            rr = np.random.default_rng(1000 + int(c * 1000) * 97 + r)
            v, n = campaign(c, C_BUDGET, KAPPA0, rr, Xte, Tte)
            vals.append(v)
        vals = np.array(vals)
        out.append((c, n, vals.mean(), vals.std(ddof=1) / np.sqrt(N_REP)))
    best = min(out, key=lambda z: z[2])
    for c, n, m, se in out:
        print(f"{c:5.2f} {n:5d} {m:10.2f} {se:7.2f} {m/best[2]:6.3f}"
              + ("   <== empirical optimum" if c == best[0] else ""))
    print(f"\nempirical c* = {best[0]:.2f} (n={best[1]}),  theory c* = {theory_cstar(KAPPA0,LAM):.3f}")
