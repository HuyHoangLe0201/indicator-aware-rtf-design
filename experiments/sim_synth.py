"""Controlled synthetic study with systematically varied Upsilon (R2.7).

World: ln T_i = ln 200 + 0.35 z_i;  health curve D(tau; a,b) = (1-exp(-a tau^b))/(1-exp(-a)), one shape per fleet
(shape differs across fleets, so Upsilon varies from ~0 to >10);  readings every T_med/60 cycles with white noise sd 0.05;
an early-window predictor f_i = ln T_i + N(0, 0.30^2) (the same for every c; only the labels depend on c).
Campaign: budget C, n = C/(kappa0+c) units, each observed to its own fraction c, label imputed by least squares on the
pooled shape (fitted once on ten complete units), exact failure time at c=1 (observation model O1).
Predictor: OLS of the label on f.  Outcome: MSE of ln T prediction on 600 fresh units.
Theory: Upsilon from the shape; threshold (1+Lam)/(1+kappa0) with Lam = sigma_res^2 / v(1-) ; predicted optimum from
R(c) = (kappa0+c)(sigma_res^2 + v(c))  (Proposition 1), v(c)= measured variance of the imputed label error.
"""
import numpy as np
from scipy.optimize import least_squares
rng = np.random.default_rng(11)
import os
K0, C, SIGN, SU, SF = 0.05, 20.0, float(os.environ.get("SIGN", "0.20")), 0.35, 0.30
TMED = 200.0
DT = TMED / 40.0
SRES2 = SU**2 * SF**2 / (SU**2 + SF**2)
CGRID = np.array([0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
POOL, NTEST, REPS = 1500, 600, 400

def Dcurve(t, a, b):
    return (1 - np.exp(-a * np.power(np.clip(t, 0, None), b))) / (1 - np.exp(-a))

def Dprime(t, a, b):
    return a * b * np.power(t, b - 1) * np.exp(-a * np.power(t, b)) / (1 - np.exp(-a))

def upsilon(a, b):
    tt = np.linspace(1e-6, 1, 200001)
    den = np.trapezoid(tt**2 * Dprime(tt, a, b)**2, tt)
    return Dprime(1.0, a, b)**2 / den

def simulate_unit(T, a, b, cfrac):
    n = int(np.floor(cfrac * T / DT))
    t = DT * np.arange(1, n + 1)
    x = Dcurve(t / T, a, b) + SIGN * rng.standard_normal(n)
    return t, x

def fit_shape(a0, b0):
    ts = np.exp(np.log(TMED) + SU * rng.standard_normal(10))
    T_all, X_all = [], []
    for T in ts:
        t, x = simulate_unit(T, a0, b0, 1.0)
        T_all.append(t / T); X_all.append(x)
    tau = np.concatenate(T_all); x = np.concatenate(X_all)
    f = lambda p: Dcurve(tau, np.exp(p[0]), np.exp(p[1])) - x
    r = least_squares(f, [np.log(max(a0, 0.05)), np.log(max(b0, 0.5))], bounds=([-5, -2], [4, 2.6]))
    return float(np.exp(r.x[0])), float(np.exp(r.x[1]))

GRIDF = np.exp(np.linspace(np.log(1.0005), np.log(12.0), 220))

def impute(t, x, ah, bh):
    if len(t) < 3:
        return t[-1] * 2.0 if len(t) else TMED
    Tc = t[-1] * GRIDF
    pred = Dcurve(t[None, :] / Tc[:, None], ah, bh)
    sse = ((pred - x[None, :])**2).sum(1)
    return Tc[np.argmin(sse)]

def label_pool(a, b, ah, bh):
    """Pool of units; imputed ln-labels for every c in CGRID; returns arrays."""
    lnT = np.log(TMED) + SU * rng.standard_normal(POOL)
    T = np.exp(lnT)
    f = lnT + SF * rng.standard_normal(POOL)
    lab = np.zeros((len(CGRID), POOL))
    for j, c in enumerate(CGRID):
        for i in range(POOL):
            if c >= 1.0:
                lab[j, i] = lnT[i]
            else:
                t, x = simulate_unit(T[i], a, b, c)
                lab[j, i] = np.log(impute(t, x, ah, bh))
    return lnT, f, lab

def campaigns(lnT, f, lab):
    out = np.zeros(len(CGRID))
    lnTt = np.log(TMED) + SU * rng.standard_normal(NTEST)
    ft = lnTt + SF * rng.standard_normal(NTEST)
    for j, c in enumerate(CGRID):
        n = max(4, int(round(C / (K0 + c))))
        mse = []
        for _ in range(REPS):
            idx = rng.choice(POOL, n, replace=False)
            X = np.c_[np.ones(n), f[idx]]
            beta = np.linalg.lstsq(X, lab[j, idx], rcond=None)[0]
            mse.append(np.mean((beta[0] + beta[1] * ft - lnTt)**2))
        out[j] = np.mean(mse)
    return out

def study(a, b):
    ah, bh = fit_shape(a, b)
    lnT, f, lab = label_pool(a, b, ah, bh)
    mse = campaigns(lnT, f, lab)
    v = np.array([np.var(lab[j] - lnT) + np.mean(lab[j] - lnT)**2 for j in range(len(CGRID))])   # label error second moment
    return dict(a=a, b=b, ah=ah, bh=bh, ups=upsilon(a, b), mse=mse, v=v)

def summarise(r):
    ups = r['ups']
    v = r['v']
    lam = SRES2 / v[-2] if v[-2] > 0 else np.inf          # v at c=0.9 stands in for v(1-)
    thr = (1 + lam) / (1 + K0)
    risk_pred = (K0 + CGRID) * (SRES2 + np.where(CGRID >= 1, 0.0, v))
    cstar_pred = CGRID[np.argmin(risk_pred)]
    cstar_emp = CGRID[np.argmin(r['mse'])]
    return ups, thr, cstar_pred, cstar_emp

if __name__ == '__main__':
    SHAPES = [(0.01, 1.0), (0.01, 2.0), (0.01, 3.0), (0.01, 4.5), (0.7, 1.0), (1.5, 1.0), (3.0, 1.0), (6.0, 1.0), (4.0, 2.0), (8.0, 3.0)]
    print('shape(a,b)  Upsilon  thr   pred c*  emp c*   MSE(corner)  MSE(emp c*)  MSE(pred c*)  MSE(0.5)   ratio 0.5/corner  regret(pred)')
    rows = []
    for a, b in SHAPES:
        r = study(a, b)
        ups, thr, cp, ce = summarise(r)
        m = r['mse']
        mp = m[int(np.argmin(np.abs(CGRID - cp)))]
        print(f'({a:4.1f},{b:3.1f})  {ups:7.2f}  {thr:5.2f}  {cp:6.2f}  {ce:6.2f}   {m[-1]:.4f}    {m[np.argmin(m)]:.4f}    {mp:.4f}    {m[4]:.4f}   {m[4]/m[-1]:.2f}   {mp/m[np.argmin(m)]-1:+.3f}')
        rows.append((a, b, ups, thr, cp, ce, m.tolist(), r['v'].tolist()))
    np.save(f'sim_synth_rows_{SIGN}.npy', np.array(rows, dtype=object), allow_pickle=True)
