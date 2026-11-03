"""Global form of Proposition 6 (the bridge between the predictive/odds estimator and Theorem 2).

The risk is G(c) = (kappa0 + c) [Lambda + 1/(eta(c) + eta0)] on the windows c in [c_min, 1].
With b = (1+kappa0)/(1+eta0) and D(c) = Lambda (1-c) + b define the envelope
        eta_*(c) = (kappa0 + c)/D(c) - eta0 .
Exact global statement: G(c) >= G(1) for all c in [c_min,1)  <=>  eta(c) <= eta_*(c) on [c_min,1).
eta_* is convex, eta_*(1) = 1 and eta_*'(1) = theta_b, so the local rule of Proposition 6(b) is the tangent of the envelope.
Corollaries checked here:
  (1) exactness of the envelope criterion against the grid minimizer, for arbitrary nondecreasing curves and any c_min;
  (2) the chord slope Ups_g = inf_c (1-eta(c))/(1-c) <= Ups: Ups_g >= theta_b implies a global corner; for concave eta Ups_g = Ups,
      so the local condition is also global;
  (3) the power class eta = c^s: the corner is global on [c_min,1] iff s >= s_g(c_min) = max_c ln eta_*(c)/ln c; s_g = theta_b iff
      c_min >= c_0 and a curvature condition, with c_0 = 0 when omega_bar >= (Lambda+b)/kappa0 (generalized Descartes rule);
  (4) the certificate on the Delta-Q(V) curve: a single measured window with eta(c) > eta_*(c) proves that stopping there beats the corner
      in the model; the share of bootstrap resamples that carry such a window, for the four values of Lambda.
Usage: python bridge_global.py
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
K0 = 0.05


def b_of(eta0, k0):
    return (1 + k0) / (1 + eta0)


def phi(c, lam, eta0, k0):
    return (k0 + c) / (lam * (1 - c) + b_of(eta0, k0)) - eta0


def theta_b(lam, eta0, k0):
    return (1 + eta0) * (1 + lam * (1 + eta0)) / (1 + k0)


def G(eta, c, lam, eta0, k0):
    return (k0 + c) * (lam + 1.0 / (eta + eta0))


rng = np.random.default_rng(2026)
GRID = np.linspace(1e-4, 1.0, 4001)


def random_curve(kind):
    c = GRID
    if kind == 'concave':
        t = rng.integers(3)
        if t == 0:
            e = c ** rng.uniform(0.2, 1.0)
        elif t == 1:
            a = rng.uniform(0.5, 12)
            e = (1 - np.exp(-a * c)) / (1 - np.exp(-a))
        else:
            p = rng.uniform(1.0, 8.0)
            e = 1 - (1 - c) ** p
        return e
    if kind == 'power':
        return c ** rng.uniform(1.0, 14.0)
    if kind == 'sigmoid':
        k, m = rng.uniform(3, 20), rng.uniform(0.2, 0.8)
        f = 1 / (1 + np.exp(-k * (c - m)))
        return (f - f[0]) / (f[-1] - f[0])
    inc = rng.gamma(rng.uniform(0.3, 3.0), 1.0, len(c))       # arbitrary monotone
    if rng.random() < 0.5:
        inc = np.convolve(inc, np.ones(60) / 60, mode='same')
    e = np.cumsum(inc)
    return (e - e[0]) / (e[-1] - e[0])


def chord_slope(e):
    m = GRID < 1.0
    return np.min((1 - e[m]) / (1 - GRID[m]))


print('(1) envelope criterion against the grid minimizer, arbitrary nondecreasing curves')
tot = ok = 0
tot_cc = ok_cc = 0
viol_chord = n_chord = 0
conc_tot = conc_ok = 0
for kind in ('concave', 'power', 'sigmoid', 'random'):
    t_k = ok_k = 0
    for _ in range(2500):
        e = random_curve(kind)
        lam, eta0, k0 = rng.uniform(0.05, 3.0), 10 ** rng.uniform(-2.0, 0.5), rng.uniform(0.01, 0.2)
        cmin = rng.choice([1e-4, 0.02, 0.05, 0.1, 0.2, 0.4])
        m = GRID >= cmin
        g = G(e, GRID, lam, eta0, k0)
        corner_grid = g[-1] <= g[m].min() * (1 + 1e-12)
        env = np.max(e[m][:-1] - phi(GRID[m][:-1], lam, eta0, k0)) <= 1e-12
        t_k += 1
        ok_k += int(corner_grid == env)
        # (2) chord criterion
        ug = np.min((1 - e[m][:-1]) / (1 - GRID[m][:-1]))
        th = theta_b(lam, eta0, k0)
        if ug >= th:
            n_chord += 1
            viol_chord += int(not corner_grid)
        if kind == 'concave':
            ups = (1 - e[-2]) / (1 - GRID[-2])
            conc_tot += 1
            conc_ok += int((ups >= th) == corner_grid)
    print(f'    {kind:8s}: envelope <=> grid minimizer in {ok_k}/{t_k} draws')
    tot += t_k
    ok += ok_k
print(f'    all     : {ok}/{tot}')
# more draws in the region where the chord condition can hold (small Lambda, small eta0, steep curves)
for _ in range(8000):
    e = random_curve(rng.choice(['concave', 'power', 'sigmoid', 'random']))
    lam, eta0, k0 = rng.uniform(0.02, 0.4), 10 ** rng.uniform(-2.5, -0.5), rng.uniform(0.01, 0.1)
    cmin = rng.choice([1e-4, 0.05, 0.2])
    m = GRID >= cmin
    ug = np.min((1 - e[m][:-1]) / (1 - GRID[m][:-1]))
    if ug >= theta_b(lam, eta0, k0):
        g = G(e, GRID, lam, eta0, k0)
        n_chord += 1
        viol_chord += int(not (g[-1] <= g[m].min() * (1 + 1e-12)))
print(f'(2) chord slope >= theta_b implies a global corner: violated in {viol_chord} of {n_chord} draws that satisfy it')
print(f'    concave curves: (local condition) <=> (global corner) in {conc_ok}/{conc_tot} draws')

print('(3) power class eta = c^s')


def s_g(lam, eta0, k0, cmin):
    c = np.linspace(cmin, 1 - 1e-7, 40001)
    f = phi(c, lam, eta0, k0)
    if np.any(f <= 0):
        return np.inf
    return np.max(np.log(f) / np.log(c))


def s_grid(lam, eta0, k0, cmin):
    lo, hi = 1.0, 400.0
    cg = np.linspace(cmin, 1.0, 20001)
    def corner(s):
        g = G(cg ** s, cg, lam, eta0, k0)
        return g[-1] <= g.min() * (1 + 1e-12)
    if not corner(hi):
        return np.inf
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if corner(mid):
            hi = mid
        else:
            lo = mid
    return hi


def root_c0(lam, eta0, k0, s):
    c = np.linspace(1e-6, 1 - 1e-6, 400001)
    P = (c ** s + eta0) * (lam * (1 - c) + b_of(eta0, k0)) - k0 - c
    sg = np.sign(P)
    idx = np.where(sg[:-1] * sg[1:] < 0)[0]
    return c[idx[0]] if len(idx) else 0.0, len(idx)


r2 = np.random.default_rng(5)
agree = tot = 0
eq_th = tot_eq = 0
extra_roots = []
case = {'a0<0': [0, 0], 'a0>0,curv': [0, 0], 'a0>0,nocurv': [0, 0]}
for _ in range(1500):
    lam, eta0, k0 = r2.uniform(0.05, 3.0), 10 ** r2.uniform(-2.0, 0.5), r2.uniform(0.01, 0.2)
    th = theta_b(lam, eta0, k0)
    if th <= 1.0:
        continue
    cmin = r2.choice([0.02, 0.05, 0.1, 0.2, 0.3, 0.5])
    a = s_g(lam, eta0, k0, cmin)
    g = s_grid(lam, eta0, k0, cmin)
    tot += 1
    agree += int((np.isinf(a) and np.isinf(g)) or abs(a - g) < 0.02 * max(1, g if np.isfinite(g) else 1))
    b = b_of(eta0, k0)
    a0 = eta0 * (lam + b) - k0
    curv = lam * (2 * b - 1 - k0) >= b * (1 - b)
    c0, nroots = root_c0(lam, eta0, k0, th)
    extra_roots.append(nroots)
    pred_eq = (a0 < 0) or (curv and cmin >= c0)
    key = 'a0<0' if a0 < 0 else ('a0>0,curv' if curv else 'a0>0,nocurv')
    equal = np.isfinite(a) and abs(a - th) < 1e-3 * th
    case[key][0] += int(equal == pred_eq)
    case[key][1] += 1
    tot_eq += 1
    eq_th += int(equal == pred_eq)
print(f'    s_g(c_min) = max ln eta_*/ln c against bisection on the grid minimizer: agree in {agree}/{tot} draws')
print(f'    s_g = theta_b  <=>  [a0 < 0]  or  [curvature condition and c_min >= c_0]: right in {eq_th}/{tot_eq}', case)
print(f'    roots of P in (0,1) at s = theta_b: max {max(extra_roots)} (Descartes bound: 1)')

print('(4) certificate on the Delta-Q(V) curve (dqv_mar.py, bootstrap over cells)')
import dqv_mar as dm
X, y, life = dm.load()
n = len(y)
om, eta, c, sl, sat, pl = dm.summarize(X, y, np.arange(n), dm.odds_mar)
OM, SL, SAT, PL, ET = dm.boot(X, y, dm.odds_mar, 2000, 20260803)
LAMS = (0.241, 0.30, 0.58, 0.86)


def certificate(eta_row, pl_, lam, informative=1.0):
    """Largest eta - eta_* over the observed windows with c < 1 and odds >= informative; and the model risk ratio there."""
    eta0 = 1.0 / pl_
    sel = np.isfinite(eta_row) & (c < 1.0) & (eta_row * pl_ >= informative)
    if not sel.any():
        return -np.inf, np.nan, np.nan
    mar = eta_row[sel] - phi(c[sel], lam, eta0, K0)
    j = int(np.argmax(mar))
    cc, ee = c[sel][j], eta_row[sel][j]
    ratio = G(ee, cc, lam, eta0, K0) / G(1.0, 1.0, lam, eta0, K0)
    return float(mar[j]), float(cc), float(ratio)


print(f'    point estimate (odds at c = 1: {pl:.2f}); windows c = {np.round(c[c < 1], 2).tolist()}')
for lam in LAMS:
    eta0 = 1 / pl
    th = theta_b(lam, eta0, K0)
    mar, cc, ratio = certificate(eta, pl, lam)
    print(f'    Lambda={lam:.3f}: theta_b={th:.2f}; best window c={cc:.2f}, eta={float(np.interp(cc, c, eta)):.2f} vs envelope '
          f'{phi(cc, lam, eta0, K0):.2f}; margin {mar:+.2f}; model risk at that stop / corner = {ratio:.2f}')
print(f'    bootstrap over cells ({len(PL)} usable resamples):')
for lam in LAMS:
    res = [certificate(ET[i], PL[i], lam) for i in range(len(PL))]
    mars = np.array([r[0] for r in res])
    ccs = np.array([r[1] for r in res])
    ratios = np.array([r[2] for r in res])
    share = np.mean(mars > 0) * 100
    slope_share = np.mean(SL < np.array([theta_b(lam, 1 / p, K0) for p in PL])) * 100
    print(f'    Lambda={lam:.3f}: a window beats the corner in {share:.1f}% of resamples (slope test: {slope_share:.1f}%); '
          f'median risk ratio {np.nanmedian(ratios):.2f} [{np.nanpercentile(ratios, 2.5):.2f}, {np.nanpercentile(ratios, 97.5):.2f}]; '
          f'median best window c={np.nanmedian(ccs):.2f}')
# raw, unsmoothed odds (no pool-adjacent-violators) as a sensitivity check
share_raw = []
for lam in LAMS:
    cnt = 0
    tot_ = 0
    for i in range(len(PL)):
        raw = OM[i]
        okm = np.isfinite(raw)
        if not okm.any():
            continue
        pl_i = PL[i]
        eta_raw = raw / pl_i
        mar = certificate(eta_raw, pl_i, lam)[0]
        tot_ += 1
        cnt += int(mar > 0)
    share_raw.append(100 * cnt / tot_)
print('    same test on the raw odds (no monotone fit):', ', '.join(f'{v:.1f}%' for v in share_raw))
# the certificate also holds on the lower pointwise bound of eta only if it clears the envelope there
lo_eta = np.nanpercentile(ET, 2.5, axis=0)
for lam in LAMS:
    mar, cc, ratio = certificate(lo_eta, pl, lam)
    print(f'    Lambda={lam:.3f}: on the lower 95% pointwise bound of eta, best margin {mar:+.2f} at c={cc:.2f}')
