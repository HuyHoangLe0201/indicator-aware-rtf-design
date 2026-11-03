"""Information curve of the Delta-Q(V) channel without the survivor bias of the window.

A window of m cycles can be read only on cells that live that long, so the leave-one-out R^2 of
dqv_odds_scale.py is computed on a sample whose log-life range shrinks with m (Var(log T | T >= m) is smaller) and
the curve bends down for a reason that has nothing to do with information.  The cells that died before m
have a known label y = ln T and a missing feature x_m, and the missingness depends on y alone: it is missing
at random given the observed y.  Selection on y does not bias the regression of x on y, so
        x_m = a_m + b_m y + e_m,   e_m ~ N(0, s_m^2)
fitted on the survivors is consistent for the whole fleet, and combined with the variance of y over all cells it gives
the odds of the population,  omega(m) = b_m^2 Var(y) / s_m^2   (exactly the Gaussian-linear odds of Proposition 5,
R^2 = b^2 Var(y) / (b^2 Var(y) + s^2)).  Nothing is assumed about which cells die before m.

Outputs: the corrected odds curve on a fine grid of windows with bootstrap intervals over cells, the saturation
point, the end-of-life slope on the normalized curve (central difference at the median life, and a smooth fit),
and the same slope on the plain survivors' R^2 for contrast.  A check on simulated data with a known curve shows that
the naive survivors' R^2 reproduces the artifact and the corrected one does not.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data', 'severson', 'dq_traj.npz')
W = (20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900)
MIN_N = 20


def load():
    d = np.load(DATA, allow_pickle=True)
    DQ, life = d['DQ'], d['life'].astype(float)
    X = np.full((len(life), len(W)), np.nan)
    for i in range(len(life)):
        cyc, val = DQ[i][0].astype(float), DQ[i][1].astype(float)
        for j, m in enumerate(W):
            if life[i] >= m and cyc[-1] >= m:
                X[i, j] = np.interp(m, cyc, val)
    return X, np.log(life), life


def odds_mar(x, y):
    """Population odds from the regression of x on y on the cells where x is observed; var(y) from all cells."""
    ok = np.isfinite(x)
    if ok.sum() < MIN_N:
        return np.nan
    xs, ys = x[ok], y[ok]
    A = np.column_stack([np.ones(len(ys)), ys])
    beta, *_ = np.linalg.lstsq(A, xs, rcond=None)
    res = xs - A @ beta
    s2 = res @ res / (len(ys) - 2)
    return float(beta[1] ** 2 * y.var(ddof=1) / s2)


def odds_naive(x, y):
    """Odds from the survivors' own R^2 (in-sample, adjusted): what a regression of y on x gives on the survivors."""
    ok = np.isfinite(x)
    if ok.sum() < MIN_N:
        return np.nan
    xs, ys = x[ok], y[ok]
    r = np.corrcoef(xs, ys)[0, 1]
    n = len(ys)
    r2 = 1 - (1 - r * r) * (n - 1) / (n - 2)
    r2 = min(max(r2, 0.0), 0.999)
    return r2 / (1 - r2)


def curve(X, y, idx, fn):
    return np.array([fn(X[idx, j], y[idx]) for j in range(X.shape[1])])


def slope_at(eta_c, c, c0=1.0, half=0.3):
    """Local slope of the normalized curve at c0: least-squares line through the points within +-half of c0."""
    sel = np.isfinite(eta_c) & (np.abs(c - c0) <= half)
    if sel.sum() < 3:
        return np.nan
    return float(np.polyfit(c[sel], eta_c[sel], 1)[0])


def pava(v):
    """Nondecreasing least-squares fit (pool adjacent violators): information cannot fall as the window grows."""
    out = []
    for a in v:
        out.append([a, 1])
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            a2, b2 = out.pop(); a1, b1 = out.pop()
            out.append([(a1 * b1 + a2 * b2) / (b1 + b2), b1 + b2])
    r = []
    for a, b in out:
        r += [a] * b
    return np.array(r)


def summarize(X, y, idx, fn):
    """Odds by window, their monotone fit, the value of the fit at c=1 (omega-bar), the normalized curve and its slope at c=1."""
    om = curve(X, y, idx, fn)
    med = np.median(np.exp(y[idx]))
    c = np.array(W) / med
    ok = np.isfinite(om)
    iso = np.full(len(om), np.nan)
    iso[ok] = pava(om[ok])
    pl = float(np.interp(1.0, c[ok], iso[ok]))
    eta = iso / pl
    sl = slope_at(eta, c)
    hit = np.where(ok & (eta >= 0.95))[0]
    sat = c[hit[0]] if len(hit) else np.nan
    return om, eta, c, sl, sat, pl


def boot(X, y, fn, B, seed):
    n = len(y)
    rng = np.random.default_rng(seed)
    OM, SL, SAT, PL, ET = [], [], [], [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        om, eta, c, sl, sat, pl = summarize(X, y, idx, fn)
        if np.isfinite(sl):
            OM.append(om); SL.append(sl); SAT.append(sat); PL.append(pl); ET.append(eta)
    return np.array(OM), np.array(SL), np.array(SAT), np.array(PL), np.array(ET)


def ci(v):
    v = v[np.isfinite(v)]
    return np.median(v), np.percentile(v, 2.5), np.percentile(v, 97.5)


def theta_b(plateau, lam, kappa0=0.05):
    eta0 = 1.0 / plateau
    return (1 + eta0) * (1 + lam * (1 + eta0)) / (1 + kappa0)


def sim_check(seed=3):
    """Known truth: the odds are constant at 6 for every window >= 150 (saturated).  Survivor bias should make the naive curve fall."""
    rng = np.random.default_rng(seed)
    n = 4000
    y = rng.normal(np.log(719), 0.42, n)
    life = np.exp(y)
    true_odds = lambda m: 6.0 * min(1.0, m / 150.0)
    out = []
    for m in (100, 200, 400, 700, 900):
        om = true_odds(m)
        s = np.sqrt(y.var() / om)           # x | y has residual s with  b=1 so that odds = Var(y)/s^2
        x = (y - y.mean()) + rng.normal(0, s, n)
        x = np.where(life >= m, x, np.nan)
        out.append((m, om, odds_naive(x, y), odds_mar(x, y), (life >= m).mean()))
    return out


if __name__ == '__main__':
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    X, y, life = load()
    n = len(y)
    print(f'{n} cells, median life {np.median(life):.0f} cycles; survivors by window:',
          dict(zip(W, (np.isfinite(X).sum(0)).tolist())))
    print('\nSimulated check (true odds saturate at 6; the share of survivors in the last column):')
    for m, tr, nv, mr, sh in sim_check():
        print(f'  m={m:4d} true {tr:5.2f}  naive {nv:6.2f}  corrected {mr:6.2f}  survivors {sh:.2f}')
    res = {}
    for name, fn in (('survivors R^2 (naive)', odds_naive), ('corrected for survival (x on y)', odds_mar)):
        om, eta, c, sl, sat, pl = summarize(X, y, np.arange(n), fn)
        print(f'\n=== {name} ===')
        print('  window  ' + ' '.join(f'{m:6d}' for m in W))
        print('  c       ' + ' '.join(f'{v:6.2f}' for v in c))
        print('  odds    ' + ' '.join(f'{v:6.2f}' for v in om))
        print('  eta     ' + ' '.join(f'{v:6.2f}' for v in eta))
        print(f'  value at c=1 of the monotone fit {pl:.2f}; 95% of it first reached at c={sat:.2f}; slope of the normalized curve at c=1: {sl:.2f}')
        OM, SL, SAT, PL, ET = boot(X, y, fn, B, 20260803)
        lo_om, hi_om = np.nanpercentile(OM, [2.5, 97.5], axis=0)
        res[name] = dict(om=om, eta=eta, c=c, sl=sl, pl=pl, OM=OM, SL=SL, SAT=SAT, PL=PL, ET=ET, lo=lo_om, hi=hi_om)
        m_, lo, hi = ci(SL)
        print(f'  bootstrap ({len(SL)} usable): slope at c=1 median {m_:.2f} 95% [{lo:.2f}, {hi:.2f}]')
        m_, lo, hi = ci(SAT)
        print(f'                   95%-of-plateau point median {m_:.2f} 95% [{lo:.2f}, {hi:.2f}]')
        m_, lo, hi = ci(PL)
        print(f'                   odds at c=1 median {m_:.2f} 95% [{lo:.2f}, {hi:.2f}]')
        for j, m in enumerate(W):
            a, b_, c_ = ci(OM[:, j])
            print(f'    odds at {m:4d}: {om[j]:6.2f}  boot median {a:6.2f} [{b_:6.2f}, {c_:6.2f}]')
        for lam in (0.241, 0.30, 0.58, 0.86):
            th = theta_b(pl, lam)
            th0 = (1 + lam) / 1.05
            print(f'  Lambda={lam:.3f}: corner threshold {th:.2f} (Theorem 2: {th0:.2f}); P(slope >= threshold) = {np.mean(SL >= th) * 100:.1f}%')
        print('  eta (normalized monotone fit) at c = 0.07, 0.14, 0.27, 0.4, 0.55, 0.8, 1.0:',
              [round(float(np.interp(v, c, eta)), 2) for v in (0.07, 0.14, 0.27, 0.4, 0.55, 0.8, 1.0)])
    r = res['corrected for survival (x on y)']
    eta_lo, eta_hi = np.nanpercentile(r['ET'], [2.5, 97.5], axis=0)
    np.savez(os.path.join(HERE, 'dqv_info_curve.npz'), W=np.array(W), c=r['c'], odds=r['om'], lo=r['lo'], hi=r['hi'],
             eta_lo=eta_lo, eta_hi=eta_hi,
             iso=r['eta'] * r['pl'], eta=r['eta'], plateau=r['pl'], slope=r['sl'], slopes=r['SL'], plateaus=r['PL'],
             median_life=np.median(life))
    print('\nsaved dqv_info_curve.npz')
