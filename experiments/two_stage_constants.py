"""Measure the two inflation constants of Theorem 5 instead of sweeping them.

Xi(n1) = 1 + u_Lambda + u0/n1 is the inflation a stage-two unit pays because its label is
imputed with a shape estimated from the n1 complete units of stage one. Section VI sweeps the
two constants over wide ranges and reports what survives, because neither is measured. Both
are measurable on a fleet that has complete trajectories, which the Severson set does.

Three references are needed and they differ only in where the shape comes from:

  own shape     each cell's own fitted exponent, so nothing is inflated          -> Var_0
  known centre  the fleet median exponent over all cells, the n1 -> infinity arm -> Var_inf
  n1 cells      the median exponent of n1 sampled cells                          -> Var(n1)

Then Xi(n1) = Var(n1)/Var_0, and 1 + u_Lambda = Var_inf/Var_0 is what survives with the centre
known exactly: the price of shrinking a cell to a fleet it does not sit at the middle of. The
label error is read on the robust scale of Section III-D, because on a class this convex the
second moment reports the width of the search box rather than the data.
"""
import numpy as np

DATA = r'D:\claudecode\generationbound\data\severson\dq_traj.npz'
CSTAR = 0.070
NS = (5, 8, 12, 20, 35, 60)
REPS = 400
GRID = np.geomspace(1.0, 25.0, 240)


def load():
    d = np.load(DATA, allow_pickle=True)
    life = d['life'].astype(float)
    traj = []
    for q in d['QD']:
        q = np.asarray(q, float)
        q0 = np.nanmedian(q[1:6])
        traj.append((q0 - q)/(0.2*q0 + 1e-12))
    return traj, life


def fit_beta(D, T):
    """Exponent of the best power-law shape for one complete trajectory."""
    n = len(D)
    tau = (np.arange(1, n + 1))/T
    m = np.isfinite(D) & (tau > 0.02) & (D > -0.4) & (D < 1.6)
    if m.sum() < 40:
        return np.nan
    t, d = tau[m], D[m]
    bs = np.linspace(0.5, 12.0, 200)
    err = [(np.clip(t, 1e-9, 3.)**b - d) for b in bs]
    return float(bs[int(np.argmin([np.mean(e*e) for e in err]))])


def impute(D, T, beta, c):
    """Life implied by a window truncated at fraction c, given an exponent."""
    m = max(6, int(round(c*T)))
    if m >= len(D):
        return np.nan
    t = np.arange(1, m + 1, dtype=float)
    d = D[:m]
    best, bt = np.inf, np.nan
    for g in GRID*t[-1]:
        r = np.clip(t/g, 1e-9, 3.)**beta - d
        v = float(np.mean(r*r))
        if v < best:
            best, bt = v, g
    return bt


def robust(e):
    e = np.abs(np.asarray(e, float))
    e = e[np.isfinite(e)]
    return float(np.median(e)/0.6745) if len(e) else np.nan


if __name__ == '__main__':
    traj, life = load()
    beta = np.array([fit_beta(traj[i], life[i]) for i in range(len(life))])
    ok = np.where(np.isfinite(beta))[0]
    print(f'{len(ok)} cells with a fitted exponent; median {np.median(beta[ok]):.3f}, '
          f'IQR [{np.percentile(beta[ok], 25):.2f}, {np.percentile(beta[ok], 75):.2f}]')

    e_own = [(impute(traj[i], life[i], beta[i], CSTAR) - life[i])/life[i] for i in ok]
    v0 = robust(e_own)
    b_all = float(np.median(beta[ok]))
    e_inf = [(impute(traj[i], life[i], b_all, CSTAR) - life[i])/life[i] for i in ok]
    vinf = robust(e_inf)
    print(f'at c = {CSTAR}: robust label error {v0:.4f} with each cell its own exponent, '
          f'{vinf:.4f} with the fleet centre known')
    print(f'so 1 + u_Lambda = {(vinf/v0)**2:.3f} in variance')
    print()

    rng = np.random.default_rng(3)
    print(f'{"n1":>5s} {"Xi(n1)":>9s} {"1+uL+u0/n1":>12s}')
    xs, ys = [], []
    for n1 in NS:
        acc = []
        for _ in range(REPS//len(NS)*2):
            s = rng.choice(ok, n1, replace=False)
            b = float(np.median(beta[s]))
            held = rng.choice([i for i in ok if i not in set(s)],
                              min(40, len(ok) - n1), replace=False)
            e = [(impute(traj[i], life[i], b, CSTAR) - life[i])/life[i] for i in held]
            acc.append(robust(e))
        xi = (float(np.mean(acc))/v0)**2
        xs.append(1.0/n1)
        ys.append(xi)
        print(f'{n1:5d} {xi:9.3f}')
    A = np.column_stack([np.ones(len(xs)), xs])
    (a, b), *_ = np.linalg.lstsq(A, np.array(ys), rcond=None)
    r2 = 1 - np.sum((np.array(ys) - A @ np.array([a, b]))**2)/np.var(ys)/len(ys)
    print()
    print(f'fit Xi = {a:.3f} + {b:.3f}/n1, R2 = {r2:.3f}')
    print(f'so u_Lambda = {a-1:.2f} and u_0 = {b:.1f}')
    print(f'the paper sweeps u_Lambda over [0.5, 8] and u_0 over [10, 100], '
          f'and Table I uses 2 and 30')
