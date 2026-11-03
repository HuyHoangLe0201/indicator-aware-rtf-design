"""What a truncated campaign costs, measured on the real cells rather than on a surrogate.

The design question is about the TRAINING campaign, not about deployment. A deployed predictor
reads a fresh unit over a fixed early window of m cycles and predicts its life. The campaign
that trained it observed each of its own units to a fraction c of that unit's life, so for
c<1 no failure was witnessed and every training label had to be imputed from the truncated
trajectory. Smaller c buys more units on the same budget and pays for them in label quality.

Two constraints that a simulation on normalised time hides, and that decide the answer:

  a unit whose campaign stops before the feature window, c*T_i < m, contributes nothing at all,
  so shrinking c does not only degrade labels, it removes the short-lived units first;

  and the features must be read at absolute cycles. Features computed on the observed window
  itself are a function of t/T alone and carry no information about T whatever, which is the
  defect this script exists to avoid.

Both channels are run. Capacity is the high-Upsilon channel the criterion says to run to
failure; the DeltaQ(V) signature is the low-Upsilon one it says may be stopped early.
"""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar

DATA = r'D:\claudecode\generationbound\data\severson\dq_traj.npz'
KAP0 = 0.05
NTEST = 30


def load():
    d = np.load(DATA, allow_pickle=True)
    return d['QD'], d['DQ'], d['life'].astype(float)


def damage(q):
    """Capacity trajectory on the scale where 1 is end of life, i.e. 80 percent of nominal."""
    q = np.asarray(q, float)
    q0 = np.nanmedian(q[1:6])
    return (q0 - q)/(0.2*q0 + 1e-12)


def feats(series, cycles, m):
    """The paper's four-feature map, read at twelve absolute cycles up to m."""
    grid = np.linspace(m/12., m, 12)
    y = np.interp(grid, cycles, series)
    return np.array([1., y[-1], y[-1] - y[5], np.log(np.var(np.diff(y)) + 1e-12)])


_PB = {}


def pool_beta(pool, dmg, dmg_cyc, life):
    """Shape exponent of the damage curve D=(t/T)^b, fitted on the complete trajectories of the TRAINING POOL only
    (the archive of cells a laboratory has already run to failure); held-out cells never enter. Median over cells."""
    key = tuple(int(i) for i in pool)
    if key not in _PB:
        bs = []
        for i in pool:
            t = dmg_cyc[i]/life[i]
            sel = t <= 1.0
            f = lambda p: np.clip(t[sel], 1e-9, 3.)**p[0] - dmg[i][sel]
            try:
                bs.append(float(least_squares(f, [4.], bounds=([0.4], [15.])).x[0]))
            except Exception:
                pass
        _PB[key] = float(np.median(bs))
    return _PB[key]


def impute(trajs, beta0):
    """Pool the shape, then solve each truncated trajectory for its own life. beta0 is the starting exponent and
    must come from cells outside the test set (see pool_beta); it has no default for that reason."""
    def life_of(t, D, b):
        return minimize_scalar(
            lambda T: np.sum((np.clip(t/T, 1e-9, 3.)**b - D)**2),
            bounds=(t[-1]*1.0001, t[-1]*25.), method='bounded',
            options={'xatol': 1e-2}).x
    b = beta0
    for _ in range(3):
        Ts = np.array([life_of(t, D, b) for t, D in trajs])
        r = lambda p: np.concatenate([np.clip(t/T, 1e-9, 3.)**p[0] - D
                                      for (t, D), T in zip(trajs, Ts)])
        try:
            b = float(least_squares(r, [b], bounds=([0.4], [15.]), max_nfev=120).x[0])
        except Exception:
            break
    return np.array([life_of(t, D, b) for t, D in trajs])


def huber(X, y, w=None, iters=12):
    """Iteratively reweighted least squares; a single ruined label must not own the fit."""
    b = np.linalg.solve(X.T @ X + 1e-3*np.eye(X.shape[1]), X.T @ y)
    for _ in range(iters):
        r = y - X @ b
        s = 1.4826*np.median(np.abs(r - np.median(r))) + 1e-9
        u = np.abs(r)/(1.345*s)
        wt = np.where(u <= 1, 1.0, 1.0/np.maximum(u, 1e-9))
        A = (X*wt[:, None]).T @ X + 1e-3*np.eye(X.shape[1])
        b = np.linalg.solve(A, (X*wt[:, None]).T @ y)
    return b


def run(channel, m, grid, C, reps=120, seed=0):
    QD, DQ, life = load()
    rng = np.random.default_rng(seed)
    n_all = len(life)

    series, cycles = [], []
    dmg, dmg_cyc = [], []
    for i in range(n_all):
        d = damage(QD[i])
        dc = np.arange(1, len(d) + 1, dtype=float)
        dmg.append(d)
        dmg_cyc.append(dc)
        if channel == 'capacity':
            s, cyc = d, dc
        else:
            cyc, s = DQ[i][0].astype(float), DQ[i][1].astype(float)
        series.append(np.asarray(s, float))
        cycles.append(cyc)

    ok = np.array([life[i] >= m and cycles[i][-1] >= m and
                   np.all(np.isfinite(series[i][np.isfinite(series[i])]))
                   for i in range(n_all)])
    idx = np.where(ok)[0]
    test = rng.choice(idx, NTEST, replace=False)
    pool = np.array([i for i in idx if i not in set(test)])

    Xte = np.array([feats(series[i], cycles[i], m) for i in test])
    yte = life[test]
    base = float(np.sqrt(np.mean((yte - life[pool].mean())**2)))

    print(f'channel {channel}, feature window {m} cycles, budget C={C}')
    print(f'   {len(idx)} of {n_all} cells reach the window; {NTEST} held out for test, '
          f'{len(pool)} in the training pool')
    print(f'   predicting the training-pool mean gives RMSE {base:.0f} cycles')
    print(f'{"c":>5s} {"n asked":>8s} {"n avail":>8s} {"RMSE":>8s} {"vs base":>8s} '
          f'{"oracle":>8s} {"lab err":>8s}')
    out = []
    for c in grid:
        elig = np.array([i for i in pool if c*life[i] >= m])
        n_ask = max(4, int(round(C/(KAP0 + c))))
        n = min(n_ask, len(elig))
        if n < 4:
            print(f'{c:5.2f} {n_ask:8d} {len(elig):8d}  no campaign is possible at this c')
            continue
        rm, orc, lab = [], [], []
        for r in range(reps):
            take = rng.choice(elig, n, replace=False)
            # the label is always imputed from the damage channel, because capacity reaching
            # 80 percent is what defines the failure the label refers to
            trajs = []
            for i in take:
                sel = dmg_cyc[i] <= c*life[i]
                trajs.append((dmg_cyc[i][sel], dmg[i][sel]))
            yl = life[take] if c >= 1.0 else impute(trajs, pool_beta(pool, dmg, dmg_cyc, life))
            X = np.array([feats(series[i], cycles[i], m) for i in take])
            mu = huber(X, yl)
            mo = huber(X, life[take])
            rm.append(np.sqrt(np.mean((Xte @ mu - yte)**2)))
            orc.append(np.sqrt(np.mean((Xte @ mo - yte)**2)))
            lab.append(np.median(np.abs(yl - life[take])/life[take]))
        out.append((c, n_ask, len(elig), float(np.mean(rm)), float(np.mean(orc)),
                    float(np.mean(lab))))
        print(f'{c:5.2f} {n_ask:8d} {len(elig):8d} {np.mean(rm):8.0f} '
              f'{np.mean(rm)/base:8.2f} {np.mean(orc):8.0f} {np.mean(lab):8.3f}')
    print()
    return base, out


if __name__ == '__main__':
    GRID = [0.35, 0.50, 0.65, 0.80, 1.00]
    run('dq', 100, GRID, C=20*1.05)
    run('capacity', 400, GRID, C=20*1.05)
