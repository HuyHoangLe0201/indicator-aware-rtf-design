"""Measure Lambda in place on the battery fleet, from the campaign the theory describes.

Theorem 2 writes the excess risk of a fixed-fraction campaign as (kappa_0+c)[K/etaT(c)+S]/n,
so the coefficient of 1/n is b(c) = S + K/etaT(c), affine in 1/etaT. Measure b at several c
and regress it on 1/etaT(c): the slope is K, the intercept is S, and Lambda = S/K. Nothing
about Upsilon enters except through etaT, and nothing is inherited.

What matters is which campaign is run. measure_b.py sweeps c but regresses on the true lives
at every c, so its b(c) reports how the FEATURES degrade under truncation while the labels stay
perfect; that is why its b(c) rises only sixfold from c=1 to c=0.2 while 1/etaT(c)=c^-9.97
rises by seven orders of magnitude. But etaT is the information for extrapolating T from a
truncated trajectory, which is label quality. So here the campaign truncates at c and imputes
every life from its own truncated capacity trajectory, exactly as a laboratory would have to,
while the deployed predictor reads a fresh cell over a fixed early window that does not move
with c. Then b(c) isolates what truncation costs, and the fit has the affine form it should.

The sweep stays near c=1, because 1/etaT spans a factor of thirty-five over [0.7,1] and seven
orders of magnitude over [0.2,1]; a regression over the wider range is decided entirely by its
smallest point.
"""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar

DATA = r'D:\claudecode\generationbound\data\severson\dq_traj.npz'
KAP0, BETA = 0.05, 4.4832
UPS = 2*BETA + 1
CS = (0.70, 0.80, 0.90, 1.00)
NS = (12, 20, 30, 45)
NTEST, REPS = 30, 60


def load():
    d = np.load(DATA, allow_pickle=True)
    return d['QD'], d['DQ'], d['life'].astype(float)


def damage(q):
    q = np.asarray(q, float)
    q0 = np.nanmedian(q[1:6])
    return (q0 - q)/(0.2*q0 + 1e-12)


def feats(series, cycles, m):
    grid = np.linspace(m/12., m, 12)
    y = np.interp(grid, cycles, series)
    return np.array([1., y[-1], y[-1] - y[5], np.log(np.var(np.diff(y)) + 1e-12)])


def impute(trajs, b0=BETA):
    def life_of(t, D, b):
        return minimize_scalar(lambda T: np.sum((np.clip(t/T, 1e-9, 3.)**b - D)**2),
                               bounds=(t[-1]*1.0001, t[-1]*25.), method='bounded',
                               options={'xatol': 1e-2}).x
    b = b0
    for _ in range(2):
        Ts = np.array([life_of(t, D, b) for t, D in trajs])
        r = lambda p: np.concatenate([np.clip(t/T, 1e-9, 3.)**p[0] - D
                                      for (t, D), T in zip(trajs, Ts)])
        try:
            b = float(least_squares(r, [b], bounds=([0.4], [15.]), max_nfev=80).x[0])
        except Exception:
            break
    return np.array([life_of(t, D, b) for t, D in trajs])


def bcoef(channel, m, c, rng):
    """Coefficient of 1/n in held-out error, for a campaign truncated at fraction c."""
    QD, DQ, life = load()
    dmg = [damage(QD[i]) for i in range(len(life))]
    dcy = [np.arange(1, len(x) + 1, dtype=float) for x in dmg]
    if channel == 'capacity':
        ser, cyc = dmg, dcy
    else:
        ser = [DQ[i][1].astype(float) for i in range(len(life))]
        cyc = [DQ[i][0].astype(float) for i in range(len(life))]

    # train and test are drawn from the same eligible population, or the shift between them
    # is a bias that does not shrink with n and the a+b/n form cannot hold
    ok = np.array([life[i] >= m and cyc[i][-1] >= m and c*life[i] >= m
                   for i in range(len(life))])
    idx = np.where(ok)[0]
    test = rng.choice(idx, NTEST, replace=False)
    pool = np.array([i for i in idx if i not in set(test)])
    Xte = np.array([feats(ser[i], cyc[i], m) for i in test])
    yte = np.log(life[test])

    mses = []
    for n in NS:
        if n > len(pool):
            return None
        acc = []
        for _ in range(REPS):
            take = rng.choice(pool, n, replace=False)
            if c >= 1.0:
                yl = life[take]
            else:
                yl = impute([(dcy[i][dcy[i] <= c*life[i]], dmg[i][dcy[i] <= c*life[i]])
                             for i in take])
            X = np.array([feats(ser[i], cyc[i], m) for i in take])
            w = np.linalg.solve(X.T @ X + 1e-2*np.eye(4), X.T @ np.log(yl))
            acc.append(float(np.mean((Xte @ w - yte)**2)))
        mses.append(float(np.mean(acc)))
    inv = 1.0/np.array(NS, float)
    D = np.column_stack([np.ones(len(inv)), inv])
    a, b = np.linalg.lstsq(D, np.array(mses), rcond=None)[0]
    return float(a), float(b), len(pool), mses


if __name__ == '__main__':
    rng = np.random.default_rng(7)
    print(f'etaT(c) = c^{UPS:.2f} on the fitted battery class')
    print(f'{"c":>5s} {"1/etaT":>9s} {"pool":>5s} {"a":>8s} {"b(c)":>8s}   MSE(n) over n='
          + ",".join(str(n) for n in NS))
    xs, ys = [], []
    for c in CS:
        r = bcoef('capacity', 400, c, rng)
        if r is None:
            print(f'{c:5.2f}  pool too small')
            continue
        a, b, npool, mses = r
        xs.append(c**-UPS)
        ys.append(b)
        print(f'{c:5.2f} {c**-UPS:9.1f} {npool:5d} {a:8.4f} {b:8.4f}   '
              + " ".join(f'{v:.4f}' for v in mses))

    xs, ys = np.array(xs), np.array(ys)
    A = np.column_stack([np.ones(len(xs)), xs])
    (S, K), *_ = np.linalg.lstsq(A, ys, rcond=None)
    pred = A @ np.array([S, K])
    r2 = 1 - np.sum((ys - pred)**2)/max(np.sum((ys - ys.mean())**2), 1e-12)
    print()
    print(f'b(c) = S + K/etaT(c):  S = {S:.4f}, K = {K:.5f}, R2 = {r2:.3f}')
    if S > 0 and K > 0:
        L = S/K
        print(f'Lambda = S/K = {L:.3f}, threshold (1+L)/(1+k0) = {(1+L)/(1+KAP0):.2f}')
        print(f'inherited value was Lambda = 0.241, threshold 1.18')
    else:
        print('the fit does not separate a fleet term from an estimation term')
