"""Information curve with long windows: how much of ln T the first m cycles explain, by channel, as m grows to the median life.

For a window m = c x (median life) the cells that reach m are kept (a cell that failed earlier cannot be read at m), the four-feature
map of the paper is read from the capacity trajectory, from Delta Q(V), and from both (seven predictors), and the cross-validated
R^2 of ln T is reported (repeated 5-fold, ridge with a fixed penalty on standardised predictors, R^2 against the training-fold mean).
The gain of Delta Q(V) given capacity is R^2(both) - R^2(capacity). SEVERSON runs the same code on the cells of the main text.

Usage: python bl_curve.py SET [SET ...]   (SET = SEVERSON or a BatteryLife set)
"""
import sys, warnings
import numpy as np
import cost_real as cr
import bl_common as bc
warnings.filterwarnings('ignore')

CFR = (0.07, 0.14, 0.27, 0.40, 0.55, 0.80, 1.00)
REP, K, LAM, MIN_N = 20, 5, 1.0, 20


def load(name):
    if name == 'SEVERSON':
        QD, DQ, life = cr.load()
        dmg = [cr.damage(QD[i]) for i in range(len(life))]
        return dmg, list(DQ), life
    QD, DQ, life, f, keys = bc.load_set(name)
    return [bc.damage_scaled(q, f) for q in QD], DQ, life


def cv_r2(X, y, rng):
    n = len(y)
    sse_m = sse_0 = 0.0
    for _ in range(REP):
        perm = rng.permutation(n)
        for fold in np.array_split(perm, K):
            tr = np.setdiff1d(perm, fold)
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
            A = (X[tr] - mu)/sd; ym = y[tr].mean()
            b = np.linalg.solve(A.T @ A + LAM*np.eye(A.shape[1]), A.T @ (y[tr] - ym))
            sse_m += np.sum((y[fold] - ym - ((X[fold] - mu)/sd) @ b)**2)
            sse_0 += np.sum((y[fold] - ym)**2)
    return 1 - sse_m/sse_0


def run(name):
    dmg, DQ, life = load(name)
    n = len(life); med = float(np.median(life))
    dc = [np.arange(1, len(d) + 1, dtype=float) for d in dmg]
    print(f'\n=== {name}: {n} cells, median life {med:.0f}')
    print(f'{"c":>5s} {"m":>5s} {"n":>4s} | {"R2 cap":>7s} {"R2 dQ":>7s} {"R2 both":>8s} | {"dQ given cap":>12s}')
    for c in CFR:
        m = int(round(c*med))
        ok = [i for i in range(n) if life[i] >= m and len(dmg[i]) >= m and DQ[i][0][-1] >= m and DQ[i][0][0] <= m]
        if len(ok) < MIN_N:
            print(f'{c:5.2f} {m:5d} {len(ok):4d} | fewer than {MIN_N} cells reach the window'); continue
        y = np.log(life[ok])
        Fc = np.array([cr.feats(dmg[i], dc[i], m)[1:] for i in ok])
        Fd = np.array([cr.feats(DQ[i][1].astype(float), DQ[i][0].astype(float), m)[1:] for i in ok])
        r = lambda X: cv_r2(X, y, np.random.default_rng(7))
        a, b, ab = r(Fc), r(Fd), r(np.hstack([Fc, Fd]))
        print(f'{c:5.2f} {m:5d} {len(ok):4d} | {a:7.2f} {b:7.2f} {ab:8.2f} | {ab - a:12.2f}', flush=True)


if __name__ == '__main__':
    for s in sys.argv[1:]:
        run(s)
