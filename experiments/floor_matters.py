"""
Why the direct-objective method returned the interior everywhere.

Equation (G) is derived up to a floor that is free of c: all the c-dependence of the risk sits
in the 1/n term, so the optimum is budget-independent and Theorem 3 can decide it from the
shape alone.  That holds when the estimator is efficient, because then truncating a trajectory
costs precision and nothing else -- with unlimited units the same model is recovered from any
c > 0.

A predictor that is a fixed map of window summaries is not efficient.  Truncating costs it
information it cannot recover from any number of units, so its floor a(c) falls with c, and
the objective at budget C is

        G(C, c) = a(c) + (kappa_0 + c) b(c) / C,

whose minimiser moves with C.  Minimising (kappa_0+c) b(c) alone -- what measure_b2.py did --
silently assumes a(c) is flat, which is exactly the assumption the measurement violates.  That
is why C-MAPSS came back interior at c* = 0.30 despite the widest corner margins in the paper.

This script measures a(c) and b(c) on both datasets and maps c*(C).  If the floor is what was
missing, large budgets should recover the corner where theory says it belongs.
"""
import numpy as np, os
from sklearn.linear_model import Ridge

FRACS = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
CS = np.arange(0.3, 1.001, 0.1)
NS = [15, 20, 25, 30, 40, 50, 60]
REPS, K0 = 700, 0.05
RNG = np.random.default_rng(41)
INV = 1.0/np.array(NS, float)
DES = np.column_stack([np.ones(len(INV)), INV])


def feats(M, c):
    j = int(np.argmin(np.abs(FRACS - c)))
    lo = max(1, j//2)
    X = np.column_stack([M[:, j], M[:, j] - M[:, 0],
                         (M[:, j] - M[:, lo])/max(FRACS[j] - FRACS[lo], 1e-9)])
    return (X - X.mean(0))/(X.std(0) + 1e-12)


def ab(X, y):
    mse = []
    for n in NS:
        acc = []
        for _ in range(REPS):
            p = RNG.permutation(len(y))
            tr, te = p[:n], p[n:n + 30]
            mo = Ridge(alpha=1.0).fit(X[tr], y[tr])
            acc.append(float(np.mean((y[te] - mo.predict(X[te]))**2)))
        mse.append(np.mean(acc))
    co = np.linalg.lstsq(DES, np.array(mse), rcond=None)[0]
    return float(co[0]), float(co[1])


def cmapss():
    ROOT = r'D:\bách khoa Đà Nẵng\nghiencuukhoahoc\baotri\data'
    d = np.loadtxt(os.path.join(ROOT, 'train_FD001.txt'))
    uid = d[:, 0].astype(int)
    keep = [j for j in range(5, d.shape[1]) if len(np.unique(d[:, j])) > 2]
    Z = (d[:, keep] - d[:, keep].mean(0))/d[:, keep].std(0)
    w = np.linalg.svd(Z - Z.mean(0), full_matrices=False)[2][0]
    pc = Z @ w
    if np.corrcoef(pc, d[:, 1])[0, 1] < 0:
        pc = -pc
    M, T = [], []
    for u in np.unique(uid):
        m = uid == u
        tr = pc[m]; n = len(tr)
        M.append([float(np.median(tr[max(0, int(round(f*(n-1)))-2):
                                     int(round(f*(n-1)))+1])) for f in FRACS])
        T.append(float(n))
    return np.array(M), np.log(np.array(T))


def battery():
    z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'relgrid.npz'))
    T, A, B = z['T'], z['A'], z['B']
    dev = np.abs(B[:, 1:-1] - 0.5*(B[:, :-2] + B[:, 2:]))
    ctr = np.median(dev, axis=0)
    sc = np.median(np.abs(dev - ctr), axis=0)/0.6745 + 1e-12
    bad = ((np.abs(dev - ctr)/sc > 10.0) & (np.abs(dev - ctr) > 1.0)).any(axis=1)
    return A[~bad], B[~bad], np.log(T[~bad])


print(__doc__)
print('=' * 78)
Mc, Yc = cmapss()
Acap, Bdq, Yb = battery()
JOBS = [('C-MAPSS PC1  (theory: CORNER, Ups=7.0)', Mc, Yc),
        ('battery capacity (theory: CORNER, Ups=9.97)', Acap, Yb),
        ('battery dQ(V)    (theory: INTERIOR, Ups~1)', Bdq, Yb)]

for name, M, Y in JOBS:
    A_, B_ = [], []
    for c in CS:
        a, b = ab(feats(M, c), Y)
        A_.append(a); B_.append(b)
    A_, B_ = np.array(A_), np.array(B_)
    print()
    print(f'{name}   (n = {len(Y)})')
    print('  c       : ' + ' '.join(f'{c:7.2f}' for c in CS))
    print('  a(c)    : ' + ' '.join(f'{v:7.4f}' for v in A_))
    print('  b(c)    : ' + ' '.join(f'{v:7.4f}' for v in B_))
    print(f'    a(c) falls by {A_[0]/A_[-1]:.2f}x from c=0.3 to c=1: '
          f'{"NOT c-free, the theory drops a real term" if A_[0]/A_[-1] > 1.15 else "roughly c-free"}')
    print(f'  {"budget C":>9s} {"c*":>6s} {"verdict":>10s}')
    for C in (30., 60., 120., 250., 500., 1000., 4000.):
        G = A_ + (K0 + CS)*B_/C
        j = int(np.argmin(G))
        print(f'  {C:9.0f} {CS[j]:6.2f} {("CORNER" if CS[j] > 0.95 else "interior"):>10s}')
