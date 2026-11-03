"""
A control for the direct-objective method: does it return the corner where it must?

measure_b2.py minimised the measured objective (kappa_0+c) b(c) and put bulk capacity at
c* = 0.90, with the corner costing 1.31x.  Theory says capacity is a corner channel by a wide
margin, Upsilon = 9.97 against a threshold of 1.18.  Before treating that as a finding about
batteries, the instrument has to be checked on a case where the answer is not in doubt.

C-MAPSS FD001 is that case.  All seven measured indicators sit at Upsilon = 7.0 to 8.5 against
a threshold of 1.11, the widest margins in the paper, and there is no candidate for an early
channel anywhere in the set.  If the direct method returns an interior optimum here too, it is
biased toward the interior and the capacity result says nothing.  If it returns the corner,
the method is sound and the battery number has to be explained some other way.

The likely explanation, if the method survives, is that b(c) is the risk of the predictor one
actually runs, whereas Upsilon is built from Fisher information and describes the efficient
one.  A predictor that does not exploit the shape of the last stretch of trajectory will not
be paid for collecting it.  So we also vary the predictor's richness and watch c* move.
"""
import numpy as np, os
from sklearn.linear_model import Ridge

ROOT = r'D:\bách khoa Đà Nẵng\nghiencuukhoahoc\baotri\data'
FRACS = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
CS = np.arange(0.3, 1.001, 0.1)
NS = [15, 20, 25, 30, 40, 50, 60]
REPS, K0 = 700, 0.05
RNG = np.random.default_rng(31)
INV = 1.0/np.array(NS, float)
DES = np.column_stack([np.ones(len(INV)), INV])


def load():
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
        tr = pc[m]
        n = len(tr)
        M.append([float(np.median(tr[max(0, int(round(f*(n-1)))-2):
                                     int(round(f*(n-1)))+1])) for f in FRACS])
        T.append(float(n))
    return np.array(M), np.array(T)


M, T = load()
Y = np.log(T)
print(f'C-MAPSS FD001: {len(T)} units, median life {np.median(T):.0f} cycles')


def feats(M, c, rich):
    j = int(np.argmin(np.abs(FRACS - c)))
    lo = max(1, j//2)
    cols = [M[:, j], M[:, j] - M[:, 0]]
    if rich >= 2:
        cols.append((M[:, j] - M[:, lo])/max(FRACS[j] - FRACS[lo], 1e-9))
    if rich >= 3:                                   # curvature over the observed window
        mid = max(1, (j + lo)//2)
        cols.append((M[:, j] - 2*M[:, mid] + M[:, lo]))
    if rich >= 4:                                   # every observed fraction, the full shape
        cols = [M[:, k] for k in range(j + 1)] + [M[:, j] - M[:, 0]]
    X = np.column_stack(cols)
    return (X - X.mean(0))/(X.std(0) + 1e-12)


def bcoef(X, y):
    mse = []
    for n in NS:
        acc = []
        for _ in range(REPS):
            p = RNG.permutation(len(y))
            tr, te = p[:n], p[n:n + 30]
            mo = Ridge(alpha=1.0).fit(X[tr], y[tr])
            acc.append(float(np.mean((y[te] - mo.predict(X[te]))**2)))
        mse.append(np.mean(acc))
    return float(np.linalg.lstsq(DES, np.array(mse), rcond=None)[0][1])


print()
for rich, tag in ((2, 'level+drift+slope (as used on the batteries)'),
                  (3, '  + curvature'),
                  (4, '  every observed fraction (closest to the full shape)')):
    G, bs = [], []
    for c in CS:
        b = bcoef(feats(M, c, rich), Y)
        bs.append(b); G.append((K0 + c)*b)
    G = np.array(G)
    j = int(np.argmin(G))
    print(f'predictor: {tag}')
    print('  c        : ' + ' '.join(f'{c:7.2f}' for c in CS))
    print('  b(c)     : ' + ' '.join(f'{v:7.4f}' for v in bs))
    print('  (k0+c)b  : ' + ' '.join(f'{v:7.4f}' for v in G))
    print(f'  -> c* = {CS[j]:.2f}   corner costs {G[-1]/G[j]:.2f}x   '
          f'{"CORNER" if CS[j] > 0.95 else "INTERIOR"}')
    print()
