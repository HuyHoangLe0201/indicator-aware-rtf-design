"""Budget-constrained campaigns on further public cell sets (the protocol of budget_sweep.py, scaled to the set).

A campaign at stop fraction c buys n = round(C/(kappa0 + c)) units, observes each to c of its life, imputes the label of every
unit stopped before failure from the shape exponent fitted on the complete trajectories of the training pool, and fits the
paper's four-feature map by Huber regression. The error on held-out cells is divided by that of the training-pool mean; the paired
gain over the corner (c = 1, N1 = C/(1 + kappa0) complete units) is taken split by split.

Scaling to a set: the held-out share is 25% of the cells; the feature window is m = 0.15 median lives for Delta Q(V) and
0.55 median lives for capacity (100 of 719 and 400 of 719 on the Severson cells); the damage scale is fixed by the failure
fraction of the set (bl_common.py).

Usage: CH=dq|capacity python bl_campaign.py SET N1,N1,... ; NSPLIT, NDRAW from the environment.
"""
import os, sys, warnings
import numpy as np
import cost_real as cr
import bl_common as bc
warnings.filterwarnings('ignore')

CH = os.environ.get('CH', 'dq')
NSPLIT = int(os.environ.get('NSPLIT', '10'))
NDRAW = int(os.environ.get('NDRAW', '8'))
GRID = (0.50, 0.65, 0.80, 1.00)

name = sys.argv[1]
N1S = [int(x) for x in sys.argv[2].split(',')]
QD, DQ, life, f, keys = bc.load_set(name)
n_all = len(life)
dmg = [bc.damage_scaled(QD[i], f) for i in range(n_all)]
dc = [np.arange(1, len(d) + 1, dtype=float) for d in dmg]
if CH == 'capacity':
    cyc = [d.copy() for d in dc]; series = [np.asarray(d, float) for d in dmg]
else:
    cyc = [DQ[i][0].astype(float) for i in range(n_all)]; series = [DQ[i][1].astype(float) for i in range(n_all)]
med = float(np.median(life))
M = int(round((0.15 if CH == 'dq' else 0.55)*med))
idx_ok = np.array([i for i in range(n_all) if life[i] >= M and cyc[i][-1] >= M and len(dmg[i]) >= M])
NT = int(round(0.25*len(idx_ok)))
rng0 = np.random.default_rng(0)
print(f'{name} channel {CH} window m={M}: {len(idx_ok)} cells, {NT} held out, f={f:.3f}, {NSPLIT} splits x {NDRAW} draws')
for N1 in N1S:
    C = N1*(1 + cr.KAP0)
    res = {}
    for c in GRID:
        R, O, NU = [], [], []
        for seed in range(NSPLIT):
            rng = np.random.default_rng(seed)
            test = rng.choice(idx_ok, NT, replace=False)
            pool = np.array([i for i in idx_ok if i not in set(test)])
            Xte = np.array([cr.feats(series[i], cyc[i], M) for i in test]); yte = life[test]
            base = np.sqrt(np.mean((yte - life[pool].mean())**2))
            B0 = cr.pool_beta(pool, dmg, dc, life)
            elig = np.array([i for i in pool if c*life[i] >= M])
            n = min(max(3, int(round(C/(cr.KAP0 + c)))), len(elig))
            if n < 3:
                continue
            r, o = [], []
            for _ in range(NDRAW):
                take = rng.choice(elig, n, replace=False)
                X = np.array([cr.feats(series[i], cyc[i], M) for i in take])
                yl = life[take] if c >= 1 else cr.impute([(dc[i][dc[i] <= c*life[i]], dmg[i][dc[i] <= c*life[i]]) for i in take], beta0=B0)
                r.append(np.sqrt(np.mean((Xte @ cr.huber(X, yl) - yte)**2))/base)
                o.append(np.sqrt(np.mean((Xte @ cr.huber(X, life[take]) - yte)**2))/base)
            R.append(np.mean(r)); O.append(np.mean(o)); NU.append(n)
        res[c] = (np.array(R), np.array(O), np.mean(NU) if NU else np.nan)
    Rc = res[1.0][0]
    print(f'\nN1={N1} (C={C:.1f}); corner buys {N1} units; corner error {Rc.mean():.2f} x pool-mean error')
    print(f'{"c":>5s} {"n":>4s} | {"imputed":>8s} {"oracle":>7s} | paired gain vs corner (imputed)')
    for c in GRID:
        R, O, nu = res[c]
        k = min(len(R), len(Rc))
        if len(R) != len(Rc):
            print(f'{c:5.2f} WARNING: {len(R)} splits at this c against {len(Rc)} at the corner; pairing is by position and not valid')
        if k < 3:
            print(f'{c:5.2f} {nu:4.0f} | not enough eligible units'); continue
        d = (Rc[:k] - R[:k])/Rc[:k]
        bs = [np.mean(d[rng0.integers(0, k, k)]) for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f'{c:5.2f} {nu:4.0f} | {R.mean():8.2f} {O.mean():7.2f} | {100*d.mean():7.1f}% [{100*lo:6.1f},{100*hi:6.1f}]', flush=True)
