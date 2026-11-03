"""Budget-matched campaigns on the DeltaQ(V) channel, including c < 0.5.
n = min(round(C/(kappa0+c)), eligible units), C = 20*1.05 as in Table 6. 8 splits x 12 draws.
Reports RMSE/base (same scale as Table 6) for imputed labels and for true lives (oracle)."""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
import numpy as np
import cost_real as cr
warnings.filterwarnings('ignore')
cr.DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'severson', 'dq_traj.npz')
QD, DQ, life = cr.load(); n_all = len(life); C = 20*1.05
dmg = [cr.damage(QD[i]) for i in range(n_all)]; dc = [np.arange(1, len(d)+1, dtype=float) for d in dmg]
cyc = [DQ[i][0].astype(float) for i in range(n_all)]; series = [DQ[i][1].astype(float) for i in range(n_all)]
rng0 = np.random.default_rng(0)
for m, grid in ((50, (0.07, 0.10, 0.20, 0.35, 0.50, 0.65, 1.00)), (100, (0.10, 0.20, 0.35, 0.50, 0.65, 1.00))):
    idx_ok = np.where(np.array([life[i] >= m and cyc[i][-1] >= m for i in range(n_all)]))[0]
    print(f'\n=== DeltaQ(V), window m={m}, budget C={C:.1f} ({len(idx_ok)} cells reach the window) ===')
    print(f'{"c":>5s} {"n asked":>8s} {"n used":>7s} {"elig":>5s} {"lab err":>8s} | {"RMSE/base imp":>13s} {"95% CI":>16s} | {"oracle":>7s}')
    for c in grid:
        R, O, L, NU, NE = [], [], [], [], []
        for seed in range(8):
            rng = np.random.default_rng(seed); test = rng.choice(idx_ok, cr.NTEST, replace=False)
            pool = np.array([i for i in idx_ok if i not in set(test)])
            Xte = np.array([cr.feats(series[i], cyc[i], m) for i in test]); yte = life[test]; base = np.sqrt(np.mean((yte - life[pool].mean())**2))
            elig = np.array([i for i in pool if c*life[i] >= m]); n_ask = max(4, int(round(C/(cr.KAP0 + c)))); n = min(n_ask, len(elig))
            if n < 8: continue
            r, o, l = [], [], []
            for _ in range(12):
                take = rng.choice(elig, n, replace=False); X = np.array([cr.feats(series[i], cyc[i], m) for i in take])
                yl = life[take] if c >= 1 else cr.impute([(dc[i][dc[i] <= c*life[i]], dmg[i][dc[i] <= c*life[i]]) for i in take], cr.pool_beta(pool, dmg, dc, life))
                r.append(np.sqrt(np.mean((Xte @ cr.huber(X, yl) - yte)**2))/base); o.append(np.sqrt(np.mean((Xte @ cr.huber(X, life[take]) - yte)**2))/base)
                l.append(np.median(np.abs(yl - life[take])/life[take]))
            R.append(np.mean(r)); O.append(np.mean(o)); L.append(np.mean(l)); NU.append(n); NE.append(len(elig)); n_ask_ = n_ask
        if not R: print(f'{c:5.2f}  infeasible (fewer than 8 eligible units)'); continue
        R = np.array(R); bs = [np.mean(R[rng0.integers(0, len(R), len(R))]) for _ in range(2000)]; lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f'{c:5.2f} {n_ask_:8d} {int(np.mean(NU)):7d} {int(np.mean(NE)):5d} {np.mean(L):8.3f} | {R.mean():13.2f} [{lo:6.2f},{hi:6.2f}] | {np.mean(O):7.2f}')
