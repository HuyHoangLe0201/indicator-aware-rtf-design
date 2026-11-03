"""What population does each campaign length actually buy?

A campaign that stops at c contributes nothing from a unit whose stop precedes the feature
window, c*T_i < m, so shrinking c does not merely shorten trajectories, it selects on life.
If the surviving training set is long-lived while the test set is not, a model that
extrapolates is punished and one that cannot extrapolate is not. This prints the selection.
"""
import numpy as np

import cost_real as CR

C_BUDGET = 21.0
for channel, m in (('capacity', 400), ('dq', 100)):
    QD, DQ, life = CR.load()
    rng = np.random.default_rng(0)
    n_all = len(life)
    series, cycles = [], []
    for i in range(n_all):
        d = CR.damage(QD[i])
        dc = np.arange(1, len(d) + 1, dtype=float)
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
    test = rng.choice(idx, CR.NTEST, replace=False)
    pool = np.array([i for i in idx if i not in set(test)])
    print(f'--- {channel}, window {m} cycles: pool {len(pool)} cells, '
          f'test {len(test)}, test life {life[test].min():.0f}-{life[test].max():.0f} '
          f'(median {np.median(life[test]):.0f}) ---')
    print(f'{"c":>5s} {"n asked":>8s} {"n avail":>8s} {"n used":>7s} '
          f'{"eligible life range":>22s} {"median":>7s}')
    for c in (0.35, 0.50, 0.65, 0.80, 1.00):
        elig = np.array([i for i in pool if c*life[i] >= m])
        n_ask = max(4, int(round(C_BUDGET/(CR.KAP0 + c))))
        n = min(n_ask, len(elig))
        if len(elig) == 0:
            print(f'{c:5.2f} {n_ask:8d} {0:8d}       -')
            continue
        L = life[elig]
        print(f'{c:5.2f} {n_ask:8d} {len(elig):8d} {n:7d} '
              f'{f"{L.min():.0f}-{L.max():.0f}":>22s} {np.median(L):7.0f}')
    print()
