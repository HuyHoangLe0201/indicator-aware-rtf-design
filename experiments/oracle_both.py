"""Two questions about the oracle result, both of which could unmake it.

The paper now claims that a campaign given the true failure times prefers to stop early while
the identical campaign that must infer them does not, and offers that as Theorem 4 measured
rather than derived. On the turbofans that claim rests on 0.79 against 0.83, a gap of five per
cent in a mean over three splits. Two things have to be true for it to survive a referee.

  1. The interior optimum must not be an artefact of averaging. So the oracle is reported
     per split, and the question asked is whether every split puts the optimum inside.

  2. It should not be peculiar to one fleet. The battery campaign of Table 6 already computes
     an oracle arm and the paper reports it only at c=1, so it is asked here across the whole
     grid: does the same divergence appear on the cells?

If the per-split argmins disagree, the honest report is that the level ordering holds and the
argmin is unresolved, and the paper must say so.
"""
import warnings

import numpy as np

import cost_real as CR
import cost_cmapss as CM

warnings.filterwarnings('ignore')

GRID = (0.50, 0.65, 0.80, 1.00)
SEEDS = (0, 1, 2)


# ---------------------------------------------------------------- batteries
def battery(channel, m, seed, reps=120):
    """cost_real's campaign, returning the realized and oracle arms per c."""
    QD, DQ, life = CR.load()
    rng = np.random.default_rng(seed)
    n_all = len(life)
    series, cycles, dmg, dmg_cyc = [], [], [], []
    for i in range(n_all):
        d = CR.damage(QD[i])
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
    test = rng.choice(idx, CR.NTEST, replace=False)
    pool = np.array([i for i in idx if i not in set(test)])
    Xte = np.array([CR.feats(series[i], cycles[i], m) for i in test])
    yte = life[test]
    base = float(np.sqrt(np.mean((yte - life[pool].mean())**2)))

    real, orac = [], []
    for c in GRID:
        elig = np.array([i for i in pool if c*life[i] >= m])
        n = min(max(4, int(round(21.0/(CR.KAP0 + c)))), len(elig))
        a, b = [], []
        for _ in range(reps):
            take = rng.choice(elig, n, replace=False)
            trajs = [(dmg_cyc[i][dmg_cyc[i] <= c*life[i]],
                      dmg[i][dmg_cyc[i] <= c*life[i]]) for i in take]
            yl = life[take] if c >= 1.0 else CR.impute(trajs, CR.pool_beta(pool, dmg, dmg_cyc, life))
            X = np.array([CR.feats(series[i], cycles[i], m) for i in take])
            rms = lambda p: float(np.sqrt(np.mean((p - yte)**2)))/base
            a.append(rms(Xte @ CR.huber(X, yl)))
            b.append(rms(Xte @ CR.huber(X, life[take])))
        real.append(float(np.mean(a)))
        orac.append(float(np.mean(b)))
    return real, orac


# ---------------------------------------------------------------- turbofans
def turbofan(m, seed, reps=40):
    """cost_cmapss's campaign, keeping the oracle arm per split."""
    _, rows, _ = CM.campaign(m, seed, reps=reps)
    real = [r[1]['Huber'] for r in rows]
    orac = [r[1]['oracle'] for r in rows]
    return real, orac


def report(name, per_split):
    """per_split: list over splits of (realized, oracle) value lists."""
    print(f'--- {name} ---')
    print(f'{"split":>7s}  ' + ' '.join(f'{c:>6.2f}' for c in GRID) + '   argmin')
    for tag, k in (('realized', 0), ('oracle', 1)):
        arg = []
        for si, s in enumerate(per_split):
            v = np.array(s[k], float)
            arg.append(GRID[int(np.argmin(v))])
            print(f'{tag[:4]+str(si):>7s}  ' + ' '.join(f'{x:6.2f}' for x in v)
                  + f'   {GRID[int(np.argmin(v))]:.2f}')
        mean = np.mean(np.array([s[k] for s in per_split], float), 0)
        print(f'{"MEAN "+tag[:4]:>7s}  ' + ' '.join(f'{x:6.2f}' for x in mean)
              + f'   {GRID[int(np.argmin(mean))]:.2f}'
              + ('   [all splits agree]' if len(set(arg)) == 1
                 else f'   [splits disagree: {arg}]'))
    print()


if __name__ == '__main__':
    print('Prediction error as a multiple of predicting the fleet mean.')
    print('"realized" imputes the lives a truncated campaign never observed; '
          '"oracle" is given them.\n')

    for m in (100, 60):
        report(f'C-MAPSS FD001, window {m} cycles (40 draws)',
               [turbofan(m, s) for s in SEEDS])

    for ch, m, label in (('capacity', 400, 'bulk capacity, Upsilon ~10'),
                         ('dq', 100, 'DeltaQ(V), Upsilon ~0')):
        report(f'Severson cells, {label} (120 draws)',
               [battery(ch, m, s) for s in SEEDS])
