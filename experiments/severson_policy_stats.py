"""Statistics of the charging policy as the one stress variable of the Severson cells (Appendix C.21 and C.20).

(1) From severson_policy.npz (the three raw .mat batches read by severson_policy.py): the mean charging rate to 80% state of charge of each cell,
    the log-linear regression of ln T on ln(rate), its bootstrap interval, the rank correlation, and the factor in life across the range.
(2) From policy.json (the policy string of every cell) and dq_traj.npz: the same regression on the 119 cells, and a regression of ln T on the three
    policy parameters (first rate, switching state of charge, second rate) with its leave-one-out error against the standard deviation of ln T.
Usage: python severson_policy_stats.py
"""
import json
import os
import re
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, 'severson_policy.npz'), allow_pickle=True)
dq0 = np.load(os.path.join(HERE, '..', 'data', 'severson', 'dq_traj.npz'), allow_pickle=True)
pool = set(dq0['name'].tolist())                       # the 119 cells of the paper
rate, life = d['rate'], d['life'].astype(float)
ok = np.isfinite(rate) & np.isfinite(life) & (life > 0) & np.array([n in pool for n in d['name']])
x, lt = np.log(rate[ok]), np.log(life[ok])
r = stats.linregress(x, lt)
rng = np.random.default_rng(1)
bs = []
for _ in range(2000):
    i = rng.integers(0, len(x), len(x))
    bs.append(np.polyfit(x[i], lt[i], 1)[0])
rho = stats.spearmanr(rate[ok], life[ok])
print(f'(1) {ok.sum()} cells, {len(set(d["policy"][ok]))} policies, rate {rate[ok].min():.1f} to {rate[ok].max():.1f} C')
print(f'    ln T ~ ln(rate): slope {r.slope:.2f} (bootstrap 95% [{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}]), R^2 {r.rvalue ** 2:.3f}, '
      f'rank correlation {rho.correlation:.2f} (p={rho.pvalue:.2f}), factor in life across the range {(rate[ok].max() / rate[ok].min()) ** r.slope:.2f}, sd of ln T {lt.std():.3f}')

pol = json.load(open(os.path.join(HERE, '..', '..', 'theory_ext', 'policy.json'))) if os.path.exists(os.path.join(HERE, '..', '..', 'theory_ext', 'policy.json')) \
    else json.load(open(os.path.join(HERE, 'policy.json')))
dq = np.load(os.path.join(HERE, '..', 'data', 'severson', 'dq_traj.npz'), allow_pickle=True)
names, lf = dq['name'], dq['life'].astype(float)
P, T = [], []
for nm, L in zip(names, lf):
    m = re.match(r'([\d.]+)C\((\d+)%\)-([\d.]+)C?', pol[nm][0])
    if m:
        P.append([float(m[1]), float(m[2]), float(m[3])])
        T.append(L)
P, T = np.array(P), np.log(np.array(T))
S = 80 / (P[:, 1] / P[:, 0] + (80 - P[:, 1]) / P[:, 2])
r2 = stats.linregress(np.log(S), T)
print(f'(2) {len(T)} cells with a parseable policy ({len(set(map(tuple, P)))} distinct policies), mean rate {S.min():.2f} to {S.max():.2f} C, '
      f'{(np.abs(S - 4.8) < 0.005).sum()} cells at 4.80 C')
print(f'    ln T ~ ln(rate): slope {r2.slope:.2f}, R^2 {r2.rvalue ** 2:.3f}')
A = np.column_stack([np.ones(len(T)), P])
beta, *_ = np.linalg.lstsq(A, T, rcond=None)
R2 = 1 - np.sum((T - A @ beta) ** 2) / np.sum((T - T.mean()) ** 2)
H = A @ np.linalg.pinv(A)
loo = (T - A @ beta) / (1 - np.diag(H))
print(f'    three policy parameters: R^2 {R2:.2f}, leave-one-out error {np.sqrt(np.mean(loo ** 2)):.3f} against a standard deviation of {T.std():.3f}')
