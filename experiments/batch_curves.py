"""Corrected odds by window for each Severson batch and for the pooled fleet (diagnostic for batch_info.py)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dqv_mar as M

X, y, life = M.load()
d = np.load(M.DATA, allow_pickle=True)
batch = np.array([str(s)[:2] for s in d['name']])
print('window            ' + ' '.join(f'{m:6d}' for m in M.W))
for name, idx in (('all', np.arange(len(y))), ('b1', np.where(batch == 'b1')[0]),
                  ('b2', np.where(batch == 'b2')[0]), ('b3', np.where(batch == 'b3')[0])):
    n_obs = np.isfinite(X[idx]).sum(0)
    om = M.curve(X, y, idx, M.odds_mar)
    print(f'{name:4s} n at window  ' + ' '.join(f'{v:6d}' for v in n_obs))
    print(f'{name:4s} odds          ' + ' '.join(f'{v:6.1f}' for v in om))
    print(f'{name:4s} life min/med/max {life[idx].min():.0f}/{np.median(life[idx]):.0f}/{life[idx].max():.0f}')
