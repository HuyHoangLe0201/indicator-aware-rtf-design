"""LaTeX rows of the survival-correction table (Appendix C.8) from dqv_info_curve.npz and the data."""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dqv_mar as M

X, y, life = M.load()
z = np.load(os.path.join(M.HERE, 'dqv_info_curve.npz'))
for j, m in enumerate(M.W):
    n = int(np.isfinite(X[:, j]).sum())
    nv = M.odds_naive(X[:, j], y)
    print(f'{m} & {n} & {z["c"][j]:.2f} & {nv:.1f} & {z["odds"][j]:.1f}\\ [{z["lo"][j]:.1f},{z["hi"][j]:.1f}] & {z["eta"][j]:.2f}\\ [{z["eta_lo"][j]:.2f},{z["eta_hi"][j]:.2f}]\\\\')
