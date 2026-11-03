"""Prediction of the budget-dependent gain from Proposition 1 using only the oracle learning curve and the measured label error."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
import numpy as np, cost_real as cr
QD, DQ, life = cr.load()
cv = life.std() / life.mean(); print('CV of life', round(cv, 3), 'mean', round(life.mean()), 'sd', round(life.std()))
# measured at the corner (oracle = imputed), RMSE/base, by N1
N1 = np.array([6, 8, 10, 12, 16, 20]); corner = np.array([0.893, 0.790, 0.716, 0.684, 0.636, 0.609])
# measured imputed at c, and label error
meas = {0.5: [0.714, 0.665, 0.655, 0.641, 0.631, 0.624], 0.65: [0.729, 0.648, 0.627, 0.613, 0.586, 0.584]}
lab = {0.5: 0.135, 0.65: 0.073}
K0 = 0.05
# oracle learning curve e^2(n) = einf2 + k/n, fitted on the corner row (oracle = imputed at c=1)
A = np.vstack([np.ones_like(N1, dtype=float), 1.0 / N1]).T
(einf2, k), *_ = np.linalg.lstsq(A, corner**2, rcond=None)
print('einf2 %.3f  k %.3f' % (einf2, k))
print('fit corner:', np.round(np.sqrt(einf2 + k / N1), 3))
for c in (0.5, 0.65):
    # label variance on the scale of base^2: robust sd = 1.4826*MAD-ish of relative error times mean life, over base (sd of life)
    v = (1.4826 * lab[c] / cv) ** 2
    n = np.round(N1 * (1 + K0) / (K0 + c))
    pred = np.sqrt(einf2 + (k / n) * (einf2 + v) / einf2)
    gain_pred = 100 * (corner - pred) / corner
    gain_meas = 100 * (corner - np.array(meas[c])) / corner
    print('c', c, 'v', round(v, 3), 'n', n)
    print('  predicted RMSE', np.round(pred, 3), ' measured', meas[c])
    print('  predicted gain %', np.round(gain_pred, 1), ' measured', np.round(gain_meas, 1))
