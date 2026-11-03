"""Measure Lambda on the milling fleet, separating S from K the way the risk defines them.

The first attempt fitted the held-out error to S + K/n and read S off the intercept.  That is
wrong.  In the risk of Theorem 2 both terms sit in the 1/n coefficient: the fleet-location
parameters get one draw per unit whatever the test length, giving S/n, and the shape
parameters give K/(n eta_T(c)).  The intercept of a learning curve is the Bayes noise floor,
which the criterion discards.  A control settles it: strip every covariate, so the predictor is
the training fleet mean and K is zero by construction.  The true Lambda is then infinite, and
the slope of the fitted curve returns the fleet variance rather than the intercept doing so.

So S is measured from the covariate-free curve and S+K from the full one, and Lambda = S/(the
difference).  Both are the same experiment run twice, which is why the ratio is meaningful even
though neither slope is an absolute risk.
"""
import numpy as np
from milling_lambda import design, add_conditions, learning_curve, fit_SK, KAPPA0

LAMS = (0.03, 0.1, 0.3, 1.0)
NS = np.arange(3, 13)
NREP = 6000


def measure_S(y):
    """S alone, from the covariate-free curve.

    With no covariate the design matrix is a single zero column, so the ridge solve returns a
    zero weight and the predictor is the training mean whatever the penalty. S therefore cannot
    depend on the penalty, and an earlier version that recomputed it inside the penalty loop was
    reporting a four-fold spread that was nothing but Monte Carlo noise on 14 cutters. It is
    measured once here, at a rep count high enough to return the fleet variance it must return.
    """
    import milling_lambda as ml
    ml.NREP, ml.RNG = NREP, np.random.default_rng(1)
    return fit_SK(NS, learning_curve(np.zeros((len(y), 1)), y, NS, 1.0))[1]


def total(X, y, lam):
    import milling_lambda as ml
    ml.NREP, ml.RNG = NREP, np.random.default_rng(1)
    e = learning_curve(X, y, NS, lam)
    return fit_SK(NS, e)[1], fit_SK(NS, e)[2], float(e[-1])


if __name__ == '__main__':
    X0, y = design()
    S = measure_S(y)
    print(f'{len(y)} cutters; Var(log life) = {y.var():.5f}')
    print(f'covariate-free slope S = {S:.4f}, which is the same quantity and is the control')
    print()
    print(f'{"features":14s} {"ridge":>6s} {"err(12)":>8s} {"S+K":>7s} {"K":>7s} {"R2":>6s} '
          f'{"Lambda":>7s} {"threshold":>10s}')
    rows, best = [], {}
    for lab, X in (('sensors', X0), ('sensors+cond', add_conditions(X0))):
        for lam in LAMS:
            tot, r2, last = total(X, y, lam)
            K = tot - S
            if K <= 0:
                continue
            L = S/K
            rows.append(L)
            if lab not in best or last < best[lab][0]:
                best[lab] = (last, lam, L)
            print(f'{lab:14s} {lam:6.2f} {last:8.4f} {tot:7.3f} {K:7.3f} {r2:6.2f} '
                  f'{L:7.3f} {(1+L)/(1+KAPPA0):10.2f}')
    lo, hi = min(rows), max(rows)
    sel = [v[2] for v in best.values()]
    print()
    print(f'across every penalty: Lambda {lo:.2f} to {hi:.2f}, threshold '
          f'{(1+lo)/(1+KAPPA0):.2f} to {(1+hi)/(1+KAPPA0):.2f}, narrowest margin on the'
          f' admissible indicators (4.7 table AE, 4.0 spindle current, 2.5 spindle'
          f' vibration) is {2.5/((1+hi)/(1+KAPPA0)):.2f}x')
    print(f'at the penalty each feature set predicts best: Lambda {min(sel):.2f} to '
          f'{max(sel):.2f}, threshold {(1+min(sel))/(1+KAPPA0):.2f} to '
          f'{(1+max(sel))/(1+KAPPA0):.2f}, narrowest margin '
          f'{2.5/((1+max(sel))/(1+KAPPA0)):.2f}x')
