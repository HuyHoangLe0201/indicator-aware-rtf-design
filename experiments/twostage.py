"""
The cold-start constraint: the design problem is two-stage.

eta_T(c) quantifies what a censored trajectory says about T GIVEN the degradation model.
The model must itself be learned from complete trajectories.  A campaign in which nothing
fails yields no labels at all, so the first campaign cannot be short whatever Upsilon is.

  stage 1: n1 units to failure.      cost n1 (kappa0 + 1).    information n1 per unit.
  stage 2: n2 units to fraction c.   cost n2 (kappa0 + c).    information
           n2 * eta_T(c) / Pi(n1),   Pi(n1) = 1 + u_Lam + u0/n1
           (Cor. known-centre inflation, plus the O(1/n1) centre uncertainty of Thm 8).

Maximise total effective sample size at fixed budget C over (n1, n2, c).
"""
import os
import numpy as np
from scipy.integrate import quad

KAP0, LAM_DESIGN = 0.05, 0.241

def make_eta(kind, par):
    if kind == 'pow':
        b = par
        return lambda c: c**(2*b+1), 2*b+1
    cg, ev = par
    return (lambda c: float(np.interp(c, cg, ev))), None

def eta_dq_info():
    """eta_T of the Delta Q(V) channel on the information scale.  R^2 saturates and is not additive; for a Gaussian
    linear model the information is proportional to the odds R^2/(1-R^2) (Proposition 5), so the curve is the
    normalized odds of the leave-one-out R^2 at each window, made monotone.  Using R^2/max(R^2) directly as an
    information fraction overstates early information by a factor of 2.5 at c = 0.07.

    The odds are those of dqv_mar.py, corrected for the cells that no longer exist at a long window (their log life is
    known and only the feature is missing, so the regression of the feature on log life is unbiased) and made monotone;
    the curve is normalized at c = 1 and held at 1 beyond it.  The earlier curve took the leave-one-out R^2 on the
    surviving cells only, which falls at long windows (0.804 at 700 cycles) because the survivors' log life varies less,
    and it was normalized by its own maximum, which placed the plateau at c = 0.27."""
    z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dqv_info_curve.npz'))
    c = np.concatenate([[0.0], z['c']])
    eta = np.concatenate([[0.0], z['eta']])
    return make_eta('emp', (c, np.minimum(eta, 1.0)))[0]


def solve(eta, u_lam, u0, C, cgrid=None, n1grid=None):
    cgrid = np.linspace(0.02, 1.0, 99) if cgrid is None else cgrid
    # n1 must be allowed all the way to the all-to-failure plan, n1 = C/(kappa0+1); a fixed
    # cap silently truncates the optimum for every budget above that cap.
    n1grid = np.arange(0, int(C/(KAP0 + 1.0)) + 2) if n1grid is None else n1grid
    best = None
    for n1 in n1grid:
        Pi = 1 + u_lam + (u0/n1 if n1 > 0 else np.inf)
        rem = C - n1*(KAP0 + 1.0)
        if rem < 0:
            continue
        if n1 == 0:
            continue                      # no labels at all: stage 2 is unusable
        for c in cgrid:
            n2 = rem/(KAP0 + c)
            neff = n1 + n2*eta(c)/Pi
            if best is None or neff > best[0]:
                best = (neff, n1, n2, c, Pi)

    # all-to-failure is the n2 = 0 member of the same family, so it must be a candidate:
    # without it an integer n1 grid can return an optimum below a feasible plan.
    n1c = C/(KAP0 + 1.0)
    if best is None or n1c > best[0]:
        best = (n1c, n1c, 0.0, 1.0, 1 + u_lam + u0/max(n1c, 1e-9))
    return best

if __name__ == '__main__':
    print('Two-stage optimum.  u_Lam = known-centre inflation, u0 = centre-uncertainty term.\n')
    # channel A: bulk capacity, convex power law beta = 4.48  -> Upsilon = 9.97
    etaA, UA = make_eta('pow', 4.483)
    # channel B: DeltaQ(V), eta on the information scale (odds of the leave-one-out R^2), saturating -> Upsilon ~ 0
    etaB = eta_dq_info()

    for nm, eta, U in [('A bulk capacity  (Ups=9.97)', etaA, UA),
                       ('B DeltaQ(V)      (Ups~0)   ', etaB, 0.0)]:
        print(f'--- channel {nm} ---')
        print(f'{"budget C":>9s} {"n1 (to fail)":>13s} {"n2 (short)":>11s} {"c*":>7s} '
              f'{"n_eff":>8s} {"one-stage n_eff":>16s}')
        for C in (50., 100., 200., 400.):
            for ul, u0 in [(2.0, 30.0), (0.5, 10.0)]:
                b = solve(eta, ul, u0, C)
                if b is None:
                    print(f'{C:9.0f}   (infeasible)'); continue
                neff, n1, n2, c, Pi = b
                one = C/(KAP0+1.0)                     # everything to failure
                star = '  <- all-to-failure wins' if one >= neff else ''
                print(f'{C:9.0f} {n1:13.1f} {n2:11.1f} {c:7.3f} {neff:8.1f} '
                      f'{one:16.1f}{star}   (u_Lam={ul}, u0={u0}; gain {100*(neff/one-1):+.1f}%)')
        print()
    print('Interpretation: n1 > 0 always, because stage 2 is worthless without labels;')
    print('the Upsilon criterion governs stage 2 only.')


def variants():
    """Gains of the model at (u_Lam, u0) = (0.5, 10) when the information curve of Delta Q(V) is taken from the survivors
    (the R^2 or the odds of dqv_odds_scale.py, normalized by their own maximum) instead of the corrected curve."""
    import dqv_mar as dm
    X, y, life = dm.load()
    om, eta, c, sl, sat, pl = dm.summarize(X, y, np.arange(len(y)), dm.odds_naive)
    ok = np.isfinite(om)
    cc = np.concatenate([[0.0], c[ok]])
    out = {}
    for nm, vec in (('odds of the survivors, monotone, / max', dm.pava(om[ok])),
                    ('R^2 of the survivors, monotone, / max', np.maximum.accumulate(om[ok] / (1 + om[ok])))):
        v = vec / vec.max()
        f = (lambda v: (lambda x: float(np.interp(x, cc, np.concatenate([[0.0], v])))))(v)
        print(f'--- {nm}: curve at c = {np.round(c[ok][:5], 2).tolist()}: {np.round(v[:5], 2).tolist()}')
        for C in (50., 100., 200., 400.):
            neff, n1, n2, cs, Pi = solve(f, 0.5, 10.0, C)
            print(f'    C={C:4.0f}: c*={cs:.2f}, gain {100 * (neff / (C / (KAP0 + 1.0)) - 1):+.0f}%')


if __name__ == '__main__':
    variants()
