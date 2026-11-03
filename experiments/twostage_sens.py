"""
Table I and Fig. 2 are computed at u_Lambda = 2 and u0 = 30.  Neither constant is measured;
they set how expensive the stage-one model fit is.  Before quoting them in the paper we ask
how much of the reported result depends on them: does c* move, and does the sign of the
two-stage advantage over an all-to-failure campaign survive?
"""
import numpy as np
from twostage import make_eta, solve, KAP0

from twostage import eta_dq_info
etaB = eta_dq_info()
etaA, UA = make_eta('pow', 4.483)

print('channel B (DeltaQ(V), Ups~0), budget C = 100, one-stage n_eff = %.1f' % (100/(KAP0+1)))
print(f'{"u_Lam":>6s} {"u0":>6s} {"n1":>5s} {"n2":>7s} {"c*":>7s} {"n_eff":>8s} {"gain":>8s}')
print('-'*50)
rows = []
for ul in (0.5, 1.0, 2.0, 4.0, 8.0):
    for u0 in (10., 30., 100.):
        b = solve(etaB, ul, u0, 100.)
        if b is None:
            print(f'{ul:6.1f} {u0:6.0f}   infeasible'); continue
        neff, n1, n2, c, Pi = b
        one = 100/(KAP0+1)
        rows.append((ul, u0, c, neff/one))
        print(f'{ul:6.1f} {u0:6.0f} {n1:5.0f} {n2:7.1f} {c:7.3f} {neff:8.1f} '
              f'{neff/one:7.2f}x')

c = np.array([r[2] for r in rows]); g = np.array([r[3] for r in rows])
print()
print('c* range over the sweep: %.3f to %.3f' % (c.min(), c.max()))
print('two-stage gain range   : %.2fx to %.2fx  (all > 1: %s)' % (g.min(), g.max(), bool((g > 1).all())))

print()
print('channel A (bulk capacity, Ups=9.97): does the corner survive the same sweep?')
allcorner = True
for ul in (0.5, 1.0, 2.0, 4.0, 8.0):
    for u0 in (10., 30., 100.):
        b = solve(etaA, ul, u0, 100.)
        if b is None:
            continue
        if b[3] < 0.999:
            allcorner = False
            print('  c* = %.3f at u_Lam=%.1f, u0=%.0f  <-- not the corner' % (b[3], ul, u0))
print('  corner c*=1 at every setting:', allcorner)
