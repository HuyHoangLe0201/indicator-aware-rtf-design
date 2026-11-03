"""Exact corner threshold under exact labels (O1) and the local-to-global check with the shape released
(Section 3.1 after Theorem 2, Section S3.8, Table S21).

Part 1: with v(c) = 1/eta(c), Theorem 2(c) gives that the corner is the global optimum iff
        Lambda <= Lam_ex = inf_{c<1} (kappa0 + c) / ((1 - c) eta(c)).
        For the power class eta = c^s (s = Upsilon) the table compares Lam_ex with the local O3 threshold (1 + kappa0) s - 1.
Part 2: for D = tau^beta with T and beta both unknown (O3), the information curve is J(c) = (a d - b^2) / d, normalized at c = 1.
        For each beta it counts the Lambda values at which the local condition holds but the corner is not the global minimum
        of (kappa0 + c)(Lambda + 1 / J(c)).
Usage: python exact_threshold.py
"""
import numpy as np
from scipy.integrate import quad

K0 = 0.05
cs = np.linspace(0.01, 0.999, 20000)

print('Part 1: Upsilon  local threshold  exact threshold (O1)  minimizing c')
for s in (1.22, 1.42, 2.0, 2.7, 3.0, 4.87, 8.76, 9.96):
    g = (K0 + cs) / ((1 - cs) * cs**s)
    i = np.argmin(g)
    print('        %6.2f  %14.2f  %20.2f  %12.2f' % (s, (1 + K0) * s - 1, g[i], cs[i]))

print('Part 2: beta   Upsilon*Pi(1)   local threshold   Lambda with local but not global (of 120)')
cg = np.linspace(0.02, 1.0, 981)
for beta in (0.12, 0.5, 1, 2, 4.48, 10):
    def J(c):
        a = quad(lambda t: beta**2 * t**(2 * beta), 0, c)[0]
        b = -beta * quad(lambda t: t**(2 * beta) * np.log(t), 0, c)[0]
        d = quad(lambda t: t**(2 * beta) * np.log(t)**2, 0, c)[0]
        return (a * d - b**2) / d
    Jv = np.array([J(c) for c in cg]) / J(1.0)
    up = (Jv[-1] - Jv[-2]) / (cg[-1] - cg[-2])
    lam_loc = (1 + K0) * up - 1
    bad = 0
    for lam in np.linspace(0.02, 3 * max(lam_loc, 0.3), 120):
        G = (K0 + cg) * (lam + 1 / Jv)
        if lam <= lam_loc and np.argmin(G) != len(cg) - 1:
            bad += 1
    print('        %5.2f  %13.2f  %15.2f  %12d' % (beta, up, lam_loc, bad))
