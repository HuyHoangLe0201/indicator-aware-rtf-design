"""Regret of the corner and of fixed early stops against the best stop (Proposition A.3, Section S3.10, Table S23).

Model: G(c) = (kappa0 + c) (Lambda + 1 / eta(c)), eta(c) = c^s (power class, s = Upsilon >= 1, so eta <= c).
Regret of a stop c at s = G(c) / min_c' G(c').
Claim: for every curve with eta(c) <= c the corner's regret is at most (1 + k0)(1 + Lam) / (1 + sqrt(k0 Lam))^2 (Lam >= k0),
whereas a fixed stop c < 1 has regret that grows without bound with s.
Usage: python corner_regret.py
"""
import numpy as np

K0 = 0.05
cs = np.linspace(0.002, 1, 20000)


def regret(c, s, lam):
    g = (K0 + cs) * (lam + cs**(-s))
    return (K0 + c) * (lam + c**(-s)) / g.min()


for lam in (0.241, 0.92):
    bound = (1 + K0) * (1 + lam) / (1 + np.sqrt(K0 * lam))**2 if lam >= K0 else 1.0
    print('Lambda = %.3f   bound on the corner regret %.3f' % (lam, bound))
    print('   %6s %9s %9s %9s' % ('s', 'corner', 'c = 0.65', 'c = 0.50'))
    for s in (1.0, 1.22, 2.0, 3.0, 5.0, 9.96, 20.0):
        print('   %6.2f %9.3f %9.2f %9.2f' % (s, regret(1.0, s, lam), regret(0.65, s, lam), regret(0.5, s, lam)))
    print('   largest corner regret over s in [1, 60]: %.3f' % max(regret(1.0, s, lam) for s in np.linspace(1, 60, 300)))
