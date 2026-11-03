"""Numerical check of the bridge between the predictive (odds) estimator and Theorem 2 (R2.2, R4.2).

Gaussian linear model: a window to fraction c leaves the label ln T with conditional variance  sigma^2 / (1 + omega(c)),
omega = R^2/(1-R^2) the odds.  Write eta = omega/omega_bar, eta0 = 1/omega_bar.  The budgeted risk is, up to a constant,
    G(c) = (kappa0 + c) * [ Lambda + 1/(eta(c) + eta0) ],          Lambda = (S + K_reg) * omega_bar / (K_lab sigma^2)
and the corner c = 1 is a local optimum iff
    Upsilon = eta'(1)  >=  theta_b = (1 + eta0) * (1 + Lambda (1 + eta0)) / (1 + kappa0).
Checks: (1) the closed form against a finite-difference derivative of G at c = 1 for random curves;
        (2) for the power-law information class eta = c^s, whether the grid minimizer of G is the corner exactly when
            Upsilon = s >= theta_b (and how often the local and global statements disagree);
        (3) the thresholds implied by the measured quantities of the battery fleets.
Usage: python bridge_check.py
"""
import numpy as np

rng = np.random.default_rng(7)
C = np.linspace(1e-3, 1.0, 20001)


def G(eta, lam, eta0, k0):
    return (k0 + C) * (lam + 1.0 / (eta + eta0))


def theta_b(lam, eta0, k0):
    return (1 + eta0) * (1 + lam * (1 + eta0)) / (1 + k0)


# (1) local condition against finite differences, random smooth monotone curves eta(c) with eta(1) = 1
bad = 0
N1 = 4000
for _ in range(N1):
    s = rng.uniform(0.2, 12.0)
    q = rng.uniform(0.0, 4.0)
    lam = rng.uniform(0.05, 5.0)
    eta0 = 10 ** rng.uniform(-2.0, 0.5)
    k0 = rng.uniform(0.0, 0.3)
    eta = lambda c: (c ** s) * np.exp(-q * (1 - c)) / 1.0
    e1 = lambda c: eta(c) / eta(1.0)
    h = 1e-6
    ups = (e1(1.0) - e1(1.0 - h)) / h
    g = lambda c: (k0 + c) * (lam + 1.0 / (e1(c) + eta0))
    dg = (g(1.0) - g(1.0 - h)) / h
    pred_corner_local = ups >= theta_b(lam, eta0, k0)
    if (dg <= 0) != pred_corner_local:
        bad += 1
print(f'(1) local condition vs finite difference: {N1 - bad}/{N1} agree')

# (2) power-law information class: global minimizer vs the local condition.  The risk is linear in the label variance, which
#     holds while the window is informative; with eta0 > 0 the variance is bounded by the prior, and a window with no information
#     (omega << 1) would seem to buy unlimited units.  The comparison is therefore made on windows with odds omega >= omega_min.
for omega_min in (0.0, 1.0, 2.0):
    agree = tot = 0
    r2 = np.random.default_rng(11)
    for _ in range(3000):
        s_ = r2.uniform(0.5, 12.0)
        lam = r2.uniform(0.05, 3.0)
        eta0 = 10 ** r2.uniform(-2.0, 0.5)
        k0 = r2.uniform(0.01, 0.2)
        eta = C ** s_
        m = (eta / eta0) >= omega_min
        if m.sum() < 5:
            continue
        g = G(eta, lam, eta0, k0)[m]
        tot += 1
        agree += ((np.argmin(g) == m.sum() - 1) == (s_ >= theta_b(lam, eta0, k0)))
    print(f'(2) power class, windows with omega >= {omega_min:3.1f}: global minimizer at the corner  <=>  Upsilon >= theta_b in {agree}/{tot} draws')

# (3) thresholds implied by measured quantities
k0 = 0.05
print('(3) thresholds, kappa0 = 0.05:  Theorem 2: (1+Lambda)/(1+kappa0)   bridge: theta_b')
for name, omega_bar, lams in (('Severson dQ(V), omega_bar=9.5', 9.5, (0.241, 0.30, 0.58, 0.86)),
                              ('window R^2 = 0.80, omega_bar=4', 4.0, (0.241, 0.30, 0.58, 0.86))):
    eta0 = 1.0 / omega_bar
    for lam in lams:
        print(f'   {name:34s} Lambda={lam:5.3f}   {(1 + lam) / (1 + k0):5.3f}   {theta_b(lam, eta0, k0):5.3f}')
