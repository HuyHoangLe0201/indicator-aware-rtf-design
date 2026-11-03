"""Check of Proposition 9(B) (Appendix B.4): label variance of the plug-in first-passage time of a Wiener process.

X(t) = mu t + sigma_W B(t), failure at the first passage of w, units mu = w = 1 so that a = sigma_W^2 and the nominal life is 1.
A unit is observed to t_c = c.  The plug-in label is  T_hat = t_c + (w - X(t_c)) / mu_hat  with  mu_hat = X(t_c) / t_c.
Given X(t_c) = x < w the remaining passage time is inverse Gaussian with mean (w - x)/mu and shape (w - x)^2 / sigma_W^2
(the draws that crossed before t_c are the unit's complete lives and are discarded, which matters only when a is not small).
Reported: the variance of T_hat - T relative to the squared nominal life, against the first-order prediction a (1-c)/c,
and the ratio to the same quantity when mu_hat is replaced by the true mu (the part of the label variance that is the
passage remainder alone, a (1-c)).
Usage: python wiener_passage_check.py
"""
import numpy as np

rng = np.random.default_rng(11)
N = 2_000_000
print('sigma_W      a     c   simulated  predicted  ratio   (remainder only: simulated, a(1-c), ratio)')
for sig in (0.05, 0.10, 0.20, 0.35):
    a = sig ** 2
    for c in (0.3, 0.6):
        x = c + sig * np.sqrt(c) * rng.standard_normal(N)
        ok = x < 1.0
        x = x[ok]
        m = (1.0 - x)                    # mean remaining time at mu = 1
        lam = (1.0 - x) ** 2 / sig ** 2
        # inverse Gaussian draws (Michael, Schucany, Haas)
        nu = rng.standard_normal(len(x))
        y = nu ** 2
        xm = m + m * m * y / (2 * lam) - m / (2 * lam) * np.sqrt(4 * m * lam * y + m * m * y * y)
        u = rng.random(len(x))
        rem = np.where(u <= m / (m + xm), xm, m * m / xm)
        T = c + rem
        mu_hat = x / c
        That = c + (1.0 - x) / mu_hat
        v_sim = np.mean((That - T) ** 2)
        v_pred = a * (1 - c) / c
        v_rem = np.mean((c + (1.0 - x) - T) ** 2)
        print(f'{sig:7.2f} {a:8.4f} {c:5.1f} {v_sim:10.5f} {v_pred:10.5f} {v_sim / v_pred:6.2f}   ({v_rem:9.5f}, {a * (1 - c):9.5f}, {v_rem / (a * (1 - c)):5.2f})')
