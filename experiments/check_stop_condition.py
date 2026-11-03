"""Numerical check of Theorem 2(f), the stationarity condition for an absolute stop time.

The risk G(t_s) = kbar(t_s) [1/itilde(t_s) + Lambda] is minimised on a grid and compared with the root of
    Ups_star(t_s) = itilde (1 + Lambda itilde) / kbar,   Ups_star = Tbar [varpi h + M] / (1 + varpi).

Fleet: lognormal lives with mean one, power-law shape (eta_T(c) = c^(2 beta + 1)).
kbar  = kappa0 + E[min(T, t)]                      mean cost per unit
ibar  = E[eta(min(1, t/T))] + varpi P(T <= t)     mean information per unit; itilde = ibar / (1 + varpi)
For varpi = 0 (O3) the two agree to the grid step in every interior case; for varpi = 5 the minimum is the corner.
Usage: python check_stop_condition.py
"""
import numpy as np
from scipy import integrate, optimize, stats

KAPPA0, LAM = 0.05, 0.241


def fleet(beta, s_log):
    dist = stats.lognorm(s=s_log, scale=np.exp(-0.5 * s_log**2))
    a = 2 * beta + 1
    return dist, a


def kbar(t, dist):
    return KAPPA0 + integrate.quad(dist.sf, 0, t)[0]


def itilde(t, dist, a, varpi):
    surv = integrate.quad(lambda T: (t / T)**a * dist.pdf(T), t, np.inf)[0]
    return (surv + dist.cdf(t) * (1 + varpi)) / (1 + varpi)


def risk(t, beta, s_log, varpi):
    dist, a = fleet(beta, s_log)
    return kbar(t, dist) * (1.0 / itilde(t, dist, a, varpi) + LAM)


def stationarity(t, beta, s_log, varpi):
    dist, a = fleet(beta, s_log)
    h = dist.pdf(t) / dist.sf(t)
    m = integrate.quad(lambda T: a * (t / T)**(a - 1) / T * dist.pdf(T), t, np.inf)[0] / dist.sf(t)
    ups = (varpi * h + m) / (1 + varpi)            # Tbar = 1
    it = itilde(t, dist, a, varpi)
    return ups - it * (1 + LAM * it) / kbar(t, dist)


if __name__ == '__main__':
    ts = np.linspace(0.15, 3.0, 60)
    for beta in (0.12, 0.5, 2.0, 4.48):
        for s_log in (0.25, 0.5):
            for varpi in (0.0, 5.0):
                r = np.array([risk(t, beta, s_log, varpi) for t in ts])
                t_arg = ts[np.argmin(r)]
                s = np.array([stationarity(t, beta, s_log, varpi) for t in ts])
                roots = [optimize.brentq(stationarity, ts[i], ts[i + 1], args=(beta, s_log, varpi))
                         for i in range(len(ts) - 1) if s[i] * s[i + 1] < 0]
                dist, _ = fleet(beta, s_log)
                print('beta %.2f sigma_log %.2f varpi %.0f | argmin G: t=%.2f (F=%.2f) | roots %s'
                      % (beta, s_log, varpi, t_arg, dist.cdf(t_arg), np.round(roots, 2)))
