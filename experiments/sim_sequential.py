"""Online plug-in stopping rule for an absolute-stop campaign (O3, varpi = 0).
Continue while Ups*(t) > eta(1+Lam eta)/kbar; stop at the first checkpoint where this fails.
Lognormal life, power-law shape.  Closed forms through lognormal partial moments.
The estimates use only the data available at time t: failures before t and the survivors censored at t."""
import sys

import numpy as np
from scipy import optimize
from scipy.special import ndtr

kap0, Lam = 0.05, 0.241


def pm(k, mu, s, t):
    """E[T^k ; T > t] for lognormal(mu, s)."""
    return np.exp(k * mu + 0.5 * k * k * s * s) * ndtr((mu + k * s * s - np.log(t)) / s)


def pm_le(k, mu, s, t):
    return np.exp(k * mu + 0.5 * k * k * s * s) * ndtr(-(mu + k * s * s - np.log(t)) / s)


def funcs(t, mu, s, b):
    """kbar, eta_tilde (varpi=0), Ups*, and F(t) for lognormal(mu,s) lives, shape exponent b."""
    a = 2 * b + 1
    Fb = 1 - ndtr((np.log(t) - mu) / s)
    F = 1 - Fb
    # Tbar = E[T] is the time unit; cost is normalised by Tbar
    Tbar = np.exp(mu + 0.5 * s * s)
    kbar = kap0 + (pm_le(1, mu, s, t) + t * Fb) / Tbar
    # information per unit (path only): complete units carry 1, survivors eta(t/T)
    it = F + t**a * pm(-a, mu, s, t)
    # marginal value per unit of stop time, divided by marginal cost: Ups*
    dens = np.exp(-0.5 * ((np.log(t) - mu) / s)**2) / (t * s * np.sqrt(2 * np.pi))
    h = dens / Fb
    M = a * t**(a - 1) * pm(-a, mu, s, t) / Fb       # F-bar^-1 int eta'(t/T) T^-1 f dT
    ups = Tbar * (0 * h + M)
    return kbar, it, ups, F


def risk(t, mu, s, b):
    kbar, it, ups, F = funcs(t, mu, s, b)
    return kbar * (1.0 / it + Lam)


def oracle(mu, s, b):
    ts = np.linspace(0.05, 8.0, 800) * np.exp(mu + 0.5 * s * s)
    r = np.array([risk(t, mu, s, b) for t in ts])
    j = np.argmin(r)
    return ts[j], r[j]


def fit_censored(x, event, mu0, s0):
    def nll(p):
        mu, ls = p
        s = np.exp(ls)
        z = (np.log(x) - mu) / s
        ll = np.where(event, -np.log(x * s) - 0.5 * z * z, np.log(np.maximum(1 - ndtr(z), 1e-300)))
        return -ll.sum()
    r = optimize.minimize(nll, [mu0, np.log(s0)], method='Nelder-Mead', options=dict(xatol=1e-3, fatol=1e-4, maxiter=200))
    return r.x[0], np.exp(r.x[1])


def one_run(n, mu, s, b, rng, tgrid, tstar_pop, minfail=5):
    T = np.exp(mu + s * rng.standard_normal(n))
    mu_h, s_h = np.log(np.median(T)), 0.4
    for t in tgrid:
        ev = T <= t
        if ev.sum() < minfail:
            continue
        x = np.where(ev, T, t)
        mu_h, s_h = fit_censored(x, ev, mu_h, max(s_h, 0.05))
        s_h = min(max(s_h, 0.05), 2.0)
        kbar, it, ups, F = funcs(t, mu_h, s_h, b)
        if ups <= it * (1 + Lam * it) / kbar:
            return t, T
    return tgrid[-1] * 1.0, T


def main(case):
    b, s = case
    mu = -0.5 * s * s
    ts, rs = oracle(mu, s, b)
    kb, itt, ups, Fs = funcs(ts, mu, s, b)
    print('\ncase beta=%.2f (Ups=%.2f) sL=%.2f | oracle t*=%.2f (F*=%.2f), risk %.3f' % (b, 2 * b + 1, s, ts, Fs, rs), flush=True)
    rng = np.random.default_rng(7)
    tgrid = np.arange(0.2, 4.0, 0.1)
    for n in (10, 20, 40, 80):
        reg_seq, F_seq, reg_h, reg_c = [], [], [], []
        for rep in range(120):
            t, T = one_run(n, mu, s, b, rng, tgrid, ts)
            reg_seq.append(risk(t, mu, s, b) / rs - 1)
            F_seq.append(funcs(t, mu, s, b)[3])
            th = np.sort(T)[int(np.ceil(0.5 * n)) - 1]
            reg_h.append(risk(th, mu, s, b) / rs - 1)
            reg_c.append(risk(np.max(T), mu, s, b) / rs - 1)
        print('  n=%3d | sequential: regret mean %.3f (median %.3f), F stop mean %.2f sd %.2f | fixed F=0.5: %.3f | corner (all fail): %.3f' % (
            n, np.mean(reg_seq), np.median(reg_seq), np.mean(F_seq), np.std(F_seq), np.mean(reg_h), np.mean(reg_c)), flush=True)


if __name__ == '__main__':
    # usage: python sim_sequential.py BETA SIGMA_LOG   e.g.  0.5 0.5   (Upsilon = 2 BETA + 1)
    main((float(sys.argv[1]), float(sys.argv[2])))
