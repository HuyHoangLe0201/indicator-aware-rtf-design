"""First-order variance of the plug-in fleet estimators (Proposition 7 of Appendix B.8) evaluated for the
Wiener simulation of sim_wiener_plan.py, and the condition on Lambda_f for an interior optimum.
V(g) = (kappa0 + E min(T,t_c)/Tbar)/C * [ s^2 (1 + z^2/2) + E v (1 + z^2) + z^2 E v^2 / (2 s^2) ],
with v_i = CV^2 / c_i for a suspended unit (c_i = t_c/T_i, the share of its life observed), 0 for a completed one."""
import sys

import numpy as np

S_A, KAP0, C = 0.30, 0.05, 21.0
GS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.6, 3.0)
rng = np.random.default_rng(7)
a = np.exp(S_A * rng.standard_normal(400000))            # drift in units of a0 = 1/median life
for cvd in (0.08, 0.15, 0.25):
    T = rng.wald(1.0 / a, 1.0 / cvd ** 2 * np.ones_like(a)) if False else rng.wald(1.0 / a, 1.0 / (cvd ** 2 * 1.0))
    tbar = T.mean()
    s2 = S_A ** 2
    out = {}
    for z, nm in ((0.0, 'median'), (-1.2815515655446004, 'q10')):
        vals = []
        for g in GS:
            c_i = np.minimum(1.0, g / T)
            v = np.where(T <= g, 0.0, cvd ** 2 / c_i)
            cost = KAP0 + np.minimum(T, g).mean() / tbar
            V = cost / C * (s2 * (1 + z * z / 2) + v.mean() * (1 + z * z) + z * z * (v ** 2).mean() / (2 * s2))
            vals.append(np.sqrt(V))
        out[nm] = np.array(vals)
    lam = s2 * (1 + 1.2815515655446004 ** 2 / 2) / ((1 + 1.2815515655446004 ** 2) * cvd ** 2)
    # interior optimum exists iff some c in (0,1) has (kappa0 + c)/c < (1 - c) Lambda_f  (eta_T(c) = c, Upsilon = 1)
    cc = np.linspace(0.01, 0.99, 981)
    need = ((KAP0 + cc) / (cc * (1 - cc))).min()
    print(f'CV {cvd}: Lambda_f = {lam:.2f}, smallest Lambda_f admitting an interior optimum = {need:.2f} '
          f'-> {"interior" if lam > need else "corner"}')
    for nm in out:
        r = out[nm] / out[nm][-1]
        k = int(np.argmin(r))
        print(f'   {nm:7s} sqrt(V)/sqrt(V at run to failure): ' + ' '.join(f'{x:.2f}' for x in r) + f'   argmin t_c = {GS[k]}')
