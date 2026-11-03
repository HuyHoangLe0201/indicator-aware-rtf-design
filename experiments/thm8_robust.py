"""
Does Theorem 8 hold on the ROBUST scale?

Thm 8 derives  Pi = 1 + u/n  from the LOCAL Fisher blocks, i.e. from the curvature of
the log-likelihood at the truth.  That curvature governs the GAUSSIAN CORE of the error
distribution -- exactly what the robust scale measures.  The heavy tail comes from
GLOBAL non-identifiability (multi-modal likelihood), a separate phenomenon.

So the prediction should be tested against Pi_rob, not against the variance.
Compare, for both benchmark domains:
   Thm 8 complete pooling :  1 + u_cp(c)/n
   Thm 8 partial pooling  :  1 + u_pp(c)/n     (C -> C + sigma^2 S_th^-1 / M)
   measured Pi_rob(c)
"""
import numpy as np
from scipy.integrate import quad

M, SH_CV, N = 60, 0.25, 120

def blocks(Dp, dth, c):
    A = quad(lambda t: (t*Dp(t))**2, 0, c, limit=400)[0]
    B = np.array([quad(lambda t: -(t*Dp(t))*d(t), 0, c, limit=400)[0] for d in dth])
    C = np.array([[quad(lambda t: d1(t)*d2(t), 0, c, limit=400)[0] for d2 in dth]
                  for d1 in dth])
    return A, B, C

def u_of(Dp, dth, c, Lpr):
    A, B, C = blocks(Dp, dth, c)
    r2 = float(B @ np.linalg.solve(C + Lpr, B) / A)
    r2 = min(max(r2, 0.0), 1-1e-12)
    return r2/(1-r2)

# ---- bearing: power law, theta = beta (log-parameterised) -------------------
BE, SIG_BE = 3.0, 0.15
bDp  = lambda t: BE*t**(BE-1)
bdth = [lambda t: BE*t**BE*np.log(t)]                 # dD/dlog beta
bS   = np.array([[SH_CV**2]])
bLpr = (SIG_BE**2/M)*np.linalg.inv(bS)

# ---- battery: stretched exp, theta = (alpha_b, b) ---------------------------
AB, BB, SIG_BA = 3.0, 0.7, 0.05
sDp  = lambda t: AB*BB*t**(BB-1)*np.exp(-AB*t**BB)/(1-np.exp(-AB))
def sdab(t):
    Mm = 1-np.exp(-AB); N_ = 1-np.exp(-AB*t**BB)
    return AB*(t**BB*np.exp(-AB*t**BB)*Mm - N_*np.exp(-AB))/Mm**2
sdb  = lambda t: BB*AB*t**BB*np.log(t)*np.exp(-AB*t**BB)/(1-np.exp(-AB))
sS   = np.diag([SH_CV**2, SH_CV**2])
sLpr = (SIG_BA**2/M)*np.linalg.inv(sS)

MEAS = {'Bearing': {0.50: 1.39, 0.65: 3.37, 0.80: 4.51},           # Pi_rob, box-averaged
        'Battery': {0.15: 29.5, 0.25: 26.3, 0.35: 28.6, 0.50: 29.9, 0.65: 24.4}}

print(f"{'domain':9s} {'c':>5s} {'u complete':>11s} {'u partial':>10s} "
      f"{'Thm8 cp':>8s} {'Thm8 pp':>8s} {'measured':>9s} {'ratio':>7s}")
for nm, Dp, dth, Lpr in [('Bearing', bDp, bdth, bLpr), ('Battery', sDp, [sdab, sdb], sLpr)]:
    k = len(dth)
    for c, meas in MEAS[nm].items():
        ucp = u_of(Dp, dth, c, 1e-14*np.eye(k))
        upp = u_of(Dp, dth, c, Lpr)
        p_cp, p_pp = 1+ucp/N, 1+upp/N
        print(f"{nm:9s} {c:5.2f} {ucp:11.1f} {upp:10.1f} {p_cp:8.2f} {p_pp:8.2f} "
              f"{meas:9.2f} {meas/p_cp:7.2f}")
    print()
