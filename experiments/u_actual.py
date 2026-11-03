"""
u with the estimator's ACTUAL penalty, and with the correct pooling structure.

Two corrections to the earlier comparison.

(1) STRUCTURE.  Theorem 8's factor is 1 + u/n because the shape is a single parameter
    SHARED by n units, so the centre is estimated and its error is O(1/n).  The oracle
    arm supplies the TRUE centre and shrinks each unit SEPARATELY toward it.  There is
    then no centre uncertainty, and the inflation is the ordinary profile-likelihood
    factor
                       Pi_rob = 1/(1-rho^2) = 1 + u,          (no 1/n)

(2) PENALTY SCALE.  With M samples spread over tau in [0,c], sum_j f(tau_j) ~ (M/c) int_0^c f.
    So the data Hessian is (M/c)*[A,B;B',C] and the penalty lambda*I enters as
                       Lam_actual = (c*lambda/M) I,     lambda = lam_mult * sigma^2 M / s^2
    i.e.  Lam_actual = lam_mult * c * sigma^2 / s^2,  versus the empirical-Bayes value
                       Lam_EB = c * sigma^2 / (M s^2).
    Ratio Lam_actual/Lam_EB = lam_mult * M = 6.
"""
import numpy as np
from scipy.integrate import quad

M, SH, LAMMULT, N = 60, 0.25, 0.1, 120

def u_of(Dp, dlog, c, Lam):
    A = quad(lambda t: (t*Dp(t))**2, 0, c, limit=400)[0]
    B = np.array([quad(lambda t: -(t*Dp(t))*d(t), 0, c, limit=400)[0] for d in dlog])
    C = np.array([[quad(lambda t: d1(t)*d2(t), 0, c, limit=400)[0] for d2 in dlog]
                  for d1 in dlog])
    r2 = float(B @ np.linalg.solve(C + Lam, B) / A)
    r2 = min(max(r2, 0.0), 1-1e-12)
    return r2/(1-r2)

AB, BB, SB = 3.0, 0.7, 0.05
sDp = lambda t: AB*BB*t**(BB-1)*np.exp(-AB*t**BB)/(1-np.exp(-AB))
def sdab(t):
    Mm = 1-np.exp(-AB); Nn = 1-np.exp(-AB*t**BB)
    return AB*(t**BB*np.exp(-AB*t**BB)*Mm - Nn*np.exp(-AB))/Mm**2
sdb = lambda t: BB*AB*t**BB*np.log(t)*np.exp(-AB*t**BB)/(1-np.exp(-AB))

BE, SG = 3.0, 0.15
bDp  = lambda t: BE*t**(BE-1)
bdlg = [lambda t: BE*t**BE*np.log(t)]

CASES = {
 'Battery': (sDp, [sdab, sdb], SB, {0.15:29.5, 0.25:26.3, 0.35:28.6, 0.50:29.9, 0.65:24.4}),
 'Bearing': (bDp, bdlg,        SG, {0.50:1.39, 0.65:3.37, 0.80:4.51}),
}

print(f"{'domain':8s} {'c':>5s} | {'1+u/n (old)':>12s} | {'1+u  EB':>8s} {'1+u actual':>11s} "
      f"| {'measured':>9s} {'ratio EB':>9s} {'ratio act':>10s}")
for nm, (Dp, dlg, sig, meas) in CASES.items():
    k = len(dlg)
    for c, mv in meas.items():
        LEB  = (c*sig**2/(M*SH**2))*np.eye(k)          # empirical-Bayes prior precision
        LACT = (LAMMULT*c*sig**2/SH**2)*np.eye(k)      # estimator's actual penalty
        uEB, uAC = u_of(Dp, dlg, c, LEB), u_of(Dp, dlg, c, LACT)
        print(f"{nm:8s} {c:5.2f} | {1+uEB/N:12.2f} | {1+uEB:8.2f} {1+uAC:11.2f} "
              f"| {mv:9.2f} {mv/(1+uEB):9.2f} {mv/(1+uAC):10.2f}")
    print()

print("geometric-mean |log ratio| over all 8 points:")
for lab, use_act in [('1+u  EB', False), ('1+u actual', True)]:
    r = []
    for nm, (Dp, dlg, sig, meas) in CASES.items():
        k = len(dlg)
        for c, mv in meas.items():
            L = ((LAMMULT*c*sig**2/SH**2) if use_act else (c*sig**2/(M*SH**2)))*np.eye(k)
            r.append(abs(np.log(mv/(1+u_of(Dp, dlg, c, L)))))
    print(f"   {lab:>11s} : {np.mean(r):.3f}   (factor {np.exp(np.mean(r)):.2f})")
