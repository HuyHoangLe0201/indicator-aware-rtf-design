"""
Recompute the misspecification row and the section-6 worked example of Paper 1 under the
corrected battery class (convex power law, beta = 4.48).

Delta(D1,D2)/Tbar^2 = inf_theta int_0^1 ( D_D1(tau) - D_D2(tau;theta) )^2 dtau,
with the exponential class extended to alpha in R so it can take convex shapes
(Paper 1, remark after Thm 8).  The existing beta = 3 row is reproduced first as a check.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, minimize

Dlin = lambda t: t
Dpow = lambda t, b: np.clip(t, 0, 1)**b
def Dexp(t, a):
    if abs(a) < 1e-8: return t
    return (1-np.exp(-a*t))/(1-np.exp(-a))
def Dstr(t, a, b):
    if abs(a) < 1e-8: return np.clip(t, 0, 1)**b
    return (1-np.exp(-a*np.clip(t, 0, 1)**b))/(1-np.exp(-a))

def L2(f, g):
    return quad(lambda t: (f(t)-g(t))**2, 0, 1, limit=300)[0]

def to_lin(true):  return L2(true, Dlin)
def to_exp(true):
    r = minimize_scalar(lambda a: L2(true, lambda t: Dexp(t, a)),
                        bounds=(-30, 30), method='bounded', options={'xatol': 1e-8})
    return r.fun, r.x
def to_pow(true):
    r = minimize_scalar(lambda b: L2(true, lambda t: Dpow(t, b)),
                        bounds=(0.05, 20), method='bounded', options={'xatol': 1e-8})
    return r.fun, r.x
def to_str(true):
    r = minimize(lambda p: L2(true, lambda t: Dstr(t, p[0], p[1])), [1.0, 1.0],
                 bounds=[(1e-6, 40), (0.05, 12)], method='L-BFGS-B')
    return r.fun, r.x

print('=== check: reproduce Paper 1 Table 5 ===')
print(f'{"true":22s} {"->linear":>9s} {"->expon.":>9s} {"->power":>9s}   (published)')
for lbl, tr, pub in [('Expon. (alpha=2)',  lambda t: Dexp(t, 2.0),        '0.030 / 0 / 0.002'),
                     ('Power-law (beta=3)', lambda t: Dpow(t, 3.0),        '0.076 / 0.001 / 0'),
                     ('Str.-exp (b=0.7)',   lambda t: Dstr(t, 3.0, 0.7),   '0.112 / 0.002 / 0.003')]:
    print(f'{lbl:22s} {to_lin(tr):9.4f} {to_exp(tr)[0]:9.4f} {to_pow(tr)[0]:9.4f}   {pub}')

BETA, TBAR, A0, B_CAP, EPS0, D_SENS = 4.483, 806.0, 0.126, 2000.0, 100.0, 6
true = lambda t: Dpow(t, BETA)
dl = to_lin(true); de, ae = to_exp(true); dp_, bp = to_pow(true); ds, ps = to_str(true)
print(f'\n=== NEW ROW: true power-law beta = {BETA} ===')
print(f'  -> linear   Delta/Tbar^2 = {dl:.4f}   sqrt(Delta) = {np.sqrt(dl)*TBAR:6.1f} cycles')
print(f'  -> expon.   Delta/Tbar^2 = {de:.4f}   sqrt(Delta) = {np.sqrt(de)*TBAR:6.1f} cycles'
      f'   (best alpha = {ae:.2f})')
print(f'  -> power    Delta/Tbar^2 = {dp_:.5f}  (best beta = {bp:.2f})')
print(f'  -> str-exp  Delta/Tbar^2 = {ds:.5f}  (alpha_b={ps[0]:.3g}, b={ps[1]:.2f})')

print('\n=== section 6 worked example, recomputed ===')
pD_new, pD_old = D_SENS+1+1, D_SENS+1+2
n1 = A0**2*B_CAP**2*320/EPS0**2
n2n, n2o = (A0**2*B_CAP**2*p/EPS0**2 for p in (pD_new, pD_old))
print(f'Step 1  unchanged:  n* = {n1:.0f} CDTs (class-independent)')
print(f'Step 2  |theta|=1 so p_D = {pD_new} (was {pD_old}):  n*_D = {n2n:.0f} (was {n2o:.0f}),'
      f' reduction {100*(1-n2n/n1):.1f}% (was {100*(1-n2o/n1):.1f}%)')
print(f'Step 3  true power-law, assumed exponential: Delta/Tbar^2 = {de:.4f},'
      f' RMSE contribution {np.sqrt(de)*TBAR:.0f} cycles')
print(f'        -> {"BELOW" if np.sqrt(de)*TBAR < EPS0 else "ABOVE"} the {EPS0:.0f}-cycle target'
      f'  (article: 36 cycles, below)')
eta_new = 0.5**(2*BETA-1); eta_old = 0.99
print(f'Step 4  eta(0.50) = 0.5^(2b-1) = {eta_new:.2e}  (article: {eta_old})')
print(f'        n_eff = 60 + 200*{eta_new:.2e} = {60+200*eta_new:.1f}'
      f'   (article: {60+200*eta_old:.0f})')
print(f'        -> the 200 partially cycled cells are worth {200*eta_new:.2f} complete ones')
kap = min(1.0, (BETA-1)/2)
rmse = A0*B_CAP*np.sqrt(pD_new/60)
print(f'Step 5  kappa = min(1,(beta-1)/2) = {kap:.2f} -> fast regime, unchanged')
print(f'        RMSE_bound(n=60) = {A0}*{B_CAP:.0f}*sqrt({pD_new}/60) = {rmse:.0f} cycles'
      f'  (article: 98)')
