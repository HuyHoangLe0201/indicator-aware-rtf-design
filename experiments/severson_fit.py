"""
Fit Paper 1's stretched-exponential degradation class to the 124 real Severson cells.

Paper 1 (Table 2) asserts alpha_b = 3, b = 0.7 for batteries.  Those values have never
been checked against data.  Here D(tau) = [Q(0)-Q(tau T)] / [Q(0)-Q(T)] is fitted with
    D(tau; a,b) = (1 - exp(-a tau^b)) / (1 - exp(-a))
per cell, and the fleet distribution of (a,b) reported, together with the resulting
Upsilon = D'(1)^2 / int_0^1 tau^2 D'^2 and the design verdict it implies.
"""
import numpy as np, os
from scipy.optimize import least_squares
from scipy.integrate import quad

HERE = os.path.join(os.path.dirname(__file__), '..', 'data', 'severson')
d = np.load(os.path.join(HERE, 'severson_traj.npz'), allow_pickle=True)
QD, life, name = d['QD'], d['life'], d['name']

def Dg(tau, a, b):
    tau = np.clip(tau, 1e-9, 1.0)
    return (1-np.exp(-a*tau**b))/(1-np.exp(-a))

def fit_cell(q):
    q = np.asarray(q, float)
    q = q[np.isfinite(q) & (q > 0.2) & (q < 1.5)]          # drop dropouts / spikes
    if len(q) < 60:
        return None
    n = len(q)
    q0 = np.median(q[:10]); qT = np.median(q[-5:])
    if q0 - qT < 0.02:
        return None
    D = (q0 - q)/(q0 - qT)
    tau = (np.arange(n)+1)/n
    keep = (D > -0.15) & (D < 1.35)
    tau, D = tau[keep], D[keep]
    try:
        r = least_squares(lambda p: Dg(tau, p[0], p[1]) - D, [3.0, 0.7],
                          bounds=([0.05, 0.15], [30., 3.0]),
                          loss='soft_l1', f_scale=0.05, max_nfev=800)
    except Exception:
        return None
    res = Dg(tau, *r.x) - D
    return r.x[0], r.x[1], float(np.sqrt(np.mean(res**2)))

rows = [fit_cell(q) for q in QD]
ok = [r for r in rows if r is not None]
A = np.array([r[0] for r in ok]); B = np.array([r[1] for r in ok])
R = np.array([r[2] for r in ok])
print(f'fitted {len(ok)}/{len(QD)} cells;  median fit RMSE = {np.median(R):.4f}')
for nm, v, p1 in [('alpha_b', A, 3.0), ('b', B, 0.7)]:
    print(f'  {nm:8s} median {np.median(v):6.3f}   mean {v.mean():6.3f}   '
          f'IQR [{np.percentile(v,25):.3f}, {np.percentile(v,75):.3f}]   '
          f'log-CV {np.std(np.log(v),ddof=1):.3f}   Paper 1: {p1}')

ab, bb = float(np.median(A)), float(np.median(B))
def ups(a, b):
    dp = lambda t: a*b*t**(b-1)*np.exp(-a*t**b)/(1-np.exp(-a))
    Z = quad(lambda t: t**2*dp(t)**2, 0, 1, limit=400)[0]
    return dp(1.0)**2/Z
print(f'\nUpsilon at fitted median (a={ab:.2f}, b={bb:.2f}) = {ups(ab,bb):.3f}')
print(f'Upsilon at Paper 1      (a=3.00, b=0.70)          = {ups(3.0,0.7):.3f}')
U = np.array([ups(a, b) for a, b in zip(A, B)])
print(f'per-cell Upsilon: median {np.median(U):.3f}  IQR [{np.percentile(U,25):.3f}, '
      f'{np.percentile(U,75):.3f}]')
print(f'\ncorner threshold needs  Pi > Pi_crit = 3.23/{ups(3.0,0.7):.3f}*Upsilon ... '
      f'see refresh_pi.py; with measured Pi=38.5, varpi = Pi*Upsilon = {38.5*np.median(U):.1f}')
np.savez(os.path.join(HERE, 'severson_shape.npz'), alpha=A, b=B, rmse=R, ups=U)
