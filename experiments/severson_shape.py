"""
What degradation class do the real Severson cells actually follow?

The stretched-exponential fit of severson_fit.py ran to both box corners
(alpha_b -> 0, b -> 3), and alpha_b -> 0 makes D(tau) -> tau^b.  That is the
CONVEX power-law class, not the concave stretched-exponential Paper 1 assumes.
Here: widen the bounds, fit power-law and stretched-exponential separately, compare
fit quality, and print the fleet-average normalised curve so the shape is visible.
"""
import numpy as np, os
from scipy.optimize import least_squares
from scipy.integrate import quad

HERE = os.path.join(os.path.dirname(__file__), '..', 'data', 'severson')
d = np.load(os.path.join(HERE, 'severson_traj.npz'), allow_pickle=True)
QD = d['QD']

def curve(q):
    q = np.asarray(q, float)
    q = q[np.isfinite(q) & (q > 0.2) & (q < 1.5)]
    if len(q) < 60: return None
    n = len(q); q0 = np.median(q[:10]); qT = np.median(q[-5:])
    if q0-qT < 0.02: return None
    D = (q0-q)/(q0-qT); tau = (np.arange(n)+1)/n
    k = (D > -0.15) & (D < 1.35)
    return tau[k], D[k]

Dpow = lambda tau, b: np.clip(tau,1e-9,1.)**b
Dstr = lambda tau, a, b: (1-np.exp(-a*np.clip(tau,1e-9,1.)**b))/(1-np.exp(-a))

def fit(tau, D):
    r1 = least_squares(lambda p: Dpow(tau,p[0])-D, [2.], bounds=([.2],[12.]),
                       loss='soft_l1', f_scale=.05, max_nfev=600)
    r2 = least_squares(lambda p: Dstr(tau,p[0],p[1])-D, [3.,0.7],
                       bounds=([1e-3,.15],[40.,12.]), loss='soft_l1', f_scale=.05,
                       max_nfev=1200)
    e1 = np.sqrt(np.mean((Dpow(tau,r1.x[0])-D)**2))
    e2 = np.sqrt(np.mean((Dstr(tau,*r2.x)-D)**2))
    ep1 = np.sqrt(np.mean((Dstr(tau,3.,0.7)-D)**2))          # Paper 1's fixed values
    return r1.x[0], e1, r2.x[0], r2.x[1], e2, ep1

res, GRID = [], np.linspace(0.02,1.,50)
mean_curve = np.zeros(len(GRID)); m = 0
for q in QD:
    c = curve(q)
    if c is None: continue
    tau, D = c; res.append(fit(tau,D))
    mean_curve += np.interp(GRID, tau, D); m += 1
res = np.array(res)
print(f'cells fitted: {m}\n')
print(f'{"model":34s} {"median fit RMSE":>16s}')
print(f'{"power-law  D=tau^beta":34s} {np.median(res[:,1]):16.4f}')
print(f'{"stretched-exp (free a,b)":34s} {np.median(res[:,4]):16.4f}')
print(f'{"Paper 1 fixed a=3, b=0.7":34s} {np.median(res[:,5]):16.4f}   <-- assumed')
print(f'\npower-law exponent beta: median {np.median(res[:,0]):.3f}  '
      f'IQR [{np.percentile(res[:,0],25):.3f}, {np.percentile(res[:,0],75):.3f}]  '
      f'log-CV {np.std(np.log(res[:,0]),ddof=1):.3f}')
print(f'stretched-exp: a median {np.median(res[:,2]):.4f}  b median {np.median(res[:,3]):.3f}')

mean_curve /= m
print('\nfleet-mean normalised degradation D(tau)  (convex if below tau):')
print(f'{"tau":>6s} ' + " ".join(f"{t:6.2f}" for t in [.1,.2,.3,.4,.5,.6,.7,.8,.9,1.]))
print(f'{"D":>6s} " + ""' .replace('" + ""',''), end='')
print(" ".join(f"{np.interp(t,GRID,mean_curve):6.3f}" for t in [.1,.2,.3,.4,.5,.6,.7,.8,.9,1.]))
print(f'{"P1 D":>6s} ' + " ".join(f"{Dstr(t,3.,.7):6.3f}" for t in
                                  [.1,.2,.3,.4,.5,.6,.7,.8,.9,1.]))

def ups_pow(b): return (2*b+1)
print(f'\nUpsilon (power-law, = 2beta+1) at median beta: {ups_pow(np.median(res[:,0])):.2f}')
print(f'Upsilon under Paper 1 stretched-exp assumption : 0.29')
np.savez(os.path.join(HERE,'severson_shape2.npz'), res=res, mean_curve=mean_curve, grid=GRID)
