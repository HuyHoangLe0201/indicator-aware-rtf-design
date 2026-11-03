r"""
Boundary term, correct structure.

The path channel must be extrapolated to failure, so it is degraded by the
nuisance-parameter (shape) confounding factor Pi of Thm 8.  The first-passage
channel is NOT: witnessing the crossing pins T without reference to the shape,
because D(1)=1 holds for every theta.  The boundary channel is SHAPE-FREE.

    Var[(That-T)/T] = [ J_T(1) eta_T(c) / Pi(c)   +   1[c>=1] J_bnd ]^{-1}
                        \_____ path, shape-confounded _____/  \_ shape-free _/

    J_T(1) = (M/sigma^2) int_0^1 tau^2 D'^2,     J_bnd = M D'(1)^2 / sigma^2 .
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

AB, BB, SIGMA, M = 3.0, 0.7, 0.05, 60
Dp = lambda t: AB*BB*t**(BB-1)*np.exp(-AB*t**BB)/(1-np.exp(-AB))
Z    = quad(lambda t: t**2*Dp(t)**2, 0, 1, limit=400)[0]
etaT = lambda c: quad(lambda t: t**2*Dp(t)**2, 0, c, limit=400)[0]/Z
JT1, JBND = M*Z/SIGMA**2, M*Dp(1.0)**2/SIGMA**2

c   = np.array([0.15,0.25,0.35,0.50,0.65,0.80,1.00])
orc = np.array([0.437,0.282,0.239,0.200,0.175,0.135,0.071])
eT  = np.array([etaT(x) for x in c])

# calibrate the single nuisance-inflation constant Pi on c <= 0.65 ONLY,
# then PREDICT c = 0.80 and c = 1.00 out of sample.
fit = c <= 0.65
Pi  = float(np.mean(orc[fit]**2*eT[fit])*JT1)
print(f"J_T(1) = {JT1:.1f}   J_bnd = {JBND:.1f}   J_bnd/J_T(1) = {JBND/JT1:.2f}")
print(f"nuisance inflation Pi calibrated on c<=0.65 : {Pi:.1f}\n")

no_b = np.sqrt(Pi/(JT1*eT))
wi_b = np.sqrt(1.0/(JT1*eT/Pi + np.where(c >= 1.0, JBND, 0.0)))
print(f"{'c':>5s} {'measured':>9s} {'path only':>10s} {'+ boundary':>11s} {'err path':>9s} {'err +bnd':>9s}")
for i, x in enumerate(c):
    tag = "  (fitted)" if fit[i] else "  <- OUT OF SAMPLE"
    print(f"{x:5.2f} {orc[i]:9.3f} {no_b[i]:10.3f} {wi_b[i]:11.3f} "
          f"{no_b[i]/orc[i]:9.2f} {wi_b[i]/orc[i]:9.2f}{tag}")

oos = ~fit
print(f"\nout-of-sample mean |log ratio|:  path only {np.mean(np.abs(np.log(no_b[oos]/orc[oos]))):.3f}"
      f"   + boundary {np.mean(np.abs(np.log(wi_b[oos]/orc[oos]))):.3f}")

K0, LAMD = 0.05, 0.241
def cs(V):
    g = lambda x: (K0+x)*(V(x)/V(1.0)/LAMD + 1.0)
    return minimize_scalar(g, bounds=(0.13,1.0), method='bounded', options={'xatol':1e-7}).x
V_no = lambda x: Pi/(JT1*etaT(x))
V_wi = lambda x: 1.0/(JT1*etaT(x)/Pi + (JBND if x >= 0.999 else 0.0))
print(f"\nc*  path only  : {cs(V_no):.3f}")
print(f"c*  + boundary : {cs(V_wi):.3f}     (oracle-arm empirical 1.000)")

# how big must J_bnd be to flip the design to the corner?
for mult in [1, 2, 5, 10]:
    V = lambda x, m=mult: 1.0/(JT1*etaT(x)/Pi + (m*JBND if x >= 0.999 else 0.0))
    print(f"   J_bnd x{mult:<3d} -> c* = {cs(V):.3f}")
