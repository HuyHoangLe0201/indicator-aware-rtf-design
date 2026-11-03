"""The out-of-sample boundary test, on the class the battery section actually fits.

boundary2.py runs this on the stretched exponential (alpha_b, b) = (3.0, 0.7), the class the
companion study assumes, whose own Upsilon is 0.29. Section VII fits the power law with
beta = 4.483 instead. The test is worth more on the fitted class, because that is where the
boundary term is large: Upsilon*Pi is about 21 there against about 11 on the stretched class,
and the whole point of the definition is that the boundary channel cannot be folded into the
inflation.

Same procedure as boundary2.py, nothing else changed. Simulate the oracle arm at each
truncation, read the robust label error, calibrate the single inflation constant Pi on
c <= 0.65 only, then predict c = 0.80 and c = 1.00 with the path channel alone and with the
boundary term added. The robust scale is forced here rather than chosen: on a class this
convex the second moment of the label error is set by the width of the estimator's constraint
set rather than by the data, which is the point of Section III-D.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from refresh_pi import AofC, dprime, oracle_label, DOM, M0, GRID

BETA, SIGMA = 4.4832, 0.05
DOM['BatteryPow'] = ('pow', SIGMA, 806.0, 0.46, 0.241, np.array([BETA]))
KIND, TH = 'pow', DOM['BatteryPow'][5]
K0, LAMD = 0.05, 0.241

Dp = dprime(KIND, TH)
Z = AofC(KIND, TH, 1.0)
etaT = lambda c: AofC(KIND, TH, c)/Z
UPS = Dp(1.0)**2/Z
JT1 = M0*Z/SIGMA**2
JBND = M0*Dp(1.0)**2/SIGMA**2


if __name__ == '__main__':
    print(f'power-law class beta={BETA}, Ups = {UPS:.2f}')
    print(f'J_T(1) = {JT1:.1f}   J_bnd = {JBND:.1f}   J_bnd/J_T(1) = {JBND/JT1:.2f} '
          f'(Lemma 2 says this is Ups)')
    print()
    c = np.array(GRID, float)
    orc = np.array([oracle_label('BatteryPow', x)[0] for x in c])

    eT = np.array([etaT(x) for x in c])
    JTc = np.array([M0*AofC(KIND, TH, x)/SIGMA**2 for x in c])
    fit = c <= 0.65
    Pi = float(np.mean(orc[fit]**2*JTc[fit]))
    print(f'nuisance inflation Pi calibrated on c<=0.65 : {Pi:.2f}, so Ups*Pi = {UPS*Pi:.1f}')
    print()

    no_b = np.sqrt(Pi/JTc)
    wi_b = np.sqrt(1.0/(JTc/Pi + np.where(c >= 1.0, JBND, 0.0)))
    print(f'{"c":>5s} {"measured":>9s} {"path only":>10s} {"+ boundary":>11s} '
          f'{"err path":>9s} {"err +bnd":>9s}')
    for i, x in enumerate(c):
        tag = '  (fitted)' if fit[i] else '  <- OUT OF SAMPLE'
        print(f'{x:5.2f} {orc[i]:9.3f} {no_b[i]:10.3f} {wi_b[i]:11.3f} '
              f'{no_b[i]/orc[i]:9.2f} {wi_b[i]/orc[i]:9.2f}{tag}')

    oos = ~fit
    mp = float(np.mean(np.abs(np.log(no_b[oos]/orc[oos]))))
    mb = float(np.mean(np.abs(np.log(wi_b[oos]/orc[oos]))))
    print()
    print(f'out-of-sample mean |log ratio|: path only {mp:.3f} ({np.exp(mp):.2f}x), '
          f'+ boundary {mb:.3f} ({np.exp(mb):.2f}x)')

    def cstar(V):
        g = lambda x: (K0 + x)*(V(x)/V(1.0)/LAMD + 1.0)
        return minimize_scalar(g, bounds=(0.13, 1.0), method='bounded',
                               options={'xatol': 1e-7}).x
    V_no = lambda x: Pi/(M0*AofC(KIND, TH, x)/SIGMA**2)
    V_wi = lambda x: 1.0/(M0*AofC(KIND, TH, x)/SIGMA**2/Pi
                          + (JBND if x >= 0.999 else 0.0))
    print()
    print(f'c* with the path channel alone : {cstar(V_no):.3f}')
    print(f'c* with the boundary term      : {cstar(V_wi):.3f}')
