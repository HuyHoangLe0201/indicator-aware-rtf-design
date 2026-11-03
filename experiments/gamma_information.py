"""Terminal information concentration of a gamma degradation process (Section S3.7, Table S21).

The shape function is k (t/T)^q with the scale known; K equally spaced readings give independent Gamma(d alpha_j) increments.
The Fisher information about ln T of the first c K increments is sum psi'(d alpha_j) (q d alpha_j)^2, and
Upsilon = eta'(1) with eta(c) = I(c) / I(1), taken from the last 2 per cent of the grid.
The result depends on the terminal increment d alpha_K = q k / K only: Upsilon -> 1 (the Wiener value) when it is small and
-> q when it is large.
Usage: python gamma_information.py
"""
import numpy as np
from scipy.special import polygamma


def upsilon(k, q, K=2000):
    j = np.arange(1, K + 1)
    da = k * ((j / K)**q - ((j - 1) / K)**q)
    cum = np.cumsum(polygamma(1, da) * (q * da)**2)
    cum /= cum[-1]
    h = int(0.02 * K)
    return (cum[-1] - cum[-1 - h]) / (h / K)


K = 2000
print('terminal increment d_alpha   ' + '   '.join('q=%.2f' % q for q in (2.0, 4.48)))
for d in (0.01, 0.1, 1, 10, 100):
    print('%26g   %s' % (d, '   '.join('%6.2f' % upsilon(d * K / q, q, K) for q in (2.0, 4.48))))
print('limits: Wiener 1; gamma, large increments q; white reading error 2q+1 (5.00 and 9.96)')
