"""Measure Pi on the class the battery section actually fits.

The Pi of 38.5 quoted beside Upsilon = 9.97 was calibrated on the stretched exponential
(alpha_b, b) = (3.0, 0.7), which is the class the companion study assumes and whose own Upsilon
is 0.29. Section VII fits the power law with beta = 4.483 instead, so quoting one class's Pi
against the other's Upsilon multiplies constants that belong to different models, and the
number of shape nuisance parameters differs between them (two against one), which is exactly
what sets Pi.

This recomputes Pi with the same estimator, the same sampling and the same fleet, on the power
law. The stretched-exponential row is kept so the two can be read against each other.
"""
import numpy as np
from refresh_pi import AofC, dprime, oracle_label, DOM, BOX, M0, GRID

DOM['BatteryPow'] = ('pow', 0.05, 806.0, 0.46, 0.241, np.array([4.4832]))

if __name__ == '__main__':
    print(f'{"class":22s} {"Ups":>6s} {"Pi":>7s} {"varpi=Pi*Ups":>13s} '
          f'{"(1+varpi)/(1+Ups)":>18s}')
    for nm, label in (('Battery', 'stretched exp (3, 0.7)'),
                      ('BatteryPow', 'power law (beta=4.483)')):
        kind, sig, Tb, cv, Ld, th0 = DOM[nm]
        A1 = AofC(kind, th0, 1.0)
        ups = dprime(kind, th0)(1.0)**2/A1
        vals = []
        for c in GRID:
            if c > 0.65:
                continue
            srob, _ = oracle_label(nm, c)
            vals.append(srob**2*M0*AofC(kind, th0, c)/sig**2)
        pi = float(np.mean(vals))
        varpi = pi*ups
        print(f'{label:22s} {ups:6.2f} {pi:7.2f} {varpi:13.1f} '
              f'{(1 + varpi)/(1 + ups):18.1f}')
