"""Pi(1) for the shapes the survey actually fits, and the corrected corner condition.

Definition 1 puts the path information at J_T(1) eta_T(c)/Pi(c). Theorem 2 omits Pi, which is
harmless only if Pi does not depend on c. It does. Carrying it through, the quantity whose
terminal slope decides the corner is eta_T/Pi normalised at c=1, and its terminal concentration
is Upsilon*Pi(1), because

    Pi'(1) = -Upsilon * Pi(1) * (Pi(1) - 1)

holds on every class checked. So the corner condition sharpens from Upsilon >= (1+Lambda)/(1+k0)
to Upsilon*Pi(1) >= (1+Lambda)/(1+k0), and since Pi(1) >= 1 the published form is conservative.

Pi(1) depends on the shape family and its parameters, so it has to be computed for the shapes
the survey fits, which are stretched exponentials, and not for the power law alone. Pi(1) is the
ratio of the known-shape information about T at c=1 to the profile information, both from the
Fisher matrix in (T, shape) over the full window.
"""
import numpy as np
from scipy.integrate import quad

H = 1e-5


def dprime(a, b):
    return lambda t: a*b*np.clip(t, 1e-12, None)**(b - 1)*np.exp(-a*t**b)/(1 - np.exp(-a))


def shape_scores(a, b):
    """dD/dlogT up to sign, and dD/da, dD/db, as functions of tau."""
    dp = dprime(a, b)
    dT = lambda t: t*dp(t)

    def D(t, aa, bb):
        return (1 - np.exp(-aa*np.clip(t, 1e-12, None)**bb))/(1 - np.exp(-aa))

    da = lambda t: (D(t, a + H, b) - D(t, a - H, b))/(2*H)
    db = lambda t: (D(t, a, b + H) - D(t, a, b - H))/(2*H)
    return dT, [da, db]


def pi_one(a, b):
    dT, dth = shape_scores(a, b)
    JTT = quad(lambda t: dT(t)**2, 0, 1, limit=300)[0]
    JTs = np.array([quad(lambda t: dT(t)*g(t), 0, 1, limit=300)[0] for g in dth])
    Jss = np.array([[quad(lambda t: g(t)*h(t), 0, 1, limit=300)[0] for h in dth]
                    for g in dth])
    prof = JTT - JTs @ np.linalg.solve(Jss + 1e-16*np.eye(2)*np.trace(Jss), JTs)
    return JTT/max(prof, 1e-300)


def ups_of(a, b):
    dp = dprime(a, b)
    Z = quad(lambda t: t**2*dp(t)**2, 0, 1, limit=300)[0]
    return dp(1.0)**2/Z if Z > 0 else np.nan


if __name__ == '__main__':
    print('Pi(1) over the stretched-exponential family the survey fits')
    print(f'{"a":>7s} {"b":>6s} {"Ups":>9s} {"Pi(1)":>9s} {"Ups*Pi(1)":>11s}')
    for a in (0.5, 1.5, 3.0, 6.0, 12.0):
        for b in (0.5, 0.7, 1.0, 1.5, 2.5):
            u, p = ups_of(a, b), pi_one(a, b)
            print(f'{a:7.2f} {b:6.2f} {u:9.4f} {p:9.3f} {u*p:11.3f}')
    print()
    print('power-law limit a -> 0 at fixed b, where D -> tau^b and Pi(1) should approach 2')
    for a in (0.3, 0.1, 0.03, 0.01):
        print(f'   a={a:5.3f}, b=4.48  ->  Pi(1) = {pi_one(a, 4.4832):7.4f}')
