"""
How does Upsilon transform when the same damage state is read through a different transducer?

Write the second readout as Dtilde = g(D) with g increasing, g(0)=0, g(1)=1.  Then
Dtilde' = g'(D) D', and with the weight w(tau) proportional to tau^2 D'(tau)^2,

        Upsilon~ / Upsilon  =  g'(1)^2 / E_w[ g'(D)^2 ].                       (*)

Three consequences, each checked numerically below.

  1. g convex  =>  g'(D(tau)) <= g'(1) on [0,1]  =>  Upsilon~ >= Upsilon.
     g concave =>  Upsilon~ <= Upsilon.  Equality iff g is affine.
     So a SATURATING readout lowers Upsilon: a sceptic could in principle manufacture a
     low-Upsilon channel out of a high-Upsilon one by squashing its top end.

  2. That loophole is closed by the noise model, not by taste.  If y = D + eps with eps
     homoscedastic, then g(y) ~ g(D) + g'(D) eps has variance g'(D)^2 sigma^2.  Homoscedastic
     on both scales forces g' constant, i.e. g affine.  So Upsilon is invariant under exactly
     those readout changes that preserve the noise model that defines it, and the scale
     selection step is what makes Upsilon well posed rather than a matter of presentation.

  3. Hence the five-fold spread across bearing indicators is not a scale artifact -- each was
     scaled independently to homoscedasticity -- but neither can 'same damage state' force
     equal Upsilon.  The physical rule therefore cannot be derived from invariance; it needs
     the conditional-independence test of readout_test2.py.
"""
import numpy as np
from scipy.integrate import quad

TAU = np.linspace(1e-9, 1.0, 200001)


def ups_of(D, dD):
    """Upsilon = D'(1)^2 / int tau^2 D'^2, by quadrature on a fine grid."""
    num = dD(1.0) ** 2
    den = np.trapezoid(TAU ** 2 * dD(TAU) ** 2, TAU)
    return num / den


def check(name, beta, g, dg, ddg_sign):
    D = lambda t: t ** beta
    dD = lambda t: beta * np.clip(t, 1e-12, None) ** (beta - 1)
    Dt = lambda t: g(D(t))
    dDt = lambda t: dg(D(t)) * dD(t)

    U, Ut = ups_of(D, dD), ups_of(Dt, dDt)

    # right-hand side of (*)
    w = TAU ** 2 * dD(TAU) ** 2
    w = w / np.trapezoid(w, TAU)
    Eg2 = np.trapezoid(w * dg(D(TAU)) ** 2, TAU)
    rhs = dg(1.0) ** 2 / Eg2

    ok = abs(Ut / U - rhs) / rhs
    pred = {'convex': 'Ups~ >= Ups', 'concave': 'Ups~ <= Ups', 'affine': 'Ups~ == Ups'}[ddg_sign]
    holds = {'convex': Ut >= U * (1 - 1e-9),
             'concave': Ut <= U * (1 + 1e-9),
             'affine': abs(Ut - U) < 1e-6 * U}[ddg_sign]
    print(f'  {name:34s} Ups {U:8.3f} -> {Ut:8.3f}  ratio {Ut/U:7.4f}  '
          f'law err {ok:.1e}  {pred}: {holds}')
    return ok, holds


print(__doc__)
print('=' * 78)
print('CHECK 1: the transformation law (*) and the convexity consequences, on D = tau^beta')
print()
for beta in (2.0, 4.483, 8.0):
    print(f'beta = {beta}')
    errs, all_ok = [], True
    # affine
    e, h = check('affine  g(x)=x', beta, lambda x: x, lambda x: np.ones_like(np.asarray(x, float)), 'affine')
    errs.append(e); all_ok &= h
    # convex
    for p in (1.5, 3.0):
        e, h = check(f'convex  g(x)=x^{p}', beta, lambda x, p=p: x ** p,
                     lambda x, p=p: p * np.clip(x, 1e-12, None) ** (p - 1), 'convex')
        errs.append(e); all_ok &= h
    # concave (saturating)
    for p in (0.5, 0.25):
        e, h = check(f'concave g(x)=x^{p}', beta, lambda x, p=p: x ** p,
                     lambda x, p=p: p * np.clip(x, 1e-12, None) ** (p - 1), 'concave')
        errs.append(e); all_ok &= h
    # a saturating map that is not a power: g(x) = log(1+ax)/log(1+a)
    a = 20.0
    e, h = check('concave g(x)=log(1+20x)/log(21)', beta,
                 lambda x: np.log1p(a * x) / np.log1p(a),
                 lambda x: a / ((1 + a * x) * np.log1p(a)), 'concave')
    errs.append(e); all_ok &= h
    print(f'  -> max relative error in law (*): {max(errs):.2e};  '
          f'all convexity predictions hold: {all_ok}')
    print()

print('=' * 78)
print('CHECK 2: how far can a saturating readout push Upsilon?  D = tau^4.483 (bulk capacity,')
print('Ups = 9.97), squashed by g(x) = x^p.  The battery threshold is 1.18.')
print()
D = lambda t: t ** 4.483
dD = lambda t: 4.483 * np.clip(t, 1e-12, None) ** 3.483
print(f'{"p":>6s} {"Ups~":>9s} {"below 1.18?":>12s}')
for p in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02):
    Dt = lambda t, p=p: D(t) ** p
    dDt = lambda t, p=p: p * np.clip(D(t), 1e-12, None) ** (p - 1) * dD(t)
    U = ups_of(Dt, dDt)
    print(f'{p:6.2f} {U:9.3f} {str(U < 1.18):>12s}')
print()
print('So the loophole is real arithmetically: enough saturation drags any channel under any')
print('threshold.  What forbids it is that g(y) has variance g\'(D)^2 sigma^2, so a squashed')
print('readout is heteroscedastic and the scale-selection step rejects it.  Upsilon is')
print('well defined only together with the scale on which the noise is homoscedastic.')
