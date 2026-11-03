"""
Independent numerical verification of the paper's mathematics.

Every claim below is recomputed from its own definition and compared against the closed form
printed in the paper, rather than re-using any intermediate the paper supplies.  Run it after
any change to the theory sections.

Summary of what it establishes:
  Theorem 1  closed forms for eta_T(c) on the power-law, exponential and stretched-exponential
             classes agree with quadrature to ~1e-11 or better.
  Lemma 2    if D' ~ tau^rho near 0 then eta ~ c^{2rho+1} and eta_T ~ c^{2rho+3}: the
             extrapolation exponent exceeds the state exponent by exactly two.
  Upsilon    eta_T'(1) = D'(1)^2 / int_0^1 tau^2 D'^2 equals 2beta+1 on the power law.
  Theorem 3  the endpoint condition Upsilon >= (1+Lambda)/(1+kappa0) is exactly the condition
             for c=1 to be a LOCAL optimum.  It is sufficient for a GLOBAL optimum only when
             the elasticity e(c) = c eta_T'/eta_T is non-increasing.  That hypothesis holds
             for every class fitted in the paper; a two-phase class violates it, and there a
             large Upsilon coexists with an interior optimum.
"""
import numpy as np
from scipy.integrate import quad
from scipy.special import gammainc

TOL = 1e-8


def etaT_num(dp, c):
    """eta_T(c) by quadrature, from the definition."""
    num = quad(lambda t: t**2*dp(t)**2, 0, c, limit=500)[0]
    den = quad(lambda t: t**2*dp(t)**2, 0, 1, limit=500)[0]
    return num/den


def check_closed_forms():
    print('Theorem 1: closed forms vs quadrature')
    worst = 0.0
    for beta in (0.7, 2.0, 4.483, 9.0):                     # power law
        dp = lambda t, b=beta: b*t**(b-1)
        for c in (0.2, 0.5, 0.8, 0.95):
            worst = max(worst, abs(etaT_num(dp, c) - c**(2*beta+1)))
    Psi = lambda x: 2 - np.exp(-x)*(x**2 + 2*x + 2)
    for al in (0.3, 1.0, 3.0, 8.0):                          # exponential
        dp = lambda t, a=al: a*np.exp(-a*t)
        for c in (0.2, 0.5, 0.8, 0.95):
            worst = max(worst, abs(etaT_num(dp, c) - Psi(2*al*c)/Psi(2*al)))
    for b, ab in ((0.5, 1.0), (1.0, 2.0), (1.5, 0.8), (3.0, 4.0)):   # stretched
        dp = lambda t, b=b, a=ab: b*a*t**(b-1)*np.exp(-a*t**b)
        for c in (0.2, 0.5, 0.8, 0.95):
            want = gammainc(2 + 1/b, 2*ab*c**b)/gammainc(2 + 1/b, 2*ab)
            worst = max(worst, abs(etaT_num(dp, c) - want))
    print(f'  worst absolute error over all three families: {worst:.2e}  '
          + ('OK' if worst < 1e-7 else 'FAIL'))
    return worst < 1e-7


def check_lemma2():
    print('Lemma 2: exponents 2rho+1 and 2rho+3')
    ok = True
    for rho in (-0.4, 0.0, 0.5, 2.0, 3.48):
        dp = lambda t, r=rho: t**r
        e = lambda c: quad(lambda t: dp(t)**2, 0, c, limit=500)[0]
        eT = lambda c: quad(lambda t: t**2*dp(t)**2, 0, c, limit=500)[0]
        c1, c2 = 1e-3, 2e-3
        pe = np.log(e(c2)/e(c1))/np.log(c2/c1)
        pT = np.log(eT(c2)/eT(c1))/np.log(c2/c1)
        ok &= abs(pe - (2*rho+1)) < 1e-6 and abs(pT - (2*rho+3)) < 1e-6
    print(f'  measured exponents match for every rho tested: {ok}')
    return ok


def check_upsilon():
    print('Upsilon on the power law equals 2beta+1')
    ok = True
    for beta in (0.7, 2.0, 4.483, 9.0):
        dp = lambda t, b=beta: b*t**(b-1)
        Z = quad(lambda t: t**2*dp(t)**2, 0, 1, limit=500)[0]
        ok &= abs(dp(1.0)**2/Z - (2*beta+1)) < 1e-9
    print(f'  identity holds: {ok}')
    return ok


def elasticity_monotone(dp, npts=500):
    cs = np.linspace(0.02, 1.0, npts)
    Z = quad(lambda t: t**2*dp(t)**2, 0, 1, limit=500)[0]
    cum = np.array([quad(lambda t: t**2*dp(t)**2, 0, c, limit=500)[0] for c in cs])/Z
    e = cs*(cs**2*dp(cs)**2/Z)/cum
    return bool(np.all(np.diff(e) <= 1e-9))


def check_thm3():
    print('Theorem 3: endpoint test is global only under non-increasing elasticity')
    fitted = []
    for beta in (0.7, 2.0, 4.483, 9.0):
        fitted.append(elasticity_monotone(lambda t, b=beta: b*t**(b-1)))
    for al in (0.3, 1.0, 3.0, 8.0):
        fitted.append(elasticity_monotone(lambda t, a=al: a*np.exp(-a*t)))
    for b, ab in ((0.5, 1.0), (1.0, 2.0), (1.5, 0.8), (3.0, 4.0), (0.46, 0.05)):
        fitted.append(elasticity_monotone(
            lambda t, b=b, a=ab: b*a*t**(b-1)*np.exp(-a*t**b)))
    print(f'  elasticity non-increasing on every fitted class: {all(fitted)}')

    # two-phase counterexample: D'^2 has an early bump and a terminal bump, D(0)=0 holds
    A, m, sd, B, sd2 = 60.0, 0.15, 0.04, 1.0, 0.03
    dp = lambda t: np.sqrt(A*np.exp(-0.5*((t-m)/sd)**2) + B*np.exp(-0.5*((t-1.)/sd2)**2))
    mono = elasticity_monotone(dp)
    k0, lam = 0.05, 0.241
    Z = quad(lambda t: t**2*dp(t)**2, 0, 1, limit=500)[0]
    ups = dp(1.0)**2/Z
    cs = np.linspace(1e-3, 1.0, 4001)
    grid = np.linspace(0, 1, 3001)
    cum = np.array([quad(lambda t: t**2*dp(t)**2, 0, g, limit=500)[0] for g in grid])/Z
    eT = np.interp(cs, grid, cum)
    G = (k0 + cs)*(1.0/np.maximum(eT, 1e-300) + lam)
    i = int(G.argmin())
    thr = (1 + lam)/(1 + k0)
    print(f'  counterexample: elasticity monotone = {mono}; Upsilon = {ups:.2f} '
          f'(threshold {thr:.2f})')
    print(f'    argmin at c = {cs[i]:.3f}, G = {G[i]:.4f} against G(1) = {G[-1]:.4f} '
          f'({G[-1]/G[i]:.2f}x better than the corner)')
    return all(fitted) and (not mono) and i < len(cs) - 1


def check_lemma_elastic():
    """Lemma (elasticity): for w = tau^p exp(-lam tau^q), e(c) = q x g'(nu,x)/g(nu,x)."""
    from scipy.special import gammainc, gamma as Gam
    print('Lemma (elasticity): closed form and monotonicity')
    ok = True
    cases = [('power beta=4.483', 2*4.483, 1.0, 0.0),
             ('exponential a=3', 2.0, 1.0, 6.0),
             ('stretched b=0.46', 2*0.46, 0.46, 0.10),
             ('stretched b=3, a=4', 6.0, 3.0, 8.0)]
    for name, pp, q, lam in cases:
        w = lambda t: t**pp*np.exp(-lam*t**q)
        cs = np.linspace(0.02, 1.0, 400)
        e = np.array([c*w(c)/quad(w, 0, c, limit=400)[0] for c in cs])
        if lam == 0:
            closed = np.full_like(cs, pp + 1)
        else:
            nu = (pp + 1)/q
            x = lam*cs**q
            closed = q*x**nu*np.exp(-x)/(gammainc(nu, x)*Gam(nu))
        # relative test: at the smallest c the incomplete gamma is ~1e-9 and the
        # quadrature carries more absolute noise than the closed form does
        agree = np.max(np.abs(e - closed)/closed) < 1e-6
        mono = bool(np.all(np.diff(e) <= 1e-9))
        ok &= agree and mono
        print(f'  {name:20s} closed form {"OK" if agree else "FAIL"}, '
              f'non-increasing {"OK" if mono else "FAIL"}')
    return ok


def check_thm4_limit():
    """Theorem 4: M(t_s) -> Upsilon/t_s only as the fleet concentrates."""
    print('Theorem 4: the marginal term M is an integral, not Upsilon/t_s')
    s_exp, ts = 9.966, 1.0
    rows = []
    for sig in (0.46, 0.15, 0.05, 0.005):
        mu = -0.5*sig**2
        f = lambda T_: np.exp(-(np.log(T_)-mu)**2/(2*sig**2))/(T_*sig*np.sqrt(2*np.pi))
        Fbar = quad(f, ts, np.inf, limit=400)[0]
        M = quad(lambda T_: s_exp*(ts/T_)**(s_exp-1)/T_*f(T_), ts, np.inf, limit=400)[0]/Fbar
        rows.append((sig, M, s_exp/ts))
        print(f'  fleet CV={sig:5.3f}  M={M:8.4f}  Upsilon/t_s={s_exp/ts:8.4f}  '
              f'ratio={s_exp/ts/M:6.3f}')
    return rows[-1][2]/rows[-1][1] < 1.05 and rows[0][2]/rows[0][1] > 2


def check_thm5_pi():
    """Theorem 5: Pi(n1) = 1 + u_Lam + u0/n1 exactly, from the arrow structure."""
    print('Theorem 5: Pi(n1) exact, u0 = g^T (C0 - a^{-1} b b^T)^{-1} g')
    a, uLam = 1.7, 0.35
    b = np.array([0.9, -0.4]); C0 = np.array([[2.3, 0.5], [0.5, 1.6]])
    g = np.array([0.8, 1.3])
    u0 = float(g @ np.linalg.inv(C0 - np.outer(b, b)/a) @ g)
    ok = True
    for n1 in (3, 20, 500):
        S = n1*C0 - np.tile(b, (n1, 1)).T @ np.linalg.solve(a*np.eye(n1), np.tile(b, (n1, 1)))
        Pi = 1 + uLam + float(g @ np.linalg.inv(S) @ g)
        ok &= abs(Pi - (1 + uLam + u0/n1)) < 1e-11
    print(f'  u0={u0:.6f}; Pi matches 1+u_Lam+u0/n1 to 1e-11 at every n1: {ok}')
    return ok


def check_appendix_b():
    """Appendix B: D-optimal = c^s, c-optimal-for-T = c^s/B(c), [J^-1]_22 ratio = c^s."""
    print('Appendix B: efficiencies of the unnormalized two-parameter model')
    ok = True
    for beta in (1.0, 2.0, 4.483):
        s = 2*beta + 1
        def J(u):
            m11 = quad(lambda x: x**(2*beta), 0, u, limit=400)[0]
            m12 = quad(lambda x: x**(2*beta)*np.log(x), 1e-14, u, limit=400)[0]
            m22 = quad(lambda x: x**(2*beta)*np.log(x)**2, 1e-14, u, limit=400)[0]
            return np.array([[m11, m12], [m12, m22]])
        for c in (0.3, 0.5, 0.8):
            Jc, J1 = J(c), J(1.0)
            L = np.log(c); B = s**2*L**2/2 - s*L + 1
            d = np.sqrt(np.linalg.det(Jc)/np.linalg.det(J1))
            cT = np.linalg.inv(J1)[0, 0]/np.linalg.inv(Jc)[0, 0]
            cb = np.linalg.inv(J1)[1, 1]/np.linalg.inv(Jc)[1, 1]
            ok &= (abs(d - c**s) < 1e-6 and abs(cT - c**s/B) < 1e-6
                   and abs(cb - c**s) < 1e-6)
    print(f'  D = c^s, c-opt-for-T = c^s/B(c), and [J^-1]_22 ratio = c^s (NOT c^s B): {ok}')
    return ok


if __name__ == '__main__':
    results = [check_closed_forms(), check_lemma2(), check_upsilon(), check_thm3(),
               check_lemma_elastic(), check_thm4_limit(), check_thm5_pi(),
               check_appendix_b()]
    print()
    print('all checks behaved as documented:', all(results))
