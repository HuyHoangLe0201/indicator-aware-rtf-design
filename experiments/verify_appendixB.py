"""
Symbolic re-verification of Appendix B (reparameterization invariance), end to end.

This appendix is the largest single block in the paper and it has already carried two errors
this project: the factorisation matrix P was missing a factor a_0 T^beta, and the quantity
c^s B(c) was mislabelled as a c-optimal efficiency when it is eta_theta of the normalized
model.  Both were fixed, but the argument has not been checked as a whole since.  Every claim
in the appendix is restated here and verified with sympy rather than by hand.

Model: X = a_0 t^beta, threshold X_f, derived T = (X_f/a_0)^(1/beta), s = 2 beta + 1,
L = ln c, B(c) = s^2 L^2/2 - s L + 1.
"""
import sympy as sp

a0, beta, T, tau, c, Xf = sp.symbols('a0 beta T tau c X_f', positive=True)
s = 2*beta + 1
L = sp.log(c)
B = s**2*L**2/2 - s*L + 1

ok = []


def check(name, expr, expect, simplify=sp.simplify):
    d = simplify(sp.together(expr - expect))
    good = d == 0
    ok.append(good)
    print(f'  [{"OK " if good else "FAIL"}] {name}')
    if not good:
        print(f'         residual: {d}')


print(__doc__)
print('=' * 74)

# --- 1. the design vector factorises as f = P ftilde -------------------------------
X = a0*(tau*T)**beta
f = sp.Matrix([sp.diff(X, a0), sp.diff(X, beta)])
P = sp.Matrix([[T**beta, 0], [a0*T**beta*sp.log(T), a0*T**beta]])
ft = tau**beta*sp.Matrix([1, sp.log(tau)])
print('1. factorisation f(tau) = P ftilde(tau), P constant in tau')
for i in (0, 1):
    check(f'   component {i+1}', sp.expand(sp.simplify((P*ft)[i])), sp.expand(sp.simplify(f[i])))
check('   P is tau-free', sp.Integer(0), sum(sp.diff(P[i, j], tau) for i in (0, 1) for j in (0, 1)))
print(f'   det P = {sp.simplify(P.det())}  (invertible for a0, T > 0)')

# --- 2. det Jtilde(c) = c^{2s}/s^4 -------------------------------------------------
J = sp.zeros(2, 2)
for i in range(2):
    for j in range(2):
        J[i, j] = sp.integrate(sp.expand(ft[i]*ft[j]), (tau, 0, c))
J = sp.simplify(J)
print()
print('2. Fisher matrix of the normalized design vector')
check('   det Jtilde = c^(2s)/s^4', sp.simplify(J.det()), c**(2*s)/s**4)

# --- 3. D-optimal efficiency = c^s -------------------------------------------------
J1 = sp.simplify(J.subs(c, 1))
print()
print('3. D-optimal efficiency, invariant to P because det ratios cancel it')
check('   (det J(c)/det J(1))^(1/2) = c^s', sp.simplify(sp.sqrt(J.det()/J1.det())), c**s)

# --- 4. and 5. the functional T, and P^-1 w along e1 -------------------------------
Tval = (Xf/a0)**(1/beta)
w = sp.Matrix([sp.diff(Tval, a0), sp.diff(Tval, beta)])
w = sp.simplify(w.subs(Xf, a0*T**beta))          # express through T
print()
print('4. gradient of the derived functional T')
check('   w1 = -T/(beta a0)', sp.simplify(w[0]), -T/(beta*a0))
check('   w2 = -T ln T/beta', sp.simplify(w[1]), -T*sp.log(T)/beta)
print()
print('5. P^-1 w is proportional to (1,0): the functional is a pure first coordinate')
Pw = sp.simplify(P.inv()*w)
check('   second component vanishes', sp.simplify(Pw[1]), sp.Integer(0))
print(f'   first component = {sp.simplify(Pw[0])}')

# --- 6. and 7. c-optimal efficiency = c^s/B(c) -------------------------------------
Ji = sp.simplify(J.inv())
print()
print('6. Var(That) proportional to [Jtilde^-1]_11')
check('   [Jtilde^-1]_11 = 2 s B(c) c^-s', sp.simplify(Ji[0, 0]), 2*s*B*c**(-s))
print()
print('7. c-optimal-for-T efficiency')
check('   [Ji]_11(1)/[Ji]_11(c) = c^s/B(c)',
      sp.simplify(Ji[0, 0].subs(c, 1)/Ji[0, 0]), c**s/B)

# --- 8. estimating beta alone --------------------------------------------------------
print()
print('8. the third efficiency: estimating beta alone is NOT c^s B(c)')
check('   [Jtilde^-1]_22 = s^3 c^-s', sp.simplify(Ji[1, 1]), s**3*c**(-s))
check('   efficiency for beta alone = c^s', sp.simplify(Ji[1, 1].subs(c, 1)/Ji[1, 1]), c**s)
same = sp.simplify(c**s - c**s)
print(f'   so it equals the D-optimal efficiency, as the appendix states: {same == 0}')
print(f'   and it is NOT c^s B(c): difference = {sp.simplify(c**s - c**s*B)}')

print()
print('=' * 74)
print(f'all {len(ok)} checks passed: {all(ok)}')
