"""Two-stage optimum with n1 continuous, as Theorem 5 states it.

twostage.py searches n1 on the integers while n2 stays continuous. That asymmetry excludes
the all-to-failure plan (n1 = C/(kappa0+1), n2 = 0) from the feasible set whenever the budget
does not divide evenly, and any leftover is then forced into stage two where the inflation
Xi(n1) charges for it. On a high-Upsilon channel the reported optimum therefore lands just
below a plan that lies inside its own family, which is an artefact of the grid and not a
property of the design.

Theorem 5 is stated for continuous n1 and n2, so this re-solves it that way. The high-Upsilon
row then collapses onto all-to-failure exactly, which is what the text says happens.
"""
import numpy as np

from twostage import KAP0, make_eta

CS = (50.0, 100.0, 200.0, 400.0)
U_LAM, U0 = 2.0, 30.0


def solve(eta, C, u_lam=U_LAM, u0=U0, n_n1=6000, n_c=393):
    """Maximise n1 + n2 eta(c)/Xi(n1) at fixed budget, both stage sizes continuous."""
    best = None
    hi = C/(KAP0 + 1.0)
    for n1 in np.linspace(hi*1e-6, hi, n_n1):
        rem = C - n1*(KAP0 + 1.0)
        if rem < -1e-12:
            continue
        Xi = 1.0 + u_lam + u0/n1
        for c in np.linspace(0.02, 1.0, n_c):
            n2 = max(rem, 0.0)/(KAP0 + c)
            v = n1 + n2*eta(c)/Xi
            if best is None or v > best[0]:
                best = (v, n1, n2, c)
    return best


if __name__ == '__main__':
    etaA, UA = make_eta('pow', 4.483)
    from twostage import eta_dq_info
    etaB = eta_dq_info()

    for nm, eta in (('A, bulk capacity, Ups=9.97', etaA),
                    ('B, DeltaQ(V), Ups~0', etaB)):
        print(f'--- channel {nm} ---')
        print(f'{"C":>6s} {"n1":>8s} {"n2":>9s} {"c*":>7s} {"n_eff":>9s} '
              f'{"all-to-fail":>12s} {"gain":>7s}')
        for C in CS:
            v, n1, n2, c = solve(eta, C)
            one = C/(KAP0 + 1.0)
            print(f'{C:6.0f} {n1:8.1f} {n2:9.1f} {c:7.3f} {v:9.1f} {one:12.1f} '
                  f'{v/one:7.3f}')
        print()
    print('the paper Table I reports, for A at C=100: n1=95, n2=0.2, c*=1.000, 95.1 vs 95.2')
    print('and for B: 63.9/47.6, 146.6/95.2, 328.3/190.5, 716.9/381.0')
