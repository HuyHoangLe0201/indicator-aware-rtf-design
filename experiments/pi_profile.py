"""What Pi actually is, in closed form, and where a scalar stops describing it.

Definition 1 writes Pi as a single number. It is not one. Pi is the ratio of the information
about T when the shape is known to the information about T when the shape must be co-estimated
from the same window, and that ratio is a function of the truncation.

For the normalised power law D(tau) = tau^beta with parameters (T, beta), the per-measurement
scores are

    dD/dT    = -(beta/T) tau^beta,      dD/dbeta = tau^beta log tau,

so with s = 2beta+1 and L = log c the profile correction is

    rho^2(c) = <tau^p log tau>^2 / (<tau^p><tau^p log^2 tau>),     p = 2beta,
    Pi(c)    = 1/(1 - rho^2) = s^2 L^2 - 2 s L + 2,

which is twice the bracket B(c) already printed in Theorem 4. It equals 2 at c = 1, the usual
price of one extra parameter, and diverges as c falls. Both facts are checked here against
direct numerical Fisher matrices, and against the penalised estimator the survey actually uses,
which shrinks the shape toward a fleet centre and therefore pays less than the profile price.
"""
import numpy as np
from scipy.integrate import quad


def moments(beta, c):
    p = 2*beta
    A = quad(lambda t: t**p, 0, c, limit=400)[0]
    B = quad(lambda t: t**p*np.log(t), 1e-12, c, limit=400)[0]
    C = quad(lambda t: t**p*np.log(t)**2, 1e-12, c, limit=400)[0]
    return A, B, C


def pi_numeric(beta, c):
    """1/(1-rho^2) from the two-parameter Fisher matrix, T profiled against beta."""
    A, B, C = moments(beta, c)
    return 1.0/(1.0 - B*B/(A*C))


def pi_closed(beta, c):
    s = 2*beta + 1
    L = np.log(c)
    return s*s*L*L - 2*s*L + 2


if __name__ == '__main__':
    print('closed form against the two-parameter Fisher matrix')
    print(f'{"beta":>7s} {"c":>6s} {"numeric":>12s} {"closed":>12s} {"rel diff":>10s}')
    worst = 0.0
    for beta in (0.74, 1.1, 3.01, 4.4832, 8.0):
        for c in (0.15, 0.35, 0.5, 0.8, 0.999):
            a, b = pi_numeric(beta, c), pi_closed(beta, c)
            d = abs(a - b)/b
            worst = max(worst, d)
            print(f'{beta:7.4f} {c:6.3f} {a:12.4f} {b:12.4f} {d:10.2e}')
    print(f'{chr(10)}largest relative difference: {worst:.2e}')

    print()
    print('Pi(c) on the classes the paper surveys, unpenalised')
    print(f'{"class":26s} ' + ' '.join(f'{("c="+str(c)):>9s}'
                                       for c in (0.15, 0.35, 0.5, 0.65, 0.8, 1.0)))
    for beta, lab in ((4.4832, 'battery power law'), (3.01, 'bearing power law'),
                      (1.1, 'turbofan power law')):
        row = [pi_closed(beta, c) for c in (0.15, 0.35, 0.5, 0.65, 0.8, 1.0)]
        print(f'{lab:26s} ' + ' '.join(f'{v:9.1f}' for v in row))
    print()
    print('Pi(1) = 2 exactly on every class: one extra parameter, and at c=1 the terminal')
    print('value D=1 is shape-free, so nothing worse than the usual doubling survives.')
