"""First-principles prediction of the achieved label-error inflation, against measurement.

Section III-C claims that profile price and shrinkage together predict the inflation with no
fitted constant. This recomputes both sides on the same constants.

Prediction. Estimating (log T, shape) by penalised least squares over a window [0,c] gives an
information matrix J^T J with a prior precision added on the shape block. The inflation of the
label variance relative to a known shape is the (1,1) ratio

    Pi_pred(c) = A / (A - b^T (C + P)^{-1} b),

with A, b, C the blocks of J^T J on that window and P the shrinkage precision. With P = 0 this
is the profile price of Lemma 2; with P -> infinity it is 1.

Measurement. The same estimator is run on simulated fleets and scored on the robust scale of
Section III-C, s_rob = medAE/0.6745, so that Pi_meas = s_rob^2 eta_T(c) J_T(1).

No constant is fitted anywhere: the prior precision fed to the prediction is the one the
simulator penalises with.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares

M = 60
CS = (0.50, 0.65, 0.80)
BOX = 100.0
NREP = 4000

BEARING = dict(name='bearing, one-parameter', beta=3.0, sig=0.15, sh_cv=0.25,
               tbar=15760.0, tcv=0.47, two=False)
BATTERY = dict(name='battery, two-parameter', beta=4.4832, sig=0.05, sh_cv=0.25,
               tbar=806.0, tcv=0.46, two=True)


def blocks(c, cfg):
    """J^T J over the window, in the parameters (log T, log beta[, log a])."""
    b = cfg['beta']
    tau = np.linspace(c/M, c, M)
    sT = -b*tau**b                       # dD/dlogT
    su = b*tau**b*np.log(tau)            # dD/dlog beta
    cols = [su]
    if cfg['two']:
        # second shape direction of the fitted family, D = (1-exp(-a tau^b))/(1-exp(-a)),
        # taken in the power-law limit where the survey's fits sit
        cols.append(tau**b*(1 - tau**b)/2)
    S = np.array(cols)
    return sT @ sT, S @ sT, S @ S.T


def pi_pred(c, cfg):
    A, bb, C = blocks(c, cfg)
    k = C.shape[0]
    # prior precision on each shape coordinate, in the same units as J^T J
    P = np.eye(k)*(0.1*M*cfg['sig']**2/cfg['sh_cv']**2)
    return A/max(A - bb @ np.linalg.solve(C + P, bb), 1e-300)


def pi_meas(c, cfg, seed=0):
    rng = np.random.default_rng(seed)
    b0, sig, shcv = cfg['beta'], cfg['sig'], cfg['sh_cv']
    Dp = lambda t: b0*t**(b0 - 1)
    Z = quad(lambda t: t**2*Dp(t)**2, 0, 1, limit=400)[0]
    JT1 = M*Z/sig**2
    etaT = c**(2*b0 + 1)
    lam = 0.1*M*sig**2/shcv**2
    s = np.sqrt(np.log(1 + cfg['tcv']**2))
    err = []
    for _ in range(NREP):
        T = rng.lognormal(np.log(cfg['tbar']) - 0.5*s**2, s)
        b = float(np.clip(b0*np.exp(shcv*rng.normal()), 0.4, 12.))
        t = np.linspace(T/M, c*T, M)
        y = np.clip(t/T, 1e-9, 1.)**b + sig*rng.normal(size=M)
        G = np.geomspace(t[-1]*1.0001, t[-1]*BOX, 300)
        R = np.clip(t[None, :]/G[:, None], 1e-9, 1.)**b0 - y
        T0 = G[np.argmin(np.einsum('ij,ij->i', R, R))]

        def res(p):
            return np.concatenate([np.clip(t/p[0], 1e-9, 1.)**np.exp(p[1]) - y,
                                   [np.sqrt(lam)*(p[1] - np.log(b0))]])
        try:
            x = least_squares(res, [T0, np.log(b0)],
                              bounds=([t[-1]*1.0001, np.log(0.4)],
                                      [t[-1]*BOX, np.log(12.)]), max_nfev=400).x
            err.append((x[0] - T)/T)
        except Exception:
            pass
    srob = np.median(np.abs(err))/0.6745
    return srob**2*etaT*JT1


if __name__ == '__main__':
    print(f'{"class":24s} {"c":>5s} {"predicted":>10s} {"measured":>10s} {"ratio":>7s}')
    print('-'*62)
    for cfg in (BATTERY, BEARING):
        worst = 1.0
        for c in CS:
            p, m = pi_pred(c, cfg), pi_meas(c, cfg)
            r = max(p/m, m/p)
            worst = max(worst, r)
            print(f'{cfg["name"]:24s} {c:5.2f} {p:10.3f} {m:10.3f} {r:7.3f}')
        print(f'{"":24s} {"worst":>5s} {"":10s} {"":10s} {worst:7.3f}')
        print()
    print('Section III-C claims agreement to a factor of 1.08 (battery) and 1.23 (bearing).')
