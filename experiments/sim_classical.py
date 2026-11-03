"""Classical planning criteria and the learning criterion, scored on the same simulated campaigns (R3.4).

The planning literature chooses the termination time t_c that minimizes the variance of an estimated fleet quantity
(mean life, a lifetime quantile) at a fixed cost.  Here every plan is generated in the literature's own setting
(units with a deterministic normalized path D(t/T) read with white noise and failing when the path reaches 1, life
T lognormal across units, Type-I censoring at an absolute time t_c, cost kappa0 + E[min(T, t_c)]/Tbar per unit) and is
then scored on BOTH objectives from the SAME campaigns:

  fleet objective     RMSE of the estimated log median life and of the estimated log 10th percentile
                      (lognormal fitted to the per-unit life estimates, measurement-error variance removed with
                      the Fisher variance of each estimate, the classical analyst's best estimator)
  learning objective  error of a life predictor trained on the same data (labels exact for failed units,
                      imputed from the truncated path for the others) and applied to fresh units from a short
                      early window, as a multiple of predicting the fleet mean.

Shapes: power law D=tau^beta (Upsilon = 2 beta + 1) and a saturating exponential (Upsilon small).
Usage: python sim_classical.py [R]   (R replications per cell, default 150)
"""
import sys

import numpy as np

R = int(sys.argv[1]) if len(sys.argv) > 1 else 150
KAP0, C_BUDGET = 0.05, 21.0
SIG_E = 0.08                     # reading noise, in units of the threshold
S_T = 0.30                       # sd of log life across units
TM = 1000.0                      # median life (time units)
DT = 10.0                        # reading interval
WINS = {'power1': 0.3, 'power3': 0.5, 'exp6': 0.3}   # early window (fraction of the median life) available to the predictor
GS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.6, 3.0)      # t_c / median life; 3.0 = run to failure
NTEST = 1500
rng_master = np.random.default_rng(12345)
TIMES = np.arange(1, int(3.2 * TM / DT) + 1) * DT
TG = TM * np.exp(np.linspace(-1.3, 1.7, 151))             # candidate life grid


def make_shape(name):
    if name.startswith('power'):
        b = float(name[5:])
        D = lambda tau: np.clip(tau, 0, None) ** b
        dD = lambda tau: b * np.clip(tau, 1e-9, None) ** (b - 1)
        ups = 2 * b + 1
    else:                                                  # saturating exponential, rate k
        k = float(name[3:])
        norm = 1 - np.exp(-k)
        D = lambda tau: (1 - np.exp(-k * np.clip(tau, 0, None))) / norm
        dD = lambda tau: k * np.exp(-k * np.clip(tau, 0, None)) / norm
        tau = np.linspace(0, 1, 20001)
        f = tau ** 2 * dD(tau) ** 2
        ups = f[-1] / np.trapezoid(f, tau)
    return D, dD, ups


def sample_lives(n, rng):
    return TM * np.exp(S_T * rng.standard_normal(n))


def fit_life(x, mask, Dg, sens_g):
    """Least-squares life per unit on the grid; returns (T_hat, Fisher variance of ln T_hat)."""
    xm = x * mask
    sse = (xm * xm).sum(1, keepdims=True) - 2 * xm @ Dg + mask @ (Dg ** 2)
    g = np.argmin(sse, axis=1)
    info = mask @ (sens_g ** 2)                            # (n, G): sum over observed readings of (tau D')^2
    v = SIG_E ** 2 / np.maximum(info[np.arange(len(g)), g], 1e-12)
    return TG[g], v


def one_cell(shape, gfrac, reps, seed):
    D, dD, ups = make_shape(shape)
    Dg = D(TIMES[:, None] / TG[None, :])
    sens_g = (TIMES[:, None] / TG[None, :]) * dD(TIMES[:, None] / TG[None, :])
    rng = np.random.default_rng(seed)
    big = sample_lives(400000, rng)
    tbar = big.mean()
    mu_true, s_true = np.log(TM), S_T
    q10_true = mu_true + (-1.2815515655446004) * s_true
    tc = gfrac * TM
    cost = KAP0 + np.minimum(big, tc).mean() / tbar
    n = max(5, int(round(C_BUDGET / cost)))
    pool_mean = big.mean()
    out = dict(n=n, ups=ups)
    e_med, e_q10, e_pred, e_base = [], [], [], []
    wmask_t = (TIMES <= WINS[shape] * TM).astype(float)
    for _ in range(reps):
        T = sample_lives(n, rng)
        tend = np.minimum(T, tc)
        mask = (TIMES[None, :] <= tend[:, None]).astype(float)
        x = D(TIMES[None, :] / T[:, None]) + SIG_E * rng.standard_normal((n, len(TIMES)))
        That, v = fit_life(x, mask, Dg, sens_g)
        failed = T <= tc
        lab = np.where(failed, T, That)
        v = np.where(failed, 0.0, v)
        ly = np.log(lab)
        m_hat = ly.mean()
        s2 = max(ly.var(ddof=1) - v.mean(), 1e-4)
        e_med.append(m_hat - mu_true)
        e_q10.append(m_hat - 1.2815515655446004 * np.sqrt(s2) - q10_true)
        # window feature of training units
        Tw, _ = fit_life(x, mask * wmask_t[None, :], Dg, sens_g)
        A = np.column_stack([np.ones(n), np.log(Tw)])
        coef = np.linalg.lstsq(A, ly, rcond=None)[0]
        Tt = sample_lives(NTEST, rng)
        mt = np.tile(wmask_t, (NTEST, 1))
        xt = D(TIMES[None, :] / Tt[:, None]) + SIG_E * rng.standard_normal((NTEST, len(TIMES)))
        Twt, _ = fit_life(xt, mt, Dg, sens_g)
        pred = np.exp(coef[0] + coef[1] * np.log(Twt))
        e_pred.append(np.sqrt(np.mean((pred - Tt) ** 2)))
        e_base.append(np.sqrt(np.mean((lab.mean() - Tt) ** 2)))
    out['rmse_med'] = float(np.sqrt(np.mean(np.square(e_med))))
    out['rmse_q10'] = float(np.sqrt(np.mean(np.square(e_q10))))
    out['pred'] = float(np.mean(e_pred) / np.mean(e_base))
    out['pred_se'] = float(np.std(np.array(e_pred) / np.array(e_base)) / np.sqrt(reps))
    out['frac_fail'] = float((big <= tc).mean())
    return out


if __name__ == '__main__':
    shapes = ('power1', 'power3', 'exp6')
    for sh in shapes:
        print(f'\n=== shape {sh}  (Upsilon = {make_shape(sh)[2]:.2f}),  R = {R} campaigns per row ===')
        print(f'{"t_c/median":>10s} {"F":>5s} {"n":>4s} {"RMSE ln median":>15s} {"RMSE ln q10":>12s} {"learning error":>15s}')
        rows = []
        for gi, g in enumerate(GS):
            r = one_cell(sh, g, R, 1000 + gi)
            rows.append((g, r))
            print(f'{g:10.2f} {r["frac_fail"]:5.2f} {r["n"]:4d} {r["rmse_med"]:15.4f} {r["rmse_q10"]:12.4f} '
                  f'{r["pred"]:10.3f} +-{r["pred_se"]:.3f}', flush=True)
        for key, nm in (('rmse_med', 'median life'), ('rmse_q10', '10th percentile'), ('pred', 'learning error')):
            best = min(rows, key=lambda t: t[1][key])
            print(f'  optimum for {nm:16s}: t_c = {best[0]:.2f} median lives')
        # regret of each optimum on the other objective, relative to that objective's own optimum
        bq = min(rows, key=lambda t: t[1]['rmse_q10']); bl = min(rows, key=lambda t: t[1]['pred'])
        lq = min(t[1]['rmse_q10'] for t in rows); ll = min(t[1]['pred'] for t in rows)
        print(f'  plan optimal for the 10th percentile: learning error {bq[1]["pred"]/ll:.3f} x its best')
        print(f'  plan optimal for learning:            10th-percentile RMSE {bl[1]["rmse_q10"]/lq:.3f} x its best')
