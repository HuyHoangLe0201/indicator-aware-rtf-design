"""Wiener-process degradation with random drift: the model of the accelerated-degradation planning literature
(Tseng et al.; Hu and Huang; Cheng and Peng), planned with the literature's criterion and scored on both objectives.

Unit i: X_i(t) = a_i t + s_B B_i(t) (the state is read exactly), a_i = lognormal across units, failure at the first
passage of X_i to the threshold 1.  A plan fixes an absolute termination time t_c (Type-I censoring, cost
kappa0 + E[min(T, t_c)]/Tbar per unit, budget C complete-test equivalents).

Fleet objective   (what the classical criteria minimize): RMSE of the log median life and of the log 10th percentile
                  life estimated from the fitted random-drift Wiener model.  The model is fitted by the per-unit
                  drift estimate X_end/t_end (the Wiener MLE), the lognormal drift spread corrected for the sampling
                  variance s_B^2/(a^2 t_end) of each estimate, and the pooled diffusion s_B.
Learning objective (what a predictor needs): a life predictor trained on the same units, labels exact for failed
                  units and, for suspended ones, the conditional mean t_c + (1 - X(t_c))/a_hat of the remaining
                  life (a label whose variance is a property of the process, not of the estimator), applied to fresh
                  units from a short early window; error as a multiple of predicting the fleet mean.
Usage: python sim_wiener_plan.py [R] [CVDIFF]
"""
import sys

import numpy as np

R = int(sys.argv[1]) if len(sys.argv) > 1 else 300
CVD = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15          # diffusion CV of the life at the median drift
KAP0, C_BUDGET = 0.05, 21.0
S_A = 0.30                       # sd of log drift across units
TM = 1000.0
A0 = 1.0 / TM
SB2 = CVD ** 2 * A0              # s_B^2 = CV^2 * omega * a0  (omega = 1)
DT = 10.0
WIN = 0.4 * TM
HOR = 6.0 * TM
GS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.6, 3.0)
NTEST = 1500
TIMES = np.arange(1, int(HOR / DT) + 1) * DT
WMASK = TIMES <= WIN


def paths(a, rng):
    n = len(a)
    inc = a[:, None] * DT + np.sqrt(SB2 * DT) * rng.standard_normal((n, len(TIMES)))
    x = np.cumsum(inc, 1)
    hit = x >= 1.0
    k = np.where(hit.any(1), hit.argmax(1), len(TIMES) - 1)
    # linear interpolation of the crossing
    xk = x[np.arange(n), k]
    xkm = np.where(k > 0, x[np.arange(n), np.maximum(k - 1, 0)], 0.0)
    frac = np.clip((1.0 - xkm) / np.maximum(xk - xkm, 1e-12), 0, 1)
    T = (k + frac) * DT
    return x, T


def drift_hat(x, tend, idx_end):
    """Wiener MLE X(t_end)/t_end, using the reading at the last grid time <= tend."""
    n = len(tend)
    i = np.clip(idx_end, 0, len(TIMES) - 1)
    xe = x[np.arange(n), i]
    te = TIMES[i]
    return xe / te, te, xe


def true_quantiles(rng):
    a = A0 * np.exp(S_A * rng.standard_normal(400000))
    mu, lam = 1.0 / a, 1.0 / SB2
    T = rng.wald(mu, lam)
    return np.quantile(T, 0.5), np.quantile(T, 0.10), T.mean()


def fleet_from_drifts(ahat, te, sb2, rng):
    la = np.log(np.maximum(ahat, 1e-9))
    v = sb2 / (np.maximum(ahat, 1e-9) ** 2 * te)
    m = la.mean()
    s2 = max(la.var(ddof=1) - v.mean(), 1e-4)
    a = np.exp(m + np.sqrt(s2) * rng.standard_normal(6000))
    T = rng.wald(1.0 / a, 1.0 / sb2)
    return np.quantile(T, 0.5), np.quantile(T, 0.10)


def one_cell(g, reps, seed):
    rng = np.random.default_rng(seed)
    qmed, q10, tbar = true_quantiles(rng)
    tc = g * TM
    big_a = A0 * np.exp(S_A * rng.standard_normal(200000))
    bigT = rng.wald(1.0 / big_a, 1.0 / SB2)
    cost = KAP0 + np.minimum(bigT, tc).mean() / tbar
    n = max(5, int(round(C_BUDGET / cost)))
    e_med, e_q10, e_pred, e_lab = [], [], [], []
    for _ in range(reps):
        a = A0 * np.exp(S_A * rng.standard_normal(n))
        x, T = paths(a, rng)
        failed = T <= tc
        tend = np.minimum(T, tc)
        idx = (np.floor(tend / DT).astype(int) - 1)
        ah, te, xe = drift_hat(x, tend, idx)
        ah = np.where(failed, 1.0 / T, ah)               # a failed unit's drift: threshold / time to failure
        te = np.where(failed, T, te)
        # pooled diffusion from increments before the end of each record
        valid = TIMES[None, :] <= tend[:, None]
        dx = np.diff(np.concatenate([np.zeros((n, 1)), x], 1), axis=1)
        resid = (dx - ah[:, None] * DT) * valid
        sb2 = max((resid ** 2).sum() / (valid.sum() * DT), 1e-9)
        mq, q1 = fleet_from_drifts(ah, te, sb2, rng)
        e_med.append(np.log(mq / qmed)); e_q10.append(np.log(q1 / q10))
        # learning labels
        lab = np.where(failed, T, tend + np.maximum(1.0 - np.minimum(xe, 1.0), 0.0) / np.maximum(ah, 1e-6))
        # window feature
        iw = int(WIN / DT) - 1
        aw = x[:, iw] / TIMES[iw]
        A = np.column_stack([np.ones(n), np.log(np.maximum(aw, 1e-6))])
        coef = np.linalg.lstsq(A, np.log(lab), rcond=None)[0]
        at = A0 * np.exp(S_A * rng.standard_normal(NTEST))
        xt, Tt = paths(at, rng)
        awt = xt[:, iw] / TIMES[iw]
        pred = np.exp(coef[0] + coef[1] * np.log(np.maximum(awt, 1e-6)))
        pred = np.clip(pred, 0.5 * lab.min(), 2.0 * lab.max())      # a predictor is bounded by what the campaign saw
        e_pred.append(np.sqrt(np.mean((pred - Tt) ** 2)))
        e_lab.append(np.sqrt(np.mean((lab.mean() - Tt) ** 2)))
    return dict(n=n, F=float((bigT <= tc).mean()), med=float(np.sqrt(np.mean(np.square(e_med)))),
                q10=float(np.sqrt(np.mean(np.square(e_q10)))),
                pred=float(np.mean(e_pred) / np.mean(e_lab)),
                se=float(np.std(np.array(e_pred) / np.array(e_lab)) / np.sqrt(reps)))


if __name__ == '__main__':
    print(f'Wiener degradation, random drift (sd of log drift {S_A}), diffusion CV of life {CVD}, R = {R} campaigns per row, budget {C_BUDGET}')
    print(f'{"t_c/median":>10s} {"F":>5s} {"n":>4s} {"RMSE ln median":>15s} {"RMSE ln q10":>12s} {"learning error":>16s}')
    rows = []
    for gi, g in enumerate(GS):
        r = one_cell(g, R, 500 + gi)
        rows.append((g, r))
        print(f'{g:10.2f} {r["F"]:5.2f} {r["n"]:4d} {r["med"]:15.4f} {r["q10"]:12.4f} {r["pred"]:10.3f} +-{r["se"]:.3f}', flush=True)
    for key, nm in (('med', 'median life'), ('q10', '10th percentile'), ('pred', 'learning error')):
        b = min(rows, key=lambda t: t[1][key]); print(f'  optimum for {nm:16s}: t_c = {b[0]:.2f} median lives (F = {b[1]["F"]:.2f})')
    bq = min(rows, key=lambda t: t[1]['q10']); bl = min(rows, key=lambda t: t[1]['pred'])
    lq = min(t[1]['q10'] for t in rows); ll = min(t[1]['pred'] for t in rows)
    print(f'  plan optimal for the 10th percentile: learning error {bq[1]["pred"]/ll:.3f} x its best')
    print(f'  plan optimal for learning:            10th-percentile RMSE {bl[1]["q10"]/lq:.3f} x its best')
