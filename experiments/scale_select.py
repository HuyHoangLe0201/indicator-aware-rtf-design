"""
Which scale is the estimator entitled to use?

Fisher information is invariant under an invertible transform of the OBSERVABLE, provided the
noise model is transformed with it.  The shape estimator of Upsilon is not: it assumes
X = D(tau) + sigma W with sigma constant, so handing it g(X) makes it estimate Upsilon for a
different model, and the answer moves (scale_check.py, scale_check2.py: up to 20x, sometimes
across the decision threshold).  The scale is therefore not free -- it is the scale on which
the measurement noise is homoscedastic, and that is an empirical question.

This script selects the scale from the data.  For each indicator and each Box-Cox lambda it
detrends each unit with a local median, then measures heteroscedasticity as the Spearman
correlation between |residual| and the fitted level.  lambda* minimises |rho_het| pooled over
units.  Upsilon is then reported at lambda*, alongside the value at the native scale, so the
question "does the corner verdict depend on a modelling choice?" gets a measured answer.
"""
import numpy as np, os, json
from scipy.stats import spearmanr
from scipy.ndimage import uniform_filter1d, median_filter
from scale_check import boxcox, load_cmapss, CACHE, THRESH, CMAPSS
from scale_check2 import ups_np

HERE = os.path.dirname(__file__)
THR = {'battery': (1+0.241)/1.05, 'C-MAPSS': (1+0.165)/1.05, 'bearing': (1+1.382)/1.05}
LAM = [('log', 0.0), ('cube-root', 1/3), ('sqrt', 0.5), ('native', 1.0), ('square', 2.0)]


def orient(h):
    """Return the indicator oriented so that it INCREASES with damage.

    The Box-Cox family is defined for positive quantities and boxcox() shifts by the series
    minimum.  For an indicator that falls as damage accumulates -- battery capacity, and the
    C-MAPSS sensors that drift downward -- that shift puts the singular point of the family
    at end of life, which is exactly where Upsilon is evaluated, so lambda = 0 explodes the
    terminal derivative and lambda = 2 flattens it.  The artefact is in the sweep, not in the
    indicator.  Orienting first removes it and leaves Upsilon unchanged, since the estimator
    normalises by (end - base) and is already sign-invariant.
    """
    h = np.asarray(h, float)
    k = max(2, len(h)//20)
    return h if np.median(h[-k:]) >= np.median(h[:k]) else -h


def het(h):
    """Spearman corr between |residual| and level after local-median detrending."""
    h = np.asarray(h, float); h = h[np.isfinite(h)]
    n = len(h)
    if n < 60:
        return np.nan
    fit = median_filter(h, size=max(5, n//25), mode='nearest')
    r = np.abs(h - fit)
    if np.allclose(r, 0):
        return np.nan
    s = spearmanr(fit, r).statistic
    return abs(s) if np.isfinite(s) else np.nan


if __name__ == '__main__':
    z = np.load(CACHE, allow_pickle=True)
    XF = {k: z[k].item() for k in z.files}
    CF = load_cmapss()
    series = {}

    # battery, bulk capacity -- the one trajectory-estimated battery channel
    SEV = os.path.join(HERE, '..', 'data', 'severson', 'dq_traj.npz')
    d = np.load(SEV, allow_pickle=True)
    cap = []
    for q in d['QD']:
        q = np.asarray(q, float)
        q = q[np.isfinite(q) & (q > .2) & (q < 1.5)]
        if len(q) < 60:
            continue
        # Severson capacity records carry isolated measurement spikes; the same 9-cycle
        # median de-spike used everywhere else in the pipeline is applied first, otherwise
        # both the heteroscedasticity statistic and the terminal derivative read the spikes.
        cap.append(np.median(np.lib.stride_tricks.sliding_window_view(
            np.pad(q, 4, mode='edge'), 9), axis=1))
    series[('battery', 'bulk capacity')] = cap

    for ind in ['RMS', 'peak', 'kurtosis', 'HF 2-5kHz', 'HF 5-10kHz', 'spec centroid']:
        series[('bearing', ind)] = [np.asarray(XF[b][ind], float) for b in XF]
    for ind, us in CF.items():
        series[('C-MAPSS', ind)] = us

    # C-MAPSS fused principal component, the seventh surveyed indicator
    dd = np.loadtxt(os.path.join(CMAPSS, 'train_FD001.txt'))
    uid = dd[:, 0].astype(int)
    cols = [6, 7, 8, 11, 12, 13, 15, 16, 17, 18, 19, 21, 24, 25]
    Xz = (dd[:, cols] - dd[:, cols].mean(0))/(dd[:, cols].std(0) + 1e-12)
    w = np.linalg.svd(Xz - Xz.mean(0), full_matrices=False)[2][0]
    series[('C-MAPSS', 'PC1 of 17')] = [
        ((dd[uid == u][:, cols] - dd[:, cols].mean(0))/(dd[:, cols].std(0) + 1e-12)) @ w
        for u in np.unique(uid)]

    hdr = (f'{"domain":9s} {"indicator":15s} {"lambda*":>10s} {"het@l*":>7s} '
           f'{"het@nat":>8s} {"Ups@l*":>8s} {"Ups@nat":>8s} {"thresh":>6s}  verdict at lambda*')
    print(hdr); print('-'*len(hdr))
    store = {}
    for (dom, ind), us in series.items():
        hs = []
        for _, lam in LAM:
            v = np.array([het(boxcox(orient(h), lam)) for h in us])
            hs.append(np.nanmedian(v))
        hs = np.array(hs)
        j = int(np.nanargmin(hs))
        jn = [n for n, _ in LAM].index('native')
        u_star = np.nanmedian([ups_np(boxcox(orient(h), LAM[j][1])) for h in us])
        u_nat = np.nanmedian([ups_np(boxcox(orient(h), 1.0)) for h in us])
        th = THR[dom]
        agree = (u_star >= th) == (u_nat >= th)
        store[f'{dom}|{ind}'] = np.array([LAM[j][1], hs[j], hs[jn], u_star, u_nat])
        print(f'{dom:9s} {ind:15s} {LAM[j][0]:>10s} {hs[j]:7.3f} {hs[jn]:8.3f} '
              f'{u_star:8.2f} {u_nat:8.2f}  '
              + f'{th:6.2f}  ' + ('corner' if u_star >= th else 'INTERIOR')
              + ('' if agree else '  (flips vs native)'))

    np.savez(os.path.join(HERE, 'scale_select.npz'),
             lam=json.dumps([n for n, _ in LAM]), **store)
    print(f'\nlambda* is the Box-Cox exponent minimising heteroscedasticity; het is the median '
          f'over units of |Spearman(|residual|, level)|.  Threshold {THRESH}.')
