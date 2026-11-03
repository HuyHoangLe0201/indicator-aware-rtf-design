"""Put an interval on the one number the paper's empirical claim rests on.

The Delta-Q(V) channel is the only entry on the short-test side of Fig. 4, so it carries the
whole claim that the indicator can change the plan, and it is the only entry without a
resampling interval: its value comes from a fleet-level predictive curve with no per-unit
quantity to resample. That is true of the per-unit decomposition, not of the curve itself. The
curve can be recomputed on a resampled fleet, which is what happens here.

Two quantities are reported. The saturation point, the earliest window at which the curve
reaches 95 per cent of its own maximum, is what Section VII-B quotes. The terminal slope is what
Fig. 4 records as Upsilon. Both are recomputed on each resample of the cells.

Leave-one-out for an ordinary least-squares fit has a closed form through the hat matrix, so the
inner loop costs one solve per window rather than one per cell. The paper fits Huber; the two
agree to about two per cent on this channel, and the question here is sampling spread rather
than the estimator.
"""
import numpy as np

DATA = r'D:\claudecode\generationbound\data\severson\dq_traj.npz'
WINDOWS = (20, 50, 100, 200, 400, 700)
B = 4000


def load():
    d = np.load(DATA, allow_pickle=True)
    DQ, life = d['DQ'], d['life'].astype(float)
    X = np.full((len(life), len(WINDOWS)), np.nan)
    for i in range(len(life)):
        cyc, val = DQ[i][0].astype(float), DQ[i][1].astype(float)
        for j, m in enumerate(WINDOWS):
            if life[i] >= m and cyc[-1] >= m:
                X[i, j] = np.interp(m, cyc, val)
    return X, np.log(life), life


def loo_r2(x, y):
    """Leave-one-out R^2 of a straight line, via the hat matrix."""
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    if len(x) < 12:
        return np.nan
    A = np.column_stack([np.ones(len(x)), x])
    G = A.T @ A
    if np.linalg.cond(G) > 1e12:
        return np.nan
    Gi = np.linalg.inv(G)
    h = np.einsum('ij,jk,ik->i', A, Gi, A)
    r = y - A @ (Gi @ (A.T @ y))
    e = r/np.maximum(1 - h, 1e-9)
    return float(1 - np.mean(e**2)/y.var())


def curve(X, y, idx):
    r2 = np.array([loo_r2(X[idx, j], y[idx]) for j in range(X.shape[1])])
    if not np.all(np.isfinite(r2)) or r2.max() <= 0:
        return None, None, None
    eta = r2/r2.max()
    # saturation: first window reaching 95 per cent of the maximum, in fraction of median life
    med = np.median(np.exp(y[idx]))
    hit = np.argmax(eta >= 0.95)
    sat = WINDOWS[hit]/med
    slope = (eta[-1] - eta[-2])/((WINDOWS[-1] - WINDOWS[-2])/med)
    return r2, sat, slope


if __name__ == '__main__':
    X, y, life = load()
    n = len(y)
    full_r2, full_sat, full_slope = curve(X, y, np.arange(n))
    print(f'{n} cells, median life {np.median(life):.0f} cycles')
    print('LOO R2 by window: ' + ' '.join(f'{m}:{v:.3f}' for m, v in zip(WINDOWS, full_r2)))
    print(f'point estimates: saturation at {full_sat:.3f} of median life, '
          f'terminal slope {full_slope:.3f}')
    print()

    rng = np.random.default_rng(20260803)
    sats, slopes, peaks = [], [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        r2, sat, sl = curve(X, y, idx)
        if sat is None:
            continue
        sats.append(sat)
        slopes.append(sl)
        peaks.append(r2.max())
    sats, slopes, peaks = map(np.array, (sats, slopes, peaks))
    print(f'{len(sats)} of {B} resamples usable')
    for nm, v in (('saturation, fraction of life', sats), ('terminal slope', slopes),
                  ('peak LOO R2', peaks)):
        lo, hi = np.percentile(v, [2.5, 97.5])
        print(f'   {nm:30s} median {np.median(v):7.3f}   95% CI [{lo:7.3f}, {hi:7.3f}]')
    thr = 1.18
    print()
    print(f'battery threshold {thr}: the terminal slope exceeds it in '
          f'{100*np.mean(slopes >= thr):.2f}% of resamples')
