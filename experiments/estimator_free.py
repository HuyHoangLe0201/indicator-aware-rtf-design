"""The same comparison across all four channels, with no shape estimator anywhere.

Section IX concedes that Upsilon for the crossing channel is estimator-dependent, and that the
whole empirical claim rests on it. The concession is smaller than it looks if the ordering the
paper needs can be read off a quantity that involves no shape fit, no terminal derivative and
no normalisation of eta_T. The predictive curve is that quantity: observe a fleet over a window,
predict log life across units by leave-one-out, and read where the curve saturates and what it
is doing at the end.

Section VII-B already shows it for the two battery channels. This puts turbofans and bearings
on the same footing, so the four sit in one table computed one way. Leave-one-out for a linear
fit has a closed form through the hat matrix, so the fleet can be resampled and each entry
carries an interval.
"""
import os

import numpy as np

SEV = r'D:\claudecode\generationbound\data\severson\dq_traj.npz'
CMAPSS = r'D:\bách khoa Đà Nẵng\nghiencuukhoahoc\baotri\data\train_FD001.txt'
XJTU = r'D:\claudecode\generationbound\experiments\xjtu_features.npz'
FRACS = (0.03, 0.07, 0.15, 0.30, 0.45, 0.60, 0.80)
B = 3000


def loo_r2(X, y):
    """Leave-one-out R^2 of a linear fit on the columns of X, via the hat matrix."""
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    if len(y) < 6 or y.var() <= 0:
        return np.nan
    A = np.column_stack([np.ones(len(y)), X])
    G = A.T @ A + 1e-8*np.eye(A.shape[1])
    if np.linalg.cond(G) > 1e12:
        return np.nan
    Gi = np.linalg.inv(G)
    h = np.einsum('ij,jk,ik->i', A, Gi, A)
    r = y - A @ (Gi @ (A.T @ y))
    e = r/np.maximum(1 - h, 1e-9)
    return float(1 - np.mean(e**2)/y.var())


def summarise(w):
    """Level and drift over the observed window; the same two summaries for every channel."""
    w = np.asarray(w, float)
    k = max(2, len(w)//10)
    return np.array([np.nanmedian(w[-k:]), np.nanmedian(w[-k:]) - np.nanmedian(w[:k])])


def build(traj, life):
    """Feature blocks at absolute windows, and log life.

    The window must be the same number of cycles for every unit. Taking a fraction of each
    unit's own life makes the window length a function of the label, which leaks it: capacity
    then appears to predict life at seven per cent of it, which Fig. 3 shows it does not.
    Windows are set as fractions of the fleet median and applied absolutely; a unit shorter
    than the window cannot contribute to it.
    """
    life = np.asarray(life, float)
    med = np.median(life)
    out = []
    for f in FRACS:
        m = max(4, int(round(f*med)))
        F = []
        for tr, T in zip(traj, life):
            F.append(summarise(tr[:m]) if len(tr) >= m else np.array([np.nan, np.nan]))
        out.append(np.array(F))
    return out, np.log(life)


def channels():
    d = np.load(SEV, allow_pickle=True)
    life = d['life'].astype(float)
    cap = []
    for q in d['QD']:
        q = np.asarray(q, float)
        q0 = np.nanmedian(q[1:6])
        cap.append((q0 - q)/(0.2*q0 + 1e-12))
    dq = []
    for i in range(len(life)):
        cyc, val = d['DQ'][i][0].astype(float), d['DQ'][i][1].astype(float)
        g = np.arange(1, int(life[i]) + 1, dtype=float)
        dq.append(np.interp(g, cyc, val))
    yield 'battery capacity', cap, life
    yield 'battery $\\Delta Q(V)$', dq, life

    raw = np.loadtxt(CMAPSS)
    uid = raw[:, 0].astype(int)
    keep = [j for j in range(2, raw.shape[1]) if raw[:, j].std() > 1e-8]
    # the component must be defined once on the pooled fleet: taking each unit's own leading
    # singular vector gives it an arbitrary sign per unit, which destroys any cross-unit fit
    P = raw[:, keep]
    P = (P - P.mean(0))/(P.std(0) + 1e-12)
    v = np.linalg.svd(P, full_matrices=False)[2][0]
    tf, tl = [], []
    for u in np.unique(uid):
        tf.append(P[uid == u] @ v)
        tl.append(int((uid == u).sum()))
    yield 'turbofan PC1', tf, tl

    F = {k: v.item() for k, v in np.load(XJTU, allow_pickle=True).items()}
    use = [b for b in F if len(F[b]['RMS'][::2]) >= 60]
    yield 'bearing RMS', [np.asarray(F[b]['RMS'][::2], float) for b in use], \
        [len(F[b]['RMS'][::2]) for b in use]


def contributing(blocks, idx):
    return [int(np.all(np.isfinite(Bk[idx]), axis=1).sum()) for Bk in blocks]


def read(blocks, y, idx):
    r2 = np.array([loo_r2(Bk[idx], y[idx]) for Bk in blocks])
    if not np.all(np.isfinite(r2)) or np.nanmax(r2) <= 0:
        return None
    eta = r2/r2.max()
    sat = FRACS[int(np.argmax(eta >= 0.95))]
    slope = (eta[-1] - eta[-2])/(FRACS[-1] - FRACS[-2])
    return r2, sat, slope


if __name__ == '__main__':
    rng = np.random.default_rng(20260803)
    print('leave-one-out R^2 for log life, by fraction of each unit observed')
    print(f'{"channel":22s} {"n":>4s} ' + ' '.join(f'{f:6.2f}' for f in FRACS))
    rows = []
    for name, traj, life in channels():
        blocks, y = build(traj, life)
        full = read(blocks, y, np.arange(len(y)))
        cnt = contributing(blocks, np.arange(len(y)))
        if full is None:
            print(f'{name:22s} {len(y):4d}   no cross-unit fit is supported on this fleet')
            continue
        print(f'{name:22s} {len(y):4d} ' + ' '.join(f'{v:6.3f}' for v in full[0]))
        print(f'{"":22s} {"n":>4s} ' + ' '.join(f'{c:6d}' for c in cnt))
        sats, slopes = [], []
        for _ in range(B):
            idx = rng.integers(0, len(y), len(y))
            r = read(blocks, y, idx)
            if r:
                sats.append(r[1])
                slopes.append(r[2])
        rows.append((name, len(y), full[1], full[2], np.array(sats), np.array(slopes)))
    print()
    print(f'{"channel":22s} {"saturates at":>13s} {"95% CI":>16s} {"end slope":>10s} '
          f'{"95% CI":>18s}')
    for name, n, sat, sl, S, L in rows:
        s1, s2 = np.percentile(S, [2.5, 97.5])
        l1, l2 = np.percentile(L, [2.5, 97.5])
        print(f'{name:22s} {sat:13.2f} [{s1:6.2f},{s2:6.2f}] {sl:10.2f} '
              f'[{l1:7.2f},{l2:7.2f}]')
