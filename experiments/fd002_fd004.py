"""C-MAPSS FD002 and FD004 (six operating conditions): terminal concentration per sensor and for PC1 (R5.3).
Operating-condition effect removed by standardising each sensor within its operating cluster (cluster = rounded
(altitude, Mach, throttle) triple), the usual preprocessing for these subsets. Estimator and smoothing sweep as in
fd003_all.py, so the numbers are comparable with the FD001/FD003 rows."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))
from fd003_all import ups, FR, THRESH
ROOT = 'D:/bách khoa Đà Nẵng/nghiencuukhoahoc/baotri/data/CMAPSSData/'
for fd in ('FD002', 'FD004', 'FD001', 'FD003'):
    d = np.loadtxt(ROOT + f'train_{fd}.txt'); uid = d[:, 0].astype(int)
    ops = np.round(d[:, 2:5], 1); keys, inv = np.unique(ops, axis=0, return_inverse=True); inv = inv.ravel()
    keep = [j for j in range(5, d.shape[1]) if len(np.unique(d[:, j])) > 2]
    Z = np.empty((len(d), len(keep)))
    for c in range(len(keys)):
        m = inv == c
        Z[m] = (d[m][:, keep] - d[m][:, keep].mean(0)) / (d[m][:, keep].std(0) + 1e-12)
    # sensors that still vary after removing the operating condition
    good = [k for k in range(len(keep)) if Z[:, k].std() > 0.5]
    w = np.linalg.svd(Z[:, good] - Z[:, good].mean(0), full_matrices=False)[2][0]
    pc = Z[:, good] @ w
    if np.corrcoef(pc, d[:, 1])[0, 1] < 0: w, pc = -w, -pc
    units = [uid == u for u in np.unique(uid)]
    print(f'\n{fd}: {len(units)} units, {len(keys)} operating clusters, {len(good)} informative sensors')
    meds, drifts = [], []
    rows = [(f's{keep[k]-4}', Z[:, k]) for k in good] + [('PC1', pc)]
    for lab, s in rows:
        vals = []
        for f in FR:
            v = np.array([ups(s[m] if np.corrcoef(s[m], np.arange(m.sum()))[0, 1] >= 0 else -s[m], f) for m in units], float)
            vals.append(np.nanmedian(v))
        vals = np.array(vals); drift = np.nanmax(vals) / max(np.nanmin(vals), 1e-9)
        med = float(np.nanmedian(vals))
        if lab != 'PC1': meds.append(med); drifts.append(drift)
        else: print(f'  PC1 : Ups {med:6.2f}  drift {drift:4.2f}x  ' + ('corner' if med >= THRESH else 'INTERIOR'))
    m = np.array(meds)
    print(f'  sensors: Ups from {np.nanmin(m):.2f} to {np.nanmax(m):.2f} (median {np.nanmedian(m):.2f}); worst drift {max(drifts):.2f}x; '
          f'{int((m>=THRESH).sum())}/{len(m)} above {THRESH}')
