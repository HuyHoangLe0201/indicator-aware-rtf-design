"""Is the FD003 measurement admissible under the paper's own stability requirement?

Section VII-E strikes any Upsilon that moves materially across the smoothing sweep.  The FD003
result is a negative one, so it would be easy to accept without checking, which is exactly when
a bad measurement slips through.  Here the sweep is applied to the two extreme sensors of FD003
and to the fused principal component.
"""
import numpy as np, os
from scipy.optimize import least_squares
from scipy.ndimage import uniform_filter1d

ROOT = r'D:\bách khoa Đà Nẵng\nghiencuukhoahoc\baotri\data'
d = np.loadtxt(os.path.join(ROOT, 'train_FD003.txt'))
uid = d[:, 0].astype(int)
keep = [j for j in range(5, d.shape[1]) if len(np.unique(d[:, j])) > 2]
Z = (d[:, keep] - d[:, keep].mean(0))/d[:, keep].std(0)
w = np.linalg.svd(Z - Z.mean(0), full_matrices=False)[2][0]
pc = Z @ w
if np.corrcoef(pc, d[:, 1])[0, 1] < 0:
    pc = -pc


def beta_of(tr, smooth):
    y = uniform_filter1d(np.asarray(tr, float), max(3, int(len(tr)*smooth)), mode='nearest')
    b, e = np.median(y[:max(2, len(y)//10)]), np.median(y[-max(2, len(y)//20):])
    if abs(e - b) < 1e-12:
        return None
    D = (y - b)/(e - b)
    t = (np.arange(len(y)) + 1)/len(y)
    m = (D > -0.4) & (D < 1.6)
    if m.sum() < 40:
        return None
    r = least_squares(lambda p: np.clip(t[m], 1e-9, 1.)**p[0] - D[m], [3.],
                      bounds=([0.05], [15.]), max_nfev=400)
    return float(r.x[0])


FR = (0.03, 0.07, 0.15)
cols = {'sensor 6': keep[[j - 5 for j in keep].index(1)] if False else None}
name = {'sensor 6': 6, 'sensor 10': 10, 'sensor 15': 15}
print(f'{"channel":11s} ' + ' '.join(f'{("Ups s="+str(f)):>12s}' for f in FR) + f' {"drift":>7s}')
print('-' * 56)
for lab, sn in list(name.items()) + [('PC1 of 16', None)]:
    vals = []
    for f in FR:
        bs = []
        for u in np.unique(uid):
            m = uid == u
            if sn is None:
                tr = pc[m]
            else:
                col = 4 + sn
                if col not in keep:
                    bs = []
                    break
                tr = d[m, col]
                if np.corrcoef(tr, np.arange(m.sum()))[0, 1] < 0:
                    tr = -tr
            b = beta_of(tr, f)
            if b:
                bs.append(b)
        vals.append(2*float(np.median(bs)) + 1 if bs else np.nan)
    if not np.isfinite(vals).all():
        print(f'{lab:11s} not measurable')
        continue
    drift = max(vals)/min(vals)
    print(f'{lab:11s} ' + ' '.join(f'{v:12.2f}' for v in vals) + f' {drift:6.2f}x'
          + ('  stable' if drift < 1.9 else '  UNSTABLE, would be struck'))
print()
print('turbofan corner threshold 1.11; a channel must fall below it to change the verdict')
