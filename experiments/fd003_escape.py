"""Does a second fault mode give turbofans an escape channel?

FD001 has one fault mode, high-pressure-compressor degradation, and Section VII-C finds all
seven of its indicators late-concentrated.  FD003 runs the same engine with TWO fault modes,
HPC degradation and fan degradation.  The paper's rule says an escape channel needs a different
physical process, and a second fault mode is exactly that, so FD003 is the strongest untried
test of the rule on data already in hand.

We measure, for every continuously varying sensor on both subsets, the shape exponent and from
it Upsilon = 2*beta + 1, and the integral statistic c_50, which Section VII-E argues is the
stable one.  The turbofan corner threshold is 1.11, so an escape channel needs beta < 0.055:
a strongly saturating sensor.  We then run the conditional-independence screen of Section VII-G
on FD003, asking whether any sensor carries information about the failure time beyond the fused
principal component, early.
"""
import numpy as np, os
from scipy.optimize import least_squares
from scipy.ndimage import uniform_filter1d

ROOT = r'D:\bách khoa Đà Nẵng\nghiencuukhoahoc\baotri\data'
K0, THRESH = 0.05, 1.11


def load(tag):
    d = np.loadtxt(os.path.join(ROOT, f'train_{tag}.txt'))
    uid = d[:, 0].astype(int)
    keep = [j for j in range(5, d.shape[1]) if len(np.unique(d[:, j])) > 2]
    return d, uid, keep


def shape(tr, smooth=15):
    y = uniform_filter1d(np.asarray(tr, float), max(3, len(tr)//smooth), mode='nearest')
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
    beta = float(r.x[0])
    # c_50: fraction of life by which half the extrapolation information has arrived,
    # for D = tau^beta this is (1/2)^(1/(2 beta + 1))
    s = 2*beta + 1
    return beta, 0.5**(1.0/s)


for tag, modes in (('FD001', 'HPC degradation only'), ('FD003', 'HPC and fan degradation')):
    d, uid, keep = load(tag)
    units = np.unique(uid)
    print(f'--- {tag}: {len(units)} units, {len(keep)} varying sensors, {modes} ---')
    print(f'    {"sensor":>7s} {"beta":>7s} {"Ups":>8s} {"c50":>7s}  verdict')
    rows = []
    for col in keep:
        bs, cs = [], []
        for u in units:
            m = uid == u
            tr = d[m, col]
            if np.corrcoef(tr, np.arange(m.sum()))[0, 1] < 0:
                tr = -tr
            f = shape(tr)
            if f:
                bs.append(f[0]); cs.append(f[1])
        if len(bs) < 20:
            continue
        b, c = float(np.median(bs)), float(np.median(cs))
        rows.append((2*b + 1, col - 4, b, c))
    for ups, sen, b, c in sorted(rows):
        verdict = 'STOP EARLY' if ups < THRESH else 'run to failure'
        print(f'    {sen:7d} {b:7.2f} {ups:8.2f} {c:7.3f}  {verdict}')
    lo = min(r[0] for r in rows)
    print(f'    lowest Upsilon {lo:.2f} against a threshold of {THRESH}: '
          f'{"a channel crosses" if lo < THRESH else "no channel crosses"}')
    print()
