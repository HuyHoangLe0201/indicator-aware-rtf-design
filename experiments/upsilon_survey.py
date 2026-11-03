"""
Measure Upsilon per health indicator per domain.

Upsilon = D'(1)^2 / int_0^1 tau^2 D'(tau)^2 dtau  is a property of the SHAPE of the
normalised indicator trajectory, so it needs no cross-unit regression and works with
as few as six units.

Per unit: D(tau) = [h(tau) - h_base] / [h(1) - h_base], h_base a robust early baseline
(sign handled automatically).  A stretched-exponential D(tau;a,b) is fitted -- that
family spans both SATURATING shapes (large a: Upsilon small, information front-loaded)
and ACCELERATING ones (a->0: D->tau^b, Upsilon = 2b+1, information at end of life).
A nonparametric estimate from a smoothed spline is reported as a check.

Corner threshold (kappa0 = 0.05): Upsilon >= 1.18 -> run to failure.
"""
import numpy as np, os, glob
from scipy.optimize import least_squares
from scipy.integrate import quad
from scipy.ndimage import uniform_filter1d

ROOT = r'D:\bách khoa Đà Nẵng\nghiencuukhoahoc\baotri\data'

def Dse(tau, a, b):
    tau = np.clip(tau, 1e-9, 1.0)
    return (1-np.exp(-a*tau**b))/(1-np.exp(-a))

def ups_of(a, b):
    dp = lambda t: a*b*t**(b-1)*np.exp(-a*t**b)/(1-np.exp(-a))
    Z = quad(lambda t: t**2*dp(t)**2, 0, 1, limit=300)[0]
    return dp(1.0)**2/Z if Z > 0 else np.nan

def ups_nonpar(tau, D):
    g = np.linspace(0.02, 1.0, 120)
    s = np.interp(g, tau, uniform_filter1d(D, max(3, len(D)//15), mode='nearest'))
    dp = np.gradient(s, g)
    Z = np.trapezoid(g**2*dp**2, g)
    return dp[-1]**2/Z if Z > 0 else np.nan

def measure(h):
    """h: raw indicator series over a unit's life -> (Upsilon_fit, Upsilon_np, a, b)"""
    h = np.asarray(h, float); h = h[np.isfinite(h)]
    if len(h) < 40: return None
    n = len(h); k = max(3, n//20)
    base = np.median(h[:k]); end = np.median(h[-max(2, n//50):])
    if abs(end-base) < 1e-9: return None
    D = (h-base)/(end-base)
    tau = (np.arange(n)+1)/n
    m = (D > -0.4) & (D < 1.6)
    tau, D = tau[m], D[m]
    if len(D) < 40: return None
    try:
        r = least_squares(lambda p: Dse(tau, p[0], p[1])-D, [3.0, 1.0],
                          bounds=([1e-3, 0.05], [60., 12.]),
                          loss='soft_l1', f_scale=0.1, max_nfev=900)
        a, b = r.x
    except Exception:
        return None
    return ups_of(a, b), ups_nonpar(tau, D), a, b

def report(dom, ind, vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x[0])])
    if len(v) == 0:
        print(f'{dom:9s} {ind:22s}   (no usable units)'); return
    U, Un = v[:, 0], v[:, 1]
    tag = 'INTERIOR' if np.median(U) < 1.18 else 'corner'
    print(f'{dom:9s} {ind:22s} {len(v):4d} {np.median(U):9.2f} '
          f'[{np.percentile(U,25):6.2f},{np.percentile(U,75):7.2f}] '
          f'{np.median(Un):9.2f}  {tag}')

# ------------------------------------------------------------------ C-MAPSS
def cmapss():
    d = np.loadtxt(os.path.join(ROOT, 'train_FD001.txt'))
    uid = d[:, 0].astype(int)
    S = {f's{j-4}': j for j in [6, 7, 8, 11, 12, 13, 15, 16, 17, 18, 19, 21, 24, 25]}
    units = [d[uid == u] for u in np.unique(uid)]
    # fused: PC1 of z-scored informative sensors, pooled
    cols = list(S.values())
    X = d[:, cols]; X = (X-X.mean(0))/(X.std(0)+1e-12)
    w = np.linalg.svd(X - X.mean(0), full_matrices=False)[2][0]
    out = {}
    for nm, j in list(S.items())[:6]:
        out[nm] = [measure(u[:, j]) for u in units]
    out['PC1 (fused)'] = [measure(((u[:, cols]-d[:, cols].mean(0)) /
                                   (d[:, cols].std(0)+1e-12)) @ w) for u in units]
    return out

# ------------------------------------------------------------------ PRONOSTIA
def pronostia(step=3):
    out = {k: [] for k in ['RMS', 'kurtosis', 'peak', 'HF band 2-5kHz']}
    for bd in sorted(glob.glob(os.path.join(
            ROOT, 'phm-ieee-2012-data-challenge-dataset-master', 'Learning_set', 'Bearing*'))):
        fs = sorted(glob.glob(os.path.join(bd, 'acc_*.csv')))[::step]
        if len(fs) < 60: continue
        rms, kur, pk, hf = [], [], [], []
        for fp in fs:
            try:
                a = np.loadtxt(fp, delimiter=',', usecols=4)
            except Exception:
                continue
            a = a - a.mean()
            rms.append(np.sqrt(np.mean(a**2)))
            sd = a.std() + 1e-12
            kur.append(np.mean((a/sd)**4))
            pk.append(np.max(np.abs(a)))
            P = np.abs(np.fft.rfft(a))**2
            f = np.fft.rfftfreq(len(a), 1/25600.)
            hf.append(P[(f > 2000) & (f < 5000)].sum()/(P.sum()+1e-12))
        for k, v in zip(out, [rms, kur, pk, hf]):
            out[k].append(measure(v))
    return out

if __name__ == '__main__':
    print(f'{"domain":9s} {"indicator":22s} {"n":>4s} {"Ups (med)":>9s} '
          f'{"[IQR]":>16s} {"Ups np":>9s}  verdict')
    print('-'*84)
    for nm, res in cmapss().items():
        report('C-MAPSS', nm, res)
    print()
    for nm, res in pronostia().items():
        report('Bearing', nm, res)
    print('\nPaper 1 asserts for bearings: RMS discriminative only after 60-80% of life,')
    print('frequency-domain indicators earlier (~40%).  A LOWER Upsilon means earlier.')
