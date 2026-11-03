"""
Persist the PER-UNIT distribution of Upsilon behind every number in the indicator survey.

The survey scripts (upsilon_survey.py, xjtu_survey.py) print medians and an IQR; the figures
need the underlying per-unit values so the spread can be drawn rather than asserted.  Nothing
about the estimator changes here -- this is the same shape estimator (a), the same units and
the same smoothing, only the intermediate array is kept.

Also computes, as a CROSS-CHECK not used in the published numbers, the shape-estimator
Upsilon for the two battery channels.  With the fixed reference cycle 10 the DeltaQ(V)
signature is a running scalar (dq_traj.py), so estimator (a) does in fact apply to it, and
its value can be compared against the predictive-R^2 estimate of 0.02 that the paper reports.

Writes upsilon_dist.npz: one array per "<domain>|<indicator>" key, plus a "meta" JSON string.
"""
import numpy as np, os, glob, json
from scipy.integrate import quad
from scipy.optimize import least_squares
from scipy.ndimage import uniform_filter1d

HERE = os.path.dirname(__file__)
CMAPSS = r'D:\bách khoa Đà Nẵng\nghiencuukhoahoc\baotri\data\CMAPSSData'
XJTU = (r'D:\bách khoa Đà Nẵng\nghiencuukhoahoc\baotri\data\XJTU-SY bearing dataset'
        r'\Data\extracted\XJTU-SY_Bearing_Datasets')
SEV = os.path.join(HERE, '..', 'data', 'severson')


def Dse(tau, a, b):
    tau = np.clip(tau, 1e-9, 1.0)
    return (1-np.exp(-a*tau**b))/(1-np.exp(-a))


def ups_of(a, b):
    dp = lambda t: a*b*t**(b-1)*np.exp(-a*t**b)/(1-np.exp(-a))
    Z = quad(lambda t: t**2*dp(t)**2, 0, 1, limit=300)[0]
    return dp(1.0)**2/Z if Z > 0 else np.nan


def measure_tau(tau, h):
    """Shape estimator (a) on an indicator sampled at arbitrary tau in (0,1]."""
    tau, h = np.asarray(tau, float), np.asarray(h, float)
    m = np.isfinite(tau) & np.isfinite(h)
    tau, h = tau[m], h[m]
    if len(h) < 40:
        return np.nan
    o = np.argsort(tau); tau, h = tau[o], h[o]
    k = max(3, len(h)//20)
    base, end = np.median(h[:k]), np.median(h[-max(2, len(h)//50):])
    if abs(end-base) < 1e-12:
        return np.nan
    D = (h-base)/(end-base)
    m = (D > -0.4) & (D < 1.6)
    tau, D = tau[m], D[m]
    if len(D) < 40:
        return np.nan
    try:
        r = least_squares(lambda p: Dse(tau, p[0], p[1])-D, [3.0, 1.0],
                          bounds=([1e-3, 0.05], [60., 12.]),
                          loss='soft_l1', f_scale=0.1, max_nfev=900)
    except Exception:
        return np.nan
    return ups_of(*r.x)


def measure(h):
    h = np.asarray(h, float); h = h[np.isfinite(h)]
    return measure_tau((np.arange(len(h))+1)/max(len(h), 1), h)


# ------------------------------------------------------------------- C-MAPSS
def cmapss():
    d = np.loadtxt(os.path.join(CMAPSS, 'train_FD001.txt'))
    uid = d[:, 0].astype(int)
    S = {'sensor 2': 6, 'sensor 3': 7, 'sensor 4': 8, 'sensor 7': 11,
         'sensor 8': 12, 'sensor 9': 13}
    cols = [6, 7, 8, 11, 12, 13, 15, 16, 17, 18, 19, 21, 24, 25]
    units = [d[uid == u] for u in np.unique(uid)]
    X = d[:, cols]; X = (X-X.mean(0))/(X.std(0)+1e-12)
    w = np.linalg.svd(X - X.mean(0), full_matrices=False)[2][0]
    out = {nm: np.array([measure(u[:, j]) for u in units]) for nm, j in S.items()}
    out['PC1 of 17'] = np.array([measure(((u[:, cols]-d[:, cols].mean(0)) /
                                          (d[:, cols].std(0)+1e-12)) @ w) for u in units])
    return out


# ---------------------------------------------------------------------- XJTU
FS = 25600.


def xjtu_features(bd):
    fs = sorted(glob.glob(os.path.join(bd, '*.csv')),
                key=lambda p: int(''.join(c for c in os.path.basename(p) if c.isdigit()) or 0))
    out = {k: [] for k in ['RMS', 'kurtosis', 'peak', 'HF 2-5kHz',
                           'HF 5-10kHz', 'spec centroid']}
    for fp in fs:
        try:
            a = np.loadtxt(fp, delimiter=',', skiprows=1, usecols=0)
        except Exception:
            continue
        if len(a) < 4096:
            continue
        a = a - a.mean(); sd = a.std() + 1e-12
        P = np.abs(np.fft.rfft(a))**2
        f = np.fft.rfftfreq(len(a), 1/FS)
        tot = P.sum() + 1e-12
        out['RMS'].append(np.sqrt(np.mean(a**2)))
        out['kurtosis'].append(np.mean((a/sd)**4))
        out['peak'].append(np.max(np.abs(a)))
        out['HF 2-5kHz'].append(P[(f > 2000) & (f < 5000)].sum()/tot)
        out['HF 5-10kHz'].append(P[(f > 5000) & (f < 10000)].sum()/tot)
        out['spec centroid'].append((f*P).sum()/tot)
    return {k: np.asarray(v, float) for k, v in out.items()}


def xjtu(frac=0.07):
    bds = sorted(glob.glob(os.path.join(XJTU, '*', 'Bearing*')))
    F = {os.path.basename(b): xjtu_features(b) for b in bds}
    F = {k: v for k, v in F.items() if len(v['RMS']) >= 60}
    out = {}
    for ind in ['RMS', 'peak', 'kurtosis', 'HF 2-5kHz', 'HF 5-10kHz', 'spec centroid']:
        vals = []
        for b in F:
            h = F[b][ind]
            n = len(h)
            s = uniform_filter1d(h, max(3, int(frac*n)), mode='nearest')
            vals.append(measure(s))
        out[ind] = np.array(vals)
    return out, len(F)


# ------------------------------------------------------------------- battery
def battery():
    t = np.load(os.path.join(SEV, 'dq_traj.npz'), allow_pickle=True)
    DQ, QD, life = t['DQ'], t['QD'], t['life']
    cap, dq = [], []
    for k in range(len(DQ)):
        q = np.asarray(QD[k], float)
        q = q[np.isfinite(q) & (q > .2) & (q < 1.5)]
        cap.append(measure(q))
        ms, vs = DQ[k]
        dq.append(measure_tau(ms/max(life[k], 1.), vs))
    return {'bulk capacity': np.array(cap),
            r'$\Delta Q(V)$ signature': np.array(dq)}


if __name__ == '__main__':
    store, meta = {}, {}
    print(f'{"domain":9s} {"indicator":22s} {"n":>4s} {"median":>8s} {"IQR":>17s}')
    print('-'*64)
    groups = [('battery', battery()), ('C-MAPSS', cmapss())]
    xj, nb = xjtu()
    groups.append(('bearing', xj))
    for dom, res in groups:
        for ind, v in res.items():
            v = np.asarray(v, float); good = v[np.isfinite(v)]
            if len(good) == 0:
                print(f'{dom:9s} {ind:22s}   (no usable units)'); continue
            store[f'{dom}|{ind}'] = good
            meta[f'{dom}|{ind}'] = dict(n=int(len(good)),
                                        med=float(np.median(good)),
                                        q1=float(np.percentile(good, 25)),
                                        q3=float(np.percentile(good, 75)))
            print(f'{dom:9s} {ind:22s} {len(good):4d} {np.median(good):8.2f} '
                  f'[{np.percentile(good,25):7.2f},{np.percentile(good,75):8.2f}]')
        print()
    out = os.path.join(SEV, '..', '..', 'experiments', 'upsilon_dist.npz')
    np.savez(os.path.abspath(out), meta=json.dumps(meta), **store)
    print(f'wrote {os.path.abspath(out)}  ({nb} bearings)')
