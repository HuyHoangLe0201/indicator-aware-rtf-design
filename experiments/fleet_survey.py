"""
The damage variable (capacity) on every cell fleet, one code path.

The survey of Section 6 measures the terminal concentration of bulk capacity on the Severson
cells; the other cell fleets entered only as a replication of the channel ordering.  A reader
who opens Table 22 sees capacity values far from the Severson one, so this script puts all
twelve fleets through the same estimators and safeguards:

  power law    the parametric arm of the survey (fd003_all.ups): per cell, smooth, normalize,
               fit D = tau**beta, Ups = 2 beta + 1; fleet median at each of three smoothing
               widths, their median as the value, the largest ratio of the three as the drift
               (the stability safeguard), and a 95% interval from resampling cells (20000 draws)
  nonparametric  the smoothed-trajectory estimate of xjtu_battery_replicate.profile at the
               same widths, the cross-check of Section 6.1
  direction    share of cells whose late median minus early median has the majority sign

Fleets: Severson (dq_traj.npz), XJTU batteries, NASA PCoE, Zhang et al., and the eight
BatteryLife sets (cells cut at the cycle life of their labels, as in batterylife_replicate.py).
Writes fleet_survey.json and prints the table.
"""
import glob
import json
import os
import re
import sys

import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scipy.ndimage import uniform_filter1d
from scipy.optimize import least_squares
from xjtu_battery_replicate import profile

FR = (0.03, 0.07, 0.15)
B = 20000
THR = 1.18                                   # battery threshold, (1+Lambda)/(1+kappa0), Lambda = 0.241
SEV = os.path.join(HERE, '..', 'data', 'severson')


def direction(units):
    s = []
    for h in units:
        h = np.asarray(h, float)
        k = max(2, len(h)//20)
        s.append(np.sign(np.median(h[-k:]) - np.median(h[:k])))
    s = [x for x in s if x != 0]
    return max(s.count(1), s.count(-1))/len(s)


def ups_pl(h, smooth):
    """Power-law arm of the survey (fd003_all.ups): smooth, normalize, fit D = tau**beta, Ups = 2 beta + 1."""
    h = np.asarray(h, float)
    y = uniform_filter1d(h, max(3, int(len(h)*smooth)), mode='nearest')
    b = np.median(y[:max(2, len(y)//10)])
    e = np.median(y[-max(2, len(y)//20):])
    if abs(e - b) < 1e-12:
        return np.nan
    D = (y - b)/(e - b)
    t = (np.arange(len(y)) + 1)/len(y)
    m = (D > -0.4) & (D < 1.6)
    if m.sum() < 40:
        return np.nan
    r = least_squares(lambda p: np.clip(t[m], 1e-9, 1.)**p[0] - D[m], [3.],
                      bounds=([0.05], [15.]), max_nfev=400)
    return 2*float(r.x[0]) + 1


def summarize(name, cells, rng):
    """cells: list of (tau, capacity) with the capacity oriented either way."""
    # per-cell value at each width; the power-law fit is on the index axis, as in the survey scripts
    A = np.array([[ups_pl(h, f) for f in FR] for _, h in cells]).T          # widths x cells
    N = np.array([[profile(h, f, t)[1] for t, h in cells] for f in FR])
    med = np.nanmedian(A, 1)
    boot = np.empty(B)
    n = A.shape[1]
    for k in range(B):
        i = rng.integers(0, n, n)
        boot[k] = np.nanmedian([np.nanmedian(A[j][i]) for j in range(A.shape[0])])
    npw = [float(np.nanmedian(r)) for r in N]
    out = dict(n=len(cells), widths=[float(x) for x in med], par=float(np.median(med)),
               lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)),
               drift=float(med.max()/max(med.min(), 1e-9)),
               np=npw, np_drift=float(max(npw)/max(min(npw), 1e-9)),
               direction=direction([h for _, h in cells]))
    print(f'{name:10s} n={out["n"]:4d}  pl {out["par"]:6.2f} [{out["lo"]:5.2f},{out["hi"]:6.2f}] '
          f'({med[0]:5.2f}/{med[1]:5.2f}/{med[2]:5.2f}, drift {out["drift"]:4.2f}x)  '
          f'np {npw[0]:5.2f}/{npw[1]:5.2f}/{npw[2]:5.2f}  dir {out["direction"]:.2f}', flush=True)
    return out


def severson():
    d = np.load(os.path.join(SEV, 'dq_traj.npz'), allow_pickle=True)
    cells = []
    for q in d['QD']:
        q = np.asarray(q, float)
        q = q[np.isfinite(q) & (q > .2) & (q < 1.5)]
        cells.append(((np.arange(len(q)) + 1)/len(q), q))
    return cells


def xjtu_batteries():
    from xjtu_battery_replicate import ROOT, cell_channels
    cells = []
    for b in ('Batch-1', 'Batch-2'):
        for p in sorted(glob.glob(os.path.join(ROOT, b, '*.mat'))):
            try:
                c = cell_channels(p)
            except Exception:
                c = None
            if c:
                cells.append((c['tau_cap'], c['capacity']))
    return cells


def nasa():
    from nasa_precursor import ROOT, load_cell
    seen, cells = set(), []
    for p in sorted(glob.glob(os.path.join(ROOT, '*', 'B*.mat'))):
        k = os.path.basename(p)
        if k in seen:
            continue
        seen.add(k)
        try:
            c = load_cell(p)
        except Exception:
            c = None
        if c:
            h = c['capacity']
            cells.append(((np.arange(len(h)) + 1)/len(h), h))
    return cells


def zhang():
    from zhang_load import discharge_capacity, eis_scalars
    CAP = r'D:\claudecode\generationbound\data\zhang\Capacity_data\Capacity data'
    EIS = r'D:\claudecode\generationbound\data\zhang\EIS_data\EIS data'
    cells = []
    for p in sorted(glob.glob(os.path.join(CAP, '*.txt'))):
        nm = re.search(r'Data_Capacity_(\w+)', p).group(1)
        ep = os.path.join(EIS, f'EIS_state_IX_{nm}.txt')
        if not os.path.exists(ep):
            continue
        q, e = discharge_capacity(p), eis_scalars(ep)
        q0 = np.nanmedian(q[:5])
        below = np.where(np.isfinite(q) & (q <= 0.8*q0))[0]
        if not len(below):
            continue
        T = below[0] + 1
        idx = [c for c in range(1, T + 1) if c in e and np.isfinite(q[c-1])]
        if len(idx) < 60:
            continue
        h = np.array([-q[c-1] for c in idx])
        cells.append(((np.arange(len(h)) + 1)/len(h), h))
    return cells


def batterylife(name):
    from batterylife_extract import LAB, OUT
    from batterylife_replicate import cell
    res = np.load(os.path.join(OUT, name + '.npy'), allow_pickle=True).item()
    lab = json.load(open(os.path.join(LAB, name + '_labels.json')))
    cells = []
    for member, rec in res.items():
        if 'err' in rec:
            continue
        key = os.path.basename(member)
        life = lab.get(key, lab.get(key.replace('--', '-#')))
        c = None if life is None else cell(rec, life)
        if c:
            cells.append((c['tau_cap'], c['capacity']))
    return cells


def turbofans(fd):
    """Fused component of one C-MAPSS subset, sensors standardized within their operating cluster
    (fd002_fd004.py); FD001 and FD003 have one condition, so this is their plain PC1."""
    from scale_check import CMAPSS
    d = np.loadtxt(os.path.join(CMAPSS, f'train_FD00{fd}.txt'))
    uid = d[:, 0].astype(int)
    _, inv = np.unique(np.round(d[:, 2:5], 1), axis=0, return_inverse=True)
    inv = inv.ravel()
    keep = [j for j in range(5, d.shape[1]) if len(np.unique(d[:, j])) > 2]
    Z = np.empty((len(d), len(keep)))
    for c in np.unique(inv):
        m = inv == c
        Z[m] = (d[m][:, keep] - d[m][:, keep].mean(0))/(d[m][:, keep].std(0) + 1e-12)
    good = [k for k in range(len(keep)) if Z[:, k].std() > 0.5]
    w = np.linalg.svd(Z[:, good] - Z[:, good].mean(0), full_matrices=False)[2][0]
    pc = Z[:, good] @ w
    if np.corrcoef(pc, d[:, 1])[0, 1] < 0:
        pc = -pc
    return [((np.arange((uid == u).sum()) + 1)/(uid == u).sum(), pc[uid == u]) for u in np.unique(uid)]


FLEETS = [('Severson', severson), ('XJTU', xjtu_batteries), ('NASA', nasa), ('Zhang', zhang)] + \
         [(n, (lambda n=n: batterylife(n))) for n in
          ('HUST', 'Tongji', 'RWTH', 'Stanford', 'MICH', 'HNEI', 'CALCE', 'SNL')]

if __name__ == '__main__':
    rng = np.random.default_rng(20250611)
    if 'milling' in sys.argv[1:]:
        # Milling (14 cutters): per-cutter power-law exponent of the table acoustic emission, the
        # best-resolved admissible signal (milling_test.py); fleet median, 95% interval over cutters.
        from milling_test import campaign, shape
        units = campaign(0.45, 0.5)
        v = []
        for _, _, _, tr in units:
            y = tr['AE_table']
            b = shape(y if np.sign(np.median(y[-2:]) - np.median(y[:2])) >= 0 else -y)
            if b:
                v.append(b)
        v = np.array(v)
        boot = [2*np.median(v[rng.integers(0, len(v), len(v))]) + 1 for _ in range(B)]
        res = dict(n=len(units), n_used=len(v), par=float(2*np.median(v) + 1),
                   lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)),
                   share_below=float(np.mean(np.array(boot) < 1.30)))
        json.dump(res, open(os.path.join(HERE, 'fleet_survey_milling.json'), 'w'), indent=1)
        print('milling AE table', res)
        sys.exit()
    if 'turbofans' in sys.argv[1:]:
        res = {f'FD00{k}': summarize(f'FD00{k}', turbofans(k), rng) for k in (1, 2, 3, 4)}
        json.dump(res, open(os.path.join(HERE, 'fleet_survey_tf.json'), 'w'), indent=1)
        print('wrote fleet_survey_tf.json')
        sys.exit()
    res = {}
    for name, fn in FLEETS:
        res[name] = summarize(name, fn(), rng)
    json.dump(res, open(os.path.join(HERE, 'fleet_survey.json'), 'w'), indent=1)
    print('wrote fleet_survey.json')
