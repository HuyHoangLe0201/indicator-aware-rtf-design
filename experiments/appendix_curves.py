"""
Normalised indicator trajectories D(tau) of every dataset of the survey, for the appendix figure.

Each unit's indicator is oriented, smoothed (7% of the record) and scaled to run from its early
baseline (0) to its terminal value (1), exactly as in the trajectory-shape estimator, and then
resampled on a common grid of fractions of life.  One indicator per dataset is the damage
variable or the indicator whose verdict the paper reports; on the four battery sets the competing
early channel (DeltaQ(V) or impedance) is kept beside it.

Writes appendix_curves.npz: for every panel P the matrices "P|main" and "P|early" (units x grid)
and the grid itself.  make_figures.py draws figure 5 from it.

Raw data are read from the roots set by set_paths.py; the two slow readers (PRONOSTIA and the
XJTU battery cells) take a few minutes.
"""
import glob
import os
import re

import numpy as np
import scipy.io as sio
from scipy.ndimage import uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
SEV = os.path.join(HERE, '..', 'data', 'severson')
GRID = np.linspace(0.0, 1.0, 201)[1:]
FRAC = 0.07


def norm(h, tau=None, frac=FRAC, k0=None, k1=None):
    """h(tau) -> D on GRID: baseline 0, terminal value 1 (the sign of the change is absorbed)."""
    h = np.asarray(h, float)
    n = len(h)
    tau = (np.arange(n) + 1)/n if tau is None else np.asarray(tau, float)
    ok = np.isfinite(h) & np.isfinite(tau)
    h, tau, n = h[ok], tau[ok], ok.sum()
    if n < 5:
        return None
    s = uniform_filter1d(h, max(3, int(frac*n)), mode='nearest') if frac else h
    base = np.median(s[:k0 or max(2, n//20)])
    end = np.median(s[-(k1 or max(2, n//50)):])
    if abs(end - base) < 1e-12:
        return None
    return np.interp(GRID, tau, (s - base)/(end - base))


def stack(rows):
    rows = [r for r in rows if r is not None]
    return np.vstack(rows) if rows else np.zeros((0, len(GRID)))


# ------------------------------------------------------------------------ batteries
def severson():
    d = np.load(os.path.join(SEV, 'dq_traj.npz'), allow_pickle=True)
    main, early = [], []
    for q, dq, life in zip(d['QD'], d['DQ'], d['life']):
        q = np.asarray(q, float)
        q = q[np.isfinite(q) & (q > .2) & (q < 1.5)]
        main.append(norm(q))
        cyc, v = np.asarray(dq[0], float), np.asarray(dq[1], float)
        early.append(norm(v, tau=cyc/life))
    return stack(main), stack(early)


def xjtu_batteries():
    from xjtu_battery_replicate import ROOT, cell_channels
    paths = []
    for b in ('Batch-1', 'Batch-2'):
        paths += sorted(glob.glob(os.path.join(ROOT, b, '*.mat')))
    main, early = [], []
    for p in paths:
        try:
            c = cell_channels(p)
        except Exception:
            c = None
        if c:
            main.append(norm(c['capacity'], tau=c['tau_cap']))
            early.append(norm(c['dqv'], tau=c['tau_dq']))
    return stack(main), stack(early)


def zhang():
    from zhang_load import discharge_capacity, eis_scalars
    CAP = r'D:\claudecode\generationbound\data\zhang\Capacity_data\Capacity data'
    EIS = r'D:\claudecode\generationbound\data\zhang\EIS_data\EIS data'
    main, early = [], []
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
        main.append(norm([-q[c-1] for c in idx]))
        early.append(norm([e[c][2] for c in idx]))
    return stack(main), stack(early)


def nasa():
    from nasa_precursor import ROOT, load_cell
    seen, main, early = set(), [], []
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
            main.append(norm(c['capacity']))
            early.append(norm(c['Rct']))
    return stack(main), stack(early)


# ------------------------------------------------------------------------ turbofans
def cmapss(subset):
    from scale_check import CMAPSS
    d = np.loadtxt(os.path.join(CMAPSS, f'train_FD00{subset}.txt'))
    uid = d[:, 0].astype(int)
    return stack([norm(d[uid == u][:, 8]) for u in np.unique(uid)]), np.zeros((0, len(GRID)))


# ------------------------------------------------------------------------ bearings
def boxcox_half(h):
    h = np.asarray(h, float)
    s = h - np.nanmin(h) + 1e-6*(np.nanmax(h) - np.nanmin(h) + 1e-12)
    return 2*(np.sqrt(s) - 1)


def xjtu_sy():
    z = np.load(os.path.join(HERE, 'xjtu_features.npz'), allow_pickle=True)
    rows = []
    for k in z.files:
        r = np.asarray(z[k].item()['RMS'], float)[::2]
        if len(r) >= 60:
            rows.append(norm(boxcox_half(r)))
    return stack(rows), np.zeros((0, len(GRID)))


def pronostia():
    from pronostia_c50 import ROOT, features
    rows = []
    for setn in ('Learning_set', 'Full_Test_Set'):
        for bd in sorted(glob.glob(os.path.join(ROOT, setn, 'Bearing*'))):
            r = features(bd)['RMS']
            if len(r) >= 60:
                rows.append(norm(r))
    return stack(rows), np.zeros((0, len(GRID)))


# ------------------------------------------------------------------------ milling
def milling():
    from milling_test import campaign
    rows = []
    for _, _, _, tr in campaign(0.45, 0.5):
        y = np.asarray(tr['AE_table'], float)
        rows.append(norm(y))
    return stack(rows), np.zeros((0, len(GRID)))


# ------------------------------------------------------------------------ replication sets
def batterylife(name):
    """Capacity and DeltaQ(V) of one BatteryLife set, cut at the cycle life of its labels
    (the cells and the cut are those of batterylife_replicate.py)."""
    import json
    from batterylife_extract import LAB, OUT
    from batterylife_replicate import cell
    res = np.load(os.path.join(OUT, name + '.npy'), allow_pickle=True).item()
    lab = json.load(open(os.path.join(LAB, name + '_labels.json')))
    main, early = [], []
    for member, rec in res.items():
        if 'err' in rec:
            continue
        key = os.path.basename(member)
        life = lab.get(key, lab.get(key.replace('--', '-#')))      # Tongji files use '--' where the labels use '-#'
        c = None if life is None else cell(rec, life)
        if c:
            main.append(norm(c['capacity'], tau=c['tau_cap']))
            early.append(norm(c['dqv'], tau=c['tau_dq']))
    return stack(main), stack(early)


def cmapss_clusters(subset):
    """FD002 / FD004: sensor 4 standardized within its operating cluster (fd002_fd004.py)."""
    from scale_check import CMAPSS
    d = np.loadtxt(os.path.join(CMAPSS, f'train_FD00{subset}.txt'))
    uid = d[:, 0].astype(int)
    _, inv = np.unique(np.round(d[:, 2:5], 1), axis=0, return_inverse=True)
    inv = inv.ravel()
    z = np.empty(len(d))
    for c in np.unique(inv):
        m = inv == c
        z[m] = (d[m, 8] - d[m, 8].mean())/(d[m, 8].std() + 1e-12)
    return stack([norm(z[uid == u]) for u in np.unique(uid)]), np.zeros((0, len(GRID)))


PANELS = [('severson', severson), ('xjtu_bat', xjtu_batteries), ('zhang', zhang), ('nasa', nasa),
          ('fd001', lambda: cmapss(1)), ('fd003', lambda: cmapss(3)),
          ('xjtu_sy', xjtu_sy), ('pronostia', pronostia), ('milling', milling),
          ('fd002', lambda: cmapss_clusters(2)), ('fd004', lambda: cmapss_clusters(4))]
PANELS += [(n.lower(), (lambda n=n: batterylife(n)))
           for n in ('HNEI', 'CALCE', 'SNL', 'MICH', 'HUST', 'RWTH', 'Stanford', 'Tongji')]

if __name__ == '__main__':
    import sys
    path = os.path.join(HERE, 'appendix_curves.npz')
    # panels named on the command line are recomputed and merged into the file; none = all
    out = dict(np.load(path)) if (len(sys.argv) > 1 and os.path.exists(path)) else {'grid': GRID}
    for name, fn in PANELS:
        if len(sys.argv) > 1 and name not in sys.argv[1:]:
            continue
        m, e = fn()
        out[f'{name}|main'], out[f'{name}|early'] = m, e
        print(f'{name:10s} units {len(m):4d} (early channel: {len(e):4d})', flush=True)
    np.savez_compressed(path, **out)
    print('wrote appendix_curves.npz')
