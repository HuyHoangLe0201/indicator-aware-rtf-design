"""Two physically distinct channels on the Zhang cells: bulk capacity against interfacial impedance.

Capacity measures the damage that defines failure.  Impedance senses the interfaces, which is
the same kind of process that makes DeltaQ(V) an escape channel on the Severson cells, so this
is the natural test of whether the crossing reproduces on an independent battery set with a
genuinely different second measurement rather than a second statistic of the same record.

Three scalars are taken from each spectrum: the ohmic resistance, read at the high-frequency
zero crossing of the imaginary part; the total resistance at the lowest frequency; and their
difference, which is the charge-transfer and diffusion arc.  Each is then treated exactly as
the paper treats any indicator: shape fit, Upsilon with a smoothing-stability sweep, and c_50.
"""
import numpy as np, os, glob, re

ROOT = r'D:\claudecode\generationbound\data\zhang'
CAP = os.path.join(ROOT, 'Capacity_data', 'Capacity data')
EIS = os.path.join(ROOT, 'EIS_data', 'EIS data')
STATE = 'IX'
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'zhang_traj.npz')


def discharge_capacity(path):
    d = np.loadtxt(path, skiprows=1)
    cyc = d[:, 1].astype(int)
    out = []
    for c in range(1, cyc.max() + 1):
        m = (cyc == c) & (d[:, 2] == 0)
        if m.sum() > 10:
            out.append(d[m, 3].max())
        else:
            out.append(np.nan)
    return np.array(out, float)


def eis_scalars(path):
    d = np.loadtxt(path, skiprows=1)
    cyc = d[:, 1].astype(int)
    rows = {}
    for c in np.unique(cyc):
        m = cyc == c
        f, re_z, im_z = d[m, 2], d[m, 3], d[m, 4]
        o = np.argsort(-f)                       # high frequency first
        f, re_z, im_z = f[o], re_z[o], im_z[o]
        # ohmic resistance: Re(Z) where -Im(Z) crosses zero coming down from high frequency
        s = np.where(np.diff(np.sign(im_z)) > 0)[0]
        if len(s):
            i = s[0]
            w = im_z[i+1] - im_z[i]
            r_ohm = re_z[i] + (re_z[i+1] - re_z[i])*(-im_z[i]/w) if abs(w) > 1e-12 else re_z[i]
        else:
            r_ohm = re_z[np.argmin(np.abs(im_z))]
        r_tot = re_z[-1]                          # lowest frequency
        rows[int(c)] = (float(r_ohm), float(r_tot), float(r_tot - r_ohm))
    return rows


if not os.path.exists(CACHE):
    cells, data = [], {}
    for p in sorted(glob.glob(os.path.join(CAP, 'Data_Capacity_*.txt'))):
        name = re.search(r'Data_Capacity_(\w+)\.txt', p).group(1)
        ep = os.path.join(EIS, f'EIS_state_{STATE}_{name}.txt')
        if not os.path.exists(ep):
            print(f'  {name}: no state {STATE} spectrum, skipped')
            continue
        cap = discharge_capacity(p)
        eis = eis_scalars(ep)
        n = min(len(cap), max(eis))
        idx = [c for c in range(1, n + 1) if c in eis and np.isfinite(cap[c-1])]
        if len(idx) < 60:
            print(f'  {name}: only {len(idx)} usable cycles, skipped')
            continue
        arr = np.array([[cap[c-1]] + list(eis[c]) for c in idx], float)
        data[name] = arr
        cells.append(name)
        print(f'  {name}: {len(idx)} cycles, capacity {arr[0,0]:.1f} -> {arr[-1,0]:.1f} mAh, '
              f'R_ohm {arr[0,1]:.3f} -> {arr[-1,1]:.3f} Ohm', flush=True)
    np.savez_compressed(CACHE, cells=np.array(cells), **{k: v for k, v in data.items()})
    print(f'cached {len(cells)} cells')

z = np.load(CACHE, allow_pickle=True)
print()
print(f'{len(z["cells"])} cells cached: {list(z["cells"])}')
