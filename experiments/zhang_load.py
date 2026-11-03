"""Robust loader for the Zhang set: the files do not share a column layout.

Six of the twelve capacity files carry four columns and six carry six, with Ewe/V and I/mA
inserted before the capacity.  Reading by position put voltage in the capacity slot for half
the fleet, which showed up as four cells whose "capacity" sat flat at 4.1 across their whole
life.  Some EIS files also have no header line at all, so a fixed skiprows drops a real
measurement.  Columns are therefore located by name, and the header is detected rather than
assumed.
"""
import numpy as np, os, re


def _read(path):
    """Return (column names or None, data array)."""
    with open(path, encoding='utf-8', errors='ignore') as fh:
        first = fh.readline()
    toks = [t.strip() for t in first.rstrip('\r\n').split('\t') if t.strip()]
    has_header = any(re.search(r'[A-Za-z]', t) and not re.fullmatch(r'[-+0-9.eE]+', t)
                     for t in toks)
    data = np.loadtxt(path, skiprows=1 if has_header else 0)
    if has_header and len(toks) != data.shape[1]:
        # some headers separate names by runs of spaces rather than tabs, which fuses
        # 'cycle number' and 'ox/red' into one token and shifts every column lookup
        toks = [t.strip() for t in re.split(r'\t|\s{2,}', first.rstrip('\r\n')) if t.strip()]
    if has_header and len(toks) != data.shape[1]:
        toks = None
    return (toks if has_header else None), data


def _col(names, want, default):
    if names is None:
        return default
    for i, n in enumerate(names):
        if want.lower() in n.lower().replace(' ', ''):
            return i
    raise KeyError(f'{want} not among {names}')


def discharge_capacity(path):
    names, d = _read(path)
    ic = _col(names, 'cyclenumber', 1)
    io = _col(names, 'ox/red', 2)
    iq = _col(names, 'capacity', 3)
    cyc = d[:, ic].astype(int)
    out = []
    for c in range(1, cyc.max() + 1):
        m = (cyc == c) & (d[:, io] == 0)
        out.append(d[m, iq].max() if m.sum() > 10 else np.nan)
    return np.array(out, float)


def eis_scalars(path):
    names, d = _read(path)
    ic = _col(names, 'cyclenumber', 1)
    iF = _col(names, 'freq', 2)
    iR = _col(names, 're(z)', 3)
    iI = _col(names, '-im(z)', 4)
    cyc = d[:, ic].astype(int)
    rows = {}
    for c in np.unique(cyc):
        m = cyc == c
        f, re_z, im_z = d[m, iF], d[m, iR], d[m, iI]
        o = np.argsort(-f)
        f, re_z, im_z = f[o], re_z[o], im_z[o]
        s = np.where(np.diff(np.sign(im_z)) > 0)[0]
        if len(s):
            i = s[0]
            w = im_z[i + 1] - im_z[i]
            r_ohm = (re_z[i] + (re_z[i + 1] - re_z[i]) * (-im_z[i] / w)
                     if abs(w) > 1e-12 else re_z[i])
        else:
            r_ohm = re_z[int(np.argmin(np.abs(im_z)))]
        rows[int(c)] = (float(r_ohm), float(re_z[-1]), float(re_z[-1] - r_ohm))
    return rows


if __name__ == '__main__':
    import glob
    CAP = r'D:\claudecode\generationbound\data\zhang\Capacity_data\Capacity data'
    EIS = r'D:\claudecode\generationbound\data\zhang\EIS_data\EIS data'
    for p in sorted(glob.glob(os.path.join(CAP, '*.txt'))):
        nm = re.search(r'Data_Capacity_(\w+)\.txt', p).group(1)
        q = discharge_capacity(p)
        q = q[np.isfinite(q)]
        ep = os.path.join(EIS, f'EIS_state_IX_{nm}.txt')
        e = eis_scalars(ep) if os.path.exists(ep) else {}
        ks = sorted(e)
        print(f'{nm}: {len(q):4d} cycles, capacity {q[0]:6.2f} -> {q[-1]:6.2f} mAh '
              f'({100*q[-1]/q[0]:5.1f}% of initial); EIS cycles {len(ks)}, '
              f'R_ohm {e[ks[0]][0]:.3f} -> {e[ks[-1]][0]:.3f}' if ks else
              f'{nm}: {len(q):4d} cycles, capacity {q[0]:6.2f} -> {q[-1]:6.2f} mAh; no EIS')
