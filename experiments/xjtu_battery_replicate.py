"""
Does the battery crossing replicate on an independent cell set?

The physical rule rests on a single positive instance: on the Severson cells, bulk capacity is
damage-driven and late-concentrated while the DeltaQ(V) signature responds to interphase
chemistry and is early. A reasonable objection is that the crossing is a property of that
dataset -- one chemistry, one fast-charge protocol, one laboratory.

This script repeats the comparison on the XJTU battery cells (Wang et al., Nat. Commun. 2024), which carry full per-cycle voltage
and capacity records, so both channels are computable on the same units. Different chemistry,
different protocol, different laboratory. If the crossing replicates, the positive instance
rests on two independent sets rather than one.

Both channels are estimated exactly as in the paper: capacity by the trajectory-shape
estimator, DeltaQ(V) by log10 Var[Q_m(V) - Q_ref(V)] against a fixed early reference cycle.
"""
import numpy as np, scipy.io as sio, glob, os
from scipy.ndimage import uniform_filter1d

ROOT = (r'D:\bách khoa Đà Nẵng\nghiencuukhoahoc\baotri\data\NASABattery'
        r'\Battery Dataset')
REF = 10                    # reference cycle for the DeltaQ(V) difference
VGRID = np.linspace(3.0, 4.0, 400)
FRACS = (0.03, 0.07, 0.15)


def profile(h, frac, tau=None):
    """(c50, Upsilon) from the normalized, smoothed trajectory -- the paper's estimator."""
    h = np.asarray(h, float); h = h[np.isfinite(h)]
    n = len(h)
    if n < 40:
        return np.nan, np.nan
    s = uniform_filter1d(h, max(3, int(frac*n)), mode='nearest')
    base, end = np.median(s[:max(2, n//20)]), np.median(s[-max(2, n//50):])
    if abs(end - base) < 1e-12:
        return np.nan, np.nan
    D = (s - base)/(end - base)
    tau = (np.arange(n) + 1)/n if tau is None else np.asarray(tau, float)[np.isfinite(np.asarray(h, float))]
    g = np.linspace(1e-3, 1.0, 400)
    dp = np.gradient(np.interp(g, tau, D), g)
    dens = g**2*dp**2
    cum = np.concatenate([[0], np.cumsum((dens[1:] + dens[:-1])/2*np.diff(g))])
    if cum[-1] <= 0:
        return np.nan, np.nan
    return float(np.interp(0.5, cum/cum[-1], g)), float(dens[-1]/cum[-1])


def qofv(cycle):
    """Discharge capacity as a function of voltage, on the common grid."""
    v = np.ravel(cycle['voltage_V']).astype(float)
    i = np.ravel(cycle['current_A']).astype(float)
    q = np.ravel(cycle['capacity_Ah']).astype(float)
    m = i < -1e-6                                   # discharge segment
    if m.sum() < 50:
        return None
    v, q = v[m], q[m]
    o = np.argsort(v)                               # increasing voltage for interpolation
    v, q = v[o], q[o]
    keep = np.concatenate([[True], np.diff(v) > 1e-9])
    v, q = v[keep], q[keep]
    if len(v) < 50 or v[0] > VGRID[0] or v[-1] < VGRID[-1]:
        return None
    return np.interp(VGRID, v, q)


def cell_channels(path):
    """Both channels on one cycle axis, cut at the same end of life.

    Capacity is the per-cycle maximum of the discharge capacity; DeltaQ(V) is log10 Var[Q_k(V) - Q_ref(V)] for cycles after the
    reference. A cycle with no usable record is skipped in its channel and filled by interpolation over the cycle index,
    so a channel never changes length. End of life is the first cycle at 80% of the median initial capacity; both channels are
    cut there. None of the 23 cells reaches that level: the dataset ends each test at 80% of the nominal 2.0 Ah (1.6 Ah), which is
    82-89% of the measured initial capacity, so the end of the record is the dataset's own failure criterion and the channels
    are normalized to the last cycle with a capacity reading.
    Returns the cycle index of each sample so the normalized time of both channels is cycle / end-of-record cycle."""
    m = sio.loadmat(path, simplify_cells=True)
    d = m['data']
    K = len(d)
    cap = np.full(K, np.nan); dq = np.full(K, np.nan)
    qref = None
    for k, c in enumerate(d):
        q = np.ravel(c['capacity_Ah']).astype(float)
        if q.size and np.isfinite(np.nanmax(q)) and np.nanmax(q) > 0:
            cap[k] = np.nanmax(q)
        qv = qofv(c)
        if qv is None:
            continue
        if k <= REF:
            qref = qv
            continue
        if qref is None:
            continue
        v = np.var(qv - qref)
        if np.isfinite(v) and v > 0:
            dq[k] = np.log10(v)
    ok = np.isfinite(cap)
    if ok.sum() < 60:
        return None
    c0 = np.median(cap[ok][:5])
    dead = np.flatnonzero(ok & (cap <= 0.8*c0))
    eol = int(dead[0]) if len(dead) else int(np.flatnonzero(ok)[-1])    # end of life if reached, else end of record
    idx = np.arange(eol + 1)
    capc, dqc = cap[:eol + 1], dq[:eol + 1]
    first = REF + 1
    good_c = np.isfinite(capc); good_d = np.isfinite(dqc)
    if good_c.sum() < 60 or good_d[first:].sum() < 60:
        return None
    capc = np.interp(idx, idx[good_c], capc[good_c])
    sel = np.arange(first, eol + 1)
    dqs = np.interp(sel, idx[good_d], dqc[good_d])
    return dict(reached=bool(len(dead)), capacity=-capc, dqv=dqs, tau_cap=(idx + 1)/(eol + 1), tau_dq=(sel + 1)/(eol + 1), eol=eol + 1)


if __name__ == '__main__':
    paths = []
    for b in ('Batch-1', 'Batch-2'):
        paths += sorted(glob.glob(os.path.join(ROOT, b, '*.mat')))
    cells = {}
    for p in paths:
        try:
            c = cell_channels(p)
        except Exception:
            c = None
        if c:
            cells[os.path.basename(p)] = c
    print(f'{len(cells)} of {len(paths)} XJTU battery cells usable on both channels; {sum(v["reached"] for v in cells.values())} reach 80% of the measured initial capacity (the data set ends each test at 80% of the nominal capacity, 82-89% of the measured one)')

    print()
    print('the rule predicts: capacity LATE (corner), DeltaQ(V) EARLY (interior)')
    print()
    hdr = (f'{"channel":10s} ' + ' '.join(f'{("Ups "+str(f)):>9s}' for f in FRACS)
           + f' {"drift":>7s} {"c50 (7%)":>9s}')
    print(hdr); print('-'*len(hdr))
    res = {}
    for ch, name in (('capacity', 'capacity'), ('dqv', 'DeltaQ(V)')):
        ups, c50 = [], None
        for fr in FRACS:
            vals = [profile(v[ch], fr, v['tau_cap' if ch == 'capacity' else 'tau_dq']) for v in cells.values()]
            ups.append(np.nanmedian([u for _, u in vals]))
            if fr == 0.07:
                c50 = np.nanmedian([c for c, _ in vals])
        ups = np.array(ups)
        drift = np.nanmax(ups)/max(np.nanmin(ups), 1e-9)
        res[ch] = (ups, drift, c50)
        print(f'{name:10s} ' + ' '.join(f'{x:9.3f}' for x in ups)
              + f' {drift:6.1f}x {c50:9.3f}')

    print()
    print('Severson reference: capacity c50 late with Upsilon 9.97; DeltaQ(V) saturates by 0.14')
    np.savez(os.path.join(os.path.dirname(__file__), 'xjtu_battery_replicate.npz'),
             n=len(cells), **{k: np.array(list(v[0]) + [v[1], v[2]]) for k, v in res.items()})
