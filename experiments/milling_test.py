"""The paper's three tests on the milling set: Upsilon, c_50, and the direction check.

Failure is the first cut at which flank wear reaches the wear limit.  ISO 8688 puts the limit
for uniform flank wear in milling at 0.3 mm and allows larger values for localised wear, and
the runs here were continued past the limit, so the limit is swept rather than fixed: a channel
that only looks early because of where the finish line was drawn is not a finding.

The damage channel is flank wear itself, the quantity the wear limit is written against.  The
candidate escape channels are the six online signals, which come from three physically distinct
transducers: acoustic emission, accelerometers, and spindle motor current.
"""
import numpy as np
from scipy.optimize import least_squares
from milling_load import load, CH

LIMITS = (0.35, 0.45, 0.55)
KEEPS = (0.4, 0.5, 0.7)


def shape(tr):
    """Upsilon exponent of a trajectory normalised to run from 0 to 1 over the campaign."""
    y = np.asarray(tr, float)
    if len(y) < 5:
        return None
    b0, e0 = np.median(y[:2]), np.median(y[-2:])
    if abs(e0 - b0) < 1e-12:
        return None
    D = (y - b0)/(e0 - b0)
    t = (np.arange(len(y)) + 1)/len(y)
    m = (D > -0.4) & (D < 1.6)
    if m.sum() < 5:
        return None
    r = least_squares(lambda p: np.clip(t[m], 1e-9, 1.)**p[0] - D[m], [2.],
                      bounds=([0.02], [20.]), max_nfev=500)
    return float(r.x[0])


def campaign(limit, keep):
    """Trajectories truncated at the wear limit, for every case that reaches it."""
    data = load(keep)
    units = []
    for c, (runs, wear, tr) in sorted(data.items()):
        hit = np.where(wear >= limit)[0]
        if not len(hit) or hit[0] < 4:
            continue
        k = hit[0] + 1
        units.append((c, k, wear[:k], {ch: tr[ch][:k] for ch in CH}))
    return units


def report(limit, keep, verbose=True):
    units = campaign(limit, keep)
    rows = {}
    for ch in ['wear'] + CH:
        vals, signs = [], []
        for _, _, wear, tr in units:
            y = wear if ch == 'wear' else tr[ch]
            signs.append(np.sign(np.median(y[-2:]) - np.median(y[:2])))
            b = shape(y if signs[-1] >= 0 else -y)
            if b:
                vals.append(b)
        if not vals:
            continue
        agree = max(signs.count(1), signs.count(-1))/len(signs)
        ups = 2*float(np.median(vals)) + 1
        c50 = float(np.median([0.5**(1.0/(2*b + 1)) for b in vals]))
        rows[ch] = (ups, c50, agree)
    if verbose:
        print(f'wear limit {limit} mm, interior fraction {keep}: '
              f'{len(units)} cutters, lives {sorted(u[1] for u in units)}')
        print(f'{"channel":13s} {"Ups":>6s} {"c50":>7s} {"dir":>6s}')
        for ch, (u, c, a) in rows.items():
            print(f'{ch:13s} {u:6.2f} {c:7.3f} {a*100:5.0f}%')
        print()
    return rows


if __name__ == '__main__':
    report(0.45, 0.5)
    print('stability of Upsilon across the wear limit and the interior window')
    grid = {(L, K): report(L, K, verbose=False) for L in LIMITS for K in KEEPS}
    print(f'{"channel":13s} ' + ' '.join(f'{("L"+str(L)+"/K"+str(K)):>9s}'
                                         for L in LIMITS for K in KEEPS) + f' {"drift":>7s}')
    for ch in ['wear'] + CH:
        vs = [grid[(L, K)][ch][0] for L in LIMITS for K in KEEPS if ch in grid[(L, K)]]
        if not vs:
            continue
        print(f'{ch:13s} ' + ' '.join(f'{v:9.2f}' for v in vs)
              + f' {max(vs)/min(vs):6.2f}x')
