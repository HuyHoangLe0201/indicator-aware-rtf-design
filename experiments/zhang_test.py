"""The paper's own tests, applied to capacity against impedance on the Zhang cells.

Failure is the first cycle at which discharge capacity falls to 80 percent of its settled
initial value.  Two of the twelve cells never reach it and are right-censored, so they are
excluded rather than treated as failures.  Each channel is oriented to increase with damage,
fitted with the same power shape the paper uses, and reported with Upsilon over the smoothing
sweep of Section VII-E and with c_50, which that section argues is the stable statistic.

The battery corner threshold is 1.18.  A channel is an escape channel only if it falls below
that, and only if its Upsilon is admissible, meaning it does not drift materially over the
sweep.
"""
import numpy as np, os, glob, re
from scipy.optimize import least_squares
from scipy.ndimage import uniform_filter1d
from zhang_load import discharge_capacity, eis_scalars

CAP = r'D:\claudecode\generationbound\data\zhang\Capacity_data\Capacity data'
EIS = r'D:\claudecode\generationbound\data\zhang\EIS_data\EIS data'
CH = ['capacity', 'R_ohm', 'R_tot', 'R_ct']
FR = (0.03, 0.07, 0.15)
THRESH = 1.18


def shape(tr, frac):
    y = uniform_filter1d(np.asarray(tr, float), max(3, int(len(tr)*frac)), mode='nearest')
    b = np.median(y[:max(2, len(y)//10)])
    e = np.median(y[-max(2, len(y)//20):])
    if abs(e - b) < 1e-12:
        return None
    D = (y - b)/(e - b)
    t = (np.arange(len(y)) + 1)/len(y)
    m = (D > -0.4) & (D < 1.6)
    if m.sum() < 40:
        return None
    r = least_squares(lambda p: np.clip(t[m], 1e-9, 1.)**p[0] - D[m], [3.],
                      bounds=([0.02], [20.]), max_nfev=500)
    return float(r.x[0])


units, censored = [], []
for p in sorted(glob.glob(os.path.join(CAP, '*.txt'))):
    nm = re.search(r'Data_Capacity_(\w+)', p).group(1)
    ep = os.path.join(EIS, f'EIS_state_IX_{nm}.txt')
    if not os.path.exists(ep):
        continue
    q = discharge_capacity(p)
    e = eis_scalars(ep)
    q0 = np.nanmedian(q[:5])
    below = np.where(np.isfinite(q) & (q <= 0.8*q0))[0]
    if not len(below):
        censored.append((nm, 100*np.nanmin(q)/q0))
        continue
    T = below[0] + 1
    idx = [c for c in range(1, T + 1) if c in e and np.isfinite(q[c-1])]
    if len(idx) < 60:
        censored.append((nm, 100*np.nanmin(q)/q0))
        continue
    tr = {'capacity': np.array([-q[c-1] for c in idx]),
          'R_ohm': np.array([e[c][0] for c in idx]),
          'R_tot': np.array([e[c][1] for c in idx]),
          'R_ct': np.array([e[c][2] for c in idx])}
    units.append((nm, T, tr))

print(f'{len(units)} cells reach 80 percent of initial capacity and are used')
print(f'{len(censored)} right-censored and excluded: '
      f'{", ".join(f"{n} (min {v:.0f} percent)" for n, v in censored)}')
print(f'lives in cycles: {sorted(u[1] for u in units)}')
print()
print(f'{"channel":10s} ' + ' '.join(f'{("Ups s="+str(f)):>11s}' for f in FR)
      + f' {"drift":>7s} {"c50":>7s}  verdict')
print('-'*74)
for ch in CH:
    # does the fleet agree on which way damage moves this channel?
    signs = [np.sign(np.median(u[2][ch][-max(2, len(u[2][ch])//20):])
                     - np.median(u[2][ch][:max(2, len(u[2][ch])//10)])) for u in units]
    agree = max(signs.count(1), signs.count(-1))/len(signs)
    ups, c50s = [], []
    for f in FR:
        bs = [shape(u[2][ch], f) for u in units]
        bs = [b for b in bs if b]
        ups.append(2*float(np.median(bs)) + 1 if bs else np.nan)
    b = shape(units[0][2][ch], FR[1])
    for u in units:
        bb = shape(u[2][ch], FR[1])
        if bb:
            c50s.append(0.5**(1.0/(2*bb + 1)))
    drift = max(ups)/min(ups)
    if agree < 0.75:
        v = f'not a damage channel ({agree*100:.0f} percent agree on direction)'
    elif drift >= 1.9:
        v = f'not a measurement (drifts {drift:.1f}x)'
    else:
        v = 'STOP EARLY' if ups[1] < THRESH else 'run to failure'
    print(f'{ch:10s} ' + ' '.join(f'{u:11.2f}' for u in ups)
          + f' {drift:6.2f}x {np.median(c50s):7.3f} {agree*100:4.0f}%  {v}')
print()
print(f'battery corner threshold {THRESH}; an escape channel must fall below it')
