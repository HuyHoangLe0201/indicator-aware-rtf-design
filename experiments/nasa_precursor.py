"""
Prospective test of the physical rule on a dataset it was not derived from.

The rule says an escape channel needs a different physical process, not merely a different
sensor.  The NASA PCoE battery set is the cleanest available test, because each cell carries
two channels driven by two processes measured on the same unit:

  capacity        the damage variable itself; end of life is defined by it
  Re, Rct         electrolyte and charge-transfer resistance from periodic EIS, driven by
                  SEI growth and electrolyte decomposition, which precede measurable fade

The rule predicts, before looking: capacity should have a large Upsilon and sit at the corner;
the two resistances should have a small Upsilon and sit interior.  Neither the rule nor any
parameter of the paper was fitted to this dataset.

Cells are the standard ambient-temperature set.  End of life follows the NASA convention of
70% of rated capacity; cells that never reach it are carried to their last cycle.
"""
import numpy as np, scipy.io as sio, os, glob
from scipy.ndimage import uniform_filter1d

ROOT = (r'D:\bách khoa Đà Nẵng\nghiencuukhoahoc\baotri\data\NASABattery'
        r'\5. Battery Data Set')
FRACS = (0.03, 0.07, 0.15)
THR = 1.18                      # battery corner threshold, (1+Lambda)/(1+kappa0)


def profile(h, frac):
    """(c50, Upsilon) from the normalized, smoothed trajectory. Same estimator as the paper."""
    h = np.asarray(h, float); h = h[np.isfinite(h)]
    n = len(h)
    if n < 40:
        return np.nan, np.nan
    s = uniform_filter1d(h, max(3, int(frac*n)), mode='nearest')
    base, end = np.median(s[:max(2, n//20)]), np.median(s[-max(2, n//50):])
    if abs(end - base) < 1e-12:
        return np.nan, np.nan
    D = (s - base)/(end - base)
    tau = (np.arange(n) + 1)/n
    g = np.linspace(1e-3, 1.0, 400)
    dp = np.gradient(np.interp(g, tau, D), g)
    dens = g**2*dp**2
    cum = np.concatenate([[0], np.cumsum((dens[1:] + dens[:-1])/2*np.diff(g))])
    if cum[-1] <= 0:
        return np.nan, np.nan
    eta = cum/cum[-1]
    return float(np.interp(0.5, eta, g)), float(dens[-1]/cum[-1])


def load_cell(path):
    key = os.path.splitext(os.path.basename(path))[0]
    m = sio.loadmat(path, simplify_cells=True)
    if key not in m:
        return None
    cyc = m[key]['cycle']
    cap, cap_i, re_, rct, imp_i = [], [], [], [], []
    for i, c in enumerate(cyc):
        t = c.get('type')
        d = c.get('data', {})
        if t == 'discharge' and 'Capacity' in d:
            v = np.ravel(d['Capacity'])
            if v.size:
                cap.append(float(v[0])); cap_i.append(i)
        elif t == 'impedance' and 'Re' in d and 'Rct' in d:
            a, b = np.ravel(d['Re']), np.ravel(d['Rct'])
            if a.size and b.size and np.isfinite(a[0]) and np.isfinite(b[0]):
                re_.append(float(np.real(a[0]))); rct.append(float(np.real(b[0])))
                imp_i.append(i)
    if len(cap) < 40 or len(re_) < 40:
        return None
    cap, cap_i = np.array(cap), np.array(cap_i, float)
    re_, rct, imp_i = np.array(re_), np.array(rct), np.array(imp_i, float)

    # end of life: 70% of the initial capacity, on the discharge index axis
    c0 = np.median(cap[:5])
    dead = np.flatnonzero(cap <= 0.7*c0)
    i_eol = cap_i[dead[0]] if len(dead) else cap_i[-1]

    keep_c = cap_i <= i_eol
    keep_i = imp_i <= i_eol
    if keep_c.sum() < 40 or keep_i.sum() < 40:
        return None
    # capacity falls, resistance rises; the estimator normalises by (end - base) so the
    # direction is handled, but we orient for readability
    return dict(capacity=-cap[keep_c], Re=re_[keep_i], Rct=rct[keep_i],
                n_cap=int(keep_c.sum()), n_imp=int(keep_i.sum()))


if __name__ == '__main__':
    paths = sorted(glob.glob(os.path.join(ROOT, '*', 'B*.mat')))
    seen, cells = set(), {}
    for p in paths:
        k = os.path.basename(p)
        if k in seen:
            continue
        seen.add(k)
        try:
            c = load_cell(p)
        except Exception as e:
            continue
        if c:
            cells[os.path.splitext(k)[0]] = c
    print(f'{len(cells)} NASA cells with both capacity and impedance to end of life')
    for k, v in sorted(cells.items())[:8]:
        print(f'   {k}: {v["n_cap"]} discharge cycles, {v["n_imp"]} impedance sweeps')

    print()
    print('the rule predicts: capacity CORNER (damage), Re and Rct INTERIOR (precursor)')
    print()
    hdr = (f'{"channel":10s} {"process":22s} ' + ' '.join(f'{("Ups "+str(f)):>9s}' for f in FRACS)
           + f' {"median":>8s} {"c50":>6s}  verdict')
    print(hdr); print('-'*len(hdr))
    proc = {'capacity': 'damage (defines EOL)', 'Re': 'precursor (SEI)',
            'Rct': 'precursor (interface)'}
    out = {}
    for ch in ('capacity', 'Re', 'Rct'):
        ups_by_frac, c50s = [], []
        for fr in FRACS:
            vals = [profile(v[ch], fr) for v in cells.values()]
            ups_by_frac.append(np.nanmedian([u for _, u in vals]))
            if fr == 0.07:
                c50s = [c for c, _ in vals]
        med = float(np.nanmedian(ups_by_frac))
        c50 = float(np.nanmedian(c50s))
        out[ch] = (ups_by_frac, med, c50)
        verdict = 'corner' if med >= THR else 'INTERIOR'
        print(f'{ch:10s} {proc[ch]:22s} ' + ' '.join(f'{x:9.3f}' for x in ups_by_frac)
              + f' {med:8.3f} {c50:6.3f}  {verdict}')

    print()
    ok = out['capacity'][1] >= THR and out['Re'][1] < THR and out['Rct'][1] < THR
    print('rule predicted all three channels correctly:', ok)
    np.savez(os.path.join(os.path.dirname(__file__), 'nasa_precursor.npz'),
             n_cells=len(cells), **{k: np.array(v[0] + [v[1], v[2]]) for k, v in out.items()})
