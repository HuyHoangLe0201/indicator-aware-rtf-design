"""
Independent replication of the bearing result on PRONOSTIA / FEMTO-ST.

Section VII-E reports that Upsilon is not measurable on PRONOSTIA: swept over the smoothing
width it drifts four-fold on RMS and over six orders of magnitude on kurtosis.  That is a
statement about Upsilon, which is a terminal derivative.  Section VII-D argues that the
integral statistic c_50 is the stable one on noisy bearing data, so the obvious test is
whether c_50 survives on PRONOSTIA even though Upsilon does not, and whether it agrees with
the XJTU-SY values.

If it does, the bearing verdict rests on two rigs rather than one, and the failure of
Section VII-E is localised to the statistic rather than to the dataset.

PRONOSTIA: 17 bearings (6 learning + 11 test), three operating conditions, horizontal
accelerometer at 25.6 kHz, 2560 samples per snapshot, one snapshot every 10 s.
"""
import numpy as np, glob, os, sys
from scipy.ndimage import uniform_filter1d

ROOT = (r'D:\bách khoa Đà Nẵng\nghiencuukhoahoc\baotri\data'
        r'\phm-ieee-2012-data-challenge-dataset-master')
FS = 25600.0
STEP = 3                      # keep every third snapshot; c_50 needs shape, not resolution
FRACS = (0.03, 0.07, 0.15)    # the same smoothing sweep as Table III
IND = ['RMS', 'peak', 'kurtosis', 'HF 2-5kHz', 'HF 5-10kHz', 'spec centroid']


def snapshots(bd):
    fs = glob.glob(os.path.join(bd, 'acc_*.csv'))
    fs.sort(key=lambda p: int(''.join(c for c in os.path.basename(p) if c.isdigit()) or 0))
    return fs[::STEP]


def features(bd):
    out = {k: [] for k in IND}
    for fp in snapshots(bd):
        try:
            a = np.loadtxt(fp, delimiter=',', usecols=4)
        except Exception:
            try:
                a = np.loadtxt(fp, delimiter=';', usecols=4)
            except Exception:
                continue
        if len(a) < 2048:
            continue
        a = a - a.mean(); sd = a.std() + 1e-12
        P = np.abs(np.fft.rfft(a))**2
        f = np.fft.rfftfreq(len(a), 1/FS)
        tot = P.sum() + 1e-12
        out['RMS'].append(np.sqrt(np.mean(a**2)))
        out['peak'].append(np.max(np.abs(a)))
        out['kurtosis'].append(np.mean((a/sd)**4))
        out['HF 2-5kHz'].append(P[(f > 2000) & (f < 5000)].sum()/tot)
        out['HF 5-10kHz'].append(P[(f > 5000) & (f < 10000)].sum()/tot)
        out['spec centroid'].append((f*P).sum()/tot)
    return {k: np.asarray(v, float) for k, v in out.items()}


def c50(h, frac):
    """Fraction of life by which half the extrapolation information has arrived.

    Identical to xjtu_survey.profile: normalise, smooth, differentiate, integrate.  No
    monotone or cumulative-maximum transform is applied -- that transform is what produced
    the spurious 0.00 reported in Section VII-E.
    """
    h = np.asarray(h, float); h = h[np.isfinite(h)]
    n = len(h)
    if n < 60:
        return np.nan
    s = uniform_filter1d(h, max(3, int(frac*n)), mode='nearest')
    base, end = np.median(s[:max(2, n//20)]), np.median(s[-max(2, n//50):])
    if abs(end - base) < 1e-12:
        return np.nan
    D = (s - base)/(end - base)
    tau = (np.arange(n) + 1)/n
    g = np.linspace(1e-3, 1.0, 400)
    dp = np.gradient(np.interp(g, tau, D), g)
    dens = g**2*dp**2
    cum = np.concatenate([[0], np.cumsum((dens[1:] + dens[:-1])/2*np.diff(g))])
    if cum[-1] <= 0:
        return np.nan
    return float(np.interp(0.5, cum/cum[-1], g))


if __name__ == '__main__':
    beds = []
    for setn in ('Learning_set', 'Full_Test_Set'):
        beds += sorted(glob.glob(os.path.join(ROOT, setn, 'Bearing*')))
    print(f'{len(beds)} PRONOSTIA bearings')

    F = {}
    for bd in beds:
        nm = os.path.basename(bd)
        F[nm] = features(bd)
        print(f'  {nm:12s} {len(F[nm]["RMS"]):5d} snapshots', flush=True)
    F = {k: v for k, v in F.items() if len(v['RMS']) >= 60}
    print(f'{len(F)} bearings with enough snapshots')

    XJTU = {'RMS': 0.963, 'peak': 0.961, 'kurtosis': 0.903,
            'HF 2-5kHz': 0.910, 'HF 5-10kHz': 0.821, 'spec centroid': 0.853}
    print()
    hdr = (f'{"indicator":15s} ' + ' '.join(f'{("c50 "+str(f)):>9s}' for f in FRACS)
           + f' {"drift":>7s} {"XJTU 7%":>8s}')
    print(hdr); print('-'*len(hdr))
    store = {}
    for ind in IND:
        row = []
        for fr in FRACS:
            v = np.array([c50(F[b][ind], fr) for b in F])
            row.append(np.nanmedian(v))
        row = np.array(row)
        store[ind] = row
        drift = np.nanmax(row)/max(np.nanmin(row), 1e-9)
        print(f'{ind:15s} ' + ' '.join(f'{x:9.3f}' for x in row)
              + f' {drift:6.2f}x {XJTU[ind]:8.3f}')

    np.savez(os.path.join(os.path.dirname(__file__), 'pronostia_c50.npz'),
             **{k: v for k, v in store.items()}, n=len(F))
    print()
    print('agreement with XJTU-SY at the 7% width:')
    d = [abs(store[i][1] - XJTU[i]) for i in IND if np.isfinite(store[i][1])]
    print(f'  mean |difference| = {np.mean(d):.3f}, max = {np.max(d):.3f}')
