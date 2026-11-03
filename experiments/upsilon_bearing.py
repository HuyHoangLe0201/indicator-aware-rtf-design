"""
Bearing Upsilon, done robustly.

The stretched-exponential fit of upsilon_survey.py ran to its bounds for every bearing
indicator (24.98 = 2b+1 at b=12; 0.00 = full saturation), and the parametric and
nonparametric estimates disagreed by up to 6x.  Vibration indicators are noisy and
non-monotone, which breaks both.

Here: enforce monotonicity by cumulative maximum (a health indicator should not recover),
then estimate Upsilon nonparametrically, and SWEEP the smoothing window.  If Upsilon is a
property of the trajectory it must be stable under that sweep, as Pi had to be stable
under the constraint-box sweep.
"""
import numpy as np, os, glob
from scipy.ndimage import uniform_filter1d

ROOT = r'D:\bách khoa Đà Nẵng\nghiencuukhoahoc\baotri\data'
LS = os.path.join(ROOT, 'phm-ieee-2012-data-challenge-dataset-master', 'Learning_set')

def features(bd, step=3):
    fs = sorted(glob.glob(os.path.join(bd, 'acc_*.csv')))[::step]
    rms, kur, pk, hf = [], [], [], []
    for fp in fs:
        try:
            a = np.loadtxt(fp, delimiter=',', usecols=4)
        except Exception:
            continue
        a = a - a.mean(); sd = a.std() + 1e-12
        rms.append(np.sqrt(np.mean(a**2)))
        kur.append(np.mean((a/sd)**4))
        pk.append(np.max(np.abs(a)))
        P = np.abs(np.fft.rfft(a))**2
        f = np.fft.rfftfreq(len(a), 1/25600.)
        hf.append(P[(f > 2000) & (f < 5000)].sum()/(P.sum()+1e-12))
    return dict(zip(['RMS', 'kurtosis', 'peak', 'HF 2-5kHz'],
                    [np.array(x) for x in (rms, kur, pk, hf)]))

def upsilon(h, frac):
    h = np.asarray(h, float); h = h[np.isfinite(h)]
    n = len(h)
    if n < 60: return np.nan
    w = max(3, int(frac*n))
    s = uniform_filter1d(h, w, mode='nearest')
    s = np.maximum.accumulate(s)                       # monotone health indicator
    base, end = s[0], s[-1]
    if end - base < 1e-12: return np.nan
    D = (s-base)/(end-base)
    tau = (np.arange(n)+1)/n
    g = np.linspace(0.02, 1.0, 200)
    Dg = np.interp(g, tau, D)
    dp = np.gradient(Dg, g)
    Z = np.trapezoid(g**2*dp**2, g)
    return dp[-1]**2/Z if Z > 0 else np.nan

if __name__ == '__main__':
    bearings = sorted(glob.glob(os.path.join(LS, 'Bearing*')))
    F = {os.path.basename(b): features(b) for b in bearings}
    print(f'bearings: {list(F)}\n')
    FRACS = [0.02, 0.05, 0.10, 0.20]
    print(f'{"indicator":12s} ' + " ".join(f"{('smooth='+str(f)):>13s}" for f in FRACS)
          + f' {"stable?":>9s}  verdict')
    print('-'*76)
    for ind in ['RMS', 'kurtosis', 'peak', 'HF 2-5kHz']:
        med = []
        for fr in FRACS:
            v = np.array([upsilon(F[b][ind], fr) for b in F], float)
            med.append(np.nanmedian(v))
        med = np.array(med)
        spread = np.nanmax(med)/max(np.nanmin(med), 1e-9)
        tag = 'yes' if spread < 3 else f'NO ({spread:.0f}x)'
        verd = ('INTERIOR' if np.nanmedian(med) < 1.18 else 'corner') if spread < 3 else '--'
        print(f'{ind:12s} ' + " ".join(f"{m:13.2f}" for m in med)
              + f' {tag:>9s}  {verd}')
    print('\nPaper 1 asserts: RMS discriminative only after 60-80% of life;')
    print('frequency-domain indicators earlier (~40%).  Lower Upsilon = earlier information.')
