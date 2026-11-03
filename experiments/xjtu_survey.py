"""
Bearing Upsilon survey on XJTU-SY (15 bearings, 3 operating conditions).

Primary statistic is c50: the fraction of life by which half the extrapolation
information has arrived,
      eta_T(c) = int_0^c tau^2 D'^2 / int_0^1 tau^2 D'^2 ,     c50 : eta_T(c50) = 0.5.
It is an integral, so it is far more stable than Upsilon = D'(1)^2 / int tau^2 D'^2,
which hinges on the terminal derivative -- precisely where a failing bearing is noisiest.
c50 is also directly comparable with Paper 1's claim that bearing RMS becomes
discriminative only after 60-80% of life while frequency-domain indicators do so by ~40%.

Upsilon is reported as secondary.  Both are swept over the smoothing width.
"""
import numpy as np, os, glob, re
from scipy.ndimage import uniform_filter1d

ROOT = ('D:\\bách khoa Đà Nẵng\\nghiencuukhoahoc\\baotri\\data\\XJTU-SY bearing dataset'
        '\\Data\\extracted\\XJTU-SY_Bearing_Datasets')
FS, STEP = 25600.0, 2

def snapshots(bd):
    fs = glob.glob(os.path.join(bd, '*.csv'))
    fs.sort(key=lambda p: int(re.findall(r'(\d+)\.csv$', p)[0]))
    return fs[::STEP]

def features(bd):
    out = {k: [] for k in ['RMS', 'kurtosis', 'peak',
                           'HF 2-5kHz', 'HF 5-10kHz', 'spec centroid']}
    for fp in snapshots(bd):
        try:
            a = np.loadtxt(fp, delimiter=',', skiprows=1, usecols=0)
        except Exception:
            continue
        if len(a) < 4096:
            continue
        a = a - a.mean(); sd = a.std() + 1e-12
        P = np.abs(np.fft.rfft(a))**2
        f = np.fft.rfftfreq(len(a), 1/FS)
        tot = P.sum() + 1e-12
        out['RMS'].append(np.sqrt(np.mean(a**2)))
        out['kurtosis'].append(np.mean((a/sd)**4))
        out['peak'].append(np.max(np.abs(a)))
        out['HF 2-5kHz'].append(P[(f > 2000) & (f < 5000)].sum()/tot)
        out['HF 5-10kHz'].append(P[(f > 5000) & (f < 10000)].sum()/tot)
        out['spec centroid'].append((f*P).sum()/tot)
    return {k: np.asarray(v, float) for k, v in out.items()}

def profile(h, frac):
    """returns (c50, Upsilon) from the normalised, smoothed indicator trajectory"""
    h = np.asarray(h, float); h = h[np.isfinite(h)]
    n = len(h)
    if n < 60: return np.nan, np.nan
    s = uniform_filter1d(h, max(3, int(frac*n)), mode='nearest')
    base, end = np.median(s[:max(2, n//20)]), np.median(s[-max(2, n//50):])
    if abs(end-base) < 1e-12: return np.nan, np.nan
    D = (s-base)/(end-base)
    tau = (np.arange(n)+1)/n
    g = np.linspace(1e-3, 1.0, 400)
    Dg = np.interp(g, tau, D)
    dp = np.gradient(Dg, g)
    dens = g**2*dp**2
    cum = np.concatenate([[0], np.cumsum((dens[1:]+dens[:-1])/2*np.diff(g))])
    if cum[-1] <= 0: return np.nan, np.nan
    eta = cum/cum[-1]
    c50 = float(np.interp(0.5, eta, g))
    return c50, float(dens[-1]/cum[-1])

if __name__ == '__main__':
    bds = sorted(glob.glob(os.path.join(ROOT, '*', 'Bearing*')))
    print(f'bearings found: {len(bds)}')
    F = {}
    for bd in bds:
        nm = os.path.basename(bd)
        F[nm] = features(bd)
        print(f'  {nm:12s} snapshots used: {len(F[nm]["RMS"]):5d}')
    FR = [0.03, 0.07, 0.15]
    print(f'\n{"indicator":15s} ' + " ".join(f"{('c50 s='+str(f)):>12s}" for f in FR)
          + f' {"stable":>7s} {"Ups(med)":>9s}')
    print('-'*72)
    for ind in ['RMS', 'peak', 'kurtosis', 'HF 2-5kHz', 'HF 5-10kHz', 'spec centroid']:
        c50s, ups = [], []
        for fr in FR:
            r = [profile(F[b][ind], fr) for b in F]
            c50s.append(np.nanmedian([x[0] for x in r]))
            ups.append(np.nanmedian([x[1] for x in r]))
        c50s = np.array(c50s)
        sp = np.nanmax(c50s)/max(np.nanmin(c50s), 1e-9)
        print(f'{ind:15s} ' + " ".join(f"{c:12.3f}" for c in c50s)
              + f' {("yes" if sp < 1.5 else f"{sp:.1f}x"):>7s} {np.nanmedian(ups):9.2f}')
    print('\nPaper 1: RMS discriminative only after 60-80% of life;')
    print('         frequency-domain indicators by ~40%.   (compare the c50 columns)')
