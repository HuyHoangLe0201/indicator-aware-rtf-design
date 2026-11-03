"""
Escape-channel screen on bearings: 12 XJTU-SY plus 17 PRONOSTIA units, two rigs.

Same question as escape_screen.py.  RMS is the damage channel -- it is what a vibration
analyst watches and what the ISO limits are written against -- and the other five indicators
are candidates.  If any of them carried information about the failure time beyond RMS, early,
it would be a bearing escape channel and the paper's corner-only verdict for this domain would
have to be qualified.

Power is the honest limit here.  With 29 units and a four-feature control the smallest
partial rank correlation resolvable at p=0.05 is about 0.5 (0.51 for 29 units against the 14 columns of the quadratic design), so this screen can rule out an
effect of the size the battery escape channel produces (0.80) but not a small one.  The
resolvable size is printed with every result.

Features are cached on first run; the two rigs together are several thousand CSV snapshots.
"""
import numpy as np, glob, os, re

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bearing_feats.npz')
IND = ['RMS', 'peak', 'kurtosis', 'HF 2-5kHz', 'HF 5-10kHz', 'spec centroid']
FS = 25600.0
XJTU = ('D:\\bách khoa Đà Nẵng\\nghiencuukhoahoc\\baotri\\data\\XJTU-SY bearing dataset'
        '\\Data\\extracted\\XJTU-SY_Bearing_Datasets')
PRON = ('D:\\bách khoa Đà Nẵng\\nghiencuukhoahoc\\baotri\\data'
        '\\phm-ieee-2012-data-challenge-dataset-master')


def feats_from(files, loader):
    out = {k: [] for k in IND}
    for fp in files:
        a = loader(fp)
        if a is None or len(a) < 2048:
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


def load_xjtu(fp):
    try:
        return np.loadtxt(fp, delimiter=',', skiprows=1, usecols=0)
    except Exception:
        return None


def load_pron(fp):
    for sep in (',', ';'):
        try:
            return np.loadtxt(fp, delimiter=sep, usecols=4)
        except Exception:
            continue
    return None


def build():
    recs = {}
    for bd in sorted(glob.glob(os.path.join(XJTU, '*', 'Bearing*'))):
        fs = sorted(glob.glob(os.path.join(bd, '*.csv')),
                    key=lambda p: int(re.findall(r'(\d+)\.csv$', p)[0]))[::2]
        d = feats_from(fs, load_xjtu)
        if len(d['RMS']) >= 30:
            recs['X:' + os.path.basename(os.path.dirname(bd)) + '/' + os.path.basename(bd)] = d
            print('  xjtu', os.path.basename(bd), len(d['RMS']), flush=True)
    for bd in sorted(glob.glob(os.path.join(PRON, '*', 'Bearing*'))):
        fs = glob.glob(os.path.join(bd, 'acc_*.csv'))
        fs.sort(key=lambda p: int(''.join(c for c in os.path.basename(p) if c.isdigit()) or 0))
        d = feats_from(fs[::3], load_pron)
        if len(d['RMS']) >= 30:
            recs['P:' + os.path.basename(bd)] = d
            print('  pron', os.path.basename(bd), len(d['RMS']), flush=True)
    flat = {}
    for name, d in recs.items():
        for k in IND:
            flat[name + '|' + k] = d[k]
    np.savez_compressed(CACHE, **flat)
    print(f'cached {len(recs)} bearings to {CACHE}', flush=True)


if __name__ == '__main__':
    if not os.path.exists(CACHE):
        print('building feature cache...', flush=True)
        build()
    z = np.load(CACHE, allow_pickle=True)
    names = sorted({k.split('|')[0] for k in z.files})
    units = []
    for n in names:
        rec = {k: np.asarray(z[n + '|' + k], float) for k in IND}
        rec['T'] = float(len(rec['RMS']))
        rec['rig'] = n[0]
        units.append(rec)
    print(f'{len(units)} bearings: '
          f'{sum(u["rig"]=="X" for u in units)} XJTU-SY, '
          f'{sum(u["rig"]=="P" for u in units)} PRONOSTIA')
    print('lives (snapshots): ' + ', '.join(str(int(u['T'])) for u in units))

    from escape_screen import screen
    cands = [k for k in IND if k != 'RMS']
    screen('Bearings, both rigs', units, 'RMS', cands, probes=(20, 40, 60))
    screen('Bearings, XJTU-SY only', [u for u in units if u['rig'] == 'X'],
           'RMS', cands, probes=(20, 40))
    screen('Bearings, PRONOSTIA only', [u for u in units if u['rig'] == 'P'],
           'RMS', cands, probes=(20, 40))
