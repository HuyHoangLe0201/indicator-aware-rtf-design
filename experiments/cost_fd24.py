"""Budget-constrained campaigns (fractional design, as Table 'cmapss') on C-MAPSS FD002 and FD004 (R5.3).
Operating condition removed by standardising every sensor within its operating cluster (as in fd002_fd004.py); sensors that
still vary after that are kept; everything downstream is cost_cmapss.campaign unchanged."""
import sys, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '.'))
import cost_cmapss as cc
ROOT = 'D:/bách khoa Đà Nẵng/nghiencuukhoahoc/baotri/data/CMAPSSData/'

def make_load(fd):
    def load():
        d = np.loadtxt(ROOT + f'train_{fd}.txt'); uid = d[:, 0].astype(int)
        ops = np.round(d[:, 2:5], 1); keys, inv = np.unique(ops, axis=0, return_inverse=True); inv = inv.ravel()
        keep = [j for j in range(5, d.shape[1]) if len(np.unique(d[:, j])) > 2]
        Z = np.empty((len(d), len(keep)))
        for c in range(len(keys)):
            m = inv == c
            Z[m] = (d[m][:, keep] - d[m][:, keep].mean(0)) / (d[m][:, keep].std(0) + 1e-12)
        good = [k for k in range(len(keep)) if Z[:, k].std() > 0.5]
        Z = Z[:, good]
        units = [Z[uid == u] for u in np.unique(uid)]
        life = np.array([len(u) for u in units], float)
        return units, life
    return load

if __name__ == '__main__':
    for fd in ('FD002', 'FD004'):
        cc.load = make_load(fd)
        units, life = cc.load()
        print(f'\n##### {fd}: {len(units)} units, life mean {life.mean():.0f}, min {life.min():.0f}, max {life.max():.0f}', flush=True)
        for m in (60, 100):
            allrows, base0, sel0 = [], None, None
            for s in (0, 1, 2):
                base, rows, sel = cc.campaign(m, s, reps=40)
                allrows.append(rows)
                if base0 is None: base0, sel0 = base, sel
            print(f'=== {fd}, window m={m}: base RMSE {base0:.0f} cycles ===', flush=True)
            print(f'{"c":>5s} {"n elig":>7s} {"n used":>7s} {"label err":>10s}   ' + ' '.join(f'{nm:>14s}' for nm in cc.NAMES))
            for k, c in enumerate(cc.GRID):
                _, ne, nu, ml = sel0[k]
                lab = np.nanmean([r[k][2] for r in allrows])
                vals = [np.nanmean([r[k][1][nm] for r in allrows]) for nm in cc.NAMES]
                print(f'{c:5.2f} {ne:7d} {nu:7d} {lab:10.3f}   ' + ' '.join(f'{v:14.2f}' for v in vals), flush=True)
