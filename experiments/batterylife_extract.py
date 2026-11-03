"""Per-cycle capacity and DeltaQ(V) channels for the BatteryLife sets (restricted unpickler, see ru.py)."""
import sys, os, json, zipfile, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
import batterylife_ru as ru

ZD = r'D:\claudecode\generationbound\data\batterylife_zenodo_19688272'
LAB = os.path.join(ZD, '_meta', 'Life labels')
OUT = os.environ.get('BL_CH', r'D:\claudecode\generationbound\data\_work\channels')
REF = 10


def one(args):
    ds, member = args
    try:
        d = ru.load(os.path.join(ZD, ds + '.zip'), member)
        cd = d['cycle_data']
        vmin = d.get('min_voltage_limit_in_V') or 2.5
        vmax = d.get('max_voltage_limit_in_V') or 4.2
        lo, hi = vmin + 0.15*(vmax - vmin), vmax - 0.30*(vmax - vmin)
        grid = np.linspace(lo, hi, 200)
        K = len(cd)
        cap = np.full(K, np.nan); dq = np.full(K, np.nan); cyc = np.zeros(K, int)
        qref = None
        for k, c in enumerate(cd):
            cyc[k] = c['cycle_number']
            q = np.asarray(c['discharge_capacity_in_Ah'], float)
            v = np.asarray(c['voltage_in_V'], float)
            if q.size < 30:
                continue
            cap[k] = np.nanmax(q)
            m = np.r_[False, np.diff(q) > 0] & (q > 0)
            if m.sum() < 30:
                continue
            vv, qq = v[m], q[m]
            if vv.min() > lo or vv.max() < hi:
                continue
            o = np.argsort(vv); vv, qq = vv[o], qq[o]
            qv = np.interp(grid, vv, qq)
            if k <= REF:
                qref = qv; continue
            if qref is None:
                continue
            x = np.var(qv - qref)
            if np.isfinite(x) and x > 0:
                dq[k] = np.log10(x)
        return member, dict(cap=cap, dq=dq, cyc=cyc, nom=d.get('nominal_capacity_in_Ah'))
    except Exception as e:
        return member, dict(err=repr(e))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for ds in sys.argv[1:]:
        with zipfile.ZipFile(os.path.join(ZD, ds + '.zip')) as z:
            mem = [n for n in z.namelist() if n.endswith('.pkl')]
        with Pool(6) as p:
            res = dict(p.map(one, [(ds, m) for m in mem], chunksize=1))
        np.save(os.path.join(OUT, ds + '.npy'), res, allow_pickle=True)
        bad = [k for k, v in res.items() if 'err' in v]
        print(ds, len(res), 'cells;', len(bad), 'errors', bad[:2], [res[b]['err'] for b in bad[:2]], flush=True)
