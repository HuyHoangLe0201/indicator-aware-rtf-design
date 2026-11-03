"""Common loader for the BatteryLife sets: per-cycle capacity and DeltaQ(V) in the format of the Severson arrays.

Reads the channels written by batterylife_extract.py (BL_CH: directory of <set>.npy) and the life labels (BL_LAB).
Returns QD (capacity per cycle), DQ ((cycle, value) pairs of the cycles with a DeltaQ(V) reading), life (cycles), and the
set's failure fraction f (median over cells of the capacity loss at the labelled life, relative to the initial capacity),
which fixes the damage scale D = (q0 - q)/(f q0), so that D = 1 at the failure the label refers to.
"""
import os, json
import numpy as np

CH = os.environ.get('BL_CH', r'D:\claudecode\generationbound\data\_work\channels')
LAB = os.environ.get('BL_LAB', r'D:\claudecode\generationbound\data\batterylife_zenodo_19688272\_meta\Life labels')
MINLIFE = 100


def load_set(name, min_dq=60):
    res = np.load(os.path.join(CH, name + '.npy'), allow_pickle=True).item()
    lab = json.load(open(os.path.join(LAB, name + '_labels.json')))
    QD, DQ, life, keys = [], [], [], []
    for member, rec in res.items():
        if 'err' in rec:
            continue
        key = os.path.basename(member)
        L = lab.get(key, lab.get(key.replace('--', '-#')))
        if L is None or L < MINLIFE:
            continue
        cap = np.asarray(rec['cap'], float)
        K = len(cap)
        ok = np.isfinite(cap) & (cap > 0)
        if ok.sum() < 50:
            continue
        idx = np.arange(K)
        cap = np.interp(idx, idx[ok], cap[ok])
        dq = np.asarray(rec['dq'], float)
        g = np.isfinite(dq)
        g[:11] = False
        if g.sum() < min_dq:
            continue
        cyc = (idx[g] + 1).astype(float)
        QD.append(cap); DQ.append((cyc, dq[g])); life.append(float(L)); keys.append(key)
    life = np.array(life)
    # failure fraction of the set, from the capacity loss at the labelled life
    fr = []
    for q, L in zip(QD, life):
        q0 = np.median(q[1:6]); k = int(min(L, len(q))) - 1
        fr.append((q0 - q[k])/q0)
    f = float(np.median(fr)) if len(fr) else np.nan
    return QD, DQ, life, f, keys


def damage_scaled(q, f):
    q = np.asarray(q, float)
    q0 = np.nanmedian(q[1:6])
    return (q0 - q)/(f*q0 + 1e-12)


if __name__ == '__main__':
    for s in ('HNEI', 'CALCE', 'CALB', 'SNL', 'MICH', 'XJTU', 'HUST', 'RWTH', 'Stanford', 'Tongji'):
        QD, DQ, life, f, keys = load_set(s)
        n100 = sum(1 for i in range(len(life)) if life[i] >= 100 and DQ[i][0][-1] >= 100)
        print(f'{s:9s} cells {len(life):3d}  life {life.min():5.0f}/{np.median(life):5.0f}/{life.max():5.0f}  '
              f'failure fraction f={f:.3f}  reach 100 cycles with dQ(V): {n100}')
