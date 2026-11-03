"""
Extract per-cell degradation trajectories from the three raw Severson batches.

Keeps only what the design analysis needs: discharge capacity vs cycle, and cycle
life.  Applies the documented cleaning from Severson et al. (2019):
  * batch 1: drop b1c8, b1c10, b1c12, b1c13, b1c22
  * batch 2: cells b2c7,8,9,15,16 are continuations of b1c0,1,2,3,4 -- concatenate
  * batch 3: drop b3c37, b3c2, b3c23, b3c32, b3c38, b3c39
yielding 41 + 43 + 40 = 124 cells.

Writes severson_traj.npz:  QD (object array of per-cell capacity curves), life (124,)
"""
import numpy as np, h5py, os, sys

HERE = os.path.join(os.path.dirname(__file__), '..', 'data', 'severson')
FILES = ['2017-05-12_batchdata_updated_struct_errorcorrect.mat',
         '2017-06-30_batchdata_updated_struct_errorcorrect.mat',
         '2018-04-12_batchdata_updated_struct_errorcorrect.mat']
DROP = {0: [8, 10, 12, 13, 22],
        1: [],
        2: [37, 2, 23, 32, 38, 39]}
CONT = {7: 0, 8: 1, 9: 2, 15: 3, 16: 4}      # batch2 index -> batch1 index

def load(path):
    f = h5py.File(path, 'r'); b = f['batch']; n = b['summary'].shape[0]
    out = []
    for i in range(n):
        s = f[b['summary'][i, 0]]
        qd = np.asarray(s['QDischarge']).ravel().astype(float)
        cl = float(np.asarray(f[b['cycle_life'][i, 0]]).ravel()[0])
        out.append((qd, cl))
    f.close()
    return out

if __name__ == '__main__':
    raw = []
    for k, fn in enumerate(FILES):
        p = os.path.join(HERE, fn)
        raw.append(load(p))
        print(f'{fn[:10]}: {len(raw[-1])} cells')

    cells = []
    # batch 1, with batch-2 continuations appended
    for i, (qd, cl) in enumerate(raw[0]):
        if i in DROP[0]:
            continue
        j = next((a for a, b_ in CONT.items() if b_ == i), None)
        if j is not None:
            qd2, cl2 = raw[1][j]
            qd = np.concatenate([qd, qd2]); cl = cl + cl2
        cells.append((qd, cl, f'b1c{i}'))
    # batch 2, excluding the continuation cells
    for i, (qd, cl) in enumerate(raw[1]):
        if i in CONT:
            continue
        cells.append((qd, cl, f'b2c{i}'))
    # batch 3
    for i, (qd, cl) in enumerate(raw[2]):
        if i in DROP[2]:
            continue
        cells.append((qd, cl, f'b3c{i}'))

    QD = np.empty(len(cells), dtype=object)
    life = np.empty(len(cells)); name = []
    for k, (qd, cl, nm) in enumerate(cells):
        QD[k] = qd; life[k] = cl; name.append(nm)
    print(f'\nkept {len(cells)} cells (target 124)')
    print(f'life: mean {life.mean():.1f}  sd {life.std(ddof=1):.1f}  '
          f'CV {life.std(ddof=1)/life.mean():.3f}  range {life.min():.0f}-{life.max():.0f}')
    np.savez_compressed(os.path.join(HERE, 'severson_traj.npz'),
                        QD=QD, life=life, name=np.array(name), allow_pickle=True)
    print('written severson_traj.npz')
