"""
Per-cycle trajectory of the DeltaQ(V) signature, for the same cells as severson_traj.npz.

The DeltaQ(V) indicator used throughout the paper is log10 Var[Q_m(V) - Q_ref(V)] with
ref = cycle 10.  e1_features.py evaluates it at the single window m = 100; here we evaluate
it at EVERY cycle m, so the indicator can be plotted as a trajectory alongside bulk capacity
and the two can be compared as curves rather than only as R^2 numbers.

Writes dq_traj.npz:
  DQ    object array, per-cell log10 Var[Q_m - Q_10] against cycle index m
  QD    object array, per-cell discharge capacity against cycle (same cells, same order)
  life  (n,) cycle life
  name  (n,) cell tag
Cell selection and cleaning follow severson_extract.py, minus the batch-1/batch-2
continuation splice: a spliced cell has two different Qdlin bases, so its DeltaQ(V)
trajectory would have a discontinuity that is an artefact of the concatenation.  Those five
cells are dropped here and the count is reported.
"""
import numpy as np, h5py, os

HERE = os.path.join(os.path.dirname(__file__), '..', 'data', 'severson')
FILES = ['2017-05-12_batchdata_updated_struct_errorcorrect.mat',
         '2017-06-30_batchdata_updated_struct_errorcorrect.mat',
         '2018-04-12_batchdata_updated_struct_errorcorrect.mat']
DROP = {0: [8, 10, 12, 13, 22, 0, 1, 2, 3, 4], 1: [7, 8, 9, 15, 16],
        2: [37, 2, 23, 32, 38, 39]}
REF = 10
STRIDE = 5          # evaluate every 5th cycle; Qdlin is 1000 floats per cycle


def run():
    DQ, QD, life, name = [], [], [], []
    for k, fn in enumerate(FILES):
        f = h5py.File(os.path.join(HERE, fn), 'r'); b = f['batch']
        for i in range(b['summary'].shape[0]):
            if i in DROP[k]:
                continue
            cy = f[b['cycles'][i, 0]]; s = f[b['summary'][i, 0]]
            qd = np.asarray(s['QDischarge']).ravel().astype(float)
            cl = float(np.asarray(f[b['cycle_life'][i, 0]]).ravel()[0])
            T = min(len(qd), cy['Qdlin'].shape[0])
            if T <= REF + 20:
                continue
            try:
                q0 = np.asarray(f[cy['Qdlin'][REF, 0]]).ravel()
            except Exception:
                continue
            if len(q0) != 1000:
                continue
            ms, vs = [], []
            for m in range(REF + STRIDE, T, STRIDE):
                try:
                    qm = np.asarray(f[cy['Qdlin'][m, 0]]).ravel()
                except Exception:
                    continue
                if len(qm) != 1000:
                    continue
                v = np.var(qm - q0)
                if not np.isfinite(v) or v <= 0:
                    continue
                ms.append(m); vs.append(np.log10(v))
            if len(ms) < 20:
                continue
            DQ.append(np.array([ms, vs]))
            QD.append(qd[:T])
            life.append(cl); name.append(f'b{k+1}c{i}')
        f.close()
        print(f'{fn[:10]}: running total {len(DQ)} cells')

    def boxed(seq):                     # ragged -> object array, without broadcasting
        a = np.empty(len(seq), dtype=object)
        for j, v in enumerate(seq):
            a[j] = v
        return a

    out = os.path.join(HERE, 'dq_traj.npz')
    np.savez(out, DQ=boxed(DQ), QD=boxed(QD),
             life=np.array(life, float), name=np.array(name))
    print(f'wrote {out}: {len(DQ)} cells, life {min(life):.0f}-{max(life):.0f}')


if __name__ == '__main__':
    run()
