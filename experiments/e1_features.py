"""
Extract Severson-style early-cycle features for all 124 cells, once, and cache them.

Features from cycles <= 100 only (their deployment window):
  f1  log10 Var[ dQ_100-10(V) ]        the variance model of Severson et al.
  f2  log10 |min dQ_100-10(V)|
  f3  log10 |skew dQ_100-10(V)|
  f4  discharge capacity at cycle 2
  f5  discharge capacity at cycle 100
  f6  slope of Q vs cycle over cycles 2-100
  f7  intercept of that fit
Target: log(cycle life).
"""
import numpy as np, h5py, os

HERE = os.path.join(os.path.dirname(__file__), '..', 'data', 'severson')
FILES = ['2017-05-12_batchdata_updated_struct_errorcorrect.mat',
         '2017-06-30_batchdata_updated_struct_errorcorrect.mat',
         '2018-04-12_batchdata_updated_struct_errorcorrect.mat']
DROP = {0: [8, 10, 12, 13, 22, 0, 1, 2, 3, 4], 1: [7, 8, 9, 15, 16],
        2: [37, 2, 23, 32, 38, 39]}
REF, WIN = 10, 100

def run():
    X, y, tag = [], [], []
    for k, fn in enumerate(FILES):
        f = h5py.File(os.path.join(HERE, fn), 'r'); b = f['batch']
        for i in range(b['summary'].shape[0]):
            if i in DROP[k]: continue
            cy = f[b['cycles'][i, 0]]; s = f[b['summary'][i, 0]]
            qd = np.asarray(s['QDischarge']).ravel().astype(float)
            T = min(len(qd), cy['Qdlin'].shape[0])
            if T <= WIN + 5: continue
            try:
                q10 = np.asarray(f[cy['Qdlin'][REF, 0]]).ravel()
                q100 = np.asarray(f[cy['Qdlin'][WIN, 0]]).ravel()
            except Exception:
                continue
            if len(q10) != 1000 or len(q100) != 1000: continue
            d = q100 - q10
            sd = d.std() + 1e-12
            w = qd[:WIN+1].copy()
            w[~np.isfinite(w) | (w < .2) | (w > 1.5)] = np.nan
            idx = np.arange(len(w))
            m = np.isfinite(w)
            if m.sum() < 50: continue
            sl, ic = np.polyfit(idx[m], w[m], 1)
            X.append([np.log10(max(np.var(d), 1e-12)),
                      np.log10(max(abs(d.min()), 1e-12)),
                      np.log10(max(abs(np.mean(((d-d.mean())/sd)**3)), 1e-12)),
                      float(np.nanmedian(w[1:4])), float(np.nanmedian(w[WIN-3:WIN+1])),
                      sl, ic])
            y.append(np.log(T)); tag.append(f'b{k+1}c{i}')
        f.close()
    X, y = np.array(X), np.array(y)
    np.savez(os.path.join(HERE, 'e1_features.npz'), X=X, y=y, tag=np.array(tag))
    print(f'cells: {len(y)}   features: {X.shape[1]}')
    print(f'log-life: mean {y.mean():.3f}  sd {y.std(ddof=1):.3f}')
    print('feature sds:', np.round(X.std(0), 3))
    return X, y

if __name__ == '__main__':
    run()
