"""
Two health channels, ABSOLUTE-cycle windows (what Severson does, and what a campaign does).

For window end m (a fixed cycle number, same for every cell), predict log(cycle life) from
  channel A  bulk capacity:      Q_discharge at cycle m
  channel B  DeltaQ(V):          log10 Var[ Q_m(V) - Q_10(V) ]
Cells still alive at m are the ones usable at that m; cells that already failed are dropped,
which is exactly the censoring structure of a fixed-t_stop campaign.

Robust (Huber) regression, leave-one-out.  Reports R^2 and the number of usable cells.
"""
import numpy as np, h5py, os
from sklearn.linear_model import HuberRegressor

HERE = os.path.join(os.path.dirname(__file__), '..', 'data', 'severson')
FILES = ['2017-05-12_batchdata_updated_struct_errorcorrect.mat',
         '2017-06-30_batchdata_updated_struct_errorcorrect.mat',
         '2018-04-12_batchdata_updated_struct_errorcorrect.mat']
DROP = {0: [8, 10, 12, 13, 22, 0, 1, 2, 3, 4], 1: [7, 8, 9, 15, 16],
        2: [37, 2, 23, 32, 38, 39]}
MGRID = [20, 50, 100, 200, 400, 700, 1200]
REF = 10

def collect():
    rows = []
    for k, fn in enumerate(FILES):
        f = h5py.File(os.path.join(HERE, fn), 'r'); b = f['batch']
        for i in range(b['summary'].shape[0]):
            if i in DROP[k]: continue
            cy = f[b['cycles'][i, 0]]; s = f[b['summary'][i, 0]]
            qd = np.asarray(s['QDischarge']).ravel().astype(float)
            T = min(len(qd), cy['Qdlin'].shape[0])
            if T < 150: continue
            try:
                qref = np.asarray(f[cy['Qdlin'][REF, 0]]).ravel()
            except Exception: continue
            if len(qref) != 1000: continue
            A, B = {}, {}
            for m in MGRID:
                if m >= T - 2: continue
                try: qm = np.asarray(f[cy['Qdlin'][m, 0]]).ravel()
                except Exception: continue
                if len(qm) != 1000: continue
                B[m] = np.log10(max(np.var(qm-qref), 1e-12))
                w = qd[max(0, m-4):m+1]; w = w[np.isfinite(w) & (w > .2) & (w < 1.5)]
                if len(w): A[m] = float(np.median(w))
            rows.append((T, A, B))
        f.close()
    return rows

def loo_r2(x, y):
    x = np.asarray(x, float).reshape(-1, 1); y = np.asarray(y, float)
    n = len(y); e = np.empty(n)
    for i in range(n):
        msk = np.ones(n, bool); msk[i] = False
        try:
            h = HuberRegressor(epsilon=1.5, max_iter=400).fit(x[msk], y[msk])
            e[i] = y[i] - h.predict(x[i:i+1])[0]
        except Exception:
            e[i] = y[i] - y[msk].mean()
    return 1 - np.mean(e**2)/np.var(y)

if __name__ == '__main__':
    rows = collect()
    print(f'cells: {len(rows)}\n')
    print(f'{"m":>6s} {"alive":>6s} {"R2 capacity":>12s} {"R2 DeltaQ(V)":>13s}  '
          f'{"median c=m/T":>13s}')
    for m in MGRID:
        sub = [(T, A, B) for T, A, B in rows if m in A and m in B]
        if len(sub) < 25: continue
        y = np.log([T for T, _, _ in sub])
        ra = loo_r2([A[m] for _, A, _ in sub], y)
        rb = loo_r2([B[m] for _, _, B in sub], y)
        cfrac = np.median([m/T for T, _, _ in sub])
        print(f'{m:6d} {len(sub):6d} {ra:12.3f} {rb:13.3f}  {cfrac:13.3f}')
    print('\nSeverson variance model (log10 Var dQ_100-10) reported ~15% test error;')
    print('R2 near 0.5-0.7 for a single feature is the expected order.')
