"""
Decisive test: do two health indicators of the SAME cells distribute their information
about cycle life differently?

  channel A (Paper 1's):  bulk discharge capacity at the end of the window
  channel B (Severson's): log10 Var[ Q_m(V) - Q_10(V) ],  the DeltaQ(V) signature

For each observation fraction c, the window is cycles [1, ceil(c*T_i)] of cell i.  A
feature is built from that window, log T is regressed on it across cells (leave-one-out),
and the empirical information fraction is

    eta_emp(c) = [Var(logT) - MSE(c)] / [Var(logT) - MSE(1)]

the direct analogue of eta_T(c).  Upsilon_emp = d eta_emp / dc at c = 1.

Continuation cells (b1c0-4, which resume in batch 2) are skipped: 119 of 124 cells.
"""
import numpy as np, h5py, os

HERE = os.path.join(os.path.dirname(__file__), '..', 'data', 'severson')
FILES = ['2017-05-12_batchdata_updated_struct_errorcorrect.mat',
         '2017-06-30_batchdata_updated_struct_errorcorrect.mat',
         '2018-04-12_batchdata_updated_struct_errorcorrect.mat']
DROP = {0: [8, 10, 12, 13, 22] + [0, 1, 2, 3, 4], 1: [7, 8, 9, 15, 16],
        2: [37, 2, 23, 32, 38, 39]}
CGRID = [0.10, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 1.00]
REF = 10                                   # reference cycle for DeltaQ

def collect():
    featA, featB, life = [], [], []
    for k, fn in enumerate(FILES):
        f = h5py.File(os.path.join(HERE, fn), 'r'); b = f['batch']
        for i in range(b['summary'].shape[0]):
            if i in DROP[k]:
                continue
            cy = f[b['cycles'][i, 0]]
            s = f[b['summary'][i, 0]]
            qd = np.asarray(s['QDischarge']).ravel().astype(float)
            n = cy['Qdlin'].shape[0]
            T = min(len(qd), n)
            if T < 120:
                continue
            try:
                qref = np.asarray(f[cy['Qdlin'][REF, 0]]).ravel()
            except Exception:
                continue
            if len(qref) != 1000:
                continue
            a_row, b_row, ok = [], [], True
            for c in CGRID:
                m = int(np.clip(round(c*T), REF+5, T-1))
                try:
                    qm = np.asarray(f[cy['Qdlin'][m, 0]]).ravel()
                except Exception:
                    ok = False; break
                if len(qm) != 1000:
                    ok = False; break
                d = qm - qref
                b_row.append(np.log10(max(np.var(d), 1e-12)))
                w = qd[:m+1]; w = w[np.isfinite(w) & (w > .2) & (w < 1.5)]
                a_row.append(float(np.median(w[-5:])) if len(w) >= 5 else np.nan)
            if ok and np.all(np.isfinite(a_row)) and np.all(np.isfinite(b_row)):
                featA.append(a_row); featB.append(b_row); life.append(float(T))
        f.close()
    return np.array(featA), np.array(featB), np.array(life)

def loo_mse(x, y):
    """leave-one-out MSE of simple linear regression of y on x"""
    n = len(y); e = np.empty(n)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        A = np.vstack([x[m], np.ones(m.sum())]).T
        cf = np.linalg.lstsq(A, y[m], rcond=None)[0]
        e[i] = y[i] - (cf[0]*x[i] + cf[1])
    return float(np.mean(e**2))

if __name__ == '__main__':
    A, B, T = collect()
    y = np.log(T)
    print(f'cells used: {len(T)}   mean life {T.mean():.0f}   Var(log T) = {y.var():.4f}\n')
    out = {}
    for nm, F in [('A capacity', A), ('B DeltaQ(V)', B)]:
        mse = np.array([loo_mse(F[:, j], y) for j in range(len(CGRID))])
        expl = y.var() - mse
        eta = expl/expl[-1]
        out[nm] = (mse, eta)
        print(f'--- channel {nm} ---')
        print(f'{"c":>6s} ' + " ".join(f"{c:7.2f}" for c in CGRID))
        print(f'{"LOO R2":>6s} ' + " ".join(f"{1-m/y.var():7.3f}" for m in mse))
        print(f'{"eta_emp":>6s} ' + " ".join(f"{e:7.3f}" for e in eta))
        dc = CGRID[-1]-CGRID[-2]
        print(f'  Upsilon_emp = d eta/dc at c=1 : {(eta[-1]-eta[-2])/dc:.2f}')
        print(f'  eta_emp(0.15) = {eta[1]:.3f}   (fraction of information by 15% of life)\n')
    print('Paper 1 model on capacity: eta_T(0.15) = %.2e' % (0.15**9.97))
