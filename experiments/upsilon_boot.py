import numpy as np, os
from scipy.signal import savgol_filter
exec(open('upsilon_eff_real.py').read().split('rows=[]')[0])
rng=np.random.default_rng(0)
def curves(series_list):
    cur=[]
    for h in series_list:
        h=np.asarray(h,float); h=h[np.isfinite(h)]
        if len(h)<30: continue
        n=len(h); base=np.median(h[:max(3,n//20)]); end=np.mean(h[-3:])
        if abs(end-base)<1e-12: continue
        z=(h-base)/(end-base); tau=np.arange(1,n+1)/n
        cur.append(np.interp(G,tau,savgol_filter(z,min(21,n//3*2+1),2)))
    return np.array(cur)
def boot(cur,B=300):
    r=[]
    for _ in range(B):
        M=cur[rng.integers(0,len(cur),len(cur))].mean(0)
        try: a,b,_=ups(M); r.append((a,b))
        except Exception: pass
    r=np.array(r); return np.percentile(r,[2.5,50,97.5],axis=0)
f=np.load('bearing_feats.npz')
print('XJTU bootstrap over units (95% CI, pinned | amplitude-free), threshold 2.27')
for ind in ("RMS","peak","HF 2-5kHz","kurtosis"):
    c=curves([f[k] for k in f.files if k.startswith("X:") and k.endswith("|"+ind)])
    p=boot(c); print(f'{ind:10s} n={len(c)}  pinned {p[0,0]:.2f}/{p[1,0]:.2f}/{p[2,0]:.2f} | free {p[0,1]:.2f}/{p[1,1]:.2f}/{p[2,1]:.2f}')
ROOT='D:/bách khoa Đà Nẵng/nghiencuukhoahoc/baotri/data'
print('C-MAPSS FD001 bootstrap, threshold 1.11')
d=np.loadtxt(os.path.join(ROOT,'train_FD001.txt')); uid=d[:,0].astype(int)
for nm,col in (("s2",6),("s4",8),("s7",11)):
    c=curves([d[uid==u][:,col] for u in np.unique(uid)]); p=boot(c,200)
    print(f'{nm:10s} n={len(c)}  pinned {p[0,0]:.2f}/{p[1,0]:.2f}/{p[2,0]:.2f} | free {p[0,1]:.2f}/{p[1,1]:.2f}/{p[2,1]:.2f}')
