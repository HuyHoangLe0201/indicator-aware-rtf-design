"""Residual structure on real indicators: lag-1/2 autocorrelation of residuals about a wide Savitzky-Golay trend,
and variance heteroscedasticity (late third / early third). Median over units with IQR. Robustness: two smoother widths."""
import numpy as np
from scipy.signal import savgol_filter
def stats(series,win):
    r1=[];r2=[];hv=[]
    for h in series:
        h=np.asarray(h,float); h=h[np.isfinite(h)]; n=len(h)
        if n<max(80,win+10): continue
        w=win if win%2 else win+1
        e=h-savgol_filter(h,w,3); e=e-e.mean()
        v=np.sum(e*e)
        r1.append(np.sum(e[1:]*e[:-1])/v); r2.append(np.sum(e[2:]*e[:-2])/v)
        k=n//3; hv.append(np.var(e[-k:])/max(np.var(e[:k]),1e-18))
    f=lambda a:(np.median(a),np.percentile(a,25),np.percentile(a,75))
    return len(r1),f(r1),f(r2),f(np.log(hv))
ROOT='D:/bách khoa Đà Nẵng/nghiencuukhoahoc/baotri/data'
d=np.loadtxt(ROOT+'/train_FD001.txt'); uid=d[:,0].astype(int)
f=np.load('bearing_feats.npz')
dq=np.load('../data/severson/dq_traj.npz',allow_pickle=True)
sets={}
for nm,col in (('C-MAPSS s2',6),('C-MAPSS s4',8),('C-MAPSS s7',11)): sets[nm]=[d[uid==u][:,col] for u in np.unique(uid)]
for ind in ('RMS','HF 2-5kHz','kurtosis'): sets['XJTU '+ind]=[f[k] for k in f.files if k.startswith('X:') and k.endswith('|'+ind)]
sets['Severson capacity']=[np.asarray(q,float) for q in dq['QD']]
print(f'{"indicator":20s} {"units":>5s} {"width":>5s}  rho1 med [IQR]         rho2 med [IQR]         ln(var late/early) med [IQR]')
for nm,s in sets.items():
    for win in (31,61):
        n,a,b,c=stats(s,win)
        if n: print(f'{nm:20s} {n:5d} {win:5d}  {a[0]:5.2f} [{a[1]:5.2f},{a[2]:5.2f}]   {b[0]:5.2f} [{b[1]:5.2f},{b[2]:5.2f}]   {c[0]:5.2f} [{c[1]:5.2f},{c[2]:5.2f}]')
