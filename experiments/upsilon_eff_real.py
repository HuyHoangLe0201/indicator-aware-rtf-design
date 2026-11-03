"""Apply Proposition 6 to real fleets: Upsilon with the terminal amplitude pinned (paper) vs treated as a nuisance.
Fleet-mean normalised curve D(tau) (baseline = median of first 5%, terminal = mean of last 3 samples), smoothed;
Upsilon computed as the secant slope of the normalised cumulant eta over c in [0.9,1]."""
import numpy as np, os, sys
from scipy.signal import savgol_filter
G=np.linspace(0.005,1.0,200); dg=G[1]-G[0]
def mean_curve(series_list):
    cur=[]
    for h in series_list:
        h=np.asarray(h,float); h=h[np.isfinite(h)]
        if len(h)<30: continue
        n=len(h); base=np.median(h[:max(3,n//20)]); end=np.mean(h[-3:])
        if abs(end-base)<1e-12: continue
        z=(h-base)/(end-base); tau=np.arange(1,n+1)/n
        cur.append(np.interp(G,tau,savgol_filter(z,min(21,n//3*2+1),2)))
    return np.mean(cur,axis=0),len(cur)
def ups(M):
    Ms=savgol_filter(M,25,3); Dp=savgol_filter(M,25,3,deriv=1,delta=dg); s=G*Dp
    def info(c,eff):
        k=G<=c+1e-12; A=np.sum(s[k]**2)*dg
        if not eff: return A
        B=np.sum(s[k]*Ms[k])*dg; C=np.sum(Ms[k]**2)*dg; return A-B*B/C
    cs=np.linspace(0.9,1.0,11); out={}
    for eff in (False,True):
        J=np.array([info(c,eff) for c in cs]); out[eff]=(np.polyfit(cs,J/J[-1],1)[0],J[-1])
    return out[False][0],out[True][0],out[True][1]/out[False][1]
rows=[]
# C-MAPSS FD001 raw sensors
ROOT=r'D:\bách khoa Đà Nẵng\nghiencuukhoahoc\baotri\data'
d=np.loadtxt(os.path.join(ROOT,'train_FD001.txt')); uid=d[:,0].astype(int)
for sname,col in (("sensor 2",6),("sensor 3",7),("sensor 4",8),("sensor 7",11),("sensor 8",12)):
    M,n=mean_curve([d[uid==u][:,col] for u in np.unique(uid)]); rows.append(("C-MAPSS "+sname,n)+ups(M))
# XJTU-SY bearings
f=np.load('bearing_feats.npz')
for ind in ("RMS","peak","HF 2-5kHz","kurtosis"):
    ser=[f[k] for k in f.files if k.startswith("X:") and k.endswith("|"+ind)]
    M,n=mean_curve(ser); rows.append(("XJTU "+ind,n)+ups(M))
# Battery capacity (amplitude IS pinned by the 80% rule; control)
dq=np.load('../data/severson/dq_traj.npz',allow_pickle=True)
ser=[]
for q in dq['QD']:
    q=np.asarray(q,float); q=q[np.isfinite(q)&(q>.2)&(q<1.5)]
    if len(q)<80: continue
    q0=np.median(q[:10]); dm=(q0-q)/(0.2*q0); T=np.flatnonzero(dm>=1.0)
    if len(T): ser.append(dm[:T[0]+1])
M,n=mean_curve(ser); rows.append(("Battery capacity (control)",n)+ups(M))
print(f'{"indicator":28s} {"units":>5s} {"Ups pinned":>10s} {"Ups amplitude-free":>18s} {"info retained":>13s}   threshold 1.11 (turbofan) / 2.27 (bearing) / 1.18 (battery)')
for nm,n,a,b,r in rows: print(f'{nm:28s} {n:5d} {a:10.2f} {b:18.2f} {r:13.3f}')
