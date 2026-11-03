"""Audit of the terminal-value assumption D(1)=1 per channel (R1.2, R4.5).
For each channel: amplitude a_i = end - base per unit; delta = robust CV of a_i across units (MAD/median);
sigma = residual noise sd about a smooth trend divided by the median amplitude; M = median samples per life.
Upsilon_lambda: terminal concentration when the amplitude has a prior N(1, delta^2) (pinned: delta->0, free: delta->inf)
  J_lam(c) = (M/s^2)[ int s^2 - (int s D)^2 / (int D^2 + s^2/(M delta^2)) ]."""
import numpy as np, os
from scipy.signal import savgol_filter
exec(open('upsilon_eff_real.py').read().split('rows=[]')[0])   # G, dg, mean_curve
def audit(name, series, mech):
    amps=[];res=[];Ms=[]
    for h in series:
        h=np.asarray(h,float); h=h[np.isfinite(h)]; n=len(h)
        if n<60: continue
        base=np.median(h[:max(3,n//20)]); end=np.mean(h[-3:]); a=end-base
        if abs(a)<1e-12: continue
        r=h-savgol_filter(h,min(31,n//2*2-1),3)
        amps.append(a); res.append(np.std(r)/abs(a)); Ms.append(n)
    amps=np.array(amps); sgn=np.sign(np.median(amps)); amps=amps*sgn
    delta=1.4826*np.median(np.abs(amps-np.median(amps)))/abs(np.median(amps))
    sig=np.median(res); M=np.median(Ms)
    Mcur,_=mean_curve([s for s in series]); Ms_=savgol_filter(Mcur,25,3); Dp=savgol_filter(Mcur,25,3,deriv=1,delta=dg); s=G*Dp
    def U(lam_inv):
        cs=np.linspace(0.9,1,11); J=[]
        for c in cs:
            k=G<=c+1e-12; A=np.sum(s[k]**2)*dg; B=np.sum(s[k]*Ms_[k])*dg; C=np.sum(Ms_[k]**2)*dg
            J.append(A-B*B/(C+lam_inv))
        J=np.array(J); return np.polyfit(cs,J/J[-1],1)[0], J[-1]
    lam_inv=sig**2/(M*max(delta,1e-6)**2)            # (s^2/(M delta^2)) in the same units as int D^2 (D normalised)
    up_pin,Jp=U(1e9); up_lam,Jl=U(lam_inv); up_free,Jf=U(0.0)
    print(f'{name:22s} n={len(amps):4d} delta={delta:6.3f} sigma={sig:6.3f} M={M:6.0f}  Ups pinned {up_pin:6.2f}  prior {up_lam:6.2f}  free {up_free:6.2f}  info kept {Jl/Jp:5.3f}  | {mech}')
ROOT='D:/bách khoa Đà Nẵng/nghiencuukhoahoc/baotri/data'
d=np.loadtxt(ROOT+'/train_FD001.txt'); uid=d[:,0].astype(int)
for nm,col in (('C-MAPSS s2',6),('C-MAPSS s3',7),('C-MAPSS s4',8),('C-MAPSS s7',11),('C-MAPSS s8',12),('C-MAPSS s9',13)):
    audit(nm,[d[uid==u][:,col] for u in np.unique(uid)],'gas-path degradation; no exogenous threshold')
cols=[6,7,8,11,12,13,15,16,17,18,19,21,24,25]; X=d[:,cols]; mu,sd=X.mean(0),X.std(0)+1e-12; w=np.linalg.svd((X-mu)/sd-((X-mu)/sd).mean(0),full_matrices=False)[2][0]
audit('C-MAPSS PC1',[((d[uid==u][:,cols]-mu)/sd)@w for u in np.unique(uid)],'fused; no exogenous threshold')
f=np.load('bearing_feats.npz')
for ind in ('RMS','peak','HF 2-5kHz','kurtosis'):
    audit('XJTU '+ind,[f[k] for k in f.files if k.startswith('X:') and k.endswith('|'+ind)],'test ends at an amplitude limit')
dq=np.load('../data/severson/dq_traj.npz',allow_pickle=True)
ser=[]
for q in dq['QD']:
    q=np.asarray(q,float); q=q[np.isfinite(q)&(q>.2)&(q<1.5)]
    if len(q)<80: continue
    q0=np.median(q[:10]); dm=(q0-q)/(0.2*q0); T=np.flatnonzero(dm>=1.0)
    if len(T): ser.append(dm[:T[0]+1])
audit('Cell capacity',ser,'80% of nominal defines end of life')
