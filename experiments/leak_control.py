"""Window-length leakage control for the own-fraction design (channel_info.py: window m_i = c*T_i).
Control feature: the cross-cell MEDIAN DeltaQ trajectory evaluated at m_i. It contains no
cell-specific signal, only the window length. Any LOO R^2 it achieves is leakage."""
import numpy as np
d=np.load('../data/severson/dq_traj.npz',allow_pickle=True)
DQ,QD,life=d['DQ'],d['QD'],d['life']; y=np.log(life); n=len(y)
grid=np.arange(15,2000,5)
M=np.full((n,len(grid)),np.nan)
for k in range(n):
    ms,vs=DQ[k]; g=(grid>=ms[0])&(grid<=ms[-1]); M[k,g]=np.interp(grid[g],ms,vs)
med=np.nanmedian(M,axis=0)
def loo(x,y):
    e=np.empty(len(y))
    for i in range(len(y)):
        m=np.ones(len(y),bool); m[i]=False
        A=np.c_[x[m],np.ones(m.sum())]; c=np.linalg.lstsq(A,y[m],rcond=None)[0]; e[i]=y[i]-(c[0]*x[i]+c[1])
    return np.mean(e**2)
print(" c     R2(real DQ at m=cT)   R2(control: window length only)   R2(real DQ, FIXED m=cT_median)")
Tmed=np.median(life)
for c in (0.10,0.15,0.25,0.35,0.50,0.65):
    m_i=np.clip(np.round(c*life/5)*5,15,None)
    real=np.array([np.interp(m_i[k],*DQ[k]) for k in range(n)])
    ctrl=np.interp(m_i,grid,med)
    mfix=c*Tmed; ok=life>mfix
    fixed=np.array([np.interp(mfix,*DQ[k]) for k in range(n)])
    r=lambda x,yy: 1-loo(x,yy)/yy.var()
    print(f"{c:4.2f}   {r(real,y):8.3f}              {r(ctrl,y):8.3f}                     {r(fixed[ok],y[ok]):8.3f}  (n={ok.sum()})")
