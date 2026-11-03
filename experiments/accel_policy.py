import numpy as np, json, re
from scipy import stats
from scipy.optimize import least_squares
exec(open('upsilon_eff_real.py').read().split('rows=[]')[0])
pol=json.load(open('policy.json'))
d=np.load('../data/severson/dq_traj.npz',allow_pickle=True)
names,QD,life=d['name'],d['QD'],d['life'].astype(float)
S=[];keep=[]
for i,nm in enumerate(names):
    m=re.match(r'([\d.]+)C\((\d+)%\)-([\d.]+)C?',pol[nm][0])
    if not m: continue
    c1,q,c2=float(m[1]),float(m[2]),float(m[3])
    S.append(80/(q/c1+(80-q)/c2)); keep.append(i)
S=np.array(S); keep=np.array(keep); T=life[keep]
print(f'{len(keep)} of {len(names)} cells with parseable policy; mean charge rate to 80% SOC: {S.min():.2f}-{S.max():.2f} C')
# 1. log-linear acceleration
lt=np.log(T)
for nm,x in (('ln s',np.log(S)),('s',S)):
    r=stats.linregress(x,lt); print(f'ln T ~ {nm}: slope {r.slope:.3f}, R2 {r.rvalue**2:.3f}, resid sd {np.std(lt-(r.intercept+r.slope*x)):.3f} vs total sd {lt.std():.3f}')
# 2. per-cell shape: D = tau^b
def dmg(q):
    q=np.asarray(q,float); q0=np.nanmedian(q[1:6]); return (q0-q)/(0.2*q0+1e-12)
B=[];curves=[];ok=[]
for j,i in enumerate(keep):
    y=dmg(QD[i]); n=len(y)
    if n<60: continue
    t=np.arange(1,n+1)/ (n if True else 1)
    # normalise by actual life so that tau=1 at failure
    tau=np.arange(1,n+1)/life[i]; k=(tau<=1.0)&(tau>0.05)&np.isfinite(y)&(y>0)
    if k.sum()<30: continue
    b=np.polyfit(np.log(tau[k]),np.log(y[k]),1)[0]; B.append(b)
    curves.append(np.interp(G,tau,savgol_filter(y,21,2))); ok.append(j)
B=np.array(B);ok=np.array(ok);curves=np.array(curves);s_ok=S[ok]
r=stats.spearmanr(s_ok,B); print(f'shape exponent b vs stress: median {np.median(B):.2f}, IQR {np.percentile(B,25):.2f}-{np.percentile(B,75):.2f}, Spearman rho {r.correlation:.2f} (p={r.pvalue:.3f})')
q=np.quantile(s_ok,[1/3,2/3]); grp=np.digitize(s_ok,q)
print('tertile  n  stress range     median b   Upsilon pinned [95% CI]   amp-free [95% CI]')
for g in range(3):
    c=curves[grp==g]
    M=c.mean(0); a,b2,_=ups(M)
    print(f'{g}  {len(c):3d}  {s_ok[grp==g].min():.2f}-{s_ok[grp==g].max():.2f}   {np.median(B[grp==g]):.2f}   {a:.2f}   {b2:.2f}')
print('Kruskal-Wallis on b across tertiles: p=%.3f'%stats.kruskal(*[B[grp==g] for g in range(3)]).pvalue)
# 3. extrapolation: fit ln T~ln s on high-stress half, predict low-stress half
x=np.log(S); hi=S>np.median(S)
r=stats.linregress(x[hi],lt[hi]); pred=r.intercept+r.slope*x[~hi]
r_in=stats.linregress(x[~hi],lt[~hi]); pin=r_in.intercept+r_in.slope*x[~hi]
print(f'extrapolation to low-stress half: RMSE ln T {np.sqrt(np.mean((pred-lt[~hi])**2)):.3f} (fit on high stress) vs {np.sqrt(np.mean((pin-lt[~hi])**2)):.3f} (fit in-group); slopes {r.slope:.2f} vs {r_in.slope:.2f}')
