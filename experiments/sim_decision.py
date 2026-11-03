"""Controlled study with known ground-truth Upsilon (R2.5, R2.7, R1.4, R2.6).
Units: D(tau)=(exp(b*tau)-1)/(exp(b)-1) with unit-specific b ~ N(b0, 0.15|b0|+0.3) (shape varies across units),
noise: white / AR(1) / heteroscedastic, sampling M points on [0,T_i], retrospective terminal normalisation
(as in the paper's estimator (a)). Estimator: pooled smoothed trajectory + numerical derivative.
Decision = Upsilon_hat >= thr, thr=(1+Lam)/(1+kappa0)=1.18. Truth from the b0-curve."""
import numpy as np, sys
from scipy.signal import savgol_filter
rng=np.random.default_rng(7)
thr=1.18
def D(t,b): return t if abs(b)<1e-9 else (np.exp(b*t)-1)/(np.exp(b)-1)
def Dp(t,b): return 1.0 if abs(b)<1e-9 else b*np.exp(b*t)/(np.exp(b)-1)
tt=np.linspace(0,1,20001)
def ups_true(b):
    w=np.trapezoid(tt**2*Dp(tt,b)**2,tt); return Dp(1.0,b)**2/w
def noise(kind,M,sig,Dv):
    if kind=="white": return sig*rng.standard_normal(M)
    if kind=="ar1":
        r=0.9; e=np.empty(M); e[0]=rng.standard_normal(); 
        for j in range(1,M): e[j]=r*e[j-1]+np.sqrt(1-r*r)*rng.standard_normal()
        return sig*e
    if kind=="het": return sig*(0.3+2.0*Dv)*rng.standard_normal(M)
def estimate(n,M,sig,kind,b0):
    grid=np.linspace(0,1,201); acc=np.zeros_like(grid)
    for _ in range(n):
        b=b0+rng.normal(0,0.15*abs(b0)+0.3); tau=np.sort(rng.uniform(0,1,M)); tau[-1]=1.0
        Dv=D(tau,b); x=Dv+noise(kind,M,sig,Dv)
        base=np.median(x[:max(3,M//20)]); end=np.median(x[-max(3,M//20):])   # robust baseline / terminal value
        z=(x-base)/(end-base)
        # local smooth onto grid by binned medians + linear interp
        bins=np.linspace(0,1,41); idx=np.clip(np.digitize(tau,bins)-1,0,39)
        cen=[];val=[]
        for k in range(40):
            s=z[idx==k]
            if len(s): cen.append(tau[idx==k].mean()); val.append(np.median(s))
        acc+=np.interp(grid,cen,val)
    m=acc/n; ms=savgol_filter(m,31,3); d=savgol_filter(m,31,3,deriv=1,delta=grid[1]-grid[0])
    d1=np.mean(d[-8:])                       # terminal derivative from last stretch
    w=np.trapezoid(grid**2*d**2,grid)
    return d1**2/w
B0=[-9,-6,-3,0,3,6,9]
print("true Upsilon by b0:",{b:round(ups_true(b),2) for b in B0})
reps=60
for kind in ("white","ar1","het"):
  print(f"\n== noise: {kind}")
  print(" n   M   sig   | P(correct) per b0 (true Ups, side)          | overall")
  for n,M,sig in ((12,100,0.03),(30,100,0.03),(100,100,0.03),(30,400,0.03),(30,100,0.10),(100,400,0.10)):
    pc=[];tot=[]
    for b0 in B0:
        tu=ups_true(b0); truth=tu>=thr
        ok=np.mean([ (estimate(n,M,sig,kind,b0)>=thr)==truth for _ in range(reps)])
        pc.append(ok)
    print(f"{n:3d} {M:4d} {sig:5.2f} | "+" ".join(f"{p:4.2f}" for p in pc)+f" | {np.mean(pc):.2f}")
