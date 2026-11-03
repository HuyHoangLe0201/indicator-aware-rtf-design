"""Recompute the empirical information curves for the two Severson channels with FIXED-cycle
windows m, comparing normalised R^2 with the odds R^2/(1-R^2) that Prop. 6 identifies as the
information scale. Uses the shipped derived arrays only."""
import numpy as np, os
H='../data/severson/dq_traj.npz'
d=np.load(H,allow_pickle=True); DQ,QD,life=d['DQ'],d['QD'],d['life']
y=np.log(life); n0=len(y)
def feat_dq(k,m):
    ms,vs=DQ[k]; return np.interp(m,ms,vs)
def feat_cap(k,m):
    q=np.asarray(QD[k],float); w=q[:m+1]; w=w[np.isfinite(w)&(w>.2)&(w<1.5)]
    return np.median(w[-5:]) if len(w)>=5 else np.nan
def loo(x,y):
    n=len(y); e=np.empty(n)
    for i in range(n):
        m=np.ones(n,bool); m[i]=False
        A=np.c_[x[m],np.ones(m.sum())]; c=np.linalg.lstsq(A,y[m],rcond=None)[0]
        e[i]=y[i]-(c[0]*x[i]+c[1])
    return np.mean(e**2)
Ms=[20,50,100,150,200,300,400,500,600]
print("n total",n0,"median life",np.median(life))
res={}
for nm,f in (("DQ",feat_dq),("cap",feat_cap)):
    rows=[]
    for m in Ms:
        ok=np.array([life[k]>m for k in range(n0)])
        x=np.array([f(k,m) if ok[k] else np.nan for k in range(n0)])
        ok&=np.isfinite(x)
        if ok.sum()<40: rows.append((m,ok.sum(),np.nan,np.nan));continue
        yy=y[ok]; r2=1-loo(x[ok],yy)/yy.var()
        rows.append((m,int(ok.sum()),r2,r2/(1-r2) if r2<1 else np.inf))
    res[nm]=rows
    print("\nchannel",nm); print(" m    n    LOO-R2   odds=R2/(1-R2)")
    for m,n,r,o in rows: print(f"{m:4d} {n:4d}  {r:7.3f}  {o:8.3f}")
# bootstrap on cells: ratio odds(m=100)/odds(m=400) and plateau timing
rng=np.random.default_rng(0)
def odds_at(m,idx,f):
    ok=[k for k in idx if life[k]>m]; x=np.array([f(k,m) for k in ok]); yy=y[ok]
    g=np.isfinite(x); x,yy=x[g],yy[g]
    r2=1-loo(x,yy)/yy.var(); return r2/(1-r2)
B=60; out={"DQ":[], "cap":[]}
for b in range(B):
    idx=rng.integers(0,n0,n0)
    for nm,f in (("DQ",feat_dq),("cap",feat_cap)):
        try: out[nm].append((odds_at(100,idx,f),odds_at(300,idx,f)))
        except Exception: pass
for nm in out:
    a=np.array(out[nm]); print(nm,"odds(100) median",np.median(a[:,0]).round(3),"[",np.percentile(a[:,0],[2.5,97.5]).round(2),"]  odds(300) median",np.median(a[:,1]).round(3),"[",np.percentile(a[:,1],[2.5,97.5]).round(2),"]")
