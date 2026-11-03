"""(a) Is test MSE ~ a(c)/n + floor(c) at fixed c? (assumption behind G(c)=(kappa0+c)a(c)).
   (b) Tobit sensitivity to its stabilisers."""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import cost_cmapss as cc
warnings.filterwarnings('ignore')
m=100
def setup(seed):
    units,life=cc.load(); rng=np.random.default_rng(seed)
    order=rng.permutation(len(life)); test,cal,pool=order[:cc.N_TEST],order[cc.N_TEST:cc.N_TEST+cc.N_CAL],order[cc.N_TEST+cc.N_CAL:]
    idx=cc.calibrate(units,cal); D=[idx(u) for u in units]
    Xte=np.array([cc.feats(D[i],m) for i in test]); yte=life[test]; base=float(np.sqrt(np.mean((yte-life[pool].mean())**2)))
    return life,D,pool,Xte,yte,base,rng,cal
print("=== (a) MSE vs n at fixed c (C-MAPSS FD001, m=100; 3 splits, 40 draws), MSE as multiple of base^2 ===")
NS=(8,12,18,26,38,55)
for c,label in ((1.0,"c=1.0 true labels"),(0.8,"c=0.8 imputed labels"),(0.8,"c=0.8 oracle labels")):
    mse={n:[] for n in NS}
    for seed in (0,1,2):
        life,D,pool,Xte,yte,base,rng,cal=setup(seed)
        elig=np.array([i for i in pool if c*life[i]>=m])
        for n in NS:
            if n>len(elig): continue
            for _ in range(40):
                take=rng.choice(elig,n,replace=False); X=np.array([cc.feats(D[i],m) for i in take])
                if c>=1 or "oracle" in label: yl=life[take]
                else: yl=cc.impute([(np.arange(1,int(c*life[i])+1,dtype=float),D[i][:int(c*life[i])]) for i in take], cc.cal_beta(D, life, cal))
                p=cc.fit_huber(X,yl,Xte); mse[n].append(np.mean((p-yte)**2)/base**2)
    ns=np.array([n for n in NS if mse[n]]); y=np.array([np.mean(mse[n]) for n in ns])
    A=np.c_[1/ns,np.ones(len(ns))]; coef=np.linalg.lstsq(A,y,rcond=None)[0]
    slope=np.polyfit(np.log(ns),np.log(y),1)[0]
    print(f"{label:24s} n={ns.tolist()}  MSE={np.round(y,3)}")
    print(f"{'':24s} fit MSE = {coef[0]:.2f}/n + {coef[1]:.3f};  log-log slope = {slope:.2f}  (1/n law => -1)")
print("\n=== (b) Tobit sensitivity (m=100), RMSE/base at F(ts)=0.25 and 0.50; ridge and sigma floor varied ===")
def tobit(X,ylog,cens,ridge,floor):
    mu,sd=X[:,1:].mean(0),X[:,1:].std(0)+1e-9; Z=np.c_[X[:,0],(X[:,1:]-mu)/sd]
    b0=np.linalg.lstsq(Z,ylog,rcond=None)[0]
    def nll(p):
        b,ls=p[:-1],max(p[-1],np.log(floor)); s=np.exp(ls); r=(ylog-Z@b)/s
        return -np.sum(np.where(cens,norm.logsf(r),norm.logpdf(r)-ls))+ridge*np.sum(b[1:]**2)
    x=minimize(nll,np.r_[b0,np.log(np.std(ylog-Z@b0)+1e-3)],method='BFGS').x
    return x[:-1],mu,sd
def pred(par,Xte): b,mu,sd=par; return np.exp(np.c_[Xte[:,0],(Xte[:,1:]-mu)/sd]@b)
for ridge,floor in ((0.05,0.05),(0.5,0.15),(2.0,0.15),(0.5,0.4)):
    res={0.25:[],0.5:[]}
    for seed in (0,1,2):
        life,D,pool,Xte,yte,base,rng,cal=setup(seed); Tbar=life[pool].mean()
        for q in (0.25,0.5):
            ts=np.quantile(life[pool],q); obs=np.minimum(life,np.floor(ts)); fail=life<=ts
            cost=cc.KAP0+np.mean(np.minimum(life[pool],ts))/Tbar
            elig=np.array([i for i in pool if obs[i]>=m]); n=min(int(round(cc.C_BUDGET/cost)),len(elig))
            for _ in range(30):
                take=rng.choice(elig,n,replace=False); X=np.array([cc.feats(D[i],m) for i in take]); f=fail[take]
                try: res[q].append(np.sqrt(np.mean((pred(tobit(X,np.log(obs[take]),~f,ridge,floor),Xte)-yte)**2))/base)
                except Exception: pass
    print(f"ridge={ridge:4.2f} sigma-floor={floor:4.2f}:  F=0.25 -> {np.mean(res[0.25]):.2f}   F=0.50 -> {np.mean(res[0.5]):.2f}")
