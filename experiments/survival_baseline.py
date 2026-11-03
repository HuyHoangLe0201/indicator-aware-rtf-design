"""Cox PH (ridge, Breslow baseline, median-survival prediction) added to the censoring baselines of cost_cmapss_ext."""
import sys,os; sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'.'))
import numpy as np, warnings
from scipy.optimize import minimize
import cost_cmapss as cc, cost_cmapss_ext as ce
warnings.filterwarnings('ignore')
def cox_fit(X, t, ev, lam=1.0):
    mu,sd=X[:,1:].mean(0),X[:,1:].std(0)+1e-9; Z=(X[:,1:]-mu)/sd
    o=np.argsort(-t); Zo,eo=Z[o],ev[o]
    def nll(b):
        eta=Zo@b; lse=np.logaddexp.accumulate(eta)
        return -np.sum((eta-lse)[eo])+lam*np.sum(b**2)
    b=minimize(nll,np.zeros(Z.shape[1]),method='BFGS').x
    eta=Z@b; ts=np.unique(t[ev]); H=0; haz=[]
    for u in ts:
        d=np.sum((t==u)&ev); r=np.sum(np.exp(eta[t>=u])); H+=d/r; haz.append(H)
    return b,mu,sd,ts,np.array(haz)
def cox_pred(par,Xte):
    b,mu,sd,ts,H=par; eta=((Xte[:,1:]-mu)/sd)@b; out=[]
    for e in eta:
        S=np.exp(-H*np.exp(e)); k=np.flatnonzero(S<=0.5)
        out.append(ts[k[0]] if len(k) else ts[-1]*1.5)
    return np.array(out,float)
def run(m,seed,designs,reps=40):
    units,life=cc.load(); rng=np.random.default_rng(seed)
    order=rng.permutation(len(life)); test,cal,pool=order[:cc.N_TEST],order[cc.N_TEST:cc.N_TEST+cc.N_CAL],order[cc.N_TEST+cc.N_CAL:]
    index=cc.calibrate(units,cal); D=[index(u) for u in units]
    Xte=np.array([cc.feats(D[i],m) for i in test]); yte=life[test]; base=float(np.sqrt(np.mean((yte-life[pool].mean())**2))); Tbar=life[pool].mean(); out={}
    for name,kind,par in designs:
        if kind=='frac': obs=np.minimum(life,np.floor(par*life)); cost=cc.KAP0+par; fail=np.full(len(life),par>=1.0)
        else:
            ts=np.inf if par>=1 else np.quantile(life[pool],par); obs=np.minimum(life,np.floor(ts)); fail=life<=ts; cost=cc.KAP0+np.mean(np.minimum(life[pool],ts))/Tbar
        elig=np.array([i for i in pool if obs[i]>=m]); n=min(max(4,int(round(cc.C_BUDGET/cost))),len(elig))
        if len(elig)<8: continue
        acc={'Tobit':[],'Cox':[],'impute+Huber':[]}
        for _ in range(reps):
            take=rng.choice(elig,n,replace=False); X=np.array([cc.feats(D[i],m) for i in take]); f=fail[take]
            r=lambda p: float(np.sqrt(np.mean((p-yte)**2)))/base
            try: acc['Tobit'].append(r(ce.tobit_predict(ce.tobit_fit(X,np.log(obs[take]),~f),Xte)))
            except Exception: acc['Tobit'].append(np.nan)
            try: acc['Cox'].append(r(cox_pred(cox_fit(X,obs[take].astype(float),f),Xte)) if f.sum()>=3 else np.nan)
            except Exception: acc['Cox'].append(np.nan)
            yl=life[take].copy()
            if not f.all():
                tr=[(np.arange(1,int(obs[i])+1,dtype=float),D[i][:int(obs[i])]) for i in take[~f]]
                yl[~f]=cc.impute(tr, cc.cal_beta(D, life, cal)) if len(tr)>=3 else obs[take[~f]]
            acc['impute+Huber'].append(r(cc.fit_huber(X,yl,Xte)))
        out[name]={k:float(np.nanmean(v)) if not np.all(np.isnan(v)) else np.nan for k,v in acc.items()}
    return out
if __name__=='__main__':
    designs=[(f'c={c:.2f}','frac',c) for c in (0.5,0.8,1.0)]+[(f'F(ts)={q:.2f}','abs',q) for q in (0.25,0.5,0.75)]
    for m in (60,100):
        res=[run(m,s,designs) for s in (0,1,2)]
        print(f'\n=== FD001 PC1, m={m}; RMSE / pool-mean RMSE, 3 splits x 40 ===\n{"design":>12s} {"impute+Huber":>12s} {"Tobit":>7s} {"Cox":>7s}')
        for nm,_,_ in designs:
            rr=[r[nm] for r in res if nm in r]
            if rr: print(f'{nm:>12s} '+' '.join(f'{np.nanmean([x[k] for x in rr]):{w}.2f}' for k,w in (('impute+Huber',12),('Tobit',7),('Cox',7))))
