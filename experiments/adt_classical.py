"""Classical ADT planning on the same fleets: lognormal life, Type-I right censoring at t_c, cost = kappa0 + E[min(T,t_c)]/Tbar per unit.
Criterion A: Fisher information on the location mu (mean log life) per unit cost.
Criterion B: asymptotic variance of the ln p-quantile (p=0.10 and 0.50) per unit cost (mu,sigma jointly)."""
import numpy as np, sys, os
from scipy.stats import norm
from scipy.optimize import minimize_scalar
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','Supplementary_Code','experiments'))
def info_matrix(zc):
    z=np.linspace(-10,zc,6001); dz=z[1]-z[0]; p=norm.pdf(z); sf=max(norm.sf(zc),1e-300); h=norm.pdf(zc)/sf
    # event part: scores (z, z^2-1); cens part: scores h*(1, zc)  [d/dmu log sf = h ; d/dsigma = h*zc]  with prob sf
    s=np.stack([z,z**2-1]); I=(s[:,None,:]*s[None,:,:]*p).sum(-1)*dz
    c=np.array([h,h*zc]); I+=np.outer(c,c)*sf
    return I   # in units (1/sigma^2)
def run(name,lives):
    ly=np.log(lives); mu,sg=ly.mean(),ly.std(); Tbar=lives.mean()
    rng=np.random.default_rng(0); sim=np.exp(mu+sg*rng.standard_normal(200000))
    print(f'\n{name}: n={len(lives)}, lognormal mu={mu:.3f}, sigma={sg:.3f}, mean life {Tbar:.0f}, CV {lives.std()/Tbar:.2f}')
    for kap in (0.05,0.2,0.5):
        res={}
        grid=np.linspace(0.2,3.0,57)   # t_c in units of the median life exp(mu)
        best={}
        for crit in ('mu','p10','p50'):
            vals=[]
            for g in grid:
                zc=np.log(g*np.exp(mu)/np.exp(mu))/sg
                I=info_matrix(zc)/sg**2; cost=kap+np.mean(np.minimum(sim,g*np.exp(mu)))/Tbar
                V=np.linalg.inv(I)
                if crit=='mu': v=V[0,0]
                else:
                    zp=norm.ppf(0.1 if crit=='p10' else 0.5); a=np.array([1,zp]); v=a@V@a
                vals.append(v*cost)   # variance x cost per unit  (smaller is better): Var of estimator at fixed budget ∝ v*cost
            k=int(np.argmin(vals)); best[crit]=(grid[k],np.mean(np.minimum(sim,grid[k]*np.exp(mu)))/Tbar,vals[k]/vals[-1])
        print(f' kappa0={kap}: '+' | '.join(f'{c}: t_c*={best[c][0]:.2f}x median life (mean exposure {best[c][1]:.2f} Tbar), var ratio vs t_c=3x {best[c][2]:.2f}' for c in best))
import cost_cmapss as cc
units,life=cc.load(); run('C-MAPSS FD001',life.astype(float))
d=np.load('../data/severson/dq_traj.npz',allow_pickle=True); run('Severson cells',d['life'].astype(float))
