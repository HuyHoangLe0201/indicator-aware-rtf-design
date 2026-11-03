"""
Test the tail-vs-variance candidate for the c* = 0.385 (theory) vs 0.50 (E3) gap.

Theory models the label channel by its VARIANCE:  Var(That) ~ sigma_T^2/eta_T(c),
optionally inflated by Thm 8's Pi(c,n) = 1 + u(c)/n.

Hypothesis: the downstream RMSE is driven by a TAIL quantile of |That-T|/T instead.

Procedure
  1. measure the label-error distribution at each c, at the budget's own n(c);
  2. compare the measured RMS curve with Thm 8's prediction;
  3. substitute each measured statistic Xi(c) for 1/eta_T(c) in the design
     objective  (k0+c)[ K*Xi(c)/Xi(1) + S ]  and read off c*.
Whichever statistic reproduces c* = 0.50 is the operative one.
"""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.integrate import quad

AB, BB = 3.0, 0.7
T_MEAN, T_CV, SIGMA, M_OBS, SH_CV = 806.0, 0.46, 0.05, 60, 0.25
K0, CBUD, LAMBDA_DESIGN = 0.05, 40*1.05, 0.241
LAM_MULT = 0.1

def Dg(tau, ab, b):
    tau = np.clip(tau, 1e-9, 1.0)
    return (1-np.exp(-ab*tau**b))/(1-np.exp(-ab))

def draw(rng):
    s = np.sqrt(np.log(1+T_CV**2))
    return (rng.lognormal(np.log(T_MEAN)-0.5*s**2, s),
            AB*np.exp(SH_CV*rng.normal()),
            float(np.clip(BB*np.exp(SH_CV*rng.normal()), .3, 1.)))

def window(T, ab, b, c, rng):
    t = np.linspace(T/M_OBS, c*T, M_OBS)
    return t, Dg(t/T, ab, b) + SIGMA*rng.normal(size=M_OBS)

def fT(t, y, a, b_):
    return minimize_scalar(lambda T_: np.sum((y-Dg(t/T_, a, b_))**2),
                           bounds=(t[-1]*1.0001, t[-1]*25.), method='bounded',
                           options={'xatol':1e-3}).x

def impute_partial(win):
    ab, bb = AB, BB
    Ts = np.array([fT(t, y, ab, bb) for t, y in win])
    for _ in range(3):
        res = lambda p: np.concatenate([y-Dg(t/T_, p[0], p[1])
                                        for (t, y), T_ in zip(win, Ts)])
        try:
            ab, bb = least_squares(res, [ab, bb], bounds=([.3,.3],[15.,1.]),
                                   method='trf', max_nfev=150).x
        except Exception: pass
        Ts = np.array([fT(t, y, ab, bb) for t, y in win])
    lam = LAM_MULT*(1./SH_CV**2)*SIGMA**2*M_OBS
    out = np.empty(len(win))
    for i, (t, y) in enumerate(win):
        lo, hi = t[-1]*1.0001, t[-1]*25.
        def r(p):
            T_, la, lb = p
            return np.concatenate([y-Dg(t/T_, np.exp(la), np.clip(np.exp(lb),.3,1.)),
                                   np.sqrt(lam)*np.array([la-np.log(ab), lb-np.log(bb)])])
        try:
            out[i] = least_squares(r, [Ts[i], np.log(ab), np.log(bb)],
                                   bounds=([lo,np.log(.3),np.log(.3)],
                                           [hi,np.log(15.),0.]),
                                   method='trf', max_nfev=400).x[0]
        except Exception:
            out[i] = Ts[i]
    return out

# ---- Thm 8 prediction (regularised u, partial pooling) ---------------------
def str_Dp(t):  return AB*BB*t**(BB-1)*np.exp(-AB*t**BB)/(1-np.exp(-AB))
def str_dab(t):
    M = 1-np.exp(-AB); N = 1-np.exp(-AB*t**BB)
    return (t**BB*np.exp(-AB*t**BB)*M - N*np.exp(-AB))/M**2
def str_db(t):  return AB*t**BB*np.log(t)*np.exp(-AB*t**BB)/(1-np.exp(-AB))
S_TH = np.diag([(SH_CV*AB)**2, (SH_CV*BB)**2])
LPR  = (SIGMA**2/M_OBS)*np.linalg.inv(S_TH)
ZT   = quad(lambda t: t**2*str_Dp(t)**2, 0, 1, limit=400)[0]
etaT = lambda c: quad(lambda t: t**2*str_Dp(t)**2, 0, c, limit=400)[0]/ZT

def u_of(c):
    A = quad(lambda t:(t*str_Dp(t))**2,0,c,limit=400)[0]
    B = np.array([quad(lambda t:-(t*str_Dp(t))*d(t),0,c,limit=400)[0] for d in (str_dab,str_db)])
    C = np.array([[quad(lambda t:d1(t)*d2(t),0,c,limit=400)[0]
                   for d2 in (str_dab,str_db)] for d1 in (str_dab,str_db)])
    r2 = min(max(float(B@np.linalg.solve(C+LPR,B)/A),0.),1-1e-12)
    return r2/(1-r2)

def cstar(Xi):
    """argmin (k0+c)[ Xi(c)/Xi(1)/Lambda + 1 ]"""
    g = lambda c: (K0+c)*(Xi(c)/Xi(1.0)/LAMBDA_DESIGN + 1.0)
    return minimize_scalar(g, bounds=(0.12,1.0), method='bounded',
                           options={'xatol':1e-6}).x

if __name__ == '__main__':
    grid = [0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    REPS = 14
    stats = {}
    print(f"{'c':>5s} {'n(c)':>5s} {'RMS':>7s} {'med':>7s} {'p75':>7s} {'p90':>7s} "
          f"| {'Thm8 pred RMS':>13s}")
    for c in grid:
        n = max(5, int(round(CBUD/(K0+c))))
        e = []
        for r in range(REPS):
            rng = np.random.default_rng(int(c*1000)*131+r)
            cells = [draw(rng) for _ in range(n)]
            win = [window(T,ab,b,c,rng) for T,ab,b in cells]
            Tt = np.array([T for T,_,_ in cells])
            e.append(np.abs(impute_partial(win)-Tt)/Tt)
        e = np.concatenate(e)
        pred = np.sqrt((1+u_of(c)/n)/etaT(c))
        stats[c] = dict(n=n, rms=np.sqrt(np.mean(e**2)), med=np.median(e),
                        p75=np.percentile(e,75), p90=np.percentile(e,90), pred=pred)
        s = stats[c]
        print(f"{c:5.2f} {n:5d} {s['rms']:7.3f} {s['med']:7.3f} {s['p75']:7.3f} "
              f"{s['p90']:7.3f} | {pred:13.3f}")

    # normalise Thm-8 prediction at c=1 for shape comparison
    pr1 = stats[1.0]['pred']; rm1 = stats[1.0]['rms']
    print(f"\nshape check (all normalised at c=1):")
    print(f"{'c':>5s} {'RMS meas':>9s} {'Thm8':>7s} {'1/sqrt(etaT)':>13s}")
    for c in grid:
        print(f"{c:5.2f} {stats[c]['rms']/rm1:9.3f} {stats[c]['pred']/pr1:7.3f} "
              f"{np.sqrt(etaT(1.0)/etaT(c))/np.sqrt(1/etaT(1.0)):13.3f}")

    print(f"\n=== c* from substituting each statistic  (E3 observed 0.50) ===")
    def mk(key, sq):
        f = lambda c: (np.interp(c, grid, [stats[g][key] for g in grid]))**(2 if sq else 1)
        return f
    print(f"{'statistic':>22s} {'c*':>7s}")
    print(f"{'theory 1/eta_T (var)':>22s} {cstar(lambda c: 1/etaT(c)):7.3f}")
    print(f"{'Thm 8 (var + pooling)':>22s} {cstar(lambda c: (1+u_of(c)/max(5,round(CBUD/(K0+c))))/etaT(c)):7.3f}")
    for key in ['rms','med','p75','p90']:
        print(f"{('measured '+key+'^2'):>22s} {cstar(mk(key,True)):7.3f}")
