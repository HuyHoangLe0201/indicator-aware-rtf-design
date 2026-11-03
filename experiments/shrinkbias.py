r"""
Shrinkage-bias model for the label channel.

Estimator:  min_{logT, l}  sum_j (y_j - D(t_j/T; e^l))^2 + lambda ||l - lhat_bar||^2
with lhat_bar the population log-shape estimated from all n units.

Linearising at the truth (l_i = mu + delta_i, lhat_bar = mu + e) and profiling out
log T gives, with A,B,C the per-sample information blocks w.r.t. (log T, l):

    Delta logT_i = g W (delta_i - e),
    g  = A^{-1} B,                      (transfer of shape error into log T)
    W  = (Ctil + Lam)^{-1} Lam,         (shrinkage operator)
    Ctil = C - B^T A^{-1} B,            (shape info after profiling out T)
    Lam  = (lambda/M) I .

Hence      E[(Delta logT)^2] = g W [ S_th (1 - 1/n) + (sigma^2/M) Ctil^{-1} / n ] W^T g^T .

Total label MSE:  L(c)^2 = sigma_T^2 / eta_T(c)  +  bias2(c, n).
                            \___ Def 4.1 ___/     \___ this file ___/
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

AB, BB = 3.0, 0.7
SIGMA, M_OBS, SH_CV = 0.05, 60, 0.25
K0, CBUD, LAM_DESIGN = 0.05, 40*1.05, 0.241
LAM_RAW = 0.1*(1.0/SH_CV**2)*SIGMA**2*M_OBS      # estimator's lambda (raw SSE scale)

def Dg(t, ab, b):
    t = np.clip(t, 1e-9, 1.0)
    return (1-np.exp(-ab*t**b))/(1-np.exp(-ab))
def dDdt(t):  return AB*BB*t**(BB-1)*np.exp(-AB*t**BB)/(1-np.exp(-AB))
def dDdab(t):
    M = 1-np.exp(-AB); N = 1-np.exp(-AB*t**BB)
    return (t**BB*np.exp(-AB*t**BB)*M - N*np.exp(-AB))/M**2
def dDdb(t):  return AB*t**BB*np.log(t)*np.exp(-AB*t**BB)/(1-np.exp(-AB))

# derivatives w.r.t. LOG parameters  (scale-free, matches the estimator)
dl = [lambda t: AB*dDdab(t), lambda t: BB*dDdb(t)]
dT = lambda t: -t*dDdt(t)                       # d D / d log T

Z    = quad(lambda t: dT(t)**2, 0, 1, limit=400)[0]
etaT = lambda c: quad(lambda t: dT(t)**2, 0, c, limit=400)[0]/Z

def ABC(c):
    A = quad(lambda t: dT(t)**2, 0, c, limit=400)[0]
    B = np.array([quad(lambda t: dT(t)*d(t), 0, c, limit=400)[0] for d in dl])
    C = np.array([[quad(lambda t: d1(t)*d2(t), 0, c, limit=400)[0] for d2 in dl] for d1 in dl])
    return A, B, C

S_TH = np.diag([SH_CV**2, SH_CV**2])
LAM  = (LAM_RAW/M_OBS)*np.eye(2)

def bias2(c, n):
    A, B, C = ABC(c)
    Ctil = C - np.outer(B, B)/A
    Ctil = Ctil + 1e-12*np.eye(2)
    W    = np.linalg.solve(Ctil + LAM, LAM)
    g    = B/A
    Mid  = S_TH*(1-1/n) + (SIGMA**2/M_OBS)*np.linalg.inv(Ctil)/n
    v    = g @ W
    return float(v @ Mid @ v)

if __name__ == '__main__':
    c   = np.array([0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60,0.70,0.80,0.90,1.00])
    Lm  = np.array([0.407,0.320,0.281,0.258,0.246,0.240,0.235,0.254,0.239,0.275,0.305,0.350])
    n   = np.array([210,168,140,120,105,93,76,65,56,49,44,40])
    eT  = np.array([etaT(x) for x in c])
    bth = np.array([bias2(x, m) for x, m in zip(c, n)])

    # fit sigma_T^2 by least squares on  L^2 = s2/etaT + bias2_model
    s2 = float(np.sum((Lm**2-bth)/eT)/np.sum(1/eT**2))
    pred = np.sqrt(np.maximum(s2/eT + bth, 1e-12))
    emp_b2 = np.maximum(Lm**2 - s2/eT, 0.0)      # empirical bias^2

    print(f"fitted sigma_T^2 = {s2:.5f}   lambda/M = {LAM[0,0]:.5f}")
    print(f"\n{'c':>5s} {'n':>4s} {'L meas':>7s} {'L model':>8s} {'bias2 model':>12s} "
          f"{'bias2 emp':>10s} {'shrink |W|':>11s}")
    for i, x in enumerate(c):
        A_, B_, C_ = ABC(x); Ct = C_-np.outer(B_,B_)/A_+1e-12*np.eye(2)
        W = np.linalg.solve(Ct+LAM, LAM)
        print(f"{x:5.2f} {n[i]:4d} {Lm[i]:7.3f} {pred[i]:8.3f} {bth[i]:12.5f} "
              f"{emp_b2[i]:10.5f} {np.linalg.norm(W,2):11.3f}")
    ss = 1-np.sum((Lm-pred)**2)/np.sum((Lm-Lm.mean())**2)
    print(f"\nR^2 of the two-term label model on L(c):  {ss:.4f}")
    print(f"argmin L model = c={c[np.argmin(pred)]:.2f}   argmin L measured = c={c[np.argmin(Lm)]:.2f}")

    # design optimum with the bias-corrected label channel
    def cs(Xi):
        g = lambda x: (K0+x)*(Xi(x)/Xi(1.0)/LAM_DESIGN + 1.0)
        return minimize_scalar(g, bounds=(0.12,1.0), method='bounded',
                               options={'xatol':1e-6}).x
    nf = lambda x: max(5.0, CBUD/(K0+x))
    print(f"\n{'label-channel model':>34s} {'c*':>7s}")
    print(f"{'Def 4.1 only  (1/eta_T)':>34s} {cs(lambda x: 1/etaT(x)):7.3f}")
    print(f"{'Def 4.1 + shrinkage bias':>34s} "
          f"{cs(lambda x: s2/etaT(x) + bias2(x, nf(x))):7.3f}")
    print(f"{'':>34s}   (E3 observed 0.50)")
