"""
The design verdict without estimating Upsilon.

The Limitations concede that the threshold crossing 'rests on one estimator': the parametric
estimator (a) returns anything between 0.13 and 8.91 for the DeltaQ(V) channel depending on a
parameterization nothing fixes, and only estimator (b) is used.  That is uncomfortable for the
one crossing the paper's empirical claim depends on.

But Upsilon is only an intermediate.  What the design actually needs is the minimiser of the
budgeted risk

        G(c)  proportional to  (kappa_0 + c) [ 1/etaT(c) + Lambda ],

and that can be computed straight from the MEASURED etaT curve of each channel, never
differentiating it at the endpoint.  This is the paper's own advice -- prefer an integral
statistic to a terminal derivative -- applied to its own headline result.

The measured curves are the leave-one-out predictive-R^2 shares of Section VII-B at cycles
20, 50, 100, 200, 400, 700.  etaT must be non-decreasing, so the raw shares are monotonised by
pool-adjacent-violators before use, clipped below at zero, and normalised to etaT(1)=1.  We
sweep Lambda over four decades and kappa_0 over the range the paper considers, and we repeat
everything under two median-life normalisations because the mapping from absolute cycles to
life fraction is itself an estimate.
"""
import numpy as np

M = np.array([20., 50., 100., 200., 400., 700.])
R2 = {'bulk capacity': np.array([-0.013, -0.048, -0.037, 0.076, 0.549, 0.485]),
      'DeltaQ(V)': np.array([0.395, 0.720, 0.795, 0.886, 0.884, 0.804])}
CGRID = np.linspace(0.02, 1.0, 400)


def pava(y, w=None):
    """Pool-adjacent-violators: nearest non-decreasing sequence in weighted least squares."""
    y = np.asarray(y, float).copy()
    w = np.ones_like(y) if w is None else np.asarray(w, float).copy()
    lvl, wt, cnt = [], [], []
    for yi, wi in zip(y, w):
        lvl.append(yi); wt.append(wi); cnt.append(1)
        while len(lvl) > 1 and lvl[-2] > lvl[-1]:
            v2, w2, c2 = lvl.pop(), wt.pop(), cnt.pop()
            v1, w1, c1 = lvl.pop(), wt.pop(), cnt.pop()
            lvl.append((v1*w1 + v2*w2)/(w1 + w2)); wt.append(w1 + w2); cnt.append(c1 + c2)
    return np.repeat(lvl, cnt)


def eta_of(name, tmed):
    """Measured etaT on the life-fraction axis, monotonised and normalised."""
    frac = M/tmed
    y = pava(np.maximum(R2[name], 0.0))
    if y[-1] <= 0:
        raise ValueError('degenerate channel')
    x = np.concatenate([[0.0], frac, [1.0]])
    v = np.concatenate([[0.0], y, [max(y[-1], y.max())]])
    v = v/v[-1]
    o = np.argsort(x)
    return lambda c: np.interp(np.clip(c, 0, 1), x[o], v[o])


def cstar(eta, lam, k0):
    g = (k0 + CGRID)*(1.0/np.maximum(eta(CGRID), 1e-12) + lam)
    return float(CGRID[int(np.argmin(g))])


def switch_lambda(eta, k0, want_corner):
    """Bisect for the Lambda at which the verdict changes."""
    lo, hi = 1e-4, 1e4
    for _ in range(60):
        mid = np.sqrt(lo*hi)
        corner = cstar(eta, mid, k0) > 0.999
        if corner == want_corner:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo*hi)


print(__doc__)
print('=' * 78)
for tmed in (730.0, 806.0):
    print()
    print(f'--- median life taken as {tmed:.0f} cycles ---')
    etas = {k: eta_of(k, tmed) for k in R2}
    print(f'  measured etaT at c = 0.1, 0.25, 0.5, 0.8:')
    for k, e in etas.items():
        print(f'    {k:16s} ' + '  '.join(f'{e(c):.3f}' for c in (0.1, 0.25, 0.5, 0.8)))
    print()
    print(f'  {"kappa0":>7s} {"Lambda":>9s} {"c* capacity":>12s} {"c* dQ(V)":>10s} '
          f'{"verdicts differ":>16s}')
    for k0 in (0.05, 0.2, 0.5):
        for lam in (0.0, 0.1, 0.241, 1.0, 5.0, 20.0):
            ca, cb = cstar(etas['bulk capacity'], lam, k0), cstar(etas['DeltaQ(V)'], lam, k0)
            print(f'  {k0:7.2f} {lam:9.3f} {ca:12.3f} {cb:10.3f} '
                  f'{str(ca > 0.999 and cb < 0.999):>16s}')
    print()
    for k0 in (0.05, 0.2, 0.5):
        la = switch_lambda(etas['bulk capacity'], k0, True)
        lb = switch_lambda(etas['DeltaQ(V)'], k0, False)
        print(f'  kappa0 = {k0:.2f}:  capacity leaves the corner above Lambda = {la:.2f};  '
              f'dQ(V) reaches it below Lambda = {lb:.3f}')
        print(f'                 the two verdicts differ for every Lambda in '
              f'({lb:.3f}, {la:.2f}), a span of {la/max(lb,1e-9):.0f}x')
