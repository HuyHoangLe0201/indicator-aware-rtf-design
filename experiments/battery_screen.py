"""The falsification test of Section VII-G, on the Severson cells.

The trichotomy rests on the battery pair, and a sceptic may hold the pair is one channel:
that DeltaQ(V) is a saturating readout of the same capacity loss. Were that so, DeltaQ(V)
could carry no information about T beyond the capacity trajectory. This runs exactly the
screen escape_screen.py runs on the other two domains, with capacity as the damage channel
and DeltaQ(V) as the candidate, and then runs it in reverse.

Method is escape_screen's, unchanged: control non-parametrically on a four-feature summary of
the damage trajectory (level, fade so far, recent rate, curvature), take the Spearman
correlation of the two residuals, and calibrate it by permuting the candidate among nearest
neighbours in the control space so the null keeps the local geometry.
"""
import os

import numpy as np

from escape_screen import RNG, knn_perm, r2, summarize  # same machinery, same defaults

HERE = os.path.dirname(os.path.abspath(__file__))
PROBES = (100, 150, 200)
MIN_LIFE_MULT = 1.3


def load():
    d = np.load(os.path.join(HERE, '..', 'data', 'severson', 'dq_traj.npz'),
                allow_pickle=True)
    DQ, QD, life = d['DQ'], d['QD'], np.asarray(d['life'], float)
    cells = []
    for i in range(len(DQ)):
        # DQ[i] is (2, N): row 0 the cycle grid it was evaluated on, row 1 the value.
        g = np.asarray(DQ[i], float)
        cyc, val = g[0], g[1]
        cap = np.asarray(QD[i], float).ravel()
        n = len(cap)
        if n < 130 or not np.isfinite(life[i]):
            continue
        ok = np.isfinite(cyc) & np.isfinite(val)
        if ok.sum() < 20:
            continue
        # put DeltaQ(V) on the per-cycle index that QD and summarize() both use
        dqi = np.interp(np.arange(n), cyc[ok], val[ok])
        okc = np.isfinite(cap)
        cap = np.interp(np.arange(n), np.flatnonzero(okc), cap[okc])
        # capacity oriented to increase with damage, as the screen requires
        cells.append({'T': float(life[i]), 'capacity': -cap, 'dqv': dqi})
    return cells


def probe(cells, damage, cand, k):
    use = [u for u in cells if u['T'] >= k*MIN_LIFE_MULT and len(u[damage]) > k]
    if len(use) < 20:
        return None
    y = np.log(np.array([u['T'] for u in use], float))
    Xd = np.array([summarize(u[damage], k) for u in use])
    x = np.array([np.median(u[cand][max(0, k-3):k+1]) for u in use], float)
    Xc = np.array([summarize(u[cand], k) for u in use])
    rho, p = knn_perm(Xd, x, y)
    base = r2(Xd, y, quad=False)
    joint = r2(np.column_stack([Xd, Xc]), y, quad=False)
    return len(use), rho, p, base, joint


if __name__ == '__main__':
    cells = load()
    print(f'{len(cells)} cells with usable paired trajectories')
    for damage, cand, tag in (('capacity', 'dqv', 'DeltaQ(V) beyond capacity'),
                              ('dqv', 'capacity', 'capacity beyond DeltaQ(V)')):
        print()
        print(tag)
        print(f'  {"cycle":>6s} {"n":>5s} {"rho|control":>12s} {"perm p":>9s} '
              f'{"R2 control":>11s} {"R2 joint":>9s}')
        for k in PROBES:
            r = probe(cells, damage, cand, k)
            if r is None:
                print(f'  {k:6d}   too few cells')
                continue
            n, rho, p, base, joint = r
            print(f'  {k:6d} {n:5d} {rho:12.3f} {p:9.4f} {base:11.3f} {joint:9.3f}')
    print()
    print('This is a different screen from the one the paper reports: it controls on capacity features alone and uses 4000 permutations')
    print('(so p cannot fall below about 2e-4). The numbers of Appendix D.1 and Section 7.7(i) come from readout_test2.py (forward test:')
    print('rho = -0.80, -0.82, -0.85 at cycles 100/150/200; R2 0.53 -> 0.84 at cycle 100) and readout_test.py (reverse test at cycle 200,')
    print('controlling on DeltaQ(V) in four bins: rho = +0.14, p = 0.16). Run those scripts to reproduce them.')
