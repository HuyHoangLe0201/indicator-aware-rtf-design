"""Window sensitivity of the capacity campaigns (Supplementary Material, campaign sizes): capacity channel at feature window m = 100 or 50, same Huber campaigns as the capacity rows of the main campaign table.
Usage: python capacity_window.py capacity M   -> mean over three seeds of RMSE / base for c = 0.35 ... 1.00."""
import contextlib
import io
import sys
import time

import numpy as np

import os
HERE = os.path.dirname(os.path.abspath(__file__))
import cost_real as CR

CR.DATA = os.path.join(HERE, '..', 'data', 'severson', 'dq_traj.npz')
GRID = (0.35, 0.50, 0.65, 0.80, 1.00)
ch, m = sys.argv[1], int(sys.argv[2])
t0 = time.time()
rows = []
for sd in (0, 1, 2):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        CR.run(ch, m, GRID, 21.0, seed=sd)
    d = {}
    el = {}
    for line in buf.getvalue().splitlines():
        p = line.split()
        if len(p) == 7 and p[0].replace('.', '', 1).isdigit():
            d[float(p[0])] = float(p[4])
            el[float(p[0])] = int(p[2])
    rows.append([d.get(c, np.nan) for c in GRID])
    print(ch, m, 'seed', sd, rows[-1], 'eligible', [el.get(c) for c in GRID], flush=True)
a = np.array(rows)
print(ch, m, 'MEAN', np.round(np.nanmean(a, 0), 2), 'sd', np.round(np.nanstd(a, 0), 2), 'time', round(time.time() - t0), flush=True)
