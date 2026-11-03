"""Load the NASA milling set and reduce each cut to one scalar per sensor.

Each run holds 9000 samples of six signals recorded across a single pass of the cutter.  The
pass begins and ends with the tool entering and leaving the workpiece, so the ends of the
record are transients that belong to the geometry of the cut rather than to the state of the
tool.  Only the interior is used, and the fraction retained is swept later, because the choice
of window is a modelling decision of exactly the kind that moved Upsilon on the battery data.

Flank wear is measured by microscope between runs, at irregular intervals and not after every
run, so it is interpolated over the run index to give a wear value for each cut.
"""
import numpy as np
import scipy.io as sio
from collections import defaultdict

CH = ['AE_table', 'AE_spindle', 'vib_table', 'vib_spindle', 'smcAC', 'smcDC']
PATH = r'D:\claudecode\generationbound\data\milling\3. Milling\mill.mat'


def _scalar(x, keep):
    """Root mean square of the interior of one cut."""
    x = np.asarray(x, float).ravel()
    n = len(x)
    lo = int(n*(1 - keep)/2)
    seg = x[lo:n - lo]
    seg = seg - np.median(seg)
    return float(np.sqrt(np.mean(seg**2)))


def load(keep=0.5):
    """Return {case: (runs, wear, {channel: trajectory})} over every run of every case."""
    d = sio.loadmat(PATH, struct_as_record=False, squeeze_me=True)['mill']
    g = defaultdict(list)
    for r in d:
        g[int(r.case)].append(r)
    out = {}
    for c, rs in g.items():
        rs = sorted(rs, key=lambda r: int(r.run))
        runs = np.array([int(r.run) for r in rs], float)
        vb = np.array([float(np.ravel(r.VB)[0]) if np.size(r.VB) else np.nan for r in rs])
        seen = np.isfinite(vb)
        if seen.sum() < 4:
            continue
        # wear is monotone by physics; the microscope reading is not, so enforce it before
        # interpolating, otherwise a single low reading invents a fall in damage
        w = np.maximum.accumulate(vb[seen])
        wear = np.interp(runs, runs[seen], w)
        tr = {ch: np.array([_scalar(getattr(r, ch), keep) for r in rs]) for ch in CH}
        out[c] = (runs, wear, tr)
    return out
