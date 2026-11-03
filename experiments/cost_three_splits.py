"""Table 4 as the caption describes it: averaged over three held-out splits.

cost_real.py runs one split. The table in the paper averages three, so this drives the same
function over three seeds and prints the mean, which is what the caption claims.
"""
import contextlib
import io as _io

import numpy as np

import cost_real as CR

GRID = (0.35, 0.50, 0.65, 0.80, 1.00)
SEEDS = (0, 1, 2)
CH = (('dq', 100, 'DeltaQ(V)'), ('capacity', 400, 'bulk capacity'))


def column(res, key):
    """cost_real.run returns rows; pull the 'vs base' ratio per c however it is shaped."""
    if isinstance(res, dict):
        return np.array([res[c][key] for c in GRID], float)
    arr = np.asarray(res, float)
    return arr


if __name__ == '__main__':
    print(f'{"channel":16s} ' + ' '.join(f'{c:>7.2f}' for c in GRID))
    print('-'*62)
    for name, m, label in CH:
        runs = []
        for sd in SEEDS:
            buf = _io.StringIO()
            with contextlib.redirect_stdout(buf):
                CR.run(name, m, GRID, 21.0, seed=sd)
            rows = []
            for line in buf.getvalue().splitlines():
                p = line.split()
                if len(p) == 7 and p[0].replace('.', '', 1).isdigit():
                    rows.append((float(p[0]), float(p[4])))
            d = dict(rows)
            runs.append([d.get(c, np.nan) for c in GRID])
        a = np.array(runs, float)
        print(f'{label:16s} ' + ' '.join(f'{v:7.2f}' for v in np.nanmean(a, 0))
              + '   <- mean of three splits')
        for sd, row in zip(SEEDS, a):
            print(f'{"  seed "+str(sd):16s} ' + ' '.join(f'{v:7.2f}' for v in row))
    print()
    print('paper Table 4, c = 0.50 0.65 0.80 1.00:')
    print('   DeltaQ(V)      0.75 0.59 0.62 0.62')
    print('   bulk capacity  3.93 2.10 0.70 0.56')
