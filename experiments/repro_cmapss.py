"""Instrumented rerun of the FD001 campaign (cost_cmapss.campaign): counts calls and failures of the least-squares solves that
impute lives, and calls that end at a search bound, so the number of valid versus failed draws can be reported (R1.5)."""
import sys, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '.'))
import cost_cmapss as cc
cnt = dict(calls=0, fail=0, upper=0, draws_bad=0)
_orig_ls = cc.least_squares
def ls(*a, **k):
    cnt['calls'] += 1
    try:
        r = _orig_ls(*a, **k)
    except Exception:
        cnt['fail'] += 1; raise
    if 'bounds' in k and len(k['bounds'][1]) == 1 and k['bounds'][1][0] > 30 and abs(r.x[0] - k['bounds'][1][0]) < 1e-3 * k['bounds'][1][0]:
        cnt['upper'] += 1
    return r
cc.least_squares = ls
for m in (60, 100):
    allrows = []
    for s in (0, 1, 2):
        base, rows, sel = cc.campaign(m, s, reps=40)
        allrows.append(rows)
        print(f'm={m} seed={s}: base RMSE {base:.0f}; (c, eligible, used, median eligible life): {[(c, ne, nu, round(ml)) for c, ne, nu, ml in sel]}', flush=True)
    for k, c in enumerate(cc.GRID):
        print(f'  c={c:.2f}: ' + ', '.join(f'{nm} {np.nanmean([r[k][1][nm] for r in allrows]):.2f}' for nm in cc.NAMES), flush=True)
print('least-squares calls', cnt['calls'], 'exceptions', cnt['fail'], 'solves ending at the upper life bound', cnt['upper'])
