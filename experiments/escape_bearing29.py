"""Re-run the bearing escape-channel screen on the fleet the paper actually reports.

The first run admitted every cached bearing with at least 30 snapshots, which kept 13 of the 15
XJTU-SY units and gave 30 bearings in total.  The rest of the paper rests on 12 XJTU-SY units,
three of the fifteen having failed too quickly, so the screen and the c_50 evidence were
quoting two different totals for the same two rigs.  Here the shortest XJTU-SY record is
dropped so that both rest on 12 + 17 = 29.
"""
import numpy as np, os
from escape_screen import screen

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bearing_feats.npz')
IND = ['RMS', 'peak', 'kurtosis', 'HF 2-5kHz', 'HF 5-10kHz', 'spec centroid']

z = np.load(CACHE, allow_pickle=True)
names = sorted({k.split('|')[0] for k in z.files})
lens = {n: len(z[n + '|RMS']) for n in names}

xj = sorted([n for n in names if n.startswith('X:')], key=lambda n: lens[n])
dropped = xj[0]
print(f'dropping the shortest XJTU-SY record: {dropped} ({lens[dropped]} snapshots)')
keep = [n for n in names if n != dropped]

units = []
for n in keep:
    rec = {k: np.asarray(z[n + '|' + k], float) for k in IND}
    rec['T'] = float(len(rec['RMS']))
    rec['rig'] = n[0]
    units.append(rec)

nx = sum(u['rig'] == 'X' for u in units)
np_ = sum(u['rig'] == 'P' for u in units)
print(f'{len(units)} bearings: {nx} XJTU-SY + {np_} PRONOSTIA')
assert (nx, np_) == (12, 17), 'fleet does not match the paper'

screen('Bearings, both rigs', units, 'RMS', [k for k in IND if k != 'RMS'],
       probes=(20, 40, 60))
