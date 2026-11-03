"""Charging policy of every Severson cell: the stress variable of an accelerated-degradation reading of the data.

The cells of the three batches are cycled under 72 fast-charging policies written 'C1(Q1%)-C2': charge at C1 up to Q1 per cent state of charge,
then at C2 up to 80 per cent.  The time to reach 80 per cent, and the equivalent mean rate, is the stress.  Writes severson_policy.npz with the
name 'b<batch>c<index>' (as in dq_traj.npz), the policy string, C1, Q1, C2, the charge time to 80 per cent (minutes) and the mean rate (C).
"""
import os
import re
import sys
import numpy as np
import h5py

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'severson')
FILES = ['2017-05-12_batchdata_updated_struct_errorcorrect.mat',
         '2017-06-30_batchdata_updated_struct_errorcorrect.mat',
         '2018-04-12_batchdata_updated_struct_errorcorrect.mat']


def chars(f, ref):
    return ''.join(chr(int(c)) for c in np.asarray(f[ref]).ravel())


names, pol, C1, Q1, C2, life = [], [], [], [], [], []
for b, fn in enumerate(FILES, start=1):
    f = h5py.File(os.path.join(ROOT, fn), 'r')
    bt = f['batch']
    for i in range(bt['summary'].shape[0]):
        s = chars(f, bt['policy_readable'][i, 0])
        m = re.match(r'\s*([\d.]+)C\((\d+)%\)-([\d.]+)C', s)
        if not m:
            m4 = re.match(r'\s*([\d.]+)C-([\d.]+)C\(', s)
            if m4:
                # 'C1-C2(...)' notation not used by these batches
                continue
            continue
        names.append(f'b{b}c{i}'); pol.append(s.strip())
        C1.append(float(m.group(1))); Q1.append(float(m.group(2))); C2.append(float(m.group(3)))
        life.append(float(np.asarray(f[bt['cycle_life'][i, 0]]).ravel()[0]))
    f.close()
C1, Q1, C2, life = map(np.array, (C1, Q1, C2, life))
tmin = 60 * ((Q1 / 100) / C1 + ((80 - Q1) / 100) / C2)
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'severson_policy.npz'), name=np.array(names), policy=np.array(pol),
         C1=C1, Q1=Q1, C2=C2, tmin=tmin, rate=0.8 / (tmin / 60), life=life)
print(len(names), 'cells with a parsed policy; example', pol[:3], 'charge time range', tmin.min().round(1), tmin.max().round(1))
