"""
Does the capacity / DeltaQ(V) ordering replicate on further public run-to-failure cell sets?

Input: per-cycle channels extracted from the BatteryLife processed sets (Zenodo record 19688272, CC-BY-4.0) by extract.py
(capacity = per-cycle maximum discharge capacity; DeltaQ(V) = log10 Var[Q_k(V) - Q_10(V)] on discharge, voltage grid
inside the cell's own voltage limits). Both channels are cut at the cycle life given in the data set's life labels and
normalized to it, then profiled with the estimator of the paper (profile() of xjtu_battery_replicate.py).

Usage: python batterylife_replicate.py <channels_dir> <labels_dir> DATASET [DATASET ...]
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xjtu_battery_replicate import profile, FRACS

REF = 10


def cell(rec, life):
    cap, dq = rec['cap'], rec['dq']
    K = min(int(life), len(cap))
    if K < 60:
        return None
    idx = np.arange(K)
    gc = np.isfinite(cap[:K]); gd = np.isfinite(dq[:K]); gd[:REF + 1] = False
    if gc.sum() < 60 or gd.sum() < 60:
        return None
    capc = np.interp(idx, idx[gc], cap[:K][gc])
    sel = np.arange(REF + 1, K)
    dqs = np.interp(sel, idx[gd], dq[:K][gd])
    return dict(capacity=-capc, dqv=dqs, tau_cap=(idx + 1)/K, tau_dq=(sel + 1)/K)


def run(name, chdir, labdir):
    res = np.load(os.path.join(chdir, name + '.npy'), allow_pickle=True).item()
    lab = json.load(open(os.path.join(labdir, name + '_labels.json')))
    cells = {}
    for member, rec in res.items():
        if 'err' in rec:
            continue
        key = os.path.basename(member)
        life = lab.get(key, lab.get(key.replace('--', '-#')))      # Tongji files use '--' where the labels use '-#'
        if life is None:
            continue
        c = cell(rec, life)
        if c:
            cells[key] = c
    out = dict(n=len(cells), total=len(res))
    if len(cells) < 5:
        return out
    for ch in ('capacity', 'dqv'):
        ups, c50s = [], {}
        for fr in FRACS:
            vals = [profile(v[ch], fr, v['tau_cap' if ch == 'capacity' else 'tau_dq']) for v in cells.values()]
            ups.append(float(np.nanmedian([u for _, u in vals])))
            c50s[fr] = [c for c, _ in vals]
        out[ch] = dict(ups=ups, c50=float(np.nanmedian(c50s[0.07])), c50_all=c50s[0.07])
    a, b = np.array(out['capacity']['c50_all']), np.array(out['dqv']['c50_all'])
    ok = np.isfinite(a) & np.isfinite(b)
    out['paired_frac'] = float(np.mean(a[ok] > b[ok])) if ok.any() else np.nan
    out['paired_n'] = int(ok.sum())
    return out


if __name__ == '__main__':
    chdir, labdir = sys.argv[1], sys.argv[2]
    print(f'{"set":9s} {"cells":>6s} | {"cap c50":>8s} {"cap Ups(3/7/15%)":>22s} | {"dqv c50":>8s} {"dqv Ups(3/7/15%)":>22s} | cap>dqv (paired)')
    for name in sys.argv[3:]:
        r = run(name, chdir, labdir)
        if 'capacity' not in r:
            print(f'{name:9s} {r["n"]:3d}/{r["total"]:<3d} too few usable cells'); continue
        f = lambda u: '/'.join(f'{x:5.2f}' for x in u)
        print(f'{name:9s} {r["n"]:3d}/{r["total"]:<3d} | {r["capacity"]["c50"]:8.3f} {f(r["capacity"]["ups"]):>22s} | '
              f'{r["dqv"]["c50"]:8.3f} {f(r["dqv"]["ups"]):>22s} | {r["paired_frac"]:.2f} of {r["paired_n"]}')
