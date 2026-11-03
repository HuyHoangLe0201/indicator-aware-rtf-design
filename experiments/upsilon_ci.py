"""Bootstrap intervals over units for every entry of the indicator figure.

Each published value is a fleet aggregate of per-unit shape estimates, so the sampling
question it faces is what happens when the fleet is redrawn. Units are resampled with
replacement and the same aggregate recomputed, which is the only resampling that respects
the estimator: the per-unit estimate is a deterministic function of one unit's trajectory,
so resampling units and refitting inside each resample give the same answer.

Two pipelines are needed because the survey uses two. Turbofans and the battery come from the
parametric shape estimator whose per-unit values are cached by upsilon_dist.py; the bearings
come from the nonparametric profile of xjtu_survey.py, because the parametric fit collapses
to zero on the frequency-domain bearing indicators. The bearing arm is rebuilt here from the
cached features and reproduces the published medians exactly, which is the check that it is
the right pipeline.
"""
import json
import numpy as np

B = 20000
KAP0 = 0.05
LAMB = {'battery': 0.241, 'C-MAPSS': 0.165, 'bearing': 1.382}
DIST = r'D:\claudecode\generationbound\experiments\upsilon_dist.npz'
FEAT = r'D:\claudecode\generationbound\experiments\xjtu_features.npz'
FR = [0.03, 0.07, 0.15]


def bearing_matrix():
    """Per-unit Upsilon at each smoothing width, on the fleet and decimation the paper used."""
    import sys
    sys.path.insert(0, r'D:\claudecode\generationbound\experiments')
    from xjtu_survey import profile
    F = {k: v.item() for k, v in np.load(FEAT, allow_pickle=True).items()}
    S = slice(None, None, 2)
    use = [b for b in F if len(F[b]['RMS'][S]) >= 60]
    out = {}
    for ind in ['spec centroid', 'kurtosis', 'HF 5-10kHz', 'HF 2-5kHz', 'RMS', 'peak']:
        out[ind] = np.array([[profile(F[b][ind][S], fr)[1] for b in use] for fr in FR])
    return out, len(use)


def boot_widths(A, rng):
    """Resample units, take the fleet median at each width, then the median over widths."""
    n = A.shape[1]
    out = np.empty(B)
    for k in range(B):
        i = rng.integers(0, n, n)
        out[k] = np.nanmedian([np.nanmedian(A[j][i]) for j in range(A.shape[0])])
    return out


def boot_flat(v, rng):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    n = len(v)
    return np.nanmedian(v[rng.integers(0, n, (B, n))], axis=1)


if __name__ == '__main__':
    rng = np.random.default_rng(20250802)
    d = np.load(DIST, allow_pickle=True)
    bear, nb = bearing_matrix()

    rows = [('battery', 'bulk capacity', 9.97, d['battery|bulk capacity']),
            ('C-MAPSS', 'sensor 2', 7.01, d['C-MAPSS|sensor 2']),
            ('C-MAPSS', 'sensor 3', 7.05, d['C-MAPSS|sensor 3']),
            ('C-MAPSS', 'sensor 4', 7.37, d['C-MAPSS|sensor 4']),
            ('C-MAPSS', 'sensor 7', 7.16, d['C-MAPSS|sensor 7']),
            ('C-MAPSS', 'sensor 8', 7.12, d['C-MAPSS|sensor 8']),
            ('C-MAPSS', 'sensor 9', 8.53, d['C-MAPSS|sensor 9']),
            ('C-MAPSS', 'PC1 of 14', 7.54, d['C-MAPSS|PC1 of 17']),
            ('bearing', 'spectral centroid', 2.37, bear['spec centroid']),
            ('bearing', 'kurtosis', 2.48, bear['kurtosis']),
            ('bearing', 'HF 5-10 kHz', 3.86, bear['HF 5-10kHz']),
            ('bearing', 'HF 2-5 kHz', 4.30, bear['HF 2-5kHz']),
            ('bearing', 'RMS', 9.43, bear['RMS']),
            ('bearing', 'peak', 11.67, bear['peak'])]

    out = {}
    print(f'{"domain":9s} {"indicator":18s} {"pub":>6s} {"median":>7s} '
          f'{"lo95":>7s} {"hi95":>7s} {"margin":>7s} {"P(wrong side)":>14s}')
    for dom, lbl, pub, arr in rows:
        arr = np.asarray(arr)
        draws = boot_widths(arr, rng) if arr.ndim == 2 else boot_flat(arr, rng)
        thr = (1 + LAMB[dom])/(1 + KAP0)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        pw = float(np.mean((draws > thr) != (pub > thr)))
        out[f'{dom}|{lbl}'] = [float(lo), float(hi), pw]
        print(f'{dom:9s} {lbl:18s} {pub:6.2f} {np.median(draws):7.2f} {lo:7.2f} {hi:7.2f} '
              f'{pub/thr:7.2f} {pw:14.3f}')
    with open(r'D:\claudecode\generationbound\experiments\upsilon_ci.json', 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'{chr(10)}written to upsilon_ci.json; bearing fleet {nb} units')
