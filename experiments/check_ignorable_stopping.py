"""Simulation behind Lemma 1 (ignorable stopping), Section 2.3 and its proof in Appendix A.

Units are removed at the first noisy reading above the failure level (observation model O3). The Fisher information
about ln T of one stopped unit should equal the expected path information up to the stopping index (Wald's identity),
with no extra term in D'(1)^2 / sigma^2. The printed ratio of E[score^2] to the expected path information is close to one.
Usage: python check_ignorable_stopping.py
"""
import numpy as np

rng = np.random.default_rng(1)
beta, sigma, n_read, reps = 4.5, 0.05, 400, 20000
tau = np.arange(1, n_read + 1) / n_read * 1.3          # readings run past nominal failure (tau = 1 at index n_read / 1.3)
mean_curve = tau**beta
s = beta * tau**beta                                   # s(tau) = tau D'(tau) = -dD / d ln T

score_sq, path_info, stop_idx = [], [], []
for _ in range(reps):
    x = mean_curve + sigma * rng.standard_normal(n_read)
    crossing = np.where(x >= 1.0)[0]
    j = crossing[0] + 1 if len(crossing) else n_read   # stop at the first noisy crossing
    score = np.sum(s[:j] * (x[:j] - mean_curve[:j])) / sigma**2
    score_sq.append(score**2)
    path_info.append(np.sum(s[:j]**2) / sigma**2)
    stop_idx.append(j)

print('E[score^2]                         %.1f' % np.mean(score_sq))
print('E[path information to the stop]    %.1f' % np.mean(path_info))
print('mean stopping index %.1f of %d (nominal failure at index %d)' % (np.mean(stop_idx), n_read, int(n_read / 1.3)))
print('ratio E[score^2] / E[path information]: %.3f' % (np.mean(score_sq) / np.mean(path_info)))
