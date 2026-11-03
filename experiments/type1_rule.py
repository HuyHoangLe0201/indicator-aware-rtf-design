"""Error control of the verdict rule "stop early only if the upper end of the interval for Upsilon is below the threshold"
(Proposition A.4, Appendix A; Section S3.11, Table S25).

Units follow D_i(tau) = tau^beta_i with beta_i = beta0 * exp(h z_i - h^2/2) (z standard normal, mean beta0), observed at M equally
spaced fractions of life with Gaussian noise of standard deviation sigma. Each unit's exponent is fitted by least squares on a grid,
Upsilon-hat = 2 mean(beta-hat) + 1 with a one-sided t interval (level 1 - alpha) over units. The rule declares "early stop" iff
Upsilon-hat + t * se < theta, with theta = (1 + Lambda) / (1 + kappa0).
The script reports, for each number of units n, the frequency of "early stop" when Upsilon = theta (should be <= alpha) and when
Upsilon = theta / 2 (power).
Usage: python type1_rule.py
"""
import numpy as np
from scipy import stats

K0, LAM, ALPHA = 0.05, 1.0, 0.025
THETA = (1 + LAM) / (1 + K0)
M = 60
REPS = 1500
tau = np.linspace(1.0 / M, 1.0, M)
grid = np.linspace(0.02, 8.0, 400)
base = tau[None, :] ** grid[:, None]          # grid x M
rng = np.random.default_rng(1)


def fit_beta(Y):
    """Least-squares exponent for each row of Y (units x M) on the grid."""
    sse = (Y**2).sum(1)[:, None] - 2 * Y @ base.T + (base**2).sum(1)[None, :]
    return grid[np.argmin(sse, axis=1)]


def one(n, ups, h, sigma):
    b0 = (ups - 1) / 2
    beta = b0 * np.exp(h * rng.standard_normal(n) - h * h / 2)
    Y = tau[None, :] ** beta[:, None] + sigma * rng.standard_normal((n, M))
    bh = fit_beta(Y)
    u = 2 * bh + 1
    se = u.std(ddof=1) / np.sqrt(n)
    return u.mean() + stats.t.ppf(1 - ALPHA, n - 1) * se < THETA


print('theta = %.3f, alpha = %.3f (one-sided)' % (THETA, ALPHA))
print('%5s %5s %6s | %12s %12s' % ('n', 'h', 'sigma', 'P(early|=th)', 'P(early|1.4)'))
for h in (0.1, 0.3):
    for sigma in (0.02, 0.05):
        for n in (10, 20, 40, 80):
            t1 = np.mean([one(n, THETA, h, sigma) for _ in range(REPS)])
            pw = np.mean([one(n, 1.4, h, sigma) for _ in range(REPS)])
            print('%5d %5.1f %6.2f | %12.3f %12.3f' % (n, h, sigma, t1, pw), flush=True)
