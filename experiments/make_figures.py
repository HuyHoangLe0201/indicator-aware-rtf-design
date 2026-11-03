"""
Paper figures: vector PDFs, typography matched to the IEEEtran/Times body text.

Form (step 1).  Every figure is asked to carry a DECISION, not a statistic, and none of them
may restate a table.
  fig_densities  pairs the three information densities with what a suspended test is worth,
                 because a bare density is not readable as a decision.
  fig_channels   shows the mechanism before the consequence: the same three exemplar cells
                 under both indicators on a log cycle axis, then the measured R^2.  Without
                 panels (a) and (b) the reader has to take the separation on trust.
  fig_twostage   plots Theorem 10's cold-start pole, which appears in no table, and then the
                 gain as a continuous function of budget rather than four tabulated points.
  fig_phase      is a diverging bar from a baseline that means something.  Each domain has
                 its own corner threshold, so the absolute Upsilon axis compared most
                 indicators against a line that did not apply to them; the margin to the
                 domain's own threshold fixes that and makes the bar length meaningful.  The
                 numeric column carries the raw Upsilon and n, so the figure replaces the
                 Upsilon columns of the survey tables instead of duplicating them.
  fig_eta        is the payoff: the measured eta_T(c) curves, which are what a stopping rule
                 actually buys.

Color (steps 2-3): four-slot categorical palette in fixed order, validated against the six
checks.  All hard checks pass; the green/purple pair sits in the 6-8 CVD floor band, so
wherever those two appear together they also carry distinct marker shapes and direct labels.
Marks (step 4): thin lines, recessive spines, no decorative grid, direct labels, text in ink
rather than series color.
"""
import json
import numpy as np, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

OUT = os.environ.get('FIG_OUT') or os.path.join(os.path.dirname(__file__), '..')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 8.4,
    'xtick.labelsize': 8.4, 'ytick.labelsize': 8.4,
    'axes.linewidth': .5, 'lines.linewidth': 1.1,
    'xtick.major.width': .5, 'ytick.major.width': .5,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'legend.frameon': False, 'pdf.fonttype': 42,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.015,
})

BLUE, VERM, GREEN, PURP = '#0072B2', '#D55E00', '#009E73', '#CC79A7'
INK, MUTED, RULE = '#1a1a1a', '#5c5c5c', '#c9c9c9'
# 6.48in is \columnwidth of cas-sc (468.33pt); drawing at the printed width means
# LaTeX scales by one and the 9pt figure text sits just under the 10pt body.
W, BETA = 6.48, 4.483
S = 2*BETA + 1

def tidy(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    return ax

# ---------------------------------------------------------------- Fig 1
def fig_densities():
    t = np.linspace(1e-3, 1, 700)
    Dp, dth = BETA*t**(BETA-1), t**BETA*np.log(t)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 4.36),
                                 gridspec_kw={'height_ratios': [1, 1.12], 'hspace': .62})
    tidy(a1); tidy(a2)

    # (a) where the information sits
    for lbl, v, c, ls in [(r'$\mathcal{I}$, state', Dp**2, BLUE, '-'),
                          (r'$\mathcal{I}_T$, failure time', t**2*Dp**2, VERM, '-'),
                          (r'$\mathcal{I}_\theta$, model parameter', dth**2, GREEN, (0, (4, 1.6)))]:
        a1.plot(t, v/np.trapezoid(v, t), color=c, ls=ls, label=lbl)
    dT = t**2*Dp**2; dT = dT/np.trapezoid(dT, t)
    a1.fill_between(t, 0, dT, where=(t >= .8), color=VERM, alpha=.13, lw=0, zorder=0)
    a1.plot([.8, .8], [0, dT[np.argmin(abs(t-.8))]], color=VERM, lw=.5,
            ls=(0, (1.5, 1.5)), zorder=1)
    a1.text(.925, 2.6, '89%', fontsize=7.9, color=INK, ha='center', va='center')
    a1.set_xlim(0, 1.005); a1.set_ylim(0, 10.4); a1.set_yticks([0, 5, 10])
    a1.set_xlabel(r'normalized time $\tau$', labelpad=1.5)
    a1.set_ylabel('information\nper unit time', linespacing=1.1)
    a1.legend(loc='upper left', borderpad=0, labelspacing=.3, handletextpad=.5,
              handlelength=1.5, fontsize=7.9)
    a1.plot([1], [0], 'o', ms=3.2, mfc='white', mec=GREEN, mew=1.0, zorder=6, clip_on=False)
    a1.annotate('parameter information\nvanishes at failure', xy=(1, 0),
                xytext=(.685, 7.6), fontsize=7.8, color=INK, ha='center', linespacing=1.15,
                arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=.55, shrinkA=2,
                                shrinkB=3, connectionstyle='arc3,rad=-0.26'))
    a1.set_title('(a) Distribution of information over the test', fontsize=8.4, color=INK, pad=3.5)

    # (b) what a test stopped at tau is worth
    c = np.linspace(.2, 1, 500); L = np.log(c)
    eT = c**S
    eth = c**S*(S**2*L**2/2 - S*L + 1)
    a2.fill_between(c, eT, eth, color=MUTED, alpha=.11, lw=0, zorder=0)
    a2.semilogy(c, eT, color=VERM, lw=1.2,
                label='information gained in training')
    a2.semilogy(c, eth, color=GREEN, ls=(0, (4, 1.6)), lw=1.2,
                label=r'$\eta_\theta$, shape information')
    a2.axhline(1, color=RULE, lw=.5)
    i5 = np.argmin(abs(c-.5))
    a2.plot([.5, .5], [eT[i5], eth[i5]], color=INK, lw=.7, zorder=4)
    a2.plot([.5, .5], [eT[i5], eth[i5]], 'o', ms=2.6, color=INK, zorder=5)
    a2.annotate(r'$32\times$', xy=(.5, np.sqrt(eT[i5]*eth[i5])), xytext=(6, 0),
                textcoords='offset points', fontsize=7.9, color=INK, va='center')
    h, l = a2.get_legend_handles_labels()
    a2.legend(h[::-1], l[::-1], loc='upper left', borderpad=0, labelspacing=.3,
              handletextpad=.5, handlelength=1.5, fontsize=7.9)
    a2.set_xlim(.2, 1.02); a2.set_ylim(1e-6, 12)
    a2.set_yticks([1e-6, 1e-4, 1e-2, 1])
    a2.set_yticklabels(['0.0001%', '0.01%', '1%', '100%'])
    a2.set_xticks([.2, .4, .6, .8, 1.0])
    a2.set_xlabel(r'stopped at fraction $c$ of life', labelpad=1.5)
    a2.set_ylabel('value relative to' + chr(10) + 'a complete test', linespacing=1.1)
    a2.set_title('(b) Value of a suspended test', fontsize=8.4, color=INK, pad=3.5)
    fig.savefig(f'{OUT}/fig1.pdf'); plt.close(fig)

# ---------------------------------------------------------------- Fig 2
#   Three exemplar cells carry the argument; the other 116 are context.  Panels (a) and (b)
#   show the SAME three cells under the two indicators, so the reader can see that the
#   ordering by lifetime is absent from capacity inside the prediction window and already
#   present in the DeltaQ(V) signature.  Panel (c) is the measured consequence.
EX = [('shortest', 148, '#E6A8C4'), ('median', 719, '#C4568B'), ('longest', 1836, '#6E2247')]


def _severson():
    return np.load(os.path.join(os.path.dirname(__file__), '..', 'data', 'severson',
                                'dq_traj.npz'), allow_pickle=True)


def fig_channels():
    d = _severson()
    DQ, QD, life = d['DQ'], d['QD'], d['life']
    order = np.argsort(life)
    picks = [order[0], order[len(order)//2], order[-1]]

    fig = plt.figure(figsize=(W, 4.55))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.08], hspace=.46, wspace=.44,
                          left=.135, right=.985, top=.905, bottom=.105)
    a1, a2, a3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    for a in (a1, a2, a3):
        tidy(a)

    def cap_curve(k):
        q = np.asarray(QD[k], float)
        q = q[np.isfinite(q) & (q > .2) & (q < 1.5)]
        if len(q) < 80:
            return None, None
        q = np.median(np.lib.stride_tricks.sliding_window_view(
            np.pad(q, 4, mode='edge'), 9), axis=1)               # de-spike, 9-cycle median
        q = q/np.median(q[:10])
        dead = np.flatnonzero(q <= .8)
        if len(dead):
            q = q[:dead[0]+1]
        return np.arange(1, len(q)+1), q

    def dq_curve(k):
        ms, vs = DQ[k]
        vs = np.median(np.lib.stride_tricks.sliding_window_view(
            np.pad(np.asarray(vs, float), 2, mode='edge'), 5), axis=1)
        return np.asarray(ms, float), vs

    # ---- (a) and (b): every cell faint, three exemplars in a light-to-dark ramp
    for ax, curve in ((a1, cap_curve), (a2, dq_curve)):
        for k in range(len(QD)):
            if k in picks:
                continue
            x, y = curve(k)
            if x is None:
                continue
            ax.plot(x, y, color=RULE, lw=.35, alpha=.75, zorder=1)
        for k, (_, yrs, col) in zip(picks, EX):
            x, y = curve(k)
            ax.plot(x, y, color=col, lw=1.15, zorder=4, solid_capstyle='round')
        ax.axvspan(10, 100, color=BLUE, alpha=.15, lw=0, zorder=0)
        ax.set_xscale('log'); ax.set_xlim(10, 2100)
        ax.set_xticks([10, 100, 1000]); ax.set_xticklabels(['10', '100', '1000'])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel('cycle', labelpad=1.5)

    a1.axhline(.8, color=INK, lw=.5, ls=(0, (2, 2)), zorder=3)
    a1.set_ylim(.788, 1.012)
    a1.set_yticks([.8, .9, 1.0]); a1.set_yticklabels(['80%', '90%', '100%'])
    a1.set_ylabel('capacity relative' + chr(10) + 'to initial cycles', linespacing=1.1,
                  fontsize=8.1)
    a1.set_title('(a) bulk capacity', fontsize=8.4, color=INK, pad=13)
    a1.text(.5, 1.030, r'at cycle 100: $|\rho|=0.05$', transform=a1.transAxes, fontsize=7.4,
            color=INK, ha='center', va='bottom')

    a2.set_ylim(-6.2, -1.4)
    a2.set_yticks([-6, -4, -2])
    a2.set_ylabel(r'$\log_{10}\mathrm{Var}\,[Q_m(V)\!-\!Q_{10}(V)]$', fontsize=7.4,
                  labelpad=1.0)
    a2.set_title(r'(b) $\Delta Q(V)$ signature', fontsize=8.4, color=INK, pad=13)
    a2.text(.5, 1.030, r'at cycle 100: $|\rho|=0.86$', transform=a2.transAxes, fontsize=7.4,
            color=INK, ha='center', va='bottom')
    a2.text(196, -5.10, 'cycle life', fontsize=7.1, color=INK, ha='left', va='center')
    for j, (lbl, yrs, col) in enumerate(EX):
        yy = -5.40 - .29*j
        a2.plot([205, 330], [yy, yy], color=col, lw=1.15, solid_capstyle='round')
        a2.text(375, yy, f'{yrs}', fontsize=7.1, color=INK, ha='left', va='center')

    # ---- (c) the measured consequence
    # Windows up to 400 cycles are read on at least 116 of the 119 cells.  Longer windows exist only for the
    # cells that live that long (86 at 500 cycles, 65 at 700), so their leave-one-out R^2 falls for a reason that has
    # nothing to do with information (the survivors' log life varies less); dqv_mar.py corrects it for DeltaQ(V).
    m = np.array([20, 50, 100, 200, 400])
    cap = np.array([-0.013, -0.048, -0.037, 0.076, 0.549])
    dq = np.array([0.395, 0.720, 0.795, 0.886, 0.884])
    a3.axhspan(-.20, 0, color=RULE, alpha=.30, lw=0, zorder=0)
    a3.axvspan(17, 100, color=BLUE, alpha=.15, lw=0, zorder=0)
    a3.axhline(0, color=MUTED, lw=.5, zorder=1)
    a3.semilogx(m, dq, 'o-', color=BLUE, ms=3.4, mfc='white', mew=1.0, zorder=3)
    a3.semilogx(m, cap, 's--', color=VERM, ms=3.2, mfc='white', mew=1.0, zorder=3)
    a3.text(170, .955, r'$\Delta Q(V)$ signature', fontsize=8.1, color=INK, ha='left',
            va='center')
    a3.text(190, .33, 'bulk capacity', fontsize=8.1, color=INK, ha='left', va='center')
    a3.text(21, -.145, 'no predictive information', fontsize=7.4, color=MUTED, va='center')
    a3.text(23, .21, 'region in which the' + chr(10) + 'design decision is made', fontsize=7.2,
            color=INK, ha='left', va='center', linespacing=1.15)
    a3.set_xlabel('observation window $m$: cycles, and (fraction of life)', labelpad=1.5)
    a3.set_ylabel(r'leave-one-out $R^2$' + chr(10) + r'for $\log T$', linespacing=1.1,
                  fontsize=8.1)
    # The window is given twice: in cycles, which is what the experiment controls, and as a
    # median fraction of life, which is the design variable c of the theory.
    frac = ['0.03', '0.07', '0.14', '0.28', '0.56']
    a3.set_xticks(m)
    a3.set_xticklabels([f'{v}\n({f})' for v, f in zip(m, frac)], linespacing=1.25)
    a3.xaxis.set_minor_locator(NullLocator())
    a3.set_xlim(17, 560); a3.set_ylim(-.20, 1.02)
    a3.set_yticks([0, .5, 1.0])
    a3.set_title('(c) Predictive content of each indicator, 119 cells', fontsize=8.4, color=INK,
                 pad=3.5)
    fig.savefig(f'{OUT}/fig3.pdf'); plt.close(fig)

# ---------------------------------------------------------------- earlier draft (not in the paper)
def fig_eta():
    """What Upsilon means on the bench: the share of the extrapolation information a test
    stopped at fraction c of life has actually collected.  Measured, not assumed."""
    z = np.load(os.path.join(os.path.dirname(__file__), 'eta_curves.npz'), allow_pickle=True)
    fig, ax = plt.subplots(figsize=(W, 3.06)); tidy(ax)

    series = [('battery capacity', VERM, '-', 'bulk capacity, 119 cells'),
              ('bearing RMS', PURP, '-', 'bearing RMS, 12 bearings'),
              ('C-MAPSS sensor 4', GREEN, (0, (4, 1.6)), 'turbofan sensor, 100 units')]
    for key, col, ls, lbl in series:
        c, med, q1, q3 = z[key]
        ax.fill_between(c, q1, q3, color=col, alpha=.13, lw=0, zorder=1)
        ax.plot(c, med, color=col, ls=ls, lw=1.15, zorder=3)

    zq = np.load(os.path.join(os.path.dirname(__file__), 'dqv_info_curve.npz'))
    ce = np.concatenate([[0], zq['c']]); re_ = np.minimum(np.concatenate([[0], zq['eta']]), 1.0)
    ax.fill_between(ce, np.concatenate([[0], zq['eta_lo']]), np.minimum(np.concatenate([[0], zq['eta_hi']]), 1.04),
                    color=BLUE, alpha=.10, lw=0, zorder=1)
    ax.plot(ce[1:], re_[1:], 'o', color=BLUE, ms=3.4, mfc='white', mew=1.0, zorder=5)
    ax.plot(ce, re_, color=BLUE, ls=(0, (1.2, 1.2)), lw=1.0, zorder=4)

    ax.axvline(.8, color=INK, lw=.5, ls=(0, (2, 2)), zorder=2)
    ax.text(.8, 1.008, 'stop at 80% of life', transform=ax.get_xaxis_transform(),
            fontsize=7.2, color=INK, ha='center', va='bottom')
    for key, col, lab, yl in [('C-MAPSS sensor 4', GREEN, '35%', .345),
                              ('bearing RMS', PURP, '16%', .180),
                              ('battery capacity', VERM, '7%', .020)]:
        c, med, _, _ = z[key]
        v = med[np.argmin(abs(c-.8))]
        ax.plot([.8], [v], 'o', ms=2.8, color=col, mec='white', mew=.5, zorder=6)
        ax.text(.818, yl, lab, fontsize=7.3, color=INK, ha='left', va='center')

    ax.text(.355, .845, r'$\Delta Q(V)$ signature', fontsize=7.9, color=INK, ha='left',
            va='center')
    ax.text(.355, .765, 'predictive odds, corrected for survival' + chr(10)
            + '(a different object, drawn here for scale)', fontsize=7.0, color=MUTED, ha='left',
            va='top', linespacing=1.15)
    hd = [plt.Line2D([], [], color=c, ls=s, lw=1.15) for _, c, s, _ in series]
    ax.legend(hd, [l for *_, l in series], loc='upper left',
              bbox_to_anchor=(.075, .60), borderpad=0, labelspacing=.32,
              handletextpad=.5, handlelength=1.6, fontsize=7.4)

    ax.set_xlim(0, 1.005); ax.set_ylim(0, 1.04)
    ax.set_xticks([0, .2, .4, .6, .8, 1.0])
    ax.set_xticklabels(['0', '20%', '40%', '60%', '80%', '100%'])
    ax.set_yticks([0, .5, 1.0]); ax.set_yticklabels(['0', '50%', '100%'])
    ax.set_xlabel('test stopped at this fraction of life', labelpad=1.5)
    ax.set_ylabel('share of the extrapolation' + chr(10) + 'information collected',
                  linespacing=1.1)
    fig.savefig(f'{OUT}/fig_eta_draft.pdf'); plt.close(fig)


# ---------------------------------------------------------------- Fig 3
#   Each domain has its OWN corner threshold (1+Lambda)/(1+kappa0), so an absolute Upsilon
#   axis compares indicators against a line that does not apply to most of them.  Plotting
#   the margin Upsilon(1+kappa0)/(1+Lambda) puts every indicator on a common decision scale
#   with a baseline at 1 that means something, so bar length encodes the safety margin
#   rather than the distance to an arbitrary axis clip.  The numeric column keeps the raw
#   Upsilon, so the figure replaces the Upsilon columns of Tables III and IV.
KAP0 = 0.05
LAMB = {'battery': 0.241, 'C-MAPSS': 0.165, 'bearing': 1.382}


def fig_upsilon():
    """Fig. 4: Upsilon relative to the threshold of its domain, in two panels on one axis.

    (a) every indicator of the three primary domains (the survey of Section 6.2-6.4);
    (b) the damage indicator of every cell and turbofan fleet of Table 7 (twelve cell fleets,
        four C-MAPSS subsets), read by the rule of Section 6.1.
    The decade width is the same in both panels and the right edge is shared, so the decision
    boundary 1 falls on one vertical line and a bar in (b) is compared directly with one in (a).
    """
    from matplotlib.transforms import blended_transform_factory as btf
    #        domain      indicator                     Upsilon   n
    rows = [('battery', r'$\Delta Q(V)$ signature',      0.24, 119),
            ('battery', 'bulk capacity',                 9.97, 119),
            ('C-MAPSS', 'sensor 2',                      7.01, 100),
            ('C-MAPSS', 'sensor 3',                      7.05, 100),
            ('C-MAPSS', 'sensor 4',                      7.37, 100),
            ('C-MAPSS', 'sensor 7',                      7.16, 100),
            ('C-MAPSS', 'sensor 8',                      7.12, 100),
            ('C-MAPSS', 'sensor 9',                      8.53,  96),
            ('C-MAPSS', 'PC1 of 14',                     7.54, 100),
            ('bearing', 'spectral centroid',             2.37,  12),
            ('bearing', 'kurtosis',                      2.48,  12),
            ('bearing', 'HF 5-10 kHz',                   3.86,  12),
            ('bearing', 'HF 2-5 kHz',                    4.30,  12),
            ('bearing', 'RMS',                           9.43,  12),
            ('bearing', 'peak',                         11.67,  12)]
    # DeltaQ(V) is read against the threshold of the predictive estimator, Proposition 6(b): theta_b = 1.27 at the
    # battery value of Lambda and the odds at c = 1 of dqv_info_curve.npz (16.7); the other rows use Theorem 2.
    THB = 1.27
    marg = [U/THB if lbl.startswith(r'$\Delta Q') else U*(1+KAP0)/(1+LAMB[d]) for d, lbl, U, _ in rows]
    # Four entries are excluded, for two different reasons.  Two bearing entries fail the
    # smoothing-stability requirement: their Upsilon drifts 6.8x and 11.2x across the sweep,
    # so the median is not a measurement and the value is withheld.  Two more fail the
    # direction check of direction_all.py -- bearing kurtosis at 67% agreement and C-MAPSS
    # sensor 9 at 69%, against a 75% bar -- where the value is measured but the channel is
    # not shown to track damage, so the value is kept.  All four are drawn hollow rather than
    # deleted, because a failed measurement is evidence too.
    UNSTABLE = {'HF 5-10 kHz', 'spectral centroid'}
    UNDIRECTED = {'kurtosis', 'sensor 9'}
    STRUCK = UNSTABLE | UNDIRECTED
    # 95% intervals from resampling units, computed by upsilon_ci.py.  The DeltaQ(V) row is
    # absent from the file: its value comes from the predictive estimator, which is a
    # fleet-level curve and has no per-unit quantity to resample.  It is drawn at the point
    # estimate 0.24 of the slope of its survival-corrected information curve at c = 1
    # (dqv_mar.py); the whisker is the 95% interval of the resampled slopes, [0.00, 2.28].
    # Bulk capacity is the pooled power-law estimate 2*beta+1 = 9.97 (severson_shape.py) with the
    # interval of upsilon_ci.json; the median of the per-cell estimates is 9.39.
    HERE = os.path.dirname(__file__)
    with open(os.path.join(HERE, 'upsilon_ci.json')) as fh:
        CI = json.load(fh)
    zq = np.load(os.path.join(HERE, 'dqv_info_curve.npz'))
    DQ_SLOPE_CI = tuple(np.percentile(zq['slopes'], [2.5, 97.5]))

    # Panel (b): fleet_survey.py, one code path for every fleet (power-law arm, interval over
    # units, nonparametric cross-check at three widths).  Order and verdicts as in Table 7.
    with open(os.path.join(HERE, 'fleet_survey.json')) as fh:
        FL = json.load(fh)
    with open(os.path.join(HERE, 'fleet_survey_tf.json')) as fh:
        TF = json.load(fh)
    cells = [('Severson', 'Severson'), ('XJTU', 'XJTU batteries'), ('MICH', 'MICH'),
             ('Stanford', 'Stanford'), ('HUST', 'HUST'), ('HNEI', 'HNEI'), ('SNL', 'SNL'),
             ('CALCE', 'CALCE'), ('NASA', 'NASA PCoE'), ('Zhang', 'Zhang et al.'),
             ('Tongji', 'Tongji'), ('RWTH', 'RWTH')]
    frows = [('battery', nm, FL[k]) for k, nm in cells] + \
            [('C-MAPSS', f'C-MAPSS FD00{i}', TF[f'FD00{i}']) for i in (1, 2, 3, 4)]

    # geometry in inches: right edge and decade width shared by the two panels
    P, TOP, GAP, BOT = .160, .42, .64, .92
    AXR, AXLA = 5.20, 1.50
    dw = (AXR - AXLA)/np.log10(20/.01)
    AXLB = AXR - dw*np.log10(20/.1)
    na, nb = len(rows), len(frows)
    ha, hb = (na + .35)*P, (nb + .35)*P
    H = TOP + ha + GAP + hb + BOT
    fig = plt.figure(figsize=(W, H))
    axA = fig.add_axes([AXLA/W, (H - TOP - ha)/H, (AXR - AXLA)/W, ha/H])
    axB = fig.add_axes([AXLB/W, (H - TOP - ha - GAP - hb)/H, (AXR - AXLB)/W, hb/H])
    tidy(axA); tidy(axB)

    def fx(ax, inches):          # x in inches from the figure edge, y in data units
        return btf(fig.transFigure, ax.transData), inches/W

    def draw_row(ax, y, M, col, solid, lo, hi, val, n, muted=False):
        ax.plot([1.0, M], [y, y], color=col, lw=2.2, alpha=.35, solid_capstyle='butt', zorder=2)
        if lo is not None:
            ax.plot([lo, hi], [y, y], color=col, lw=.7, alpha=.75, zorder=4, solid_capstyle='butt')
            for x in (lo, hi):
                ax.plot([x, x], [y - .21, y + .21], color=col, lw=.7, alpha=.75, zorder=4)
        ax.plot([M], [y], 'o', ms=3.6, color=col if solid else 'white', mec=col,
                mew=.6 if solid else 1.0, zorder=5)
        tr, _ = fx(ax, 0)
        ax.text((AXR + .185)/W, y, val, transform=tr, fontsize=7.2,
                color=MUTED if muted else INK, ha='right', va='center')
        ax.text((AXR + .72)/W, y, f'{n}', transform=tr, fontsize=7.2, color=MUTED,
                ha='right', va='center')

    # ---- panel (a)
    ys = np.arange(na)[::-1]
    axA.axvline(1.0, color=INK, lw=.9, zorder=3)
    for y, (dom, lbl, U, n), M in zip(ys, rows, marg):
        col = BLUE if M < 1 else VERM
        bad = lbl in STRUCK
        ci = CI.get(f'{dom}|{lbl}')
        lo = hi = None
        if lbl.startswith(r'$\Delta Q'):
            # 95% interval of the slope over resamples of the cells (dqv_mar.py): [0.00, 3.37] with the monotone fit
            lo, hi = 0.0108, DQ_SLOPE_CI[1]/THB
            axA.plot([lo, hi], [y, y], color=col, lw=.7, alpha=.75, zorder=4, solid_capstyle='butt')
            axA.plot([hi, hi], [y - .21, y + .21], color=col, lw=.7, alpha=.75, zorder=4)
            axA.annotate('', xy=(lo, y), xytext=(lo*1.5, y), zorder=4,
                         arrowprops=dict(arrowstyle='-|>', color=col, lw=.7, mutation_scale=6,
                                         shrinkA=0, shrinkB=0))
            lo = hi = None
        elif ci is not None:
            lo, hi = (v*(1 + KAP0)/(1 + LAMB[dom]) for v in ci[:2])
        val = '$-$' if lbl in UNSTABLE else f'{U:.2f}' if U >= .1 else r'$\approx0$'
        draw_row(axA, y, M, col, not bad, lo, hi, val, n, muted=bad)

    # ---- panel (b)
    ysb = np.arange(nb)[::-1]
    axB.axvline(1.0, color=INK, lw=.9, zorder=3)
    for y, (dom, nm, r) in zip(ysb, frows):
        k = (1 + KAP0)/(1 + LAMB[dom])
        thr = 1/k
        M, lo, hi = r['par']*k, r['lo']*k, r['hi']*k
        c1, c2 = min(r['np'])*k, max(r['np'])*k
        run = lo > 1 and min(r['np']) > thr
        axB.plot([c1, c2], [y - .30, y - .30], color=INK, lw=.9, alpha=.9, zorder=4,
                 solid_capstyle='butt')
        for x in (c1, c2):
            axB.plot([x, x], [y - .30 - .12, y - .30 + .12], color=INK, lw=.9, alpha=.9, zorder=4)
        draw_row(axB, y, M, VERM, run, lo, hi, f'{r["par"]:.2f}', r['n'], muted=not run)

    for ax, ys_, labels, n_ in ((axA, ys, [r[1] for r in rows], na),
                                (axB, ysb, [r[1] for r in frows], nb)):
        ax.set_yticks(ys_); ax.set_yticklabels(labels)
        ax.tick_params(axis='y', length=0)
        ax.set_ylim(-.7, n_ - .35)
        ax.set_xscale('log')
        ax.xaxis.set_minor_locator(NullLocator())
        tr = btf(fig.transFigure, ax.transAxes)
        ax.text((AXR + .185)/W, 1.012, r'$\Upsilon$', transform=tr, fontsize=7.6, color=INK,
                ha='right', va='bottom')
        ax.text((AXR + .72)/W, 1.012, 'units', transform=tr, fontsize=7.2, color=MUTED,
                ha='right', va='bottom')
    axA.set_xlim(.010, 20); axB.set_xlim(.1, 20)
    axA.set_xticks([.01, .1, 1, 10]); axA.set_xticklabels(['1/100', '1/10', '1', '10'])
    axB.set_xticks([.1, 1, 10]); axB.set_xticklabels(['1/10', '1', '10'])
    for ax, xl in ((axA, .013), (axB, .11)):
        for xx, tt, ha_ in [(xl, 'early stop optimal', 'left'), (7.2, 'run to failure optimal', 'right')]:
            ax.text(xx, 1.012, tt, transform=ax.get_xaxis_transform(), fontsize=7.6, color=INK,
                    ha=ha_, va='bottom')
    axB.set_xlabel(r'$\Upsilon$ relative to the threshold of its domain', labelpad=2.0)

    # domain brackets in the left margin, one column for both panels
    GX, GR = .40, .52
    for ax, rws, ys_ in ((axA, rows, ys), (axB, frows, ysb)):
        prev, spans = None, []
        for y, rw in zip(ys_, rws):
            if rw[0] != prev:
                spans.append([rw[0], y, y]); prev = rw[0]
            else:
                spans[-1][2] = y
        tr = btf(fig.transFigure, ax.transData)
        for dom, y0, y1 in spans:
            ax.text(GX/W, (y0 + y1)/2, dom, transform=tr, rotation=90, ha='center', va='center',
                    fontsize=8.1, color=INK)
            ax.plot([GR/W, GR/W], [y1 - .38, y0 + .38], transform=tr, color=RULE, lw=.8,
                    clip_on=False)
    for ax, t in ((axA, '(a) Indicators of the three primary domains'),
                  (axB, '(b) Damage indicator of every cell and turbofan fleet')):
        y_top = ax.get_position().y1 + .27/H
        fig.text(.05/W, y_top, t, fontsize=8.2, color=INK, ha='left', va='bottom',
                 fontweight='bold')

    L = plt.Line2D
    hd = [L([], [], marker='o', ms=3.6, color=VERM, mec=VERM, mew=.6, ls='none'),
          L([], [], marker='o', ms=3.6, color='white', mec=VERM, mew=1.0, ls='none'),
          L([], [], color=VERM, lw=.7, alpha=.75, marker='|', ms=5),
          L([], [], color=INK, lw=.9, alpha=.9, marker='|', ms=4)]
    fig.legend(hd, ['run to failure', 'excluded (a); not determined (b)',
                    '95% interval over units', 'nonparametric cross-check, three widths (b)'],
               loc='lower center', bbox_to_anchor=(.5, .004), ncol=2, fontsize=7.2,
               handlelength=2.0, handletextpad=.5, columnspacing=2.4, labelspacing=.5,
               borderaxespad=0)
    fig.savefig(f'{OUT}/fig4.pdf'); plt.close(fig)


def fig_twostage():
    """(a) The cold-start pole -- n_eff collapses to zero as n1 -> 0, so no plan is purely short;
       (b) the gain over an all-to-failure campaign as a function of budget, for two settings of the
       inflation constants (the gain is set by them, not by the channel alone).
       Channel B enters on the information scale (odds of the leave-one-out R^2), not as R^2."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from twostage import solve, make_eta, eta_dq_info, KAP0
    etaB = eta_dq_info()
    etaA, _ = make_eta('pow', 4.483)
    SETS = [((0.5, 10.), VERM, '-', r'$u_\Lambda=0.5,\ u_0=10$'),
            ((2.0, 30.), BLUE, (0, (4, 1.6)), r'$u_\Lambda=2,\ u_0=30$')]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 3.96),
                                 gridspec_kw={'height_ratios': [1, 1], 'hspace': .70})
    tidy(a1); tidy(a2)

    # ---- (a) how n_eff depends on the number of run-to-failure units, at a fixed budget C = 400
    cg = np.linspace(.02, 1., 99)
    C = 400.
    for (UL, U0), col, ls, lab in SETS:
        n1s = np.arange(1, int(C/(KAP0+1.))+1)
        best = []
        for n1 in n1s:
            Pi = 1 + UL + U0/n1
            rem = C - n1*(KAP0+1.)
            n2 = rem/(KAP0+cg)
            best.append((n1 + n2*np.array([etaB(c) for c in cg])/Pi).max())
        best = np.array(best)
        a1.plot(n1s/n1s[-1], best, color=col, ls=ls, lw=1.15, zorder=3)
        j = int(np.argmax(best))
        a1.plot([n1s[j]/n1s[-1]], [best[j]], 'o', ms=3.4, color=col, mec='white', mew=.7, zorder=5)
    a1.axhline(C/(KAP0+1.), color=MUTED, lw=.6, ls=':', zorder=2)
    a1.text(1.0, C/(KAP0+1.)-18, 'all to failure', fontsize=7.0, color=MUTED, ha='right', va='top')
    a1.plot([0], [0], 'o', ms=3.4, mfc='white', mec=INK, mew=.8, zorder=5, clip_on=False)
    a1.annotate('no unit run to failure,' + chr(10) + 'no labels: $n_{\\mathrm{eff}}=0$',
                xy=(0, 0), xytext=(.13, 100), fontsize=7.3, color=INK, ha='left',
                va='center', linespacing=1.15,
                arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=.55, shrinkA=1, shrinkB=3,
                                connectionstyle='arc3,rad=0.25'))
    a1.set_xlim(-.012, 1.02); a1.set_ylim(0, 500)
    a1.set_xticks([0, .25, .5, .75, 1.0])
    a1.set_xticklabels(['0', '25%', '50%', '75%', '100%'])
    a1.set_yticks([0, 250, 500])
    a1.set_xlabel('share of the budget spent running units to failure', labelpad=1.5)
    a1.set_ylabel('effective' + chr(10) + r'samples $n_{\mathrm{eff}}$', linespacing=1.1)
    a1.set_title('(a) Necessity of a run-to-failure stage ($C=400$)', fontsize=8.4, color=INK, pad=3.5)

    # ---- (b) the gain, as a continuous function of budget
    Cg = np.geomspace(20, 800, 46)
    a2.axhline(1, color=INK, lw=.8, zorder=3)
    for (UL, U0), col, ls, lab in SETS:
        gain = []
        for Cb in Cg:
            bB = solve(etaB, UL, U0, float(Cb))
            gain.append(bB[0]/(Cb/(KAP0+1.)) if bB else np.nan)
        a2.semilogx(Cg, gain, color=col, ls=ls, lw=1.2, zorder=4)
        a2.fill_between(Cg, 1, gain, color=col, alpha=.09, lw=0, zorder=1)
    a2.text(23, 1.145, r'$\Delta Q(V)$: $u_\Lambda=0.5,\ u_0=10$', fontsize=7.6, color=VERM, ha='left', va='center')
    a2.text(23, 1.115, r'$\Delta Q(V)$: $u_\Lambda=2,\ u_0=30$ (the corner at every budget)', fontsize=7.6, color=BLUE, ha='left', va='center')
    a2.text(23, 1.07, r'bulk capacity, $\Upsilon=9.97$: two-stage plan coincides with the corner',
            fontsize=7.4, color=INK, ha='left', va='center')
    a2.set_xlim(20, 800); a2.set_ylim(.99, 1.17)
    a2.set_xticks([20, 50, 100, 200, 400, 800])
    a2.set_xticklabels(['20', '50', '100', '200', '400', '800'])
    a2.xaxis.set_minor_locator(NullLocator())
    a2.set_yticks([1.0, 1.05, 1.1, 1.15])
    a2.set_yticklabels([r'$1\times$', r'$1.05\times$', r'$1.1\times$', r'$1.15\times$'])
    a2.set_xlabel('budget $C$ (units of rig time)', labelpad=1.5)
    a2.set_ylabel('effective samples,' + chr(10) + 'relative to all-to-failure',
                  linespacing=1.1)
    a2.set_title('(b) Gain of the two-stage plan', fontsize=8.4, color=INK, pad=3.5)
    fig.savefig(f'{OUT}/fig2.pdf'); plt.close(fig)

# ---------------------------------------------------------------- Fig 5
def fig_curves():
    """The indicator of every dataset, scaled from its baseline (0) to its end of life (1).

    This is the object the estimator reads (Section 6.1): the shape of D(tau).  A curve that
    stays near zero until late and then rises, below the dotted diagonal, is the late
    concentration that puts the corner at c = 1; a curve above the diagonal has delivered its
    change early.  The battery sets carry the competing early channel beside the damage
    variable, the one place where the choice of indicator changed the plan.  Bands are the
    interquartile range over units (single units are drawn when there are fewer than ten).
    All nineteen datasets of Table 7 are drawn, the survey sets first.
    Data: appendix_curves.npz, built by appendix_curves.py."""
    z = np.load(os.path.join(os.path.dirname(__file__), 'appendix_curves.npz'))
    g = z['grid']
    panels = [('severson', 'Severson', VERM), ('xjtu_bat', 'XJTU batteries', VERM),
              ('zhang', 'Zhang et al.', VERM), ('nasa', 'NASA PCoE', VERM),
              ('hnei', 'HNEI', VERM), ('calce', 'CALCE', VERM),
              ('snl', 'SNL', VERM), ('mich', 'MICH', VERM),
              ('hust', 'HUST', VERM), ('rwth', 'RWTH', VERM),
              ('stanford', 'Stanford', VERM), ('tongji', 'Tongji', VERM),
              ('fd001', 'C-MAPSS FD001', GREEN), ('fd002', 'C-MAPSS FD002', GREEN),
              ('fd003', 'C-MAPSS FD003', GREEN), ('fd004', 'C-MAPSS FD004', GREEN),
              ('xjtu_sy', 'XJTU-SY', PURP), ('pronostia', 'PRONOSTIA', PURP),
              ('milling', 'Milling', BLUE)]
    fig, axs = plt.subplots(5, 4, figsize=(W, 8.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=.075, right=.955, top=.972, bottom=.062, wspace=.10, hspace=.40)
    i80 = int(np.argmin(abs(g - .8)))
    fmt = lambda v: f'{v:.2f}'.replace('-', '−')
    flat = axs.ravel()
    for ax, (key, name, col) in zip(flat, panels):
        tidy(ax)
        M, E = z[f'{key}|main'], z[f'{key}|early']
        n = len(M)
        ax.plot([0, 1], [0, 1], color=RULE, lw=.65, ls=(0, (1.5, 1.8)), zorder=1)
        if n < 10:
            for row in M:
                ax.plot(g, row, color=col, lw=.55, alpha=.55, zorder=2)
        else:
            q1, q3 = np.percentile(M, [25, 75], 0)
            ax.fill_between(g, q1, q3, color=col, alpha=.20, lw=0, zorder=2)
        med = np.median(M, 0)
        ax.plot(g, med, color=col, lw=1.6, zorder=4, solid_capstyle='butt')
        dots = [(med[i80], col, 3.2)]
        if len(E):
            if len(E) >= 10:
                e1, e3 = np.percentile(E, [25, 75], 0)
                ax.fill_between(g, e1, e3, color=INK, alpha=.08, lw=0, zorder=2)
            me = np.median(E, 0)
            ax.plot(g, me, color=INK, lw=.95, ls=(0, (3.4, 1.7)), zorder=3)
            dots.append((me[i80], INK, 2.9))
        ax.plot([.8, .8], [-.15, min(d[0] for d in dots)], color=RULE, lw=.55, zorder=1)
        top = max(d[0] for d in dots)
        for v, c, ms in dots:
            ax.plot([.8], [v], 'o', ms=ms, color=c, mec='white', mew=.6, zorder=6)
            if len(dots) > 1 and v == top or key == 'pronostia':
                ax.text(.775, v + .05, fmt(v), fontsize=6.8, color=INK, ha='right', va='bottom',
                        zorder=6)
            else:
                ax.text(.83, v - .045, fmt(v), fontsize=6.8, color=INK, ha='left', va='top',
                        zorder=6)
        ax.set_title(name, loc='left', fontsize=8.4, color=INK, pad=3.0)
        ax.set_title(f'$n={n}$', loc='right', fontsize=7.0, color=MUTED, pad=3.4)
        ax.set_xlim(0, 1.0); ax.set_ylim(-.15, 1.12)
        ax.set_xticks([0, .5, 1]); ax.set_xticklabels(['0', '0.5', '1'])
        ax.set_yticks([0, .5, 1]); ax.set_yticklabels(['0', '0.5', '1'])
        ax.tick_params(labelsize=7.4, length=2.2)
    axs[3, 3].tick_params(labelbottom=True)

    axl = axs[4, 3]
    axl.axis('off')
    hd = [plt.Line2D([], [], color=INK, lw=1.6),
          plt.Rectangle((0, 0), 1, 1, fc=INK, alpha=.2, lw=0),
          plt.Line2D([], [], color=INK, lw=.95, ls=(0, (3.4, 1.7))),
          plt.Line2D([], [], color=RULE, lw=.65, ls=(0, (1.5, 1.8))),
          plt.Line2D([], [], color=INK, marker='o', ms=3.2, mec='white', mew=.6, ls='none')]
    axl.legend(hd, ['median', 'interquartile range', 'early channel', 'linear degradation',
                    'value at 80% of life'], loc='center left', bbox_to_anchor=(-.02, .5),
               fontsize=7.2, handlelength=1.8, handletextpad=.5, labelspacing=.7, borderaxespad=0)
    for r0, r1, lab in [(0, 0, 'Li-ion cells'), (1, 2, 'Li-ion cells, BatteryLife'),
                        (3, 3, 'Turbofans'), (4, 4, 'Bearings, cutters')]:
        y1, y0 = axs[r0, 0].get_position().y1, axs[r1, 0].get_position().y0
        fig.add_artist(plt.Line2D([.968, .968], [y0, y1], color=RULE, lw=.8))
        fig.text(.985, (y0 + y1)/2, lab, rotation=270, ha='center', va='center', fontsize=7.2,
                 color=MUTED)
    fig.text(.515, .018, 'fraction of life', ha='center', va='center', fontsize=8.6)
    fig.text(.011, .515, 'indicator, from baseline (0) to end of life (1)',
             rotation=90, ha='center', va='center', fontsize=8.6)
    fig.savefig(f'{OUT}/fig5.pdf'); plt.close(fig)


if __name__ == '__main__':
    fig_densities(); fig_channels(); fig_upsilon(); fig_twostage(); fig_eta(); fig_curves()
    for f in sorted(os.listdir(OUT)):
        if f.endswith('.pdf'):
            print(f'{f:22s} {os.path.getsize(os.path.join(OUT, f))/1024:6.1f} KB')
