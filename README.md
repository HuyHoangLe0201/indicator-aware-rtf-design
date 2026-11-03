# Reproduction code

Supplementary material for **"Indicator-aware design of run-to-failure degradation tests for
data-driven prognostics"** (Huy Hoang Le, Kim-Anh Nguyen).

Every measured quantity, table and figure in the paper is traced in section 5 to a script in
`experiments/`. Three kinds of numbers are not recomputed by a script and are listed in section 8: constants typed
into the figure scripts, the model constants of the two-stage plan, and values read off another table.

---

## 1. Requirements

Python 3.11 or later.

```
pip install -r requirements.txt
```

`numpy`, `scipy` and `scikit-learn` are needed throughout; `matplotlib` only for the figures
and `sympy` only for the symbolic re-verification of the theory (Sections 8.1 and 8.3, Appendix A).

## 2. Setting the paths

The scripts use absolute paths, which is how they were run. Rewrite them once:

```
python set_paths.py
```

That points them at this directory. If you also want the results that need the raw datasets
listed in section 4, add the directory you unpacked those into:

```
python set_paths.py --external "C:/data/prognostics"
```

`--dry-run` shows what would change without writing.

## 3. What is included, and what you can run without downloading anything

`data/severson/` and `experiments/*.npz`, `*.json` hold the derived arrays behind the battery
results: per-cell capacity and ΔQ(V) trajectories, cycle lives, extracted features, bootstrap
intervals and the cached bearing and PRONOSTIA features. They are about 4 MB, and with them
the battery and bearing analyses run with no download at all. The raw Severson `.mat` batches
are 8 GB and are not shipped; `severson_extract.py` is included so the derived arrays can be
rebuilt from them if you want to check that step too.

Derived arrays are redistributed under the terms of their source datasets, with attribution
as given in section 4 and in the paper's reference list.

## 4. Raw datasets not shipped

All are public. Put them under one directory and pass it to `set_paths.py --external`.

| Dataset | Files expected | Source |
|---|---|---|
| C-MAPSS | `train_FD001.txt`, `train_FD003.txt`, `RUL_FD001.txt`, `RUL_FD003.txt`, `CMAPSSData/` | NASA PCoE prognostics data repository |
| XJTU-SY bearings | `XJTU-SY bearing dataset/` | Xi'an Jiaotong University / Changxing Sumyoung |
| PRONOSTIA / FEMTO-ST | `phm-ieee-2012-data-challenge-dataset-master/` | IEEE PHM 2012 data challenge |
| NASA PCoE batteries | `NASABattery/B0005.mat` … | NASA PCoE prognostics data repository |
| NASA Ames milling | `mill.mat` (under `data/milling/`) | NASA PCoE prognostics data repository |
| Zhang et al. impedance cells | `data/zhang/` | Zenodo; `zhang_fetch.py` downloads it |
| BatteryLife processed sets | `batterylife_zenodo_19688272/` with the zip files of the sets and `_meta/Life labels/<set>_labels.json`; `batterylife_extract.py <set> ...` writes the channels to `data/_work/channels/` (override with `BL_CH`; labels with `BL_LAB`) | Zenodo record 19688272 (Tan et al., KDD 2025), CC-BY-4.0 |
| XJTU battery cells | `NASABattery/Battery Dataset/` (`.mat` files of the 23 cells) | Wang et al., Nature Communications 15, 4332 (2024) |
| Severson cells (raw) | three `.mat` batches in `data/severson/`; read only by `dq_traj.py`, `severson_extract.py`, `severson_policy.py`, `e1_features.py`, `channel_abs.py`, `channel_info.py` and `crossing_free.py`; the other Severson analyses use the shipped derived arrays | `data.matr.io`, Severson et al. 2019 |

## 5. What produces what

Table and figure numbers are those of the revised manuscript; numbers with the prefix S (Tables S1 to S26, Sections S1 to S6) are those of the Supplementary Material (`Supplementary_Material_R1.pdf`), and Appendices A and B are those of the paper.

| Paper element | Script |
|---|---|
| Table 2 (all budgets: caption), Fig. 5 (Appendix B; file `fig2.pdf`): two-stage plan | `twostage.py` (information-scale curve `eta_dq_info()`, read from `dqv_info_curve.npz`, the survival-corrected curve of `dqv_mar.py`; run with `(u_Λ, u0) = (2, 30)` and `(0.5, 10)`), continuous-`n₁` check `twostage_cont.py`, constants swept in `twostage_sens.py`, measured in `two_stage_constants.py`; the figure is `make_figures.py`, `fig_twostage()` |
| Table 7 (Appendix B): Υ, interval, cross-check and verdict of all nineteen datasets | `fleet_survey.py`: `python fleet_survey.py` (twelve cell fleets → `fleet_survey.json`), `python fleet_survey.py turbofans` (FD001–FD004 → `fleet_survey_tf.json`), `python fleet_survey.py milling` (→ `fleet_survey_milling.json`); every fleet goes through one code path (power-law arm with a three-width sweep, bootstrap interval over units, nonparametric cross-check, direction share) and the verdict rule of Section 5.4; the bearing rows come from `upsilon_ci.json` and `pronostia_c50.py` |
| Table 8 (Appendix B), Fig. 3: every indicator | Υ: `upsilon_survey.py`, `upsilon_bearing.py`, `xjtu_survey.py`, `milling_test.py`; direction column: `direction_all.py`; drift column: `scale_select.py`, `scale_span.py`; intervals: `upsilon_ci.py` (→ `upsilon_ci.json`); ΔQ(V) slope on the information scale, corrected for the cells that no longer exist at a long window: `dqv_mar.py` (→ `dqv_info_curve.npz`), checks in `dqv_mar_check.py`; the uncorrected curve of the survivors remains in `dqv_odds_scale.py`; figure: `make_figures.py`, `fig_upsilon()`; panel (b) of Fig. 3 reads `fleet_survey.json` and `fleet_survey_tf.json` (Table 7) |
| Table 9: XJTU-SY `c₅₀` at three widths | `xjtu_survey.py` |
| Table 3: battery campaigns | `cost_real.py`, averaged over three splits by `cost_three_splits.py` |
| Table 4: effect of stopping early against the budget | `budget_sweep.py` (`CH=dq` or `capacity`, `WIN`, `NSPLIT`, `N1S`; `BMODE=pool` is the reported one, `leak` and `free` reproduce the sensitivity checks) |
| Table S22 (Section S4.6): campaigns scored on a held-out manufacturing batch, with the ΔQ(V) information curve of the training pool | `batch_campaign.py` (`CH=dq WIN=100`, `CH=capacity WIN=400`; outputs `batch_campaign_dq.txt`, `batch_campaign_cap.txt`), `batch_info.py`, `batch_curves.py` (outputs `batch_info_out.txt`, `batch_curves_out.txt`) |
| Table 4 intervals (Nadeau–Bengio correction for overlapping splits) | `budget_nb.py` (`CH=dq WIN=100` and `WIN=50`; outputs `budget_nb_dq.txt`, `budget_nb_dq50.txt`) |
| Table S18: five learners (Section S4.2) | `confirm_learner120.py` (learners in `learner_robust.py`) |
| Table 5: turbofan campaigns | `cost_cmapss.py` |
| Table 6: the verdict against the campaigns at a small and a large budget | assembled from existing results, no script of its own: the Severson rows from Tables 3 and 4 and Table 7, the further fleets from the `bl_campaign_<set>_<channel>.txt` outputs (paired gain of the stop at c = 0.65 at the smallest and the largest budget of Table S23) and from Table S5 (best stop) |
| Table 10 (Appendix B): limitations of the evidence | each row names the Supplementary Section and the script of the analysis it summarizes |
| Table S9 (Section S2.1): channel audit, terminal amplitude pinned, prior and free (point values from `channel_audit.py`; resampled medians and intervals from `upsilon_boot.py`) | `upsilon_boot.py` (medians and intervals), `upsilon_eff_real.py` (point values; `channel_audit.py` and Table S9 use it) |
| Table S10 (Section S2.2): residual structure | `resid_check.py` |
| Noise-exponent sweep (Section S2.2, variance proportional to (D+0.1)^p) | `noise_exponent_sweep.py` (output `theory_ext/noise_exponent_sweep_out.txt`) |
| Charging policy as the only stress (Section S5.2): log-linear acceleration, three-parameter regression, terciles | `severson_policy_stats.py` (output `theory_ext/severson_policy_stats_out.txt`; reads `severson_policy.npz`, `policy.json`), `accel_policy.py` |
| Classical test plan on the same fleets (Section S1.6) | `adt_classical.py` |
| Theorem 5(4), model (B) (Section 8.3): label variance of the plug-in first-passage time | `wiener_passage_check.py` (output `theory_ext/wiener_passage_check_out.txt`) |
| Gains of the two-stage model when the survivors' curve is used instead of the corrected one (Section S3.3) | `twostage.py` (`variants()`; output `theory_ext/twostage_out.txt`) |
| Table S21 (Section S4.5): censoring-aware learners on FD001 | `survival_baseline.py` (the entries; uses the functions of `cost_cmapss_ext.py`, whose own run is a second random stream, `theory_ext/pool_cost_cmapss_ext.txt`, and also gives the rows for c = 0.65 and F = 0.10) |
| Table S12: decision reliability | `sim_decision.py` |
| Lemma 1 (Section 2.3, proof in Appendix A): stopping at the first noisy crossing adds no information beyond the path | `check_ignorable_stopping.py` (prints the ratio of E[score²] to the expected path information) |
| Theorem 2(f) (Section 3.2): the stationarity condition of an absolute stop time against the minimiser of the risk | `check_stop_condition.py` |
| Proposition 1 (Section 3.2), Table S2 (Section S1.2): online plug-in stopping rule against fixed rules | `sim_sequential.py BETA SIGMA_LOG` (Υ = 2·BETA + 1; the table uses `0.12 0.5`, `0.5 0.5`, `2.0 0.5`, `4.48 0.3`) |
| Proposition A.1 (Appendix A), Table S4 (Section S1.4): risk of a least-squares learner, bias and variance terms against the Monte-Carlo campaign | `risk_decomposition.py {dq\|capacity} WINDOW` (the table uses `dq 100`, `capacity 400`, `capacity 100`) |
| Table S5 (Section S1.4): Proposition A.1 against the campaigns on the five further cell fleets; error with true lives of the same cells | `risk_decomposition_sets.py` (`CH=dq|capacity python risk_decomposition_sets.py SET N1`; the table uses N1 = 10 and 24 (HUST, Tongji), 8 and 14 (RWTH), 8 and 12 (Stanford, MICH)) |
| Section S1.4: choosing c from an archive of completed cells (prospective rule) | `archive_rule.py {dq\|capacity} WINDOW N_ARCHIVE` (the text uses `dq 100 20` and `capacity 400 20`) |
| Section S1.4: the same rule on the five further cell fleets | `CH=dq\|capacity python archive_rule_sets.py SET N1` (larger budget of each fleet: HUST, Tongji 24; RWTH 14; Stanford, MICH 12) |
| Window sensitivity of the capacity campaigns ($m=100$, $50$; Supplementary Material, campaign sizes) | `python capacity_window.py capacity {100\|50}` |
| Section S2.6 (Proposition A.3): size and power of the verdict rule | `type1_rule.py` |
| Section 8.3, Table S24 (Section S5.1): Υ of a gamma degradation process between the Wiener value 1 and the exponent q | `gamma_information.py` |
| Theorem 2 (Section 3.2), Table S1 (Section S1.1): exact corner threshold under exact labels; local-to-global check with the shape released | `exact_threshold.py` |
| Proposition A.2 (Appendix A), Table S6 (Section S1.5): excess risk at fixed ratio of label variance to units, four learners | `learner_invariance.py` |
| Proposition 2 (Section 3.4), Table S3 (Section S1.3): regret of the corner against fixed early stops, power class; bound for the front-loaded ΔQ(V) curve | `corner_regret.py` |
| Table S11: controlled study with known Υ | `sim_synth.py` |
| Table S20 (Section S4.5): absolute stop time on the Severson cells | `cost_batt_abs.py` |
| Table S7 (Section S1.6), Theorem 4 (Section 4.2): classical plans scored on both objectives | `sim_classical.py` (deterministic known-shape case), `sim_wiener_plan.py R CV` (Wiener random drift; CV = 0.08, 0.15, 0.25), `wiener_plan_formula.py` (first-order variance and the constant Λ_f) |
| Theorem 5(3)(i)-(c) (Section 3.3; proof in Appendix A): decision rule for the predictive estimator, local form | `bridge_check.py` |
| Theorem 5(3)(ii)(d)-(g) (Section 3.3; proof in Appendix A): exact global criterion (envelope), chord condition, power-class threshold; certificate on the ΔQ(V) curve (Section S3.3) | `bridge_global.py` (output `theory_ext/bridge_global_out.txt`) |
| Table S13 (Section S3.3): ΔQ(V) odds by window, survivors and corrected; selection tests; slope at c = 1 | `dqv_mar.py` (estimator, simulated check, bootstrap, `dqv_info_curve.npz`), `dqv_mar_check.py` (selection imposed by hand on the real cells, capacity channel, slope variants), `survcorr_table.py` (the rows of Table S13) |
| Table S8 (Section S1.6): the same absolute-stop campaigns scored on the learner and on the classical fleet quantities; charging policy as a stress | `cost_classical_abs.py` (→ `cost_classical_abs.json`), `severson_policy.py` (reads the three raw `.mat` files, writes `severson_policy.npz`) |
| Tables S15 and S16 (Section S3.4): C-MAPSS FD002 and FD004 | `fd002_fd004.py` (Υ), `cost_fd24.py` (campaigns) |
| Table S19 (Section S4.4): campaigns with the cost of the stage-one archive charged to the budget | `stage1_cost.py` (output in `stage1_cost_out.txt`) |
| Table S17 (Section S4.1): campaign settings and realized sizes | settings are in the scripts above; `repro_counts.py` (battery) and `repro_cmapss.py` (turbofan) print the counters |
| Fig. 1, Fig. 2 (files `fig1.pdf`, `fig3.pdf`) | `make_figures.py`, `fig_densities()` and `fig_channels()`, reading `data/severson/dq_traj.npz` built by `dq_traj.py` |
| Fig. 4 (Appendix B; file `fig5.pdf`): the indicator of every dataset | `appendix_curves.py` reads the raw sets and the BatteryLife channel files of `batterylife_extract.py` (the PRONOSTIA and XJTU battery readers take several minutes) and writes `appendix_curves.npz`, which ships; `python appendix_curves.py hnei fd002` recomputes only the named panels; figure: `make_figures.py`, `fig_curves()` |
| Section S2.3 and Section 9.2: thresholds Λ measured in place | `battery_lambda2.py`, `turbofan_lambda.py`, `milling_lambda2.py` |
| Section 4, Sections 8.1 and 8.3, Appendix A: divergence from planning criteria, symbolic checks | `verify_theory.py`, `verify_appendixB.py` |
| Section 5.2 and Section S2.2: scale selection and stability | `scale_check.py`, `scale_check2.py`, `scale_select.py`, `scale_span.py` |
| Section 6.3: bearing survey | `xjtu_survey.py`, `upsilon_bearing.py`, `upsilon_ci.py`, `upsilon_dist.py` |
| Section 5.3: where the estimator fails | `pronostia_c50.py`, `escape_bearing.py` (30 bearings), `escape_bearing29.py` (the 29 of the paper) |
| Section S4.7, Table S23 (all further sets, one column per set): partial correlation, transfer and campaigns on the further public cell fleets | `bl_common.py` (loader), `bl_readout.py` (output `bl_readout_out.txt`), `bl_campaign.py` run by `bl_campaign_all.sh` (outputs `bl_campaign_<set>_<channel>.txt`); `bl_info.py` (output `bl_info_out.txt`) computes the information curve and the covariate ablation, whose results the paper does not use. The channels come from `batterylife_extract.py` |
| Section S3.3 (Table S14) and Section S4.7 (protocol paragraph): information curve to the median life, and leave-protocol-out transfer on the further cell fleets | `bl_curve.py` (output `bl_curve_out.txt`; the Severson cells are run as `SEVERSON`), `bl_protocol.py` (output `bl_protocol_out.txt`) |
| Section S6.1 and Table S23: ordering of the two battery channels on eight further public sets | `batterylife_extract.py` (per-cycle channels from the BatteryLife processed sets, Zenodo record 19688272, CC-BY-4.0; unpickling restricted by `batterylife_ru.py`), then `python batterylife_replicate.py <channels_dir> <labels_dir> HNEI CALCE SNL MICH HUST RWTH Stanford Tongji` (the CALB set is left out because its records stop after about 100 cycles); output `batterylife_replicate_out.txt` |
| Section 9.3 and Section S6.1: the physical rule and its tests | forward and reverse tests on the battery: `readout_test2.py`, `readout_test.py`; screens: `escape_screen.py`, `screen_shapes.py`; `nasa_precursor.py`, `zhang_test.py`, `xjtu_battery_replicate.py` (XJTU battery cells; records end at the data set's failure criterion, 80% of nominal capacity, 82 to 89% of the measured initial capacity), `battery_screen.py` (a different screen; its output is not the paper's) |
| Section S6.2 (milling, the fourth domain) and FD003 (Section 6.2) | `milling_load.py`, `milling_test.py`, `milling_lambda2.py`, `fd003_all.py`, `fd003_escape.py`, `fd003_stability.py` |
| Section S4.2 and Section 9.2: oracle arm per split, richer feature map | `oracle_both.py`, `feature_robust2.py` |
| Bias floor, Π and the shape inflation | `measure_b.py`, `measure_b2.py`, `measure_B3.py` (low c; the 13.2 and 11.3 of Section S4.3), `measure_B4.py` (campaigns including c < 0.5), `pi_survey.py`, `pi_one.py`, `pi_powerlaw.py`, `pi_profile.py`, `measure_pi.py` |
| Supporting diagnostics | `diag_elig.py`, `diag_huber.py`, `inflation_predict.py`, `dqv_interval.py`, `dq_odds.py`, `leak_control.py`, `pred_gain.py`, `verify_scaling.py` |

Scripts not named above are the exploratory and superseded runs kept for completeness; each
states in its own docstring what it does and why.

## 6. A note on internal numbering

Some docstrings and comments refer to element numbers from earlier drafts of this work
("Table VIII", "Table I", "Section VII-G", "Theorem 8", "Paper 1"). The table in section 5 above is the
authoritative map to the revised manuscript.

## 7. Reproducing the campaign tables

These are the paper's operational results and the slowest to run, so their settings are
stated explicitly. Run them from `experiments/`.

```
python cost_three_splits.py     # Table 3:  120 draws, three splits
python budget_sweep.py          # Table 4:  CH=dq WIN=100 NSPLIT=16; CH=dq WIN=50; CH=capacity WIN=400 NSPLIT=12 N1S=6,10,20
python budget_nb.py             # Table 4 intervals: CH=dq WIN=100; CH=dq WIN=50
python batch_campaign.py        # Table S22: CH=dq WIN=100; CH=capacity WIN=400
python batch_info.py            # Table S22: information curve of the pools
python confirm_learner120.py    # Table S18: 120 draws, three splits, five learners
python cost_cmapss.py           # Table 5:   40 draws, three splits, two feature windows
python cost_batt_abs.py         # Table S20: absolute stop times
python stage1_cost.py           # Table S19: stage-one archive charged (N1S=8,12,20 STAGE1=3,4,6 NSPLIT=16 NDRAW=12)
```

The label imputation starts from a shape exponent fitted on the complete trajectories of the
training pool (`cost_real.pool_beta`; the stage-one units for the turbofans, `cost_cmapss.cal_beta`), never on test
cells, and `impute()` has no default start for that reason. Each script prints the table as it appears in the paper. Expect tens of
minutes apiece: the cost is dominated by imputing lives for every draw. All are seeded, so repeated runs reproduce the printed numbers exactly.

The figures need nothing downloaded:

```
cd experiments
python make_figures.py
```

This writes `fig1.pdf` … `fig5.pdf` one directory up, the five figures of the manuscript (`fig1.pdf` is Fig. 1, `fig3.pdf` Fig. 2, `fig4.pdf` Fig. 3, `fig5.pdf` Fig. 4, which reads
`appendix_curves.npz`, and `fig2.pdf` Fig. 5). It also writes `fig_eta_draft.pdf`, an extrapolation-information plot from an
earlier draft that the manuscript does not use.

## 8. Numbers that are typed, not computed

So that nothing is read as more than it is: `make_figures.py` types the Υ values and unit counts of Fig. 3(a) (they equal
Table 8, whose sources are in section 5), the thresholds Λ (0.241, 0.165, 1.382) and the excluded sets; `upsilon_ci.py` and
`threshold_stability.py` repeat the same typed values. The bulk-capacity Υ of 9.97 is the pooled power-law value 2β+1 of
`severson_shape.py` (the median of the per-cell estimates in `upsilon_dist.npz` is 9.39). Table 7 reads every cell fleet unit by unit with the power-law arm of `fleet_survey.py`, which gives 8.76 for the Severson capacity; its bearing rows (XJTU-SY peak 11.67 with the interval of `upsilon_ci.json`, PRONOSTIA `c₅₀`) are typed from the earlier scripts. The information curve behind `twostage.py` is not typed: it is read from `dqv_info_curve.npz`, which `dqv_mar.py` writes. `direct_cstar.py`, `estimator_free.py`, `crossing_free.py` and
`eta_curves.py` use R² as an information fraction and are superseded by the information-scale treatment of Section 3.3. `boundary2.py` prints
standard-deviation ratios where the text says variance ratios.
