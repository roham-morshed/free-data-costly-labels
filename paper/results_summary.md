# Results summary

Every number below is computed by `scripts/make_paper_artifacts.py` directly from
the files in `results/`. If a number is missing here it was not measured.

- datasets with complete results: **40**
- result rows: **16600**, cells: **2000**, failed cells: **0**
- label budgets: [1, 2, 5, 10, 20], seeds: 10, methods: ['1nn_dtw', 'sup_strict', 'sup_tf', 'self_train', 'label_spread', 'augment', 'safe_ssl', 'safe_ssl_novalve']

## RQ1 -- Recovery

Fraction of the sup-strict -> sup-full gap that each method closes:

Datasets whose ceiling is within 2 accuracy point of the low-label floor are excluded (the denominator would be a rounding artefact). Median across datasets; mean in parentheses.

| method | k=1 | k=2 | k=5 | k=10 | k=20 |
|---|---|---|---|---|---|
| 1nn_dtw | -0.018 (-0.038) | -0.026 (-0.140) | -0.200 (-0.371) | -0.458 (-0.667) | -0.554 (-0.722) |
| sup_strict | +0.000 (+0.000) | +0.000 (+0.000) | +0.000 (+0.000) | +0.000 (+0.000) | +0.000 (+0.000) |
| sup_tf | +0.004 (+0.018) | +0.005 (+0.001) | +0.008 (+0.009) | -0.003 (-0.014) | -0.019 (-0.015) |
| self_train | -0.010 (+0.011) | +0.085 (+0.090) | +0.133 (+0.183) | +0.124 (+0.151) | +0.077 (+0.189) |
| label_spread | -0.051 (-0.374) | -0.067 (-0.444) | +0.026 (-0.209) | -0.034 (-0.133) | -0.023 (-0.010) |
| augment | +0.005 (+0.011) | +0.009 (+0.007) | +0.077 (+0.019) | +0.099 (-0.012) | +0.061 (-0.007) |
| safe_ssl | +0.008 (+0.075) | +0.037 (+0.056) | +0.027 (+0.066) | +0.000 (+0.011) | +0.033 (+0.041) |
| safe_ssl_novalve | +0.008 (+0.075) | +0.065 (+0.086) | +0.060 (+0.028) | +0.008 (-0.073) | +0.066 (-0.059) |

| k | datasets used | excluded |
|---|---|---|
| 1 | 40 | 0 |
| 2 | 40 | 0 |
| 5 | 39 | 1 |
| 10 | 35 | 5 |
| 20 | 31 | 9 |

## RQ2 -- Attribution (three tiers)

- **k=1** (N=40): sup-strict 0.5199 -> sup-tf 0.5232 -> SAFE-SSL 0.5412. Free features give **+0.32** accuracy points (sd 0.90, Wilcoxon p=1.30e-01); pseudo-labeling adds **+1.81** points (sd 7.79, p=3.83e-01); without the safety valve +1.81 points (p=3.83e-01).
- **k=2** (N=40): sup-strict 0.5772 -> sup-tf 0.5777 -> SAFE-SSL 0.5930. Free features give **+0.05** accuracy points (sd 0.87, Wilcoxon p=2.29e-01); pseudo-labeling adds **+1.53** points (sd 3.38, p=4.47e-03); without the safety valve +2.09 points (p=1.20e-02).
- **k=5** (N=40): sup-strict 0.6337 -> sup-tf 0.6348 -> SAFE-SSL 0.6458. Free features give **+0.11** accuracy points (sd 0.73, Wilcoxon p=1.17e-01); pseudo-labeling adds **+1.10** points (sd 3.23, p=7.71e-02); without the safety valve +1.30 points (p=1.13e-01).
- **k=10** (N=40): sup-strict 0.6884 -> sup-tf 0.6866 -> SAFE-SSL 0.6943. Free features give **-0.18** accuracy points (sd 0.76, Wilcoxon p=1.60e-01); pseudo-labeling adds **+0.77** points (sd 2.86, p=1.04e-01); without the safety valve +0.46 points (p=4.01e-01).
- **k=20** (N=40): sup-strict 0.7350 -> sup-tf 0.7326 -> SAFE-SSL 0.7391. Free features give **-0.24** accuracy points (sd 0.87, Wilcoxon p=2.26e-01); pseudo-labeling adds **+0.64** points (sd 1.74, p=2.80e-02); without the safety valve -0.32 points (p=6.77e-01).

## RQ3 -- Ranking

- **k=1** (N=40, CD=1.660, Friedman F p=1.72e-02): sup_tf 4.06, safe_ssl 4.15, safe_ssl_novalve 4.15, augment 4.21, sup_strict 4.24, self_train 4.47, 1nn_dtw 4.88, label_spread 5.84
- **k=2** (N=40, CD=1.660, Friedman F p=1.46e-05): self_train 3.26, safe_ssl_novalve 3.69, safe_ssl 4.03, augment 4.22, sup_tf 4.81, 1nn_dtw 5.05, sup_strict 5.19, label_spread 5.75
- **k=5** (N=40, CD=1.660, Friedman F p=1.77e-07): self_train 2.60, safe_ssl_novalve 3.98, augment 4.20, safe_ssl 4.50, sup_tf 4.85, label_spread 4.95, sup_strict 5.24, 1nn_dtw 5.69
- **k=10** (N=40, CD=1.660, Friedman F p=5.60e-07): self_train 2.83, augment 4.09, safe_ssl 4.30, safe_ssl_novalve 4.40, sup_strict 4.49, sup_tf 4.76, label_spread 5.05, 1nn_dtw 6.09
- **k=20** (N=40, CD=1.660, Friedman F p=3.68e-07): self_train 2.96, safe_ssl 4.10, augment 4.29, safe_ssl_novalve 4.41, sup_strict 4.51, label_spread 4.61, sup_tf 4.78, 1nn_dtw 6.34

Pairwise Wilcoxon signed-rank vs. sup-tf (Holm-corrected):

| k | method | wins/ties/losses | median diff | p (Holm) |
|---|---|---|---|---|
| 1 | label_spread | 11/0/29 | -1.50 pts | 0.0235 |
| 1 | 1nn_dtw | 13/0/27 | -0.23 pts | 0.709 |
| 1 | sup_strict | 16/1/23 | -0.12 pts | 0.709 |
| 1 | safe_ssl | 21/0/19 | +0.10 pts | 1 |
| 1 | safe_ssl_novalve | 21/0/19 | +0.10 pts | 1 |
| 1 | augment | 21/1/18 | +0.14 pts | 1 |
| 1 | self_train | 18/1/21 | -0.07 pts | 1 |
| 2 | self_train | 30/0/10 | +1.76 pts | 0.000308 |
| 2 | safe_ssl | 25/2/13 | +0.63 pts | 0.0268 |
| 2 | safe_ssl_novalve | 27/0/13 | +1.89 pts | 0.0601 |
| 2 | label_spread | 16/0/24 | -2.00 pts | 0.412 |
| 2 | sup_strict | 13/1/26 | -0.20 pts | 0.687 |
| 2 | augment | 20/2/18 | +0.05 pts | 0.687 |
| 2 | 1nn_dtw | 19/0/21 | -0.34 pts | 0.687 |
| 5 | self_train | 34/1/5 | +2.10 pts | 2.72e-05 |
| 5 | augment | 22/0/18 | +1.05 pts | 0.318 |
| 5 | safe_ssl | 23/2/15 | +0.20 pts | 0.386 |
| 5 | safe_ssl_novalve | 23/1/16 | +0.87 pts | 0.451 |
| 5 | sup_strict | 13/2/25 | -0.08 pts | 0.451 |
| 5 | 1nn_dtw | 14/0/26 | -2.08 pts | 0.751 |
| 5 | label_spread | 22/0/18 | +0.26 pts | 0.795 |
| 10 | self_train | 28/1/11 | +1.23 pts | 0.000823 |
| 10 | 1nn_dtw | 11/1/28 | -3.12 pts | 0.313 |
| 10 | safe_ssl | 23/3/14 | +0.15 pts | 0.519 |
| 10 | sup_strict | 22/2/16 | +0.06 pts | 0.64 |
| 10 | augment | 22/1/17 | +0.39 pts | 0.64 |
| 10 | safe_ssl_novalve | 21/2/17 | +0.09 pts | 0.802 |
| 10 | label_spread | 18/0/22 | -0.90 pts | 0.837 |
| 20 | self_train | 28/2/10 | +0.76 pts | 0.00149 |
| 20 | 1nn_dtw | 10/1/29 | -4.20 pts | 0.0558 |
| 20 | safe_ssl | 22/7/11 | +0.04 pts | 0.14 |
| 20 | sup_strict | 21/4/15 | +0.07 pts | 0.905 |
| 20 | augment | 21/1/18 | +0.04 pts | 1 |
| 20 | safe_ssl_novalve | 20/3/17 | +0.03 pts | 1 |
| 20 | label_spread | 20/1/19 | +0.05 pts | 1 |

## RQ4 -- Predictability (LFA)

Correlations across datasets. `delta` = acc(SAFE-SSL) - acc(sup-tf);
`headroom` = acc(sup-full) - acc(sup-tf). The last column is the partial
correlation of LFA with the gain, controlling for headroom.

| k | #ds | rho(LFA, delta) | rho(LFA, headroom) | rho(headroom, delta) | partial rho(LFA, delta | headroom) |
|---|---|---|---|---|---|
| 1 | 40 | -0.446 (p=0.00394) | +0.151 (p=0.352) | +0.121 (p=0.458) | -0.473 (p=0.00236) |
| 2 | 40 | -0.349 (p=0.0274) | +0.135 (p=0.405) | +0.201 (p=0.215) | -0.387 (p=0.0148) |
| 5 | 40 | -0.216 (p=0.18) | +0.152 (p=0.348) | +0.337 (p=0.0335) | -0.288 (p=0.0759) |
| 10 | 40 | -0.339 (p=0.0326) | +0.137 (p=0.398) | +0.325 (p=0.0406) | -0.409 (p=0.00971) |
| 20 | 40 | -0.390 (p=0.0129) | +0.146 (p=0.37) | +0.370 (p=0.0188) | -0.483 (p=0.00185) |

Per-component Spearman rho with delta:

| component | k=1 | k=2 | k=5 | k=10 | k=20 |
|---|---|---|---|---|---|
| LFA | -0.446 (p=0.00394) | -0.349 (p=0.0274) | -0.216 (p=0.18) | -0.339 (p=0.0326) | -0.390 (p=0.0129) |
| chi | -0.260 (p=0.105) | -0.141 (p=0.385) | -0.001 (p=0.995) | -0.135 (p=0.407) | -0.011 (p=0.948) |
| knn_purity_proxy | -0.400 (p=0.0106) | -0.426 (p=0.00618) | -0.193 (p=0.233) | -0.371 (p=0.0184) | -0.333 (p=0.0359) |
| sil | -0.372 (p=0.018) | -0.248 (p=0.123) | -0.158 (p=0.332) | -0.199 (p=0.218) | -0.353 (p=0.0257) |
| stab | -0.421 (p=0.00684) | -0.416 (p=0.00764) | -0.344 (p=0.0298) | -0.400 (p=0.0106) | -0.474 (p=0.00201) |

## RQ5 -- Safety

- augment|k=1: regression rate **37.5%** of 400 (dataset, seed) pairs; mean loss when regressing 4.75 pts; worst 22.62 pts; mean delta +0.45 pts.
- augment|k=10: regression rate **35.5%** of 400 (dataset, seed) pairs; mean loss when regressing 4.88 pts; worst 33.33 pts; mean delta +1.76 pts.
- augment|k=2: regression rate **35.8%** of 400 (dataset, seed) pairs; mean loss when regressing 5.46 pts; worst 28.00 pts; mean delta +1.28 pts.
- augment|k=20: regression rate **38.0%** of 400 (dataset, seed) pairs; mean loss when regressing 4.17 pts; worst 35.19 pts; mean delta +0.98 pts.
- augment|k=5: regression rate **35.8%** of 400 (dataset, seed) pairs; mean loss when regressing 4.82 pts; worst 39.29 pts; mean delta +1.93 pts.
- label_spread|k=1: regression rate **59.5%** of 400 (dataset, seed) pairs; mean loss when regressing 13.23 pts; worst 68.90 pts; mean delta -4.67 pts.
- label_spread|k=10: regression rate **38.2%** of 400 (dataset, seed) pairs; mean loss when regressing 7.84 pts; worst 43.01 pts; mean delta +0.07 pts.
- label_spread|k=2: regression rate **51.5%** of 400 (dataset, seed) pairs; mean loss when regressing 13.00 pts; worst 58.84 pts; mean delta -3.25 pts.
- label_spread|k=20: regression rate **35.2%** of 400 (dataset, seed) pairs; mean loss when regressing 5.51 pts; worst 23.64 pts; mean delta +0.41 pts.
- label_spread|k=5: regression rate **45.5%** of 400 (dataset, seed) pairs; mean loss when regressing 8.82 pts; worst 57.56 pts; mean delta -0.50 pts.
- safe_ssl_novalve|k=1: regression rate **35.8%** of 400 (dataset, seed) pairs; mean loss when regressing 7.95 pts; worst 64.86 pts; mean delta +1.81 pts.
- safe_ssl_novalve|k=10: regression rate **29.5%** of 400 (dataset, seed) pairs; mean loss when regressing 6.06 pts; worst 30.24 pts; mean delta +0.46 pts.
- safe_ssl_novalve|k=2: regression rate **32.8%** of 400 (dataset, seed) pairs; mean loss when regressing 6.73 pts; worst 35.04 pts; mean delta +2.09 pts.
- safe_ssl_novalve|k=20: regression rate **30.0%** of 400 (dataset, seed) pairs; mean loss when regressing 5.79 pts; worst 35.82 pts; mean delta -0.32 pts.
- safe_ssl_novalve|k=5: regression rate **30.0%** of 400 (dataset, seed) pairs; mean loss when regressing 6.57 pts; worst 41.04 pts; mean delta +1.30 pts.
- safe_ssl|k=1: regression rate **35.8%** of 400 (dataset, seed) pairs; mean loss when regressing 7.95 pts; worst 64.86 pts; mean delta +1.81 pts.
- safe_ssl|k=10: regression rate **10.2%** of 400 (dataset, seed) pairs; mean loss when regressing 5.62 pts; worst 19.79 pts; mean delta +0.77 pts.
- safe_ssl|k=2: regression rate **8.2%** of 400 (dataset, seed) pairs; mean loss when regressing 5.80 pts; worst 26.74 pts; mean delta +1.53 pts.
- safe_ssl|k=20: regression rate **10.0%** of 400 (dataset, seed) pairs; mean loss when regressing 3.29 pts; worst 8.63 pts; mean delta +0.64 pts.
- safe_ssl|k=5: regression rate **11.2%** of 400 (dataset, seed) pairs; mean loss when regressing 5.29 pts; worst 16.88 pts; mean delta +1.10 pts.
- self_train|k=1: regression rate **40.8%** of 400 (dataset, seed) pairs; mean loss when regressing 5.12 pts; worst 21.24 pts; mean delta +0.62 pts.
- self_train|k=10: regression rate **20.0%** of 400 (dataset, seed) pairs; mean loss when regressing 4.25 pts; worst 16.67 pts; mean delta +2.70 pts.
- self_train|k=2: regression rate **26.8%** of 400 (dataset, seed) pairs; mean loss when regressing 4.80 pts; worst 18.89 pts; mean delta +2.20 pts.
- self_train|k=20: regression rate **19.2%** of 400 (dataset, seed) pairs; mean loss when regressing 3.26 pts; worst 10.79 pts; mean delta +2.24 pts.
- self_train|k=5: regression rate **22.2%** of 400 (dataset, seed) pairs; mean loss when regressing 4.06 pts; worst 27.58 pts; mean delta +2.87 pts.

Safety-valve firing rate (fraction of cells where SAFE-SSL reverted to the supervised model):

- k=1: 0.0%
- k=2: 66.2%
- k=5: 59.2%
- k=10: 60.5%
- k=20: 54.5%

## RQ6 -- Compute

- sup_tf: 0.066 s/cell median (0.083 s incl. amortised unsupervised cost), mean accuracy 0.6348
- 1nn_dtw: 0.001 s/cell median (0.235 s incl. amortised unsupervised cost), mean accuracy 0.6279
- safe_ssl_novalve: 0.286 s/cell median (0.300 s incl. amortised unsupervised cost), mean accuracy 0.6478
- sup_strict: 0.543 s/cell median (0.543 s incl. amortised unsupervised cost), mean accuracy 0.6337
- label_spread: 0.539 s/cell median (0.558 s incl. amortised unsupervised cost), mean accuracy 0.6298
- augment: 0.835 s/cell median (0.835 s incl. amortised unsupervised cost), mean accuracy 0.6541
- safe_ssl: 1.615 s/cell median (1.642 s incl. amortised unsupervised cost), mean accuracy 0.6458
- self_train: 5.788 s/cell median (5.825 s incl. amortised unsupervised cost), mean accuracy 0.6634
- deep_ssl: 63.268 s/cell median (63.268 s incl. amortised unsupervised cost), mean accuracy 0.5812

## Verdict

- **Tier 1 (free unsupervised feature fitting) does not pay.** sup-tf beat sup-strict at 3/5 budgets, by +0.01 accuracy points on average, and the difference was significant (Wilcoxon, p<0.05) at 0/5 budgets. Fitting MiniRocket on the unlabeled pool instead of on the labeled subset is free, and it is also worth essentially nothing.
- **Tier 2 (pseudo-labeling) is where the benefit is.** SAFE-SSL beat sup-tf by +1.17 accuracy points on average (5/5 budgets positive, significant at 2/5).
- **THE LFA HYPOTHESIS IS REJECTED, WITH THE SIGN REVERSED.** The paper hypothesised that unlabeled data helps when the label-free geometry is already aligned with the class structure, i.e. a POSITIVE correlation between LFA and the SSL gain. Measured: Spearman rho is NEGATIVE at every budget (-0.45 to -0.22), significant at p<0.05 for 4/5 budgets. Semi-supervised learning helps MOST where the unlabeled feature geometry is WORST.
- The obvious confound -- that well-clustered datasets are already solved and so have no headroom -- does not explain it: LFA correlates with headroom at p<0.05 for only 0/5 budgets, and the partial correlation of LFA with the gain controlling for headroom is -0.48 to -0.29, i.e. as strong or stronger than the raw correlation.
- LFA is therefore still a usable free decision statistic -- it predicts the gain with |rho| up to 0.45 without touching a single label -- but the mechanism claimed in the hypothesis is wrong, and the paper must invert it.
- **The safety valve is what SAFE-SSL actually buys.** Averaged over the budgets where it can run (k>=2), it cuts the regression rate by 20.6 percentage points and the worst-case single loss from 35.5 to 18.0 accuracy points.
