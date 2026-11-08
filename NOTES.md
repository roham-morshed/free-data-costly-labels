# NOTES

Running log of every decision, deviation and surprise.  If a number in the paper
depends on a judgement call, that call is written down here.

---

## 1. API corrections against the installed versions

Verified before writing any code against them (aeon 1.5.0, scikit-learn 1.8.0):

| spec says | installed reality | action |
|---|---|---|
| `from aeon.datasets import load_classification` | correct | none |
| `aeon.transformations.collection.convolution_based.{MiniRocket,Rocket,MultiRocket}` | correct | none |
| `aeon.classification.convolution_based.MiniRocketClassifier` | exists | not used; we need the transform and the classifier separately to implement the three tiers |
| `aeon.classification.distance_based.KNeighborsTimeSeriesClassifier` | correct, but the signature is `(distance, distance_params, n_neighbors, ...)` — `distance` is **first** | keyword arguments used everywhere |
| `SelfTrainingClassifier(LogisticRegression, ...)` | in sklearn 1.8 the parameter is `estimator`, not `base_estimator` | passed positionally, which is stable across both |
| `RidgeClassifierCV(alphas=...)` | no `normalize` argument any more (removed in sklearn 1.2) | see §4 below |

`MultiRocket`'s default is `n_kernels=6250` with 4 features per kernel, so the
three transforms do not produce comparable feature dimensionalities at equal
`n_kernels`.  The realised `feature_dim` is recorded in every feature manifest
and in every result row instead of being assumed.

---

## 2. Dataset selection — a deviation from §4, and why

**The specification's compute caps make the target unreachable.**  §4 asks for 40
datasets satisfying

* `n_train + n_test <= 2000`, `series_length <= 500`, `n_classes <= 10`,
* every class has at least 25 training instances.

Scanning the entire aeon index of univariate equal-length UCR datasets (112 of
them; `results/archive_scan.json` has the full table) yields **18**.  §4
anticipates a shortfall and says to extend by scanning the archive index — we
did, exhaustively, and the archive simply does not contain 40 such datasets.
The binding rejections are:

| reason | count |
|---|---|
| `length > 500` | 33 |
| smallest training class `< 25` | 28 |
| `n_train + n_test > 2000` | 26 |
| `n_classes > 10` | 7 |

Only 46 datasets in the whole archive satisfy the *scientific* constraints
(`n_classes <= 10`, min class in train `>= 25`, univariate, equal length, no
missing values), regardless of size.

**Resolution.**  The two size caps are compute budget, not science, so they were
relaxed to the smallest values that yield exactly 40 datasets:

* `n_train <= 1500` (per-cell SSL fits scale with `n_train`; the ridge on the
  full propagated training set is `O(n_train^2 * D)`)
* `n_train + n_test <= 7500` (feature-cache size: 7500 x 9996 float32 = 300 MB)
* `length <= 2000` (the DTW distance matrix costs `O(n_test * n_train * L^2)`)

The scientific constraint — **min class in train >= 25**, which is what makes
`k=20` labels per class feasible with a non-empty unlabeled pool — is untouched.

Six datasets pass the scientific rules but are excluded as too expensive, and
they are listed by name in `configs/datasets.yaml` under `rejected`:
`ElectricDevices` (n_train 8926), `FordA` (3601), `FordB` (3636),
`PhalangesOutlinesCorrect` (1800), `StarLightCurves` (n 9236),
`HandOutlines` (length 2709).  **This is a size-based exclusion, announced here
and in the paper; it is not a result-based one.**

The final 40 datasets range from 57 to 1000 training instances, 15 to 1751
timepoints, and 2 to 8 classes.

---

## 3. Feature caching and what it means for the timings

The MiniRocket transform, the standardiser and the kNN graphs are all fitted on
`L u U`, which **is** the whole training set, so none of them depends on `k`,
on the evaluation seed, or on which points are labeled.  They are computed once
per dataset and reused by every cell.  This is not a shortcut — it is the
operational form of the paper's central claim.

Consequences to keep honest:

* the one-time cost is measured (`features/*/*/manifest.json`) and reported;
* every result row carries `unsup_seconds_amortized`, the one-time cost of the
  dataset, so the RQ6 Pareto figure can charge it;
* `SUP-strict` is deliberately **not** cached: its transform is fitted on
  `X_train[L]` and therefore depends on `(k, seed)`.  It is recomputed per cell.

### LabelSpreading with a precomputed graph

sklearn's `LabelSpreading(kernel="knn")` rebuilds the kNN graph on every `fit`.
We instead pass a callable kernel that returns the cached affinity matrix;
sklearn still computes the normalised graph Laplacian and runs the same power
iteration.  `tests/test_methods_sanity.py::test_precomputed_graph_matches_sklearn_knn_kernel`
asserts the two paths produce **bit-identical** `label_distributions_`.

### Single transform seed

The cached `L u U` features use one transform seed (`features.transform_seed: 0`)
for all 10 evaluation seeds.  Using a fresh transform seed per evaluation seed
would multiply the feature cache by 10 (~47 GB) for a variance component we
expected to be negligible.  The evaluation seed still controls the thing we
actually care about — which instances are labeled.

**We measured this rather than asserting it.**
`python scripts/run_experiments.py --transform-seed-check` runs 5 datasets x 3
transform seeds x 5 evaluation seeds at k=5 and decomposes the accuracy
standard deviation (`results/transform_seed_check.json`):

| method | sd across transform seeds | sd across evaluation seeds | ratio |
|---|---|---|---|
| `sup_tf` | 0.0048 | 0.0591 | 0.08 |
| `safe_ssl` | 0.0071 | 0.0634 | 0.11 |

The MiniRocket seed contributes 8–11% of the variability that the label draw
contributes, so fixing it costs almost nothing and is reported in the paper.

---

## 4. Standardisation, and where it is (not) applied

Per §5: `StandardScaler` is fitted on `L u U` only and applied before graph
construction and before logistic regression.  It is **not** applied before the
ridge.  Reason: the canonical MiniRocket pipeline is ridge on raw PPV features,
and `Delta(SAFE-SSL - SUP-TF)` must isolate pseudo-labeling, not a change of
feature scaling — both arms therefore use identical (unscaled) features and
differ only in the training set.  `RidgeClassifierCV` no longer has a
`normalize` argument in scikit-learn 1.8, so there is no hidden scaling either.

## 5. Probabilities

`RidgeClassifierCV` has no `predict_proba`.  Rather than manufacturing
probabilities from `decision_function`:

* `self_train` uses `LogisticRegression(max_iter=2000)` on standardised features
  (this is what sklearn's `SelfTrainingClassifier` needs);
* `label_spread` and `safe_ssl` take their posteriors from the label-spreading
  distribution itself, which is a genuine distribution over classes;
* every other method uses the ridge and never needs probabilities.

## 6. Ridge alpha selection at k=1

`RidgeClassifierCV` picks `alpha` by closed-form leave-one-out.  With one
labeled instance per class, leaving one out removes an entire class, so the
selection is meaningless.  Rule implemented in `fit_ridge`: use
`RidgeClassifierCV` when `n >= 4` **and** `n >= 2 * n_classes`, otherwise
`RidgeClassifier(alpha=1.0)`.  In practice this means k=1 always uses the fixed
alpha.  The branch taken is recorded per row as `extra.ridge_cv`.

## 7. The safety valve at k=1

The valve compares cross-validated accuracy of the SSL model against `sup_tf`
**using labeled data only**.  At `k=1` there is no held-out labeled data at all,
so no honest estimate exists.  Policy: the valve **abstains** and the SSL model
is kept; the row records `valve.status = "abstain_k_too_small"` and
`safe_ssl_fired = false`.  This is a documented policy choice, not an estimate.

Inside each fold, the held-out labeled instances *join the unlabeled pool* for
the SSL run — otherwise the SSL model would be evaluated on points it had
already seen as pseudo-labels.

## 8. `safe_ssl_novalve` — an eighth arm

§11 asks table T4 to report the safety numbers "with and without the safety
valve".  That counterfactual cannot be inferred from the `safe_ssl` rows, so
SAFE-SSL with the valve removed is run as its own arm (`safe_ssl_novalve`) and
reported as a measured number.  It shares `_ssl_core` with `safe_ssl`, so the
only difference between the two arms is the valve.

## 9. 1NN-DTW: cached distance matrix and a warping window

DTW distances do not depend on labels, so the `(n_test, n_train)` matrix is
computed once per dataset and every `(k, seed)` cell reduces to an `argmin` over
the columns in `L`.  Without this, 1NN-DTW alone would dominate the runtime of
the entire study.

A Sakoe-Chiba band of 10% of the series length is used
(`dtw.window: 0.1`), which is the standard configuration for the UCR 1NN-DTW
baseline and the only way the longest datasets fit the budget.  The window is
recorded in the DTW cache manifest and in `paper/reproducibility.json`.

## 10. `sup_full` is computed once per dataset

With a fixed transform seed and all training labels, `sup_full` has no
dependence on `k` or on the evaluation seed — there is nothing for a seed to
vary.  It is computed once per dataset (`results/sup_full.json`) and reused as
the recovery-ratio ceiling.  The alternative (recomputing 10 identical numbers)
would only add noise-free duplication.

## 11. Recovery ratio when the ceiling equals the floor

`recovery = (acc - acc_strict) / (acc_full - acc_strict)` is undefined when the
denominator is zero.  Those cells are recorded as `null`, excluded from the
aggregate, and counted.  They are never clipped, imputed or set to 1.

## 12. BLAS threads

Every worker pins BLAS to a single thread.  With 14 joblib processes each
spawning 16 BLAS threads the machine oversubscribes 14x and gets *slower*; it
also introduces run-to-run nondeterminism in reduction order.

## 13. Recovery ratio: a guard the specification did not anticipate

The raw recovery ratio divides by `acc_full - acc_strict`.  On several datasets
the fully-supervised ceiling is *at or below* the low-label floor (MiniRocket +
ridge is already saturated, and more labels change nothing), so the denominator
is a rounding artefact and the ratio explodes: the first aggregation produced a
mean recovery of **-1.54** for 1NN-DTW at k=20, driven entirely by a handful of
datasets with a 0.3-point denominator.

Aggregation therefore drops datasets whose ceiling exceeds the floor by less
than `stats.recovery_min_gap` (default 0.02 = 2 accuracy points) and reports the
**median** as the headline with the mean beside it.  The number of datasets used
and dropped is printed in the table note (at k=20, 9 of 40 are dropped).  The
per-row `recovery_ratio` in `results/raw/` is left raw and unguarded; the guard
lives only in the aggregation, and is stated wherever the aggregate appears.

## 14. Surprises — and they are the paper

Three findings contradict or invert the specification's hypotheses.  They are
reported as measured, in `paper/results_summary.md`, under **Verdict**.

**(a) Tier 1 is worth nothing.**  `SUP-TF − SUP-strict` — the gain from fitting
the MiniRocket transform on the unlabeled pool rather than on the labeled subset
— averages **+0.01 accuracy points** across budgets and is *never* significant
(Wilcoxon p ∈ [0.12, 0.23] at all five budgets); it is negative at k=10 and
k=20.  The mechanism is visible in the MiniRocket construction: only the **bias
quantiles** depend on the data, while the 84 kernel patterns and the dilations
depend on the series length alone.  Free unlabeled data buys better bias
estimates, and better bias estimates buy nothing.

**(b) The LFA hypothesis is inverted, not merely unsupported.**  The
specification predicts a POSITIVE correlation between label-free alignment and
the realised SSL gain.  The measurement is NEGATIVE at every budget (Spearman
ρ −0.45 to −0.22, significant at 4/5 budgets).  SSL helps most where the
unlabeled feature geometry is *worst*.

The obvious confound — clean geometry implies the supervised model is already
near the ceiling, so there is no headroom — is ruled out: LFA does not correlate
with headroom (ρ +0.13 to +0.15, p > 0.34 at every budget), and the partial
correlation of LFA with the gain controlling for headroom is *stronger* than the
raw one (−0.48 to −0.29).  `stab` (bootstrap cluster stability) is the single
most predictive component and is significant at all five budgets.

LFA is still useful as a free, label-free decision statistic — it just points
the other way.

**(c) The valve, not the pseudo-labels, is the contribution.**  `self_train` has
the best mean rank at every budget k ≥ 2, but regresses against `sup_tf` on
19–41% of (dataset, seed) pairs.  SAFE-SSL trades roughly one rank position for
a regression rate of 8–11% at k ≥ 2 and roughly halves the worst-case loss.  The
`safe_ssl_novalve` arm isolates this: identical pseudo-labels, no valve, 30–36%
regression rate.
