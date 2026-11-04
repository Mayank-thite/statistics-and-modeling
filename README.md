# Statistics and Modeling

Built in 2026 on public UCI datasets.

This repo is where I work through the core ideas of data science on real data, from descriptive statistics and hypothesis tests to predictive models I would be comfortable defending. Six notebooks, executed with outputs saved, backed by a small tested Python package in `src/`. Every number below comes from code in this repo.

## What this shows

| Concept | Where |
|---|---|
| Distributions, skew, outliers (IQR), correlation, class imbalance, spotting leakage | [01 EDA and descriptive statistics](notebooks/01_eda_descriptive_stats.ipynb) |
| Sampling distributions, central limit theorem by simulation, CI coverage, bootstrap CIs | [02 Probability and sampling](notebooks/02_probability_and_sampling.ipynb) |
| Welch t-test, Mann-Whitney U, chi-square of independence, one-way ANOVA (classic, Welch, Kruskal-Wallis) with Tukey HSD, assumption checks, effect sizes, power analysis, Bonferroni and Benjamini-Hochberg | [03 Hypothesis testing](notebooks/03_hypothesis_testing.ipynb) |
| A/B test sample-size planning, simulated false positive rate and power, the cost of peeking | [03 Hypothesis testing](notebooks/03_hypothesis_testing.ipynb) |
| OLS with statsmodels: coefficients and CIs, VIF, residual diagnostics, HAC standard errors | [04 Regression](notebooks/04_regression_bike_sharing.ipynb) |
| Ridge, lasso and LightGBM with time-aware validation (chronological holdout, `TimeSeriesSplit`) | [04 Regression](notebooks/04_regression_bike_sharing.ipynb) |
| Logistic regression, random forest, LightGBM in pipelines, stratified CV, ROC-AUC and PR-AUC, imbalance, calibration, cost-based threshold, permutation importance, SHAP, a time-ordered stress test | [05 Classification](notebooks/05_classification_bank_marketing.ipynb) |
| Model cards, what works, what does not, limitations | [06 Model card and summary](notebooks/06_model_card_summary.ipynb) |

## Data

| Dataset | Used for | Source | License |
|---|---|---|---|
| Bike Sharing (Fanaee-T, 2013). Capital Bikeshare, Washington DC, 2011 to 2012. 731 days, 17,379 hours. | Regression, sampling, tests | [UCI ML Repository, id 275](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), DOI 10.24432/C5W894 | CC BY 4.0 |
| Bank Marketing (Moro, Rita and Cortez, 2014). Portuguese bank phone campaigns, May 2008 to Nov 2010, `bank-additional-full.csv`, 41,188 calls. | Classification, tests | [UCI ML Repository, id 222](https://archive.ics.uci.edu/dataset/222/bank+marketing), DOI 10.24432/C5K306 | CC BY 4.0 |

Raw files are not committed. `scripts/download_data.py` fetches them from UCI into `data/raw/`.

## Key results

### Statistics

- **CLT on skewed data.** Hourly rental counts have skew 1.277. By n = 30 the sample mean is close to normal, and its spread matches sigma over root n (33.29 measured vs 33.12 predicted).
- **Interval coverage.** Over 2,000 repeated samples of n = 50, t intervals covered the true mean 95.3% of the time. Percentile bootstrap intervals for the median covered 94.3%.
- **Working days vs days off, total rentals.** Welch t-test p = 0.11, Cohen's d = 0.132. No evidence of a difference, but the test had only 37.9% power for an effect that size. It would take 908 days per group to reach 80%.
- **Casual riders.** Mann-Whitney p = 3.79e-29, rank-biserial r = 0.515. Casual riders are much busier on days off (median 1,338 vs 616).
- **Contact type and subscribing.** Chi-square p = 1.53e-189, Cramer's V = 0.145. Mobile 14.74% vs landline 5.23%.
- **Weather and daily rentals.** Classic, Welch and Kruskal-Wallis ANOVA all reject H0 (p = 3.11e-17, 1.52e-15, 2.59e-15). Eta squared 0.099. Tukey: every pair of weather levels differs.
- **Multiple testing.** With 20 pure-noise features, 64.6% of rounds had at least one false hit at raw p < 0.05. Benjamini-Hochberg cut that to 5.8% and Bonferroni to 5.6%.
- **A/B test.** To detect 11.27% to 12.77% at alpha 0.05 and 80% power needs 7,375 per arm. In 5,000 simulations the false positive rate was 0.0524 and power 0.7902. Peeking 10 times raised the false positive rate to 0.1975.

![Central limit theorem on hourly bike counts](docs/img/clt.png)

### Regression: hourly bike rentals

Daily OLS (731 days): R squared 0.851. Residuals are autocorrelated (Durbin-Watson 1.210), so I report Newey-West standard errors, which are up to 2.08 times the plain OLS ones. Predicted rentals peak at about 27.6 C.

Hourly prediction, trained on Jan 2011 to Sep 2012 and tested on the 2,168 hours of Oct to Dec 2012:

| Model | RMSE | MAE | R squared |
|---|---|---|---|
| Baseline (hour x working-day mean) | 119.59 | 79.95 | 0.648 |
| Ridge | 70.08 | 43.40 | 0.879 |
| Lasso | 70.38 | 44.24 | 0.878 |
| LightGBM | 66.66 | 42.20 | 0.891 |

![LightGBM vs actual, one test week](docs/img/bike_forecast.png)

### Classification: term deposit subscription

`duration` is dropped because call length is only known after the call ends. Positive rate 11.27%. Stratified 80/20 split, 5-fold stratified CV on the 80%.

| Model | CV ROC-AUC | CV PR-AUC | Test ROC-AUC | Test PR-AUC | Test Brier |
|---|---|---|---|---|---|
| Logistic regression | 0.789 | 0.450 | 0.801 | 0.465 | 0.077 |
| Logistic, class-weighted | 0.790 | 0.444 | 0.801 | 0.460 | 0.162 |
| Random forest | 0.792 | 0.469 | 0.809 | 0.494 | 0.074 |
| LightGBM | 0.795 | 0.469 | 0.814 | 0.495 | 0.074 |

Class weighting did not help ranking and wrecked calibration (Brier 0.162 vs 0.077), so I kept unweighted models and handled the imbalance in the threshold.

**Threshold from costs.** Assuming a wasted call costs 5 and a missed subscriber costs 50, theory says call when p > 0.091. The threshold picked on out-of-fold training predictions was 0.11. On the test set:

| Policy | Calls | Subscribers reached (of 928) | Total cost |
|---|---|---|---|
| Call everyone | 8,238 | 928 | 36,550 |
| Default threshold 0.5 | 407 | 268 | 33,695 |
| Cost-based threshold 0.11 | 1,627 | 615 | 20,710 |

![Test-set cost by threshold](docs/img/bank_cost_threshold.png)

**The honest caveat.** Trained on the first 80% of calls in time order and tested on the last 20%, LightGBM's ROC-AUC fell to 0.658 and logistic regression's to 0.748. The later period converted at 30.83% against 6.37% earlier, with interest rates and employment outside the training range. The top features by permutation importance are exactly those macro columns (`nr_employed`, `euribor3m`), so the random-split scores flatter every model here. Notebooks 5 and 6 go through this.

![ROC and PR curves](docs/img/bank_roc_pr.png)

## How to run

Tested with Python 3.13 on Linux. Needs `make`.

```bash
make all        # venv + pinned install, download data, run tests, execute all notebooks in order
```

Or step by step:

```bash
make setup      # create .venv and install requirements.txt
make data       # download the UCI files into data/raw/
make test       # pytest on src/
make notebooks  # execute every notebook in place with nbconvert
```

A full `make all` takes a few minutes on a laptop. All randomness is seeded (`src.SEED = 42`).

## Layout

```
notebooks/            numbered, executed notebooks
src/
  data.py             loaders for both datasets (unit conversion, pdays flag, duration drop)
  stats_utils.py      effect sizes, bootstrap CI, IQR outliers, two-proportion sample size
  modeling.py         VIF table, chronological split, regression metrics, cost-based thresholds
  plotting.py         one chart style for every figure
tests/                pytest tests for src/
scripts/download_data.py
results/              metrics written by notebooks 4 and 5, read by notebook 6
docs/img/             charts used in this README
```

## Citations

- Fanaee-T, H. (2013). Bike Sharing. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894
- Moro, S., Rita, P., and Cortez, P. (2014). Bank Marketing. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306
- Moro, S., Cortez, P., and Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. Decision Support Systems, 62, 22 to 31.

## License

Code is MIT licensed, see [LICENSE](LICENSE). The datasets belong to their authors and are used under CC BY 4.0.
