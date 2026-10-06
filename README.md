# Marketing Campaign Analysis with pandas

End-to-end exploratory data analysis (EDA) and regression modelling on a 200,000-row marketing campaign dataset, built as preparation for a data science internship at Opteo.

The whole project lives in one Jupyter notebook: [`opteo_prep_marketing_analysis.ipynb`](opteo_prep_marketing_analysis.ipynb).

## TL;DR

- The data is cleaned, enriched with standard ad metrics (CTR, CPC, CPA, CPM) and explored by channel, campaign type, audience and time.
- **The dataset turns out to be synthetic random data.** Distributions are roughly uniform, correlations are close to zero, and no segment clearly does better than another.
- An OLS regression of ROI on 27 predictors gets **R² ≈ 0.0002**. Only 1 of 27 predictors is significant at α = 0.05, which is about what chance alone would produce (≈ 1.4).
- So the modelling section is about **methodology**: setting up a regression correctly, reading its diagnostics, and recognising when a model has found nothing.

## Contents

- [Dataset](#dataset)
- [Getting started](#getting-started)
- [Notebook walkthrough](#notebook-walkthrough)
- [Derived metrics](#derived-metrics)
- [Results](#results)
- [Lessons learned](#lessons-learned)
- [Possible next steps](#possible-next-steps)

## Dataset

The notebook expects a CSV at `data/marketing_campaign_dataset.csv`. **The data file is not committed to this repository**, so download it yourself and put it at that path.

The columns match the public *Marketing Campaign Performance Dataset* on Kaggle. It has 200,000 rows, 16 columns and no missing values.

| Column | Raw type | Description |
|---|---|---|
| `Campaign_ID` | int | Unique campaign identifier |
| `Company` | str | Company running the campaign |
| `Campaign_Type` | str | Campaign type, e.g. Email, Influencer, Search, Social Media, Display |
| `Target_Audience` | str | Demographic target, e.g. `Men 18-24`, `Women 35-44` |
| `Duration` | str | Campaign length as text, e.g. `"30 days"` (15, 30, 45 or 60) |
| `Channel_Used` | str | Facebook, Instagram, YouTube, Google Ads, Email, Website |
| `Conversion_Rate` | float | Fraction of clicks that converted (0.01–0.15) |
| `Acquisition_Cost` | str | Spend as a currency string, e.g. `"$16,174.00"` |
| `ROI` | float | Return on investment (2.0–8.0) |
| `Location` | str | City, e.g. New York, Los Angeles, Chicago, Houston, Miami |
| `Language` | str | Campaign language |
| `Clicks` | int | Total clicks (100–1,000) |
| `Impressions` | int | Total impressions (1,000–10,000) |
| `Engagement_Score` | int | Engagement score (1–10) |
| `Customer_Segment` | str | e.g. Foodies, Tech Enthusiasts, Health & Wellness |
| `Date` | str | Campaign date, starting 2021-01-01 |

## Getting started

### Requirements

The notebook was last run with **Python 3.14**. These libraries are needed:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `statsmodels`
- `scikit-learn`
- `jupyter`

### Setup

```bash
git clone <this-repo-url>
cd pandas-marketingdata-prep

python3 -m venv .venv
source .venv/bin/activate

pip install pandas numpy matplotlib seaborn statsmodels scikit-learn jupyter

mkdir -p data
# put marketing_campaign_dataset.csv inside data/

jupyter notebook opteo_prep_marketing_analysis.ipynb
```

Then run all the cells from top to bottom. On 200k rows the whole notebook runs in well under a minute on a laptop.

## Notebook walkthrough

### 1. Loading the data

Reads the CSV and inspects its shape, dtypes and summary statistics. First observations:

- There are no missing values.
- `Acquisition_Cost`, `Duration` and `Date` are stored as strings and need parsing.
- `Conversion_Rate` and `ROI` are already numeric, so they are probably pre-computed.
- There are no `CTR` or `CPC` columns, so those have to be derived.

### 2. Data cleaning and feature engineering

| Step | Transformation |
|---|---|
| Missing values | Checked with `isna().sum()`; none found |
| `Acquisition_Cost` | Removed `$` and `,`, then cast to `float` |
| `Duration` | Extracted the number of days into `Duration_Days` (int) |
| `Date` | Parsed with `pd.to_datetime`; derived `Year`, `Month`, `DayOfWeek` |
| Ad metrics | Derived `CTR`, `CPC`, `Conversions`, `CPA`, `CPM` (see [below](#derived-metrics)) |
| Sanity checks | Confirmed there are no infinities and no negative values in the cost and rate metrics |

### 3. Exploratory data analysis

| Section | What it shows |
|---|---|
| 3.1 Distributions | Histograms of Impressions, Clicks, Acquisition Cost, CTR, Conversion Rate and ROI |
| 3.2 By channel | Aggregated table of average CTR, CPC, ROI, conversion rate, total spend and campaign count per channel, plus an ROI boxplot |
| 3.3 By campaign type and audience | ROI boxplot by campaign type; heatmap of average ROI for each Campaign Type × Channel pair |
| 3.4 Correlations | Correlation heatmap of 11 numeric features, raw and derived |
| 3.5 Time trends | Daily total spend and average ROI over time; ROI by day of week |
| 3.6 Cost vs. clicks | Hexbin plot of Acquisition Cost against Clicks |

### 4. Predictive modelling

| Section | What it does |
|---|---|
| 4.1 Target and predictors | Target: `ROI`. Predictors: 6 numeric features plus 5 one-hot encoded categoricals (`drop_first=True` to avoid the dummy-variable trap), giving 27 features |
| 4.2 OLS fit | `statsmodels` OLS with an intercept; full summary table; counts significant p-values against the number expected by chance |
| 4.3 Diagnostics | Residuals-vs-fitted plot, Q-Q plot, residual histogram, and the variance inflation factor (VIF) for the numeric predictors |
| 4.4 Alternative specification | Replaces `Acquisition_Cost` with `log1p(Acquisition_Cost)` and compares R² and AIC |

## Derived metrics

| Metric | Formula | Meaning |
|---|---|---|
| **CTR** (click-through rate) | `Clicks / Impressions` | Share of impressions that got a click |
| **CPC** (cost per click) | `Acquisition_Cost / Clicks` | Average cost of one click |
| **Conversions** | `Clicks × Conversion_Rate` | Estimated number of conversions |
| **CPA** (cost per acquisition) | `Acquisition_Cost / Conversions` | Average cost of one conversion |
| **CPM** (cost per mille) | `Acquisition_Cost / Impressions × 1000` | Cost per 1,000 impressions |

## Results

### EDA

Every channel performs almost the same:

| Channel | Avg CTR | Avg CPC | Avg ROI | Avg Conv. Rate | Campaigns |
|---|---|---|---|---|---|
| Facebook | 0.1405 | $32.13 | 5.019 | 0.0800 | 32,819 |
| Website | 0.1410 | $31.78 | 5.014 | 0.0802 | 33,360 |
| Google Ads | 0.1392 | $32.31 | 5.003 | 0.0802 | 33,438 |
| Email | 0.1405 | $31.88 | 4.996 | 0.0803 | 33,599 |
| YouTube | 0.1412 | $31.87 | 4.994 | 0.0799 | 33,392 |
| Instagram | 0.1400 | $32.08 | 4.989 | 0.0799 | 33,392 |

The best and worst channels differ in average ROI by only **0.03**, against a standard deviation of about 1.73. The same holds for campaign type, audience, location, day of week and date. Raw features are almost uncorrelated with each other; the only notable correlations are between derived metrics and the inputs they are computed from (e.g. CPC and Clicks).

### Regression

| Model | R² | AIC |
|---|---|---|
| OLS, raw `Acquisition_Cost` | 0.000190 | 787,877.4 |
| OLS, `log1p(Acquisition_Cost)` | 0.000184 | 787,878.7 |

- The overall F-test has **p = 0.078**, so the model as a whole is not significant at α = 0.05.
- Only `Acquisition_Cost` has p < 0.05 (p ≈ 0.040). With 27 predictors, about 1.4 false positives are expected at that threshold, so this is consistent with noise.
- The residuals are close to uniform rather than normal (kurtosis ≈ 1.8), which matches a target drawn uniformly from 2 to 8.
- The numeric predictors' VIFs range from 4.1 to 6.8. Without an intercept, VIF values are inflated, so they are not a sign of real multicollinearity here.
- Log-transforming cost changes nothing, which confirms that there is no hidden non-linear signal in this feature.

## Lessons learned

1. **Check whether the data has signal before modelling.** Flat distributions and correlations near zero in the EDA predicted the regression result.
2. **Significance depends on how many tests you run.** One "significant" predictor out of 27 is what chance alone would give.
3. **Large samples make tiny effects significant.** With n = 200,000, standard errors are small, so effect sizes and R² matter more than p-values.
4. **A null result is still a result.** Reporting clearly that a model found nothing is more useful than over-fitting noise.

## Possible next steps

- Run the same pipeline on real campaign data, such as a Google Ads export, where the channel and audience effects should be real.
- Add a train/test split and cross-validation, and compare OLS with regularised models (Ridge, Lasso) and tree-based models.
- Correct for multiple comparisons (Bonferroni or Benjamini–Hochberg).
- Model CTR or conversion rate with a binomial/logistic GLM instead of OLS.
- Add a `requirements.txt` and commit a small data sample so the notebook runs out of the box.

## Author

Raphael Vermeil
