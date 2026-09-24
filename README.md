# Credit Risk Scorecard: Predicting Loan Default

Predicting the probability that a borrower will default, and turning it into a credit score a lender can use to approve or decline applications.

![Default rate by score band](images/default_by_band.png)

## Business problem
Approving borrowers who later default costs a lender money; rejecting good borrowers loses revenue. This project builds a **probability of default (PD)** model and a **points-based credit scorecard**, then tests where the lender should set its approval cut-off.

## Data
**163,987 real personal loans** from Lending Club (US peer-to-peer lender), each labelled as repaid or defaulted. The overall default rate is **18.3%**.

The notebook downloads the data automatically. To work offline, save the CSV as `data/loan.csv`:
`https://raw.githubusercontent.com/h2oai/app-consumer-loan/master/data/loan.csv`

## Approach
1. **Exploratory analysis:** default rates by loan term, purpose and income
2. **Feature selection:** excluded `int_rate` (it reflects the lender's own risk grading, which would be data leakage) and `addr_state` (a potential proxy for protected characteristics)
3. **Weight of Evidence (WoE) and Information Value (IV):** binned each feature, measured its predictive power and dropped features with IV below 0.02
4. **Logistic regression scorecard:** fitted on WoE features and scaled to points (600 points = 50:1 odds, 20 points to double the odds)
5. **Gradient boosting challenger model** for comparison
6. **Validation** on a 30% hold-out test set using AUC, Gini and KS, plus a rank-ordering check
7. **Business cut-off analysis:** the trade-off between approval rate and default rate

## Results (test set: 49,197 loans)

| Model | AUC | Gini | KS |
|---|---|---|---|
| Logistic regression scorecard | 0.673 | 0.35 | 0.25 |
| Gradient boosting | 0.686 | 0.37 | 0.27 |

- **The scorecard rank-orders risk cleanly.** Default rates fall in every score band, from **38.8%** in the lowest band to **6.7%** in the highest.
- **Business impact:** declining the lowest-scoring 20% of applicants cuts the default rate among approved loans from **18.3% to 14.5%**, avoiding about **670 defaults per 10,000 applications**.
- **Strongest predictors:** loan term, revolving credit utilisation, debt-to-income ratio and annual income.
- **Gradient boosting is only slightly better** than the scorecard. The scorecard is fully explainable, which is why banks typically use it for decisions and keep machine learning as a challenger.

![ROC curve and score distribution](images/roc_and_scores.png)

![Approval vs risk trade-off](images/cutoff_tradeoff.png)

## How to run
```bash
pip install -r requirements.txt
jupyter notebook credit_risk_scorecard.ipynb
```

## Limitations and next steps
- **The data is old.** Loans are from 2007–2011, so a production model would be validated on recent data and monitored for drift (Population Stability Index).
- **This covers PD only.** A full expected loss model adds Loss Given Default and Exposure at Default: **EL = PD × LGD × EAD** (the IFRS 9 / Basel framework).
- **Binning is simple.** Bins are quantile-based; production scorecards use optimised bins and include multicollinearity and fairness testing.

## Tools
Python · Pandas · NumPy · Scikit-learn · Matplotlib · Jupyter
