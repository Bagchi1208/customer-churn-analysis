# Customer Churn Analysis

A leakage-audited, cross-validated classification pipeline predicting customer churn on the IBM Watson Telco Customer Churn dataset (extended version, 7043 customers, 33 raw columns).

## Problem

Telecom companies lose revenue every time a customer churns. The goal here is to build a model that flags at-risk customers early enough for a retention team to act — and to do that with a pipeline that's actually trustworthy, not one that looks good on paper because of hidden leakage.

## Dataset

IBM Watson Telco Customer Churn (extended), including demographic info, service subscriptions, billing details, and business metrics (Churn Score, CLTV, Churn Reason). [Add your source link here, e.g. Kaggle/IBM's dataset page.]

## Approach

1. **Leakage audit** — identified and excluded columns that wouldn't be available at real prediction time (`Churn Reason`, only populated post-outcome) or that were themselves outputs of another churn model (`Churn Score`). `CLTV` was checked empirically (no clean separation across churn labels) and kept.
2. **Cleaning** — resolved a dtype issue in `Total Charges` (11 rows stored as strings for zero-tenure customers), tied the fix to the actual cause rather than blind imputation.
3. **Target-aware EDA** — tested hypotheses against the data rather than plotting everything: found fiber-optic internet and month-to-month contracts both strongly associated with higher churn.
4. **Preprocessing** — stratified train/test split before any encoding, `ColumnTransformer` with ordinal encoding for `Contract` (explicit duration order), one-hot for nominal categoricals, standard scaling for numeric features — fit only on training data.
5. **Imbalance handling** — SMOTE applied strictly inside cross-validation folds (via `imblearn.Pipeline`) to avoid synthetic-sample leakage across folds.
6. **Model comparison** — 5-fold stratified CV across Logistic Regression, Decision Tree, Random Forest, SVC, and XGBoost, scored on accuracy, precision, recall, F1, and ROC-AUC.
7. **Metric-driven selection** — chose **recall** as the priority metric (a missed churner costs more than a wasted retention offer) and selected Logistic Regression over Random Forest on that basis, despite its lower raw accuracy.
8. **Tuning** — grid search over `C`/`penalty`, optimized directly for recall.
9. **Interpretation** — examined which features survived L1 regularization to identify the strongest churn drivers.

## Key Result

The tuned Logistic Regression model (`C=0.01`, `penalty='l1'`) catches **306 of 374 churners (82%)** on the held-out test set, at the cost of 307 false positives — a deliberate trade-off given that missing an actual churner is more costly to the business than one unnecessary retention offer.

**Strongest churn drivers identified by the model:**
- Higher risk: fiber-optic internet, higher monthly charges, month-to-month contracts, electronic check payment
- Lower risk: longer tenure, having dependents, longer contract terms, online security/tech support add-ons

## Business Takeaway

Retention efforts should prioritize newer, month-to-month customers on fiber-optic plans paying by electronic check — the segment the model consistently flags as highest-risk.

## Repo Structure

```
customer_churn_documented.ipynb   # full analysis notebook
churn_model.pkl                    # saved tuned model
preprocessor.pkl                   # saved fitted ColumnTransformer
README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn
jupyter notebook customer_churn_documented.ipynb
```

## Author

[Your name] — [LinkedIn/portfolio link]
