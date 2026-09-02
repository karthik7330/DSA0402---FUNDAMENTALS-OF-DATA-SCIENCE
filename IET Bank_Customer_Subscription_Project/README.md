
# Bank Customer Subscription Prediction and Customer Segmentation

## Files
- `bank_marketing_solution.py` – complete end-to-end implementation.
- `report_draft.docx` – report structured to the uploaded DSA0402 assignment.
- `REPORT_GENERATION_NOTES.txt` – how to replace placeholders with actual execution results.
- `data/` – put `bank-full.csv` here if automatic UCI retrieval is unavailable.
- `outputs/` – generated tables and figures.

## Dataset
Use the UCI Machine Learning Repository Bank Marketing dataset, specifically `bank-full.csv`.
The dataset contains 45,211 records and 17 columns: 16 input attributes and the binary target `y`.
The target indicates whether the customer subscribed to a term deposit.

## Run
```bash
pip install pandas numpy scipy scikit-learn matplotlib python-docx ucimlrepo
python bank_marketing_solution.py
```

If internet access is unavailable, manually download the UCI dataset and place
`bank-full.csv` in the `data` directory.

## Important modelling decision
`duration` is included in descriptive analysis/EDA but excluded from the predictive
feature set. It is the duration of the current call and is only known after the
contact has taken place, so using it for a pre-campaign targeting model would cause
information leakage.

## Required outputs
The program creates:
- preprocessing summary
- descriptive statistics and variances
- covariance and correlation matrices
- 95% confidence interval
- EDA graphs
- confusion matrices
- model comparison
- K selection by silhouette score
- customer cluster profiles
- customer segment CSV
- execution summary

The exact numerical model results should be copied into the final report only after
running the program on the dataset. This avoids inventing results.
