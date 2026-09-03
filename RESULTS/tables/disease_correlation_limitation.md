# Disease-correlation limitation

The CKD, diabetes, heart-disease, and breast-cancer training CSVs are independent
patient populations. They contain no shared patient identifier or common disease
indicators, so this project intentionally produces **no disease-to-disease medical
correlation** from them. A valid future comorbidity dataset must contain one row per
patient and `patient_id` plus binary `diabetes`, `heart_disease`, `ckd`, and
`breast_cancer` columns. `ANALYTICS.disease_correlation.analyse_comorbidity_dataset`
validates that structure before calculating a Pearson association matrix. Correlation
does not imply causation.
