# Federated Healthcare Diagnosis System

## Overview

A privacy-preserving Federated Learning system for healthcare diagnosis.

The system trains machine learning models collaboratively across multiple clients without sharing patient data.

Supported Diseases:

- Chronic Kidney Disease (CKD)
- Diabetes
- Heart Disease
- Breast Cancer

---

## Features

- Federated Learning using Flower
- Privacy-Preserving Training
- Multiple Disease Prediction
- Unified Server Architecture
- Unified Client Architecture
- Automatic Accuracy Graph Generation
- Modular Disease Registry
- Research analytics: metrics, client comparisons, charts, and report-ready tables

---

## Technologies Used

- Python
- PyTorch
- Flower Framework
- NumPy
- Pandas
- Scikit-Learn
- Matplotlib

---

## Project Structure

```text
Federated_Healthcare_Diagnosis_System/
│
├── DATASETS/
├── MODELS/
├── UTILS/
├── GRAPHS/
│
├── disease_registry.py
├── server_unified.py
├── client_unified.py
├── launcher.py
├── requirements.txt
└── README.md
```

---

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python launcher.py
```

---

## Results

| Disease | Final Accuracy |
|----------|----------|
| Chronic Kidney Disease | 97.4% |
| Diabetes | 74.7% |
| Heart Disease | 80.0% |
| Breast Cancer | 97.4% |

### Accuracy Graphs

The system automatically generates and stores training graphs inside the GRAPHS folder after federated training.
---

## Research Analytics

Each federated evaluation now records loss, accuracy, precision, recall (sensitivity),
F1-score, specificity, TP, TN, FP, FN, and the number of evaluation samples. The
server saves these measurements by disease, federated round, client, and experiment
run in `RESULTS/<disease>_results.csv`. Results are append-only, so a subsequent
experiment does not overwrite previous runs.

The two federation participants are **simulated clients**, not real hospitals. In
charts and report tables they should be described as `Client 0 (Simulated Hospital A)`
and `Client 1 (Simulated Hospital B)` where that context is useful.

After running a federated experiment, generate the research outputs with:

```bash
python analytics.py
# or
python -m ANALYTICS.generate_all
```

This generates CSV and Markdown tables in `RESULTS/tables/`, including overall
performance, client comparisons, dataset characteristics, and client-accuracy
correlations across federated rounds. It also generates performance and class-
distribution charts in disease-specific `GRAPHS/` folders. The existing
accuracy-vs-round graph remains unchanged.

### Disease-correlation limitation

The four provided disease datasets are independent populations, without shared
patient IDs or multiple disease indicators per patient. They **cannot** be used to
claim diabetes–heart disease, diabetes–CKD, or other medical correlations. The
project deliberately does not fabricate those results.

`ANALYTICS/disease_correlation.py` is ready for a future, scientifically valid
comorbidity dataset containing one row per patient and at least:

```text
patient_id, diabetes, heart_disease, ckd, breast_cancer
```

Each disease indicator must be binary (0/1). The module validates this shape before
producing an association matrix; correlation remains association, not causation.

---

## Future Scope

- Differential Privacy
- Secure Aggregation
- More Diseases
- Real Hospital Integration

---

## Author

Devansh
