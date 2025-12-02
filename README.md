# Target Trial Emulation & Causal Inference Pipeline

This repository contains a pipeline for performing **Target Trial Emulation (TTE)** on observational healthcare data (e.g., CPRD). It uses **PySpark** for data extraction, **R** for missing data imputation, and **Python** for causal estimation.

## 📂 Project Structure

| File | Engine | Description |
| :--- | :--- | :--- |
| **`build_target_trial.py`** | PySpark | **Step 1:** Extracts data, constructs sequential trials, applies eligibility criteria, and creates a consolidated cohort. |
| **`impute.R`** | R | **Step 2:** Performs Multiple Imputation (MICE) on the cohort to handle missing baseline covariates. |
| **`estimate_treatment_effects.py`** | Python | **Step 3:** Runs causal estimators (PSM, IPTW, TMLE) on imputed data and pools results using Rubin's Rules. |
| **`simulate.py`** | Python | **Benchmarking:** Generates a semi-synthetic dataset with a known ground truth to validate the estimators. |

---

## 🚀 Workflow

### 1. Cohort Construction
**Script:** `build_target_trial.py`

Processes demographics, medications, and measurements to create the study cohort. Handles "New User" design and covariate extraction.

* **Input:** Raw Parquet files.
* **Output:** A consolidated cohort file (e.g., `./target_trial_data/.../all.parquet`).

```bash
python build_target_trial.py
````

### 2\. Multiple Imputation

**Script:** `impute.R`

Loads the cohort file and performs multiple imputation for missing variables (e.g., BMI, SBP) using the `mice` package.

  * **Input:** `all.parquet`
  * **Output:** 5 imputed datasets saved in `./imputed_data/`.

<!-- end list -->

```bash
Rscript impute.R
```

### 3\. Causal Estimation

**Script:** `estimate_treatment_effects.py`

Iterates through imputed datasets, runs analysis, and pools estimates (Hazard Ratios / Risk Ratios).

  * **Methods:**
      * **PSM:** Propensity Score Matching (1:1 with caliper).
      * **IPTW:** Inverse Probability of Treatment Weighting (Stabilized).
      * **TMLE:** Targeted Maximum Likelihood Estimation.
  * **Output:** Pooled results with 95% Confidence Intervals.

<!-- end list -->

```bash
python estimate_treatment_effects.py
```

-----

## 🧪 Simulation

**Script:** `simulate.py`

Validates the estimators using semi-synthetic data. It uses real covariates from the cohort but simulates **Treatment** and **Outcome** based on a known Hazard Ratio.

1.  Run `python simulate.py` to generate `simulated_all.parquet`.
2.  Point `estimate_treatment_effects.py` to this file to check if it recovers the true effect size.

-----

## ⚙️ Configuration

Update the `CONFIG` block at the top of each script to match your local file paths before running.

**Example (`build_target_trial.py`):**

```python
CONFIG = {
    "paths": {
        "cohort_output": "./your/local/path/",
    },
    # ...
}
```

## 📦 Requirements

**Python**

  * `pyspark`
  * `pandas`
  * `numpy`
  * `scipy`
  * `scikit-learn`
  * `lifelines`
  * `joblib`

**R**

  * `arrow`
  * `tidyverse`
  * `mice`
