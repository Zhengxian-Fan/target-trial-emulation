# Target Trial Emulation & Causal Inference Pipeline

This repository contains a pipeline for performing **Target Trial Emulation (TTE)** on observational healthcare data (e.g., CPRD). It uses **PySpark** for data extraction, **R** for missing data imputation, and **Python** for causal estimation.

## 📂 Project Structure

| File | Engine | Description |
| :--- | :--- | :--- |
| **`generate_demo_data.py`** | Python | **Demo:** Generates a purely synthetic cohort from scratch to demonstrate the pipeline without needing real data access. |
| **`build_target_trial.py`** | PySpark | **Step 1:** Extracts data, constructs sequential trials, applies eligibility criteria, and creates a consolidated cohort. |
| **`impute.R`** | R | **Step 2:** Performs Multiple Imputation (MICE) on the cohort to handle missing baseline covariates. |
| **`estimate_treatment_effects.py`** | Python | **Step 3:** Runs causal estimators (PSM, IPTW, TMLE) on imputed data and pools results using Rubin's Rules. |
| **`simulate.py`** | Python | **Benchmarking:** Generates a semi-synthetic dataset based on *real* covariates but simulated outcomes to validate estimators. |

---

## ⚡ Quick Start (Demo)

You can run the full causal estimation pipeline immediately using the demo generator. **This does not require access to raw data, extraction, or imputation steps.**

**Expected Run Time:** < 1 minute on a standard desktop computer.

1.  **Generate Demo Data:**
    This script creates synthetic imputed datasets (e.g., `synthetic_imputed_data/`) from scratch.
    ```bash
    python generate_demo_data.py
    ```

2.  **Run Estimation:**
    Run the analysis (PSM, IPTW, TMLE) on the generated demo data.
    ```bash
    python estimate_treatment_effects.py
    ```

---

## 🚀 Full Workflow (Real Data)

For the full analysis using real observational data, follow these steps:

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

> **Note:** As shown in the **Quick Start**, this step can be run directly on **simulated/demo data** without running Steps 1 & 2.

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

## 🧪 Simulation (Benchmarking)

**Script:** `simulate.py`

This script is for method validation. Unlike the demo generator, it requires the **real cohort** (`all.parquet`) to preserve the true covariate structure, but simulates **Treatment** and **Outcome** based on a known Hazard Ratio.

1.  Ensure `all.parquet` exists (run Step 1).
2.  Run `python simulate.py` to generate `simulated_all.parquet`.
3.  Point `estimate_treatment_effects.py` to this file to check if it recovers the true effect size.

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

-----

## 📦 Installation & Requirements

**Estimated Installation Time:** \~5-10 minutes on a standard desktop computer.

### Python Environment

Tested on Python 3.8+.

  * `pyspark` (\>= 3.2.0)
  * `pandas` (\>= 1.3.0)
  * `numpy` (\>= 1.21.0)
  * `scipy` (\>= 1.7.0)
  * `scikit-learn` (\>= 1.0.0)
  * `lifelines` (\>= 0.26.0)
  * `joblib` (\>= 1.1.0)

To install Python dependencies:

```bash
pip install pyspark pandas numpy scipy scikit-learn lifelines joblib
```

### R Environment

Tested on R 4.0+.

  * `arrow` (\>= 6.0.0)
  * `tidyverse` (\>= 1.3.0)
  * `mice` (\>= 3.14.0)

To install R dependencies:

```r
install.packages(c("arrow", "tidyverse", "mice"))
```