import os
import numpy as np
import pandas as pd
from datetime import timedelta

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = "./synthetic_data"
N_PATIENTS = 200       # Size of synthetic cohort
N_IMPUTATIONS = 1       # Number of imputed datasets to generate
SEED = 2024

# Column definitions (Must match estimate_treatment_effects.py COVARIATES_CONFIG)
CATEGORICALS = {
    "gender": ["Male", "Female"],
    "gen_ethnicity_mapped": ["White", "Black", "Asian", "Mixed", "Other"],
    "region": ["London", "North West", "South East", "Midlands", "South West"],
    "imd2015_5": ["1", "2", "3", "4", "5"],
    "smoke": ["Non-smoker", "Ex-smoker", "Active smoker"]
}

NUMERICS = ["bmi", "sbp", "dbp", "creatinine", "tchdl_rat"]

HISTORY_FLAGS = [
    "hypertension_history", "diabetes_history", "ischaemic_heart_disease_history",
    "atrial_fibrillation_history", "stroke_history", "dyslipidaemia_history",
    "asthma_history", "peripheral_arterial_disease_history",
    "valve_disorders_history", "chronic_kidney_disease_history",
    "pulmonary_embolism_history", "malignant_cancer_history"
]

MEDS = ["ACE", "ARB", "Diuretics", "CCB", "statin", "digoxin", "anticoagulant", "antiplatelet"]

# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_synthetic_dataset(n, seed):
    np.random.seed(seed)
    df = pd.DataFrame()
    
    # 1. Core Identifiers & Dates
    df["patid"] = np.arange(1, n + 1)
    df["trial_start"] = pd.to_datetime("2010-01-01")
    
    # 2. Categorical Covariates
    for col, categories in CATEGORICALS.items():
        df[col] = np.random.choice(categories, n)
        
    # 3. Numeric Covariates (Random Normal with plausible means)
    df["bmi"] = np.random.normal(28, 5, n)
    df["sbp"] = np.random.normal(130, 15, n)
    df["dbp"] = np.random.normal(80, 10, n)
    df["creatinine"] = np.random.normal(90, 20, n)
    df["tchdl_rat"] = np.random.normal(4, 1, n)
    
    # 4. Binary Flags (History & Meds)
    # Generate random binary 0/1
    for col in HISTORY_FLAGS + MEDS:
        df[col] = np.random.binomial(1, 0.2, n) # ~20% prevalence
        
    # 5. Treatment Assignment (Simulated based on confounding)
    # Sicker patients (more history flags) are more likely to be treated
    risk_score = df[HISTORY_FLAGS].sum(axis=1)
    prob_treat = 1 / (1 + np.exp(-(risk_score - 2))) # Sigmoid
    df["treatment"] = np.random.binomial(1, prob_treat)
    
    # 6. Outcome: All-Cause Mortality (Simulated)
    # Treatment is protective (coeff = -0.5), Risk increases mortality
    linear_pred = 0.2 * risk_score - 0.5 * df["treatment"] - 2
    prob_death = 1 / (1 + np.exp(-linear_pred))
    df["all_cause_mortality_2y"] = np.random.binomial(1, prob_death)
    
    # 7. Dates (Survival Logic)
    # If dead, random date within 2 years. If alive, future date.
    random_days = np.random.randint(1, 730, n)
    df["dod"] = df["trial_start"] + pd.to_timedelta(random_days, unit="D")
    
    # Set dod to NaT (Not a Time) for survivors
    mask_survivor = df["all_cause_mortality_2y"] == 0
    df.loc[mask_survivor, "dod"] = pd.NaT
    
    # Admin censoring dates
    df["enddate"] = pd.to_datetime("2020-01-01")
    df["end_followup_date_2y"] = df["trial_start"] + pd.Timedelta(days=730)
    
    return df

def main():
    print(f"Generating {N_IMPUTATIONS} synthetic datasets with N={N_PATIENTS} patients...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for i in range(1, N_IMPUTATIONS + 1):
        # Vary seed slightly so imputed datasets differ
        df_sim = generate_synthetic_dataset(N_PATIENTS, SEED + i)
        
        filename = f"imputed_dataset_{i}.parquet"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        df_sim.to_parquet(filepath)
        print(f"  Saved: {filepath}")
        
    print("\nDone! You can now run 'estimate_treatment_effects.py'.")
    print(f"Ensure CONFIG path in that script points to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()