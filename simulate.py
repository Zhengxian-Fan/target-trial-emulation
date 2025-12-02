import os
import numpy as np
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

CONFIG = {
    # Paths (Relative to project root)
    "input_path": "./target_trial_data/trial_BB_HFrEF_2010_60/all.parquet",
    "output_path": "./target_trial_data/trial_BB_HFrEF_2010_60/simulated_all.parquet",
    
    # Simulation Parameters
    "seed": 88,
    "max_time": 24,       # Follow-up months
    "H0": 0.0001,         # Baseline hazard
    "lambda_decay": 0.25, # Lambda (Hazard decay rate)
    "theta_treat": -0.5,  # Treatment effect (Log Hazard Ratio)
    "time_decay": 1.01,   # Time-dependent decay factor
    "no_censoring": 1.0,  # Probability of NOT being censored (1.0 = no censoring)
    "alpha": 1.0          # Multiplier for confounding coefficients
}

# Comorbidities used to construct the confounding score (Z*)
CONFOUNDER_COLS = [
    'hypertension_history', 
    'diabetes_history', 
    'ischaemic_heart_disease_history', 
    'thyroid_disorders_history', 
    'chronic_obstructive_pulmonary_disease_history'
]

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def generate_z_star(df):
    """
    Calculates composite comorbidity score Z*.
    Z* = 1 if sum(comorbidities) > 2, else 0.
    """
    # Ensure columns exist in dataframe
    for col in CONFOUNDER_COLS:
        if col not in df.columns:
            df[col] = 0
            
    # Calculate sum of conditions
    z_sum = df[CONFOUNDER_COLS].sum(axis=1)
    
    # Apply threshold
    return (z_sum > 2).astype(int)

def event_ascertainment(hazards_matrix, no_censoring_prob, seed):
    """
    Simulates event and censoring times based on hazard probabilities.
    
    Args:
        hazards_matrix (np.array): (N_patients x T_time) matrix of hazard probabilities.
        no_censoring_prob (float): Probability of NOT being censored at each step.
        seed (int): Random seed for reproducibility.
        
    Returns:
        times (np.array): Time to event or censoring.
        labels (np.array): 1 if event occurred, 0 if censored.
    """
    np.random.seed(seed)
    n_patients, n_time = hazards_matrix.shape
    
    labels = []
    times = []
    
    for i in range(n_patients):
        patient_time = n_time # Default to max follow-up
        patient_label = 0     # Default to censored
        
        for t in range(n_time):
            # Hazard at time t for patient i
            h_it = hazards_matrix[i, t]
            
            # 1. Check Censoring status
            # 1 = Not Censored, 0 = Censored
            not_censored = np.random.binomial(1, p=no_censoring_prob)
            
            if not_censored == 0:
                patient_time = t + 1
                patient_label = 0
                break
            
            # 2. Check Event status
            # 1 = Event Occurred
            event_happened = np.random.binomial(1, p=h_it)
            
            if event_happened == 1:
                patient_time = t + 1
                patient_label = 1
                break
        
        times.append(patient_time)
        labels.append(patient_label)
        
    return np.array(times), np.array(labels)

# =============================================================================
# 3. MAIN SIMULATION
# =============================================================================

def main():
    print(f"Loading data from: {CONFIG['input_path']}")
    df = pd.read_parquet(CONFIG['input_path'])
    
    np.random.seed(CONFIG['seed'])
    
    # --- A. Preprocessing ---
    # Normalize Age
    if 'age_at_trial' in df.columns:
        df['age_minmax'] = (df['age_at_trial'] - df['age_at_trial'].min()) / \
                           (df['age_at_trial'].max() - df['age_at_trial'].min())
    else:
        df['age_minmax'] = 0.5 
        
    # Standardize Gender to numeric (0/1)
    if df['gender'].dtype == object:
        df['gender_numeric'] = df['gender'].apply(lambda x: 1 if str(x).lower() in ['male', '1'] else 0)
    else:
        df['gender_numeric'] = df['gender'].astype(int)

    # --- B. Generate Z* & Treatment ---
    print("Generating Z* and Treatment Assignment...")
    df['Zstar'] = generate_z_star(df)
    
    # Assign Treatment based on Z* (Confounding by indication)
    # Z*=1 (High Comorbidity) -> High probability of treatment (0.95)
    # Z*=0 (Low Comorbidity)  -> Low probability of treatment (0.05)
    prob_treat = np.where(df['Zstar'] == 0, 0.05, 0.95)
    df['sim_treatment'] = np.random.binomial(1, prob_treat)
    
    print(f"Treatment Rate: {df['sim_treatment'].mean():.2%}")

    # --- C. Define Coefficients ---
    low, high = 0.0, 0.5
    alpha = CONFIG['alpha']
    
    beta = {
        'age_minmax': np.random.uniform(low, high),
        'gender_numeric': np.random.uniform(low, high),
    }
    for col in CONFOUNDER_COLS:
        beta[col] = np.random.uniform(low, high) * alpha

    # --- D. Calculate Hazards Matrix (Vectorized) ---
    print("Calculating Hazards Matrix...")
    
    # 1. Static Confounder Score (exp(X*Beta))
    lp = (beta['age_minmax'] * df['age_minmax'].values + 
          beta['gender_numeric'] * df['gender_numeric'].values)
    
    for col in CONFOUNDER_COLS:
        lp += beta[col] * df[col].values
    
    conf_score = np.exp(lp)
    
    # 2. Treatment Score (exp(theta*T))
    treat_score = np.exp(CONFIG['theta_treat'] * df['sim_treatment'].values)
    
    # 3. Create Hazard Matrix (Patients x Time)
    # Formula: H(t) = H0 * exp(-lambda*t) * exp(theta*T) * exp(X*Beta) * exp(log(decay)*t*Z*)
    
    time_steps = np.arange(1, CONFIG['max_time'] + 1)
    
    # Expand scores to dimensions (N, 1) for broadcasting
    conf_score_matrix = conf_score[:, np.newaxis]
    treat_score_matrix = treat_score[:, np.newaxis]
    z_star_matrix = df['Zstar'].values[:, np.newaxis]
    
    # Time-dependent Baseline Hazard (1, T)
    baseline_term = CONFIG['H0'] * np.exp(-CONFIG['lambda_decay'] * time_steps)
    
    # Time-dependent Interaction Term (N, T)
    time_decay_term = np.exp(np.log(CONFIG['time_decay']) * time_steps * z_star_matrix)
    
    # Combine terms
    hazards_matrix = baseline_term * treat_score_matrix * conf_score_matrix * time_decay_term
    
    # Clip hazards to valid probability range [0, 1]
    hazards_matrix = np.clip(hazards_matrix, 1e-8, 1.0)
    
    # --- E. Simulate Outcomes ---
    print("Simulating Events...")
    times, labels = event_ascertainment(hazards_matrix, CONFIG['no_censoring'], CONFIG['seed'])
    
    df['sim_outcome'] = labels
    df['sim_time_to_event'] = times
    
    # --- F. Save Results ---
    print("\nSimulation Statistics:")
    print(f"  Event Rate: {df['sim_outcome'].mean():.2%}")
    print(f"  Mean Time:  {df['sim_time_to_event'].mean():.2f}")
    
    print(f"Saving to: {CONFIG['output_path']}")
    df.to_parquet(CONFIG['output_path'])
    print("Done.")

if __name__ == "__main__":
    main()