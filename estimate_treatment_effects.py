import os
import glob
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from lifelines import CoxPHFitter
from joblib import Parallel, delayed
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
np.random.seed(1234)

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

DATA_PATHS = {
    "imputed_data_dir": "./imputed_data/",
    "output_dir": "./results/"
}

STUDY_PARAMS = {
    "treatment": "treatment",
    "outcome": "all_cause_mortality_2y",
    "caliper": 0.01,
    "trimming_thresholds": np.round(np.arange(0.01, 0.11, 0.01), 2), # Matching your notebook range
    "n_folds": 5
}

COVARIATES_CONFIG = {
    "core": ["patid", "trial_start"],
    "dates": ["dod", "enddate", "end_followup_date_2y"],
    "categorical": ["gender", "gen_ethnicity_mapped", "region", "imd2015_5", "smoke"],
    "numeric": ["bmi", "sbp", "dbp", "creatinine", "tchdl_rat"],
    "history_flags": [
        "hypertension_history", "diabetes_history", "ischaemic_heart_disease_history",
        "atrial_fibrillation_history", "stroke_history", "dyslipidaemia_history",
        "asthma_history", "peripheral_arterial_disease_history",
        "valve_disorders_history", "chronic_kidney_disease_history",
        "pulmonary_embolism_history", "malignant_cancer_history"
    ],
    "meds": ["ACE", "ARB", "Diuretics", "CCB", "statin", "digoxin", "anticoagulant", "antiplatelet"]
}

# =============================================================================
# 2. RUBIN'S RULES (Pooling)
# =============================================================================

class RubinsRules:
    @staticmethod
    def pool_estimates(estimates, std_errors):
        m = len(estimates)
        if m == 0: return np.nan, np.nan, np.nan
        
        pooled_est = np.mean(estimates)
        within_var = np.mean(np.array(std_errors)**2)
        between_var = np.var(estimates, ddof=1)
        total_var = within_var + (1 + 1/m) * between_var
        pooled_se = np.sqrt(total_var)
        
        lower_ci = pooled_est - 1.96 * pooled_se
        upper_ci = pooled_est + 1.96 * pooled_se
        
        return np.exp(pooled_est), np.exp(lower_ci), np.exp(upper_ci)

# =============================================================================
# 3. CV-TMLE LOGIC (Matches your CV_TMLE.py)
# =============================================================================

class CVTMLE_Sturmer:
    """
    Implements the targeting step of TMLE on Cross-Validated predictions.
    Logic mirrored from CVTMLE_PERCENTILE class.
    """
    def __init__(self, q_t0, q_t1, g, t, y, truncate_level=0.05):
        self.q_t0 = q_t0
        self.q_t1 = q_t1
        self.g = g
        self.t = t
        self.y = y
        self.truncate_level = truncate_level

    def _trim_data(self, q_t0, q_t1, g, t, y, level):
        """
        Applies Stürmer trimming based on group-specific quantiles of g.
        """
        treated_mask = (t == 1)
        untreated_mask = (t == 0)
        
        # Calculate thresholds
        lb_treated = np.quantile(g[treated_mask], level) if np.sum(treated_mask) > 0 else 0.0
        ub_untreated = np.quantile(g[untreated_mask], 1 - level) if np.sum(untreated_mask) > 0 else 1.0
            
        # Create mask
        keep_mask = np.zeros_like(g, dtype=bool)
        keep_mask[treated_mask] = g[treated_mask] >= lb_treated
        keep_mask[untreated_mask] = g[untreated_mask] <= ub_untreated
        
        return q_t0[keep_mask], q_t1[keep_mask], g[keep_mask], t[keep_mask], y[keep_mask]

    def _perturbed_model(self, q_t0, q_t1, g, t, eps):
        # Clever covariate H
        g_clipped = np.clip(g, 1e-6, 1 - 1e-6)
        h = t * (1. / g_clipped) - (1. - t) / (1. - g_clipped)
        
        # Offset
        q_t0_c = np.clip(q_t0, 1e-8, 1-1e-8)
        q_t1_c = np.clip(q_t1, 1e-8, 1-1e-8)
        full_log_q = (1. - t) * logit(q_t0_c) + t * logit(q_t1_c)
        
        return expit(full_log_q + eps * h)

    def fit_rr(self, n_bootstrap=500):
        # 1. Apply Trimming
        q0, q1, g_trim, t_trim, y_trim = self._trim_data(self.q_t0, self.q_t1, self.g, self.t, self.y, self.truncate_level)
        if len(t_trim) < 10: return np.nan, np.nan

        # 2. Define Minimization Routine (matches your fit_rr)
        def solve_eps(curr_q0, curr_q1, curr_g, curr_t, curr_y):
            def loss_fn(eps):
                p_pert = self._perturbed_model(curr_q0, curr_q1, curr_g, curr_t, eps)
                # Minimize Negative Log-Likelihood
                return -np.mean(curr_y * np.log(p_pert + 1e-9) + (1 - curr_y) * np.log(1 - p_pert + 1e-9))
            
            # Find epsilon
            eps_hat = minimize(loss_fn, 0.0, method="Nelder-Mead").x.item()
            
            # Update Q
            qq1 = self._perturbed_model(curr_q0, curr_q1, curr_g, np.ones_like(curr_t), eps_hat)
            qq0 = self._perturbed_model(curr_q0, curr_q1, curr_g, np.zeros_like(curr_t), eps_hat)
            return np.mean(qq1) / np.mean(qq0)

        # 3. Point Estimate
        rr_point = solve_eps(q0, q1, g_trim, t_trim, y_trim)

        # 4. Bootstrap for SE
        n = len(t_trim)
        seeds = np.arange(n_bootstrap) + 12345
        
        def bootstrap_step(seed):
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, n, size=n)
            return solve_eps(q0[idx], q1[idx], g_trim[idx], t_trim[idx], y_trim[idx])

        results = Parallel(n_jobs=-1)(delayed(bootstrap_step)(s) for s in seeds)
        results = np.array([r for r in results if not np.isnan(r) and r > 0])
        
        if len(results) < (n_bootstrap * 0.5): return rr_point, np.nan
        
        # Return Log Scale SE for Rubin's Rules
        log_results = np.log(results)
        se_log_rr = np.std(log_results, ddof=1)
        
        return rr_point, se_log_rr

# =============================================================================
# 4. DATA PROCESSING
# =============================================================================

def load_and_preprocess(file_path, treatment_col, outcome_col):
    cols = (COVARIATES_CONFIG["core"] + COVARIATES_CONFIG["dates"] + 
            COVARIATES_CONFIG["categorical"] + COVARIATES_CONFIG["numeric"] + 
            COVARIATES_CONFIG["history_flags"] + COVARIATES_CONFIG["meds"] + 
            [treatment_col, outcome_col])
    
    df = pd.read_parquet(file_path, columns=list(set(cols)))
    
    for col in COVARIATES_CONFIG["dates"] + ["trial_start"]:
        if col in df.columns: df[col] = pd.to_datetime(df[col], errors='coerce')
        
    df['event_date'] = df['dod']
    df['end_followup'] = df[['end_followup_date_2y', 'enddate', 'event_date']].min(axis=1)
    df['time_to_event'] = (df['end_followup'] - df['trial_start']).dt.days
    df = df[df['time_to_event'] > 0]
    
    df[treatment_col] = pd.to_numeric(df[treatment_col], errors='coerce')
    df[outcome_col] = pd.to_numeric(df[outcome_col], errors='coerce').fillna(0)

    cat_vars = [c for c in COVARIATES_CONFIG["categorical"] if c in df.columns]
    for c in cat_vars: df[c] = df[c].astype(str)
    df = pd.get_dummies(df, columns=cat_vars, drop_first=True, dummy_na=False)
    
    return df.dropna()

def get_covariates(df, treatment_col, outcome_col):
    exclude = [treatment_col, outcome_col, 'patid', 'trial_start', 'dod', 'enddate', 
               'end_followup_date_2y', 'event_date', 'end_followup', 'time_to_event']
    return [c for c in df.columns if c not in exclude]

# =============================================================================
# 5. ANALYSIS RUNNERS
# =============================================================================

def run_psm(df, treatment_col, outcome_col, feats, caliper):
    # Scale Features
    X = StandardScaler().fit_transform(df[feats])
    ps_model = LogisticRegression(max_iter=2000, solver='lbfgs').fit(X, df[treatment_col])
    df['propensity_score'] = ps_model.predict_proba(X)[:, 1]
    
    t, c = df[df[treatment_col]==1], df[df[treatment_col]==0]
    if len(c) < 1 or len(t) < 1: return None
    
    nn = NearestNeighbors(n_neighbors=1).fit(c[['propensity_score']])
    dists, idxs = nn.kneighbors(t[['propensity_score']])
    match_mask = dists.flatten() <= caliper
    matched = pd.concat([t[match_mask], c.iloc[idxs.flatten()[match_mask]]])
    
    try:
        cph = CoxPHFitter()
        cph.fit(matched[[treatment_col, 'time_to_event', outcome_col]], 'time_to_event', outcome_col, robust=True)
        return cph.summary.loc[treatment_col, 'coef'], cph.summary.loc[treatment_col, 'se(coef)']
    except: return None

def run_iptw(df, treatment_col, outcome_col, feats, thresholds):
    X = StandardScaler().fit_transform(df[feats])
    ps_model = LogisticRegression(max_iter=2000, solver='lbfgs').fit(X, df[treatment_col])
    df['propensity_score'] = ps_model.predict_proba(X)[:, 1]
    
    p_t = df[treatment_col].mean()
    df['w'] = np.where(df[treatment_col]==1, p_t/df['propensity_score'], (1-p_t)/(1-df['propensity_score']))
    results = []
    
    for trim in thresholds:
        lt = df[df[treatment_col]==1]['propensity_score'].quantile(trim)
        uc = df[df[treatment_col]==0]['propensity_score'].quantile(1-trim)
        sub = df[((df[treatment_col]==1) & (df['propensity_score'] >= lt)) | 
                 ((df[treatment_col]==0) & (df['propensity_score'] <= uc))]
        try:
            cph = CoxPHFitter()
            cph.fit(sub[[treatment_col, 'time_to_event', outcome_col, 'w']], 'time_to_event', outcome_col, weights_col='w', robust=True)
            results.append({'trim': trim, 'coef': cph.summary.loc[treatment_col, 'coef'], 'se': cph.summary.loc[treatment_col, 'se(coef)']})
        except: pass
    return results

def run_cv_tmle(df, treatment_col, outcome_col, feats, thresholds, n_folds=5):
    """
    Runs 5-Fold CV-TMLE generating predictions using 'T-Learner' approach 
    (Separate outcome models for T=0 and T=1).
    """
    y = df[outcome_col].values
    t = df[treatment_col].values
    X = df[feats].values
    
    # Initialize arrays for cross-validated predictions (stacked)
    n = len(df)
    g_cv = np.zeros(n)
    q0_cv = np.zeros(n)
    q1_cv = np.zeros(n)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1234)
    
    # Cross-Validation Loop
    for train_idx, test_idx in skf.split(X, t):
        X_train, X_test = X[train_idx], X[test_idx]
        t_train, _      = t[train_idx], t[test_idx]
        y_train, _      = y[train_idx], y[test_idx]
        
        # Scale (fit on train, transform test)
        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # 1. Fit Propensity Score (g)
        g_model = LogisticRegression(max_iter=1000, solver='lbfgs').fit(X_train_s, t_train)
        g_cv[test_idx] = g_model.predict_proba(X_test_s)[:, 1]
        
        # 2. Fit Outcome Model T=0 (on control subset of training)
        mask_t0 = (t_train == 0)
        q0_model = LogisticRegression(max_iter=1000, solver='lbfgs')
        q0_model.fit(X_train_s[mask_t0], y_train[mask_t0])
        q0_cv[test_idx] = q0_model.predict_proba(X_test_s)[:, 1]
        
        # 3. Fit Outcome Model T=1 (on treated subset of training)
        mask_t1 = (t_train == 1)
        q1_model = LogisticRegression(max_iter=1000, solver='lbfgs')
        q1_model.fit(X_train_s[mask_t1], y_train[mask_t1])
        q1_cv[test_idx] = q1_model.predict_proba(X_test_s)[:, 1]

    # Run Targeting Step on Full Stacked Predictions
    results = []
    for trim in thresholds:
        tmle = CVTMLE_Sturmer(q0_cv, q1_cv, g_cv, t, y, truncate_level=trim)
        rr, se_log = tmle.fit_rr(n_bootstrap=200) 
        if not np.isnan(rr) and rr > 0:
            results.append({'trim': trim, 'log_rr': np.log(rr), 'se_log_rr': se_log})
            
    return results

# =============================================================================
# 6. MAIN PIPELINE
# =============================================================================

def main():
    print("--- Starting Causal Inference Pipeline (Multiple Imputation + CV-TMLE) ---")
    
    imputed_files = glob.glob(os.path.join(DATA_PATHS["imputed_data_dir"], "*.parquet"))
    if not imputed_files: raise FileNotFoundError("No imputed parquet files found.")
    
    drug, outcome = STUDY_PARAMS["treatment"], STUDY_PARAMS["outcome"]
    
    psm_estimates, psm_ses = [], []
    iptw_storage = {t: {'coefs': [], 'ses': []} for t in STUDY_PARAMS['trimming_thresholds']}
    tmle_storage = {t: {'log_rrs': [], 'ses': []} for t in STUDY_PARAMS['trimming_thresholds']}
    
    for i, fpath in enumerate(imputed_files):
        print(f"Processing Imputation {i+1}/{len(imputed_files)}: {os.path.basename(fpath)}")
        df = load_and_preprocess(fpath, drug, outcome)
        feats = get_covariates(df, drug, outcome)
        
        # 1. PSM
        psm_res = run_psm(df, drug, outcome, feats, STUDY_PARAMS['caliper'])
        if psm_res:
            psm_estimates.append(psm_res[0])
            psm_ses.append(psm_res[1])
            
        # 2. IPTW
        iptw_res = run_iptw(df, drug, outcome, feats, STUDY_PARAMS['trimming_thresholds'])
        for res in iptw_res:
            iptw_storage[res['trim']]['coefs'].append(res['coef'])
            iptw_storage[res['trim']]['ses'].append(res['se'])
            
        # 3. CV-TMLE
        tmle_res = run_cv_tmle(df, drug, outcome, feats, STUDY_PARAMS['trimming_thresholds'], n_folds=STUDY_PARAMS['n_folds'])
        for res in tmle_res:
            tmle_storage[res['trim']]['log_rrs'].append(res['log_rr'])
            tmle_storage[res['trim']]['ses'].append(res['se_log_rr'])

    print("\n--- POOLED RESULTS (Rubin's Rules) ---")
    
    hr, lb, ub = RubinsRules.pool_estimates(psm_estimates, psm_ses)
    print(f"\nPSM (Caliper {STUDY_PARAMS['caliper']}): HR={hr:.3f} [{lb:.3f}, {ub:.3f}]")
    
    print("\nStabilized IPTW Results:")
    print(f"{'Trim':<10} {'HR':<10} {'95% CI':<20}")
    for trim in STUDY_PARAMS['trimming_thresholds']:
        coefs, ses = iptw_storage[trim]['coefs'], iptw_storage[trim]['ses']
        if coefs:
            hr, lb, ub = RubinsRules.pool_estimates(coefs, ses)
            print(f"{trim:<10} {hr:.3f}      [{lb:.3f}, {ub:.3f}]")
            
    print("\nCV-TMLE Results (Risk Ratio):")
    print(f"{'Trim':<10} {'RR':<10} {'95% CI':<20}")
    for trim in STUDY_PARAMS['trimming_thresholds']:
        log_rrs, ses = tmle_storage[trim]['log_rrs'], tmle_storage[trim]['ses']
        if log_rrs:
            rr, lb, ub = RubinsRules.pool_estimates(log_rrs, ses)
            print(f"{trim:<10} {rr:.3f}      [{lb:.3f}, {ub:.3f}]")

if __name__ == "__main__":
    main()