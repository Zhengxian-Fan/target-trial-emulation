"""
Target Trial Emulation Builder (PySpark)

Description:
    This script constructs a sequence of target trials from observational healthcare data 
    (e.g., CPRD). It processes demographics, measurements, and medication history to 
    create monthly cohorts, applying eligibility criteria, washout periods, and 
    covariate extraction for causal inference analysis.

Usage:
    Ensure Spark is configured and paths in CONFIG are updated before running.
"""

import os
import sys
from functools import reduce
from typing import List, Dict

import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame, Window

# ---------------------------------------------------------
# 1. CONFIGURATION & SETUP
# ---------------------------------------------------------
CONFIG = {
    "paths": {
        # Output directory (Use ./ for relative path to avoid permission issues)
        "cohort_output": "./target_trial_data/",
    },
    "params": {
        "start_year": 2010,
        "num_months": 60,
        "lookback_months_meds": 12,     # Baseline medication usage lookback
        "lookback_months_meas": 24,     # Baseline measurement lookback
        "washout_months": 12,           # Washout period for new user design
        "age_min": 35,
        "age_max": 85,
        "min_years_registration": 2
    }
}

# List of conditions to extract as binary history covariates
CONDITIONS_LIST = [
    'hypertension', 'diabetes', 'ischaemic heart disease', 'atrial fibrillation',
    'stroke', 'dyslipidaemia', 'asthma', 'peripheral arterial disease',
    'valve disorders', 'infective endocarditis', 'thyroid disorders',
    'chronic kidney disease', 'pulmonary embolism', 'malignant cancer',
    'chronic obstructive pulmonary disease'
]

# Mapping: Standardized Name -> Raw Column Name
MEASUREMENTS_MAP = {
    "bmi": "bmi",
    "systolic": "sbp",
    "diastolic": "dbp",
    "creatinine": "creatinine",
    "tchdl_rat": "tchdl_rat",
    "smoke": "smoking_status"
}

# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------

def add_disease_history_flags(df: DataFrame, conditions: List[str], baseline_col: str = "enrollment_end") -> DataFrame:
    """
    Converts *_date columns into binary history flags (1 if date <= baseline, else 0).
    """
    history_exprs = []
    cols_to_drop = []

    for cond in conditions:
        col_base = cond.lower().replace(" ", "_")
        date_col = f"{col_base}_date"
        hist_col = f"{col_base}_history"
        
        if date_col in df.columns:
            # Logic: If event happened on or before enrollment end -> 1, else 0
            expr = F.when(F.col(date_col) <= F.col(baseline_col), 1).otherwise(0).alias(hist_col)
            history_exprs.append(expr)
            cols_to_drop.append(date_col)

    if history_exprs:
        df = df.select("*", *history_exprs).drop(*cols_to_drop)
    
    return df

def get_last_measurements(
    df_cohort: DataFrame,
    unified_meas_df: DataFrame,
    baseline_col: str = "enrollment_end",
    months_lookback: int = 24
) -> DataFrame:
    """
    Retrieves the most recent measurement value within a lookback window.
    """
    joined = df_cohort.select("patid", baseline_col).join(unified_meas_df, on="patid", how="left")

    window_start = F.expr(f"add_months({baseline_col}, -{months_lookback})")
    
    joined = joined.filter(
        (F.col("eventdate") >= window_start) & 
        (F.col("eventdate") < F.col(baseline_col))
    )

    window_spec = Window.partitionBy("patid", "measure_name").orderBy(F.col("eventdate").desc())
    
    pivoted = (
        joined
        .withColumn("rn", F.row_number().over(window_spec))
        .filter(F.col("rn") == 1)
        .groupBy("patid")
        .pivot("measure_name")
        .agg(F.first("measure_value"))
    )
    
    return pivoted

# ---------------------------------------------------------
# 3. TRIAL BUILDER CLASS
# ---------------------------------------------------------

class TargetTrialBuilder:
    def __init__(self, spark: SparkSession, config: Dict):
        self.spark = spark
        self.cfg = config
        self.save_path = config['paths']['cohort_output']
        
    def add_medication_usage_flags(
        self, 
        df_cohort: DataFrame, 
        med_df: DataFrame, 
        baseline_col: str = "enrollment_end"
    ) -> DataFrame:
        """
        Flags medication usage in the lookback period before baseline.
        """
        lookback = self.cfg['params']['lookback_months_meds']
        
        df_cohort = df_cohort.withColumn(
            "lookback_start", F.expr(f"add_months({baseline_col}, -{lookback})")
        )

        med_usage = (
            df_cohort.select("patid", "lookback_start", baseline_col)
            .join(med_df, on="patid", how="left")
            .filter(
                (F.col("eventdate") >= F.col("lookback_start")) &
                (F.col("eventdate") < F.col(baseline_col))
            )
            .select("patid", "med_name").distinct()
        )

        med_names_row = med_usage.select("med_name").distinct().collect()
        med_names = [row.med_name for row in med_names_row]
        
        if not med_names:
            return df_cohort.drop("lookback_start")

        med_flags = (
            med_usage.groupBy("patid")
            .pivot("med_name", med_names)
            .agg(F.lit(1))
        )
        
        df_cohort = df_cohort.join(med_flags, on="patid", how="left")
        df_cohort = df_cohort.fillna({m: 0 for m in med_names}).drop("lookback_start")
        
        return df_cohort

    def build_trial(
        self,
        demographics: DataFrame,
        measurements: DataFrame,
        treatment_df: DataFrame,
        all_meds_df: DataFrame,
        conditions: List[str],
        hf_type_filter: str = "HFrEF",
        drug_name_label: str = "BB"
    ):
        start_year = self.cfg['params']['start_year']
        num_months = self.cfg['params']['num_months']
        
        trial_starts = [
            F.expr(f"add_months(to_date('{start_year}-01-01'), {i})") 
            for i in range(num_months)
        ]

        # Use demographics base (contains 'dod')
        demo_base = demographics

        # Define the specific folder for this trial run
        trial_folder_name = f"trial_{drug_name_label}_{hf_type_filter}_{start_year}_{num_months}"
        trial_output_dir = os.path.join(self.save_path, trial_folder_name)

        # ---------------------------------------------------------
        # 1. Generate Monthly Cohorts
        # ---------------------------------------------------------
        for i, month_expr in enumerate(trial_starts):
            print(f"Building Trial Month {i}...")

            # --- A. Define Baseline & Eligibility ---
            df = (
                demo_base
                .withColumn("trial_start", month_expr)
                .withColumn("enrollment_end", F.expr("add_months(trial_start, 1)"))
                .withColumn("age_at_trial", F.year("trial_start") - F.year("dob"))
                .withColumn("years_since_reg", F.year("trial_start") - F.year("startdate"))
                .withColumn("months_since_hf", F.months_between("trial_start", "HF_date"))
                .withColumn("end_followup_2y", F.expr("add_months(trial_start, 24)"))
                .withColumn("end_followup_5y", F.expr("add_months(trial_start, 60)"))
            )

            if hf_type_filter != "all":
                df = df.filter(F.col("hf_type") == hf_type_filter)

            df = df.filter(
                (F.col("age_at_trial").between(self.cfg['params']['age_min'], self.cfg['params']['age_max'])) &
                (F.col("years_since_reg") >= self.cfg['params']['min_years_registration']) &
                (F.col("months_since_hf") >= 0) &
                (F.col("enddate") > F.col("trial_start"))
            )

            # --- B. Washout (New User Design) ---
            washout = self.cfg['params']['washout_months']
            df = df.withColumn("washout_start", F.expr(f"add_months(trial_start, -{washout})"))
            
            df = df.join(
                treatment_df,
                on=(
                    (df.patid == treatment_df.patid) &
                    (treatment_df.eventdate >= df.washout_start) &
                    (treatment_df.eventdate < df.trial_start)
                ),
                how="left_anti"
            ).drop("washout_start")

            # --- C. Add Covariates (History & Measures) ---
            df = add_disease_history_flags(df, conditions, baseline_col="enrollment_end")

            meas_pivot = get_last_measurements(
                df, measurements, baseline_col="enrollment_end", 
                months_lookback=self.cfg['params']['lookback_months_meas']
            )
            df = df.join(meas_pivot, "patid", "left")

            df = self.add_medication_usage_flags(df, all_meds_df, baseline_col="enrollment_end")

            # --- D. Assign Treatment Arm (ITT) ---
            treated = df.join(
                treatment_df,
                on=(
                    (df.patid == treatment_df.patid) &
                    (treatment_df.eventdate >= df.trial_start) &
                    (treatment_df.eventdate < df.enrollment_end)
                ),
                how="left_semi"
            ).withColumn("treatment", F.lit(1))

            controls = df.join(
                treated.select("patid"), on="patid", how="left_anti"
            ).withColumn("treatment", F.lit(0))

            df_final = treated.unionByName(controls)

            # --- E. Define Outcomes ---
            df_final = df_final.withColumn(
                "all_cause_mortality_2y", 
                F.when(F.col("dod") <= F.col("end_followup_2y"), 1).otherwise(0)
            ).withColumn(
                "all_cause_mortality_5y", 
                F.when(F.col("dod") <= F.col("end_followup_5y"), 1).otherwise(0)
            )

            # --- F. Save Individual Month ---
            output_file = os.path.join(trial_output_dir, f"month_{i}.parquet")
            df_final.write.mode("overwrite").parquet(output_file)
            
        print("All monthly cohorts generated successfully.")

        # ---------------------------------------------------------
        # 2. Merge and Save Consolidated File
        # ---------------------------------------------------------
        print("Merging monthly files into 'all.parquet'...")
        
        # Read back all files matching the pattern "month_*.parquet" within the folder
        # This wildcard prevents reading the 'all.parquet' itself if re-running
        all_months_df = self.spark.read.parquet(os.path.join(trial_output_dir, "month_*.parquet"))
        
        # Save consolidated file
        final_path = os.path.join(trial_output_dir, "all.parquet")
        all_months_df.write.mode("overwrite").parquet(final_path)
        
        print(f"Consolidated data saved to: {final_path}")
        
# ---------------------------------------------------------
# 4. MAIN EXECUTION
# ---------------------------------------------------------

def main():
    spark = SparkSession.builder.appName("CPRD_Trial_Emulation").getOrCreate()
    base_path = CONFIG['paths']['cohort_output']
    
    # 1. Demographics (Contains 'dod' for outcomes)
    patient_df = spark.read.parquet(os.path.join(base_path, 'patient_step1.parquet'))
    
    # 2. Measurements
    meas_dfs = []
    for name, col_map in MEASUREMENTS_MAP.items():
        path = os.path.join(base_path, f"{name}.parquet")
        tmp = spark.read.parquet(path).select(
            "patid", "eventdate", F.col(col_map).alias("measure_value")
        ).withColumn("measure_name", F.lit(name))
        meas_dfs.append(tmp)
    
    unified_measurements = reduce(DataFrame.unionByName, meas_dfs).cache()

    # 3. Medications
    med_list = ["BB", "ACE", "ARB", "Diuretics", "CCB", "statin", "digoxin", "anticoagulant", "antiplatelet"]
    med_dfs = []
    for m in med_list:
        path = os.path.join(base_path, f"{m}.parquet")
        tmp = spark.read.parquet(path).select("patid", "eventdate").withColumn("med_name", F.lit(m))
        med_dfs.append(tmp)
    
    unified_meds = reduce(DataFrame.unionByName, med_dfs).cache()

    # 4. Treatment Data
    treatment_df = spark.read.parquet(os.path.join(base_path, "BB.parquet")).cache()

    # --- Run Builder ---
    builder = TargetTrialBuilder(spark, CONFIG)
    
    builder.build_trial(
        demographics=patient_df,
        measurements=unified_measurements,
        treatment_df=treatment_df,
        all_meds_df=unified_meds,
        conditions=CONDITIONS_LIST,
        hf_type_filter="HFrEF",
        drug_name_label="BB",
    )

if __name__ == "__main__":
    main()