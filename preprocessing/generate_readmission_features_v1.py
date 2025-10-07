"""
Generate enhanced MIMIC-IV readmission feature dataset (v1).

Usage:
------
you can pass arg: --window_days 30 or 60 to generate 30-day or 60-day readmission dataset.
30-days means using readmission within 30 days as label.
60-days means using readmission within 60 days as label.

Outputs:
--------
The resulting CSV will be saved to the top-level 'datasets/' directory:
    datasets/readmission_features_30d_v1.csv
    datasets/readmission_features_60d_v1.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# -----------------------------
# Helpers
# -----------------------------
def _to_datetime(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def _safe_col(df, name, default=np.nan):
    if name not in df.columns:
        df[name] = default
    return df

def _agg_labs(labs_df, d_lab, wanted_labels):
    # Find target lab itemid from d_labitems
    sub = d_lab[d_lab["label"].isin(wanted_labels)][["itemid", "label"]].drop_duplicates()
    labs = labs_df.merge(sub, on="itemid", how="inner")

    # Keep only necessary columns to save memory
    labs = labs[["hadm_id", "label", "valuenum"]].dropna(subset=["valuenum"])

    # Aggregate min/median/max for each hadm_id + lab
    agg = labs.groupby(["hadm_id", "label"])["valuenum"].agg(["min", "median", "max"]).reset_index()
    # Flatten columns: HGB_min, HGB_median, HGB_max ...
    agg = agg.pivot(index="hadm_id", columns="label")
    agg.columns = [f"{lab}_{stat}".replace(" ", "").replace(",", "") for stat, lab in agg.columns]
    agg = agg.reset_index()
    return agg

# -----------------------------
# Main
# -----------------------------
def generate_readmission_dataset(data_root: str, window_days: int = 30):
    """
    Generate MIMIC-IV readmission features (30/60-day window),
    output saved to top-level 'datasets/' with suffix '_v1'.
    Input directory: datasets/mimic-iv-3.1/ (must contain hosp/)
    """
    data_root = Path(data_root)
    hosp_path = data_root / "hosp"

    # Save output to top-level datasets/
    project_root = data_root.parent  # i.e., datasets/
    output_path = project_root
    output_path.mkdir(exist_ok=True)

    print("🩺 [1/7] Loading base tables ...")
    patients = pd.read_csv(hosp_path / "patients.csv")
    admissions = pd.read_csv(hosp_path / "admissions.csv")
    diagnoses = pd.read_csv(hosp_path / "diagnoses_icd.csv", usecols=["subject_id", "hadm_id", "icd_code"])
    transfers = pd.read_csv(hosp_path / "transfers.csv", usecols=["hadm_id", "careunit"])
    services = pd.read_csv(hosp_path / "services.csv", usecols=["hadm_id", "curr_service"]) if (hosp_path / "services.csv").exists() else pd.DataFrame(columns=["hadm_id","curr_service"])
    omr = pd.read_csv(hosp_path / "omr.csv") if (hosp_path / "omr.csv").exists() else pd.DataFrame(columns=["subject_id","chartdate","result_name","result_value"])
    d_labitems = pd.read_csv(hosp_path / "d_labitems.csv", usecols=["itemid", "label"]) if (hosp_path / "d_labitems.csv").exists() else pd.DataFrame(columns=["itemid","label"])
    # labevents is large, read only necessary columns
    labevents_path = hosp_path / "labevents.csv"
    has_labs = labevents_path.exists() and len(d_labitems) > 0

    # Convert time columns
    admissions = _to_datetime(admissions, ["admittime", "dischtime", "deathtime", "edregtime", "edouttime"])
    patients = _to_datetime(patients, ["dod"])
    omr = _to_datetime(omr, ["chartdate"])

    # Ensure missing columns exist
    for c in ["language","ethnicity","marital_status","insurance","admission_type","admission_location","discharge_location"]:
        admissions = _safe_col(admissions, c, default=np.nan)

    print("✅ Base tables loaded successfully.")

    # -----------------------------
    # [2] Build readmission label & visit history
    # -----------------------------
    print(f"⚙️ [2/7] Computing {window_days}-day readmission label & history ...")

    admissions = admissions.sort_values(["subject_id", "admittime"])
    admissions["next_admittime"] = admissions.groupby("subject_id")["admittime"].shift(-1)
    admissions["prev_dischtime"] = admissions.groupby("subject_id")["dischtime"].shift(1)

    admissions["days_to_next"] = (admissions["next_admittime"] - admissions["dischtime"]).dt.days
    admissions["days_since_prev_discharge"] = (admissions["admittime"] - admissions["prev_dischtime"]).dt.days

    admissions["readmit_label"] = (
        (admissions["days_to_next"] >= 0) &
        (admissions["days_to_next"] <= window_days)
    ).astype(int)

    admissions["has_prior_admission"] = admissions["prev_dischtime"].notna().astype(int)

    # ED visit flag and ED length of stay
    ed_present = admissions["edregtime"].notna() | admissions["edouttime"].notna()
    admissions["ed_visit_flag"] = ed_present.astype(int)
    admissions["ed_los_hours"] = ((admissions["edouttime"] - admissions["edregtime"]).dt.total_seconds() / 3600.0).fillna(0)

    # Hospital length of stay (days)
    admissions["length_of_stay"] = (admissions["dischtime"] - admissions["admittime"]).dt.total_seconds() / (3600 * 24)

    # -----------------------------
    # [3] Mortality-related features
    # -----------------------------
    print("⚙️ [3/7] Generating mortality-related features ...")
    base = admissions.merge(
        patients[["subject_id", "gender", "anchor_age", "dod"]],
        on="subject_id",
        how="left"
    )

    base["died_in_hospital"] = base["deathtime"].notna().astype(int)
    # Death within window_days after discharge
    base["death_within_window"] = (
        (base["dod"].notna()) &
        ((base["dod"] - base["dischtime"]).dt.days >= 0) &
        ((base["dod"] - base["dischtime"]).dt.days <= window_days)
    ).astype(int)

    # -----------------------------
    # [4] Diagnosis complexity proxy
    # -----------------------------
    print("⚙️ [4/7] Counting diagnosis codes ...")
    diag_counts = diagnoses.groupby("hadm_id")["icd_code"].nunique().reset_index().rename(columns={"icd_code": "num_diagnoses"})
    base = base.merge(diag_counts, on="hadm_id", how="left")
    base["num_diagnoses"] = base["num_diagnoses"].fillna(0)

    # -----------------------------
    # [5] Transfer and service-related features
    # -----------------------------
    print("⚙️ [5/7] Extracting transfer & service features ...")
    if len(transfers) > 0:
        transfers["careunit"] = transfers["careunit"].astype(str)
        t_agg = transfers.groupby("hadm_id").agg(
            num_transfers=("careunit", "size"),
            unique_careunits=("careunit", pd.Series.nunique),
            had_icu_transfer_flag=("careunit", lambda s: int(any("ICU" in x.upper() for x in s if isinstance(x, str))))
        ).reset_index()
        base = base.merge(t_agg, on="hadm_id", how="left")
    else:
        base["num_transfers"] = 0
        base["unique_careunits"] = 0
        base["had_icu_transfer_flag"] = 0

    if len(services) > 0:
        # Use the last recorded service as the primary discharge service
        srv = services.groupby("hadm_id")["curr_service"].last().reset_index().rename(columns={"curr_service":"last_service"})
        srv["last_service"] = srv["last_service"].astype(str)
        srv["is_surgical_service"] = srv["last_service"].str.upper().str.contains("SURG").astype(int)
        base = base.merge(srv, on="hadm_id", how="left")
    else:
        base["last_service"] = np.nan
        base["is_surgical_service"] = 0

    # -----------------------------
    # [6] OMR baseline features (latest record)
    # -----------------------------
    print("⚙️ [6/7] Aggregating OMR features ...")
    if len(omr) > 0:
        omr = omr.rename(columns={"result_name":"name","result_value":"value"})
        # Keep BMI/eGFR and select last observation <= discharge time
        omr_sub = omr[omr["name"].isin(["Body Mass Index","Estimated Glomerular Filtration Rate"])]
        sub = base[["subject_id","hadm_id","dischtime"]]
        omr_join = omr_sub.merge(sub, on="subject_id", how="inner")
        omr_join = omr_join[ (omr_join["chartdate"] <= omr_join["dischtime"]) | omr_join["chartdate"].isna() ]

        # Take last record per hadm_id + metric
        omr_join = omr_join.sort_values(["hadm_id","name","chartdate"])
        omr_last = omr_join.groupby(["hadm_id","name"]).tail(1)
        omr_pvt = omr_last.pivot(index="hadm_id", columns="name", values="value").reset_index()
        omr_pvt = omr_pvt.rename(columns={
            "Body Mass Index":"bmi_last",
            "Estimated Glomerular Filtration Rate":"egfr_last"
        })
        base = base.merge(omr_pvt, on="hadm_id", how="left")
    else:
        base["bmi_last"] = np.nan
        base["egfr_last"] = np.nan

    # -----------------------------
    # [7] Aggregate key lab tests during stay
    # -----------------------------
    print("🧪 [7/7] Aggregating key labs (HGB/WBC/PLT/Na/K/Cr/BUN/Glucose min/median/max) ...")
    if has_labs:
        # Read only necessary columns, filter by hadm_id (chunking optional)
        needed_hadm = set(base["hadm_id"].unique().tolist())
        labs = pd.read_csv(labevents_path, usecols=["hadm_id","itemid","valuenum"])
        labs = labs[labs["hadm_id"].isin(needed_hadm)]

        wanted = ["Hemoglobin", "WBC", "Platelet Count", "Sodium", "Potassium", "Creatinine", "Urea Nitrogen", "Glucose"]
        lab_agg = _agg_labs(labs, d_labitems, wanted)
        base = base.merge(lab_agg, on="hadm_id", how="left")
    else:
        # Skip if no lab table/dictionary
        pass

    # -----------------------------
    # Final cleanup & export
    # -----------------------------
    keep_cols = [
        # Keys & label
        "subject_id","hadm_id","admittime","dischtime","readmit_label",
        # Utilization
        "days_since_prev_discharge","has_prior_admission","ed_visit_flag","ed_los_hours",
        # Stay characteristics
        "length_of_stay","num_transfers","unique_careunits","had_icu_transfer_flag",
        # Services
        "last_service","is_surgical_service",
        # Demographics
        "gender","anchor_age","language","ethnicity","marital_status","insurance",
        "admission_type","admission_location","discharge_location",
        # Diagnosis complexity
        "num_diagnoses",
        # Competing risk
        "died_in_hospital","death_within_window",
        # OMR
        "bmi_last","egfr_last"
    ] + [c for c in base.columns if any(c.startswith(x) for x in [
        "Hemoglobin_", "WBC_", "PlateletCount_", "Sodium_", "Potassium_", "Creatinine_", "UreaNitrogen_", "Glucose_"
    ])]

    # Keep intersection to ensure robustness
    keep_cols = [c for c in keep_cols if c in base.columns]
    base = base[keep_cols].drop_duplicates(subset=["hadm_id"]).reset_index(drop=True)

    out_file = output_path / f"readmission_features_{window_days}d_v1.csv"
    base.to_csv(out_file, index=False)
    print(f"✅ File generated: {out_file} (total {len(base):,} records)")
    return base


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate enhanced MIMIC-IV readmission dataset")
    parser.add_argument("--data_root", type=str, default="datasets/mimic-iv-3.1", help="MIMIC-IV root directory (must contain hosp/)")
    parser.add_argument("--window_days", type=int, default=30, help="Readmission window (30 or 60)")
    args = parser.parse_args()
    generate_readmission_dataset(args.data_root, args.window_days)
