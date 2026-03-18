#!/usr/bin/env python3
"""
Prepare MIMIC-IV data for AKI disease progression modeling.
All features are sourced from the pipeline's ICU preprocessed outputs.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ============================================================
# Constants
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data"

# TODO: clinical evaluation

BIN_HOURS = 6 # 6 hours per bin
N_DAYS = 7 # 7 days of data
N_TIMESTEPS = N_DAYS * 24 // BIN_HOURS # how many bins
MORTALITY_WINDOW_DAYS = 28 

# ------------------------------------------------------------
# Feature Names
# ------------------------------------------------------------

FEATURE_NAMES = [
    "creatinine",
    "bun",
    "urine_output",
    "potassium",
    "map",
]

# ------------------------------------------------------------
# Item IDs
# ------------------------------------------------------------

# TODO: clinical evaluation

# preproc_chart_icu
CHART_ITEMS = {
    "creatinine": [220615], # Creatinine (serum)
    "bun":        [225624], # BUN
    "potassium":  [227442, 227464], # Potassium (serum), Potassium (whole blood)
    "map":        [220052, 220181, 225312], # ABP mean, NIBP mean, ART BP Mean
}

# preproc_out_icu
URINE_OUTPUT_ITEMS = [
    226559, # Foley
    226560, # Void
    226561, # Condom Cath
    226563, # Suprapubic
    226564, # R Nephrostomy
    226565, # L Nephrostomy
    226566, # Urine and GU Irrigant Out
    226567, # Straight Cath
    226627, # OR Urine
    226631, # PACU Urine
    227489, # GU Irrigant/Urine Volume Out
]

# AKI ICD-10
AKI_ICD10_ROOTS = ["N17"]


# ============================================================
# Helper Functions
# ============================================================

# ------------------------------------------------------------
# Cohort
# ------------------------------------------------------------

def build_aki_cohort() -> pd.DataFrame:
    """
    Build AKI cohort from pipeline outputs.

    Criteria:
      - Adult ICU patients
      - First stay per patient
      - AKI

    Returns:
        pd.DataFrame: cohort dataframe with columns:
            - subject_id: patient ID
            - stay_id: stay ID
            - intime: admission time
            - dod: death time
    """

    print("=" * 60)
    print("STEP 1: Building AKI cohort")
    print("=" * 60)

    cohort = pd.read_csv(
        DATA_PATH / "cohort" / "cohort_icu_mortality_0_.csv.gz",
        compression="gzip",
        parse_dates=["intime", "outtime"],
    )

    cohort["dod"] = pd.to_datetime(cohort["dod"], errors="coerce")

    print(f"  ICU cohort: {len(cohort):,} stays, "
          f"{cohort['subject_id'].nunique():,} patients")

    # AKI diagnoses
    diag = pd.read_csv(
        DATA_PATH / "features" / "preproc_diag_icu.csv.gz",
        compression="gzip",
        usecols=["stay_id", "new_icd_code"],
    )
    aki_mask = diag["new_icd_code"].astype(str).str[:3].isin(AKI_ICD10_ROOTS)
    aki_stay_ids = diag.loc[aki_mask, "stay_id"].unique()
    print(f"  Stays with AKI diagnosis (N17.x): {len(aki_stay_ids):,}")

    cohort = cohort[cohort["stay_id"].isin(aki_stay_ids)].copy()
    print(f"  After AKI filter: {len(cohort):,} stays")

    # first stay per patient
    cohort = cohort.sort_values("intime").groupby("subject_id").first().reset_index()
    print(f"  First stay per patient: {len(cohort):,} patients")

    return cohort[["subject_id", "stay_id", "intime", "dod"]]


# ------------------------------------------------------------
# Feature Extraction
# ------------------------------------------------------------

def _assign_bin(hours: pd.Series) -> pd.Series:
    """
    For a given time series, assign a bin index based on the hour.

    Args:
        hours (pd.Series): time series of hours

    Returns:
        pd.Series: bin index for each hour
    """
    
    return (hours // BIN_HOURS).astype(int)


def extract_chart_features(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Extract chart features from preproc_chart_icu.

    Args:
        cohort (pd.DataFrame): cohort dataframe

    Returns:
        pd.DataFrame: dataframe with columns:
            - stay_id: stay ID
            - feature: feature name
            - bin_idx: bin index
            - value: feature value
    """

    print("\n  Loading preproc_chart_icu.csv.gz...")

    # map item IDs to feature names
    item_to_feat = {}
    all_ids = set()
    for feat, ids in CHART_ITEMS.items():
        for iid in ids:
            item_to_feat[iid] = feat
            all_ids.add(iid)

    chart = pd.read_csv(
        DATA_PATH / "features" / "preproc_chart_icu.csv.gz",
        compression="gzip",
    )

    # filter for cohort stays and item IDs
    cohort_stays = set(cohort["stay_id"])
    chart = chart[chart["stay_id"].isin(cohort_stays) & chart["itemid"].isin(all_ids)] # filter for cohort stays and item IDs
    chart["feature"] = chart["itemid"].map(item_to_feat)

    # filter for hours within the defined window
    chart["event_time_from_admit"] = pd.to_timedelta(chart["event_time_from_admit"])
    chart["hours"] = chart["event_time_from_admit"].dt.total_seconds() / 3600.0
    chart = chart[(chart["hours"] >= 0) & (chart["hours"] < N_DAYS * 24)]

    # assign bin index
    chart["bin_idx"] = _assign_bin(chart["hours"])

    # select columns and rename value column
    result = chart[["stay_id", "feature", "bin_idx", "valuenum"]].rename(
        columns={"valuenum": "value"}
    )
    print(f"         {len(result):,} rows")

    return result


def extract_urine_output(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Extract urine output from preproc_out_icu.

    Args:
        cohort (pd.DataFrame): cohort dataframe

    Returns:
        pd.DataFrame: dataframe with columns:
    """

    print("\n  Loading preproc_out_icu.csv.gz...")

    out = pd.read_csv(
        DATA_PATH / "features" / "preproc_out_icu.csv.gz",
        compression="gzip",
    )

    # filter for cohort stays and item IDs
    cohort_stays = set(cohort["stay_id"])
    out = out[out["stay_id"].isin(cohort_stays) & out["itemid"].isin(URINE_OUTPUT_ITEMS)]
    out = out.dropna(subset=["value"])

    # filter for hours within the defined window
    out["event_time_from_admit"] = pd.to_timedelta(out["event_time_from_admit"])
    out["hours"] = out["event_time_from_admit"].dt.total_seconds() / 3600.0
    out = out[(out["hours"] >= 0) & (out["hours"] < N_DAYS * 24)]

    # assign bin index
    out["bin_idx"] = _assign_bin(out["hours"])

    # set feature name
    out["feature"] = "urine_output"

    result = out[["stay_id", "feature", "bin_idx", "value"]]
    print(f"         {len(result):,} rows")

    return result


# ------------------------------------------------------------
# Data Formatting
# ------------------------------------------------------------

def build_tensor(cohort: pd.DataFrame, events: pd.DataFrame) -> np.ndarray:
    """
    Format events into a (n_patients, N_TIMESTEPS, n_features) array.

    Aggregation per bin:
      - urine_output = sum of values (total mL in the window) for each bin (TODO: clinical evaluation)
      - others = last value (most recent measurement) for each bin (TODO: clinical evaluation)

    Imputation rules:
      - urine_output = 0 for empty bins (TODO: clinical evaluation)
      - others = forward-fill, then population median (TODO: clinical evaluation)

    Standardization rules:
      - zero-mean, unit-variance

    Args:
        cohort (pd.DataFrame): cohort dataframe
        events (pd.DataFrame): events dataframe

    Returns:
        np.ndarray: (n_patients, N_TIMESTEPS, n_features) array/tensor
    """

    print("\n" + "=" * 60)
    print("STEP 3: Binning, imputation, standardization")
    print("=" * 60)

    # get stay IDs and convert to indices mapping
    stay_ids = cohort["stay_id"].values
    sid_to_idx = {s: i for i, s in enumerate(stay_ids)}

    n_patients = len(stay_ids)
    n_features = len(FEATURE_NAMES)

    X = np.full((n_patients, N_TIMESTEPS, n_features), np.nan)

    # ------------------------------------------------------------
    # Binning
    # ------------------------------------------------------------

    print("\n  Binning...")
    for fi, fname in enumerate(FEATURE_NAMES):

        # filter for current feature
        fd = events[events["feature"] == fname]
        
        if fd.empty:
            print(f"  WARNING  no data for '{fname}'")
            continue

        # aggregate by feature
        if fname == "urine_output":
            agg = fd.groupby(["stay_id", "bin_idx"])["value"].sum().reset_index() # sum of values
        else:
            agg = fd.sort_values("bin_idx").groupby(["stay_id", "bin_idx"])["value"].last().reset_index() # last value

        # fill tensor
        for sid, b, v in zip(agg["stay_id"], agg["bin_idx"].astype(int), agg["value"]):
            
            # get patient index
            pi = sid_to_idx.get(sid)
            
            # fill tensor
            if pi is not None and 0 <= b < N_TIMESTEPS:
                X[pi, b, fi] = v 

        # count filled bins
        filled = np.sum(~np.isnan(X[:, :, fi]))
        total = n_patients * N_TIMESTEPS
        print(f"  {fname:15s}  {filled:>9,}/{total:,} bins filled "
              f"({100 * filled / total:.1f}%)")

    # ------------------------------------------------------------
    # Imputation
    # ------------------------------------------------------------

    print("\n  Imputing...")
    for fi, fname in enumerate(FEATURE_NAMES):
        
        if fname == "urine_output":
            np.nan_to_num(X[:, :, fi], copy=False, nan=0.0) # set empty bins to 0
        else:

            # forward-fill
            for p in range(n_patients):
                last = np.nan
                for t in range(N_TIMESTEPS):
                    if np.isnan(X[p, t, fi]): 
                        X[p, t, fi] = last # fill with last value
                    else:
                        last = X[p, t, fi]

            # set median
            med = np.nanmedian(X[:, :, fi]) # population median
            if np.isnan(med):
                med = 0.0 # set to 0 if median is NaN

            # fill with median if NaN
            X[:, :, fi] = np.where(np.isnan(X[:, :, fi]), med, X[:, :, fi])

        print(f"    {fname:15s}  NaNs remaining: {np.isnan(X[:, :, fi]).sum()}")

    # ------------------------------------------------------------
    # Standardization
    # ------------------------------------------------------------

    print("\n  Standardizing...")
    for fi, fname in enumerate(FEATURE_NAMES):

        # get flat array of current feature
        flat = X[:, :, fi].ravel()

        mu, sigma = flat.mean(), flat.std()

        # set to 1 if standard deviation is too small
        if sigma < 1e-8:
            sigma = 1.0

        # standardize
        X[:, :, fi] = (X[:, :, fi] - mu) / sigma

        print(f"    {fname:15s}  μ={mu:.4f}  σ={sigma:.4f}")

    return X


# ---------------------------------------------------------------------------
# Outcome
# ---------------------------------------------------------------------------

def compute_outcomes(cohort: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mortality outcome as per predefined window.

    Args:
        cohort (pd.DataFrame): cohort dataframe

    Returns:
        tuple[np.ndarray, np.ndarray]: duration and event arrays (duration in days, event is 1 if patient died within window, 0 otherwise)
    """

    print("\n" + "=" * 60)
    print("STEP 4  Mortality outcome")
    print("=" * 60)

    n = len(cohort)
    duration = np.full(n, float(MORTALITY_WINDOW_DAYS))
    event = np.zeros(n, dtype=np.float64)

    # compute duration and event
    for i, (_, row) in enumerate(cohort.iterrows()):

        # check if patient died within window
        if pd.notna(row["dod"]):
            days = (row["dod"] - row["intime"]).total_seconds() / 86400.0 # convert to days

            # set event and duration
            if 0 <= days <= MORTALITY_WINDOW_DAYS:
                event[i] = 1.0
                duration[i] = days

    print(f"  Patients:   {n:,}")
    print(f"  Deaths <= {MORTALITY_WINDOW_DAYS}d: {int(event.sum()):,}  ({100 * event.mean():.1f}%)")
    print(f"  Median dur: {np.median(duration):.1f} d")

    return duration, event


# ============================================================
# Pipeline
# ============================================================

def main():
    """
    Main function to prepare AKI data.

    Steps:
      1. build AKI cohort
      2. extract features
      3. build tensor
      4. compute outcomes
      5. save data

    Args:
        --output: output .pt file name
    """

    parser = argparse.ArgumentParser(
        description="Prepare MIMIC-IV AKI data"
    )
    parser.add_argument(
        "--output", default="aki_data.pt", help="Output .pt file (default: aki_data.pt)"
    )
    args = parser.parse_args()

    output_path = Path("exports") / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # STEP 1: build AKI cohort
    cohort = build_aki_cohort()

    # STEP 2: extract features
    print("\n" + "=" * 60)
    print("STEP 2: Extracting features")
    print("=" * 60)

    chart_df = extract_chart_features(cohort)
    urine_df = extract_urine_output(cohort)

    all_events = pd.concat([chart_df, urine_df], ignore_index=True)
    print(f"\n  Total event rows: {len(all_events):,}")

    # STEP 3: build tensor
    X = build_tensor(cohort, all_events)
    n_patients = X.shape[0]
    X_flat = X.reshape(n_patients * N_TIMESTEPS, len(FEATURE_NAMES))

    # STEP 4: compute outcomes
    duration, event = compute_outcomes(cohort)

    # STEP 5: save data
    print("\n" + "=" * 60)
    print("STEP 5: Saving data")
    print("=" * 60)

    # convert to tensor
    X_tensor = torch.FloatTensor(X_flat)

    # check feature names
    assert len(FEATURE_NAMES) == X_tensor.shape[1], (
        f"feature_names length ({len(FEATURE_NAMES)}) != X.shape[1] ({X_tensor.shape[1]})"
    )

    # create data dictionary
    data = {
        "X": X_tensor,
        "feature_names": list(FEATURE_NAMES),
        "n_patients": n_patients,
        "n_timesteps": N_TIMESTEPS,
        "duration": duration,
        "event": event,
    }

    # save data
    torch.save(data, output_path)

    # print data summary
    print(f"\n  → {output_path}")
    print(f"    X.shape       = {data['X'].shape}")
    print(f"    feature_names = {data['feature_names']}")
    print(f"    n_patients    = {data['n_patients']}")
    print(f"    n_timesteps   = {data['n_timesteps']}")
    print(f"    duration.shape= {data['duration'].shape}")
    print(f"    event.shape   = {data['event'].shape}")
    print(f"    mortality rate= {data['event'].mean():.3f}")


if __name__ == "__main__":
    main()
