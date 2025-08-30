
# this script applies different linear correction on the final data reported by quantaq website for o3
# I did not change the name of hdf files
# name of quantaq final data file are : "MOD-*-York.csv" where * is the sensor ID

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
import warnings
from scipy.optimize import OptimizeWarning
from scipy.stats import pearsonr

warnings.simplefilter("ignore", OptimizeWarning)

# === SETTINGS ===
var           = "O3 CONC"
var_q         = "o3"
var_t         = "temp"
var_rh        = "rh"
start_date    = "2024-12-14"
end_date      = "2025-01-14"
time_interval = 2

start_date = pd.to_datetime(start_date).tz_localize("UTC")
end_date   = pd.to_datetime(end_date).tz_localize("UTC") + pd.Timedelta(days=1)


def find_files_in_date_range(base_dirs, start_date, end_date):
    selected = []
    for base in base_dirs:
        for f in glob.glob(os.path.join(base, "**", "*.hdf"), recursive=True):
            try:
                d = pd.to_datetime(os.path.basename(f)[:10], errors='coerce')
                if pd.notna(d):
                    d = d.tz_localize("UTC")
                    if start_date <= d < end_date:
                        selected.append(f)
            except:
                pass
    return sorted(selected)


def read_variable_from_files(files, var, start_date, end_date):
    frames = []
    for f in files:
        try:
            df = pd.read_hdf(f)
            if var in df.columns and "Time_ISO" in df.columns:
                df = df[["Time_ISO", var]].dropna()
                df["Time_ISO"] = pd.to_datetime(df["Time_ISO"])
                if df["Time_ISO"].dt.tz is None:
                    df["Time_ISO"] = df["Time_ISO"].dt.tz_localize("UTC")
                else:
                    df["Time_ISO"] = df["Time_ISO"].dt.tz_convert("UTC")
                df = df[(df[var] > 0) & (df[var] <= 100)]
                df = df[(df["Time_ISO"] >= start_date) & (df["Time_ISO"] < end_date)]
                frames.append(df)
        except:
            pass
    if frames:
        return pd.concat(frames).sort_values("Time_ISO").reset_index(drop=True)
    return pd.DataFrame(columns=["Time_ISO", var])


def remove_outliers_rolling_zscore(df, var, window_size=60, z_thresh=3.0):
    m = df[var].rolling(window=window_size, center=True, min_periods=1).mean()
    s = df[var].rolling(window=window_size, center=True, min_periods=1).std()
    z = (df[var] - m) / s
    return df[z.abs() <= z_thresh].reset_index(drop=True)


def time_average(df, var, interval):
    df = df.copy()
    df["Time_ISO"] = pd.to_datetime(df["Time_ISO"])
    df = df.set_index("Time_ISO")
    return df[[var]].resample(f"{interval}min").mean().reset_index()


# === MODEL FUNCTIONS ===

def Linear_model(x, a, b):
    return a * x + b

def model_T(X, a, b, c):
    x, T = X
    return a * x + b + c * T

def model_T_avg(X, a, b, c):
    x, t = X
    return a * x + b + c * (t - t.mean())

def model_T2(X, a, b, c, d):
    x, T = X
    return a * x + b + c * T + d * (T ** 2)

def model_T2_avg(X, a, b, c, d):
    x, t = X
    t0   = t - t.mean()
    return a * x + b + c * t0 + d * (t0 ** 2)

def model_RH(X, a, b, c):
    x, RH = X
    return a * x + b + c * RH

def model_RH_avg(X, a, b, c):
    x, RH = X
    rh0    = RH - RH.mean()
    return a * x + b + c * rh0

def model_RH2(X, a, b, c, d):
    x, RH = X
    return a * x + b + c * RH + d * (RH ** 2)

def model_RH2_avg(X, a, b, c, d):
    x, RH = X
    rh0    = RH - RH.mean()
    return a * x + b + c * rh0 + d * (rh0 ** 2)

def model_RH_plus_T(X, a, b, c, d):
    x, T, RH = X
    return a * x + b + c * T + d * RH

def model_RH_plus_T_avg(X, a, b, c, d):
    x, t, rh = X
    t0       = t - t.mean()
    rh0      = rh - rh.mean()
    return a * x + b + c * t0 + d * rh0

def model_RH_x_T(X, a, b, c):
    x, T, RH = X
    return a * x + b + c * (T * RH)

def model_RH_x_T_avg(X, a, b, c):
    x, t, rh = X
    t0       = t - t.mean()
    rh0      = rh - rh.mean()
    return a * x + b + c * (t0 * rh0)


def compute_residuals(y_true, y_pred):
    return y_true - y_pred


# === LOAD CALIBRATED DATA ===
base_folders = ["2024", "2025"]
hdf_files    = find_files_in_date_range(base_folders, start_date, end_date)
df_raw       = read_variable_from_files(hdf_files, var, start_date, end_date)
df_clean     = remove_outliers_rolling_zscore(df_raw, var)
df_averaged  = time_average(df_clean, var, time_interval)
df_cal       = df_averaged.copy()


# === LOAD CSV SOURCES  ===
csv_files   = glob.glob("MOD-*-York.csv")
csv_sources = []
for f in csv_files:
    label = os.path.basename(f).replace(".csv", "")
    df    = pd.read_csv(f)
    df["Time_ISO"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[(df["Time_ISO"] >= start_date) & (df["Time_ISO"] < end_date)]
    if var_q in df.columns and var_t in df.columns and var_rh in df.columns:
        df_q  = time_average(df, var_q,  time_interval)
        df_T  = time_average(df, var_t,  time_interval)
        df_rh = time_average(df, var_rh, time_interval)
        df_rh = df_rh.rename(columns={var_rh: "RH"})
        csv_sources.append((label, df_q, df_T, df_rh))


# === BUILD DICTS OF MODEL PARAMETERS ===
popt_T_map          = {}
popt_T_avg_map      = {}
popt_RH_map         = {}
popt_RH_avg_map     = {}
popt_T2_map         = {}
popt_T2_avg_map     = {}
popt_RH2_map        = {}
popt_RH2_avg_map    = {}
popt_RH_plus_T_map  = {}
popt_RH_plus_T_avg_map = {}
popt_RH_x_T_map     = {}
popt_RH_x_T_avg_map = {}
popt_Linear_map     = {}
x_store             = {}
y_store             = {}

# 1) T‐model
for label, df_q, df_T, df_RH in csv_sources:
    m = pd.merge(df_cal, df_q, on="Time_ISO", how="inner").merge(df_T, on="Time_ISO", how="inner")
    x, y, T = m[var].values, m[var_q].values, m[var_t].values
    mask = (~np.isnan(x) & ~np.isnan(y) & ~np.isnan(T) & np.isfinite(x) & np.isfinite(y) & np.isfinite(T))
    x_fit, y_fit, T_fit = x[mask], y[mask], T[mask]
    if len(x_fit) < 2: 
        continue
    popt, _ = curve_fit(model_T, (x_fit, T_fit), y_fit)
    popt_T_map[label] = popt
    # also fit "_avg" version
    popt_avg, _ = curve_fit(model_T_avg, (x_fit, T_fit), y_fit)
    popt_T_avg_map[label] = popt_avg

# 2) RH‐model
for label, df_q, df_T, df_RH in csv_sources:
    m = pd.merge(df_cal, df_q, on="Time_ISO", how="inner").merge(df_RH, on="Time_ISO", how="inner")
    x, y, RH = m[var].values, m[var_q].values, m["RH"].values
    mask = (~np.isnan(x) & ~np.isnan(y) & ~np.isnan(RH) & np.isfinite(x) & np.isfinite(y) & np.isfinite(RH))
    x_fit, y_fit, RH_fit = x[mask], y[mask], RH[mask]
    if len(x_fit) < 2:
        continue
    popt, _ = curve_fit(model_RH, (x_fit, RH_fit), y_fit)
    popt_RH_map[label] = popt
    popt_avg, _ = curve_fit(model_RH_avg, (x_fit, RH_fit), y_fit)
    popt_RH_avg_map[label] = popt_avg

# 3) T²‐model
for label, df_q, df_T, df_RH in csv_sources:
    m = pd.merge(df_cal, df_q, on="Time_ISO", how="inner").merge(df_T, on="Time_ISO", how="inner")
    x, y, T = m[var].values, m[var_q].values, m[var_t].values
    mask = (~np.isnan(x) & ~np.isnan(y) & ~np.isnan(T) & np.isfinite(x) & np.isfinite(y) & np.isfinite(T))
    x_fit, y_fit, T_fit = x[mask], y[mask], T[mask]
    if len(x_fit) < 2:
        continue
    popt, _ = curve_fit(model_T2, (x_fit, T_fit), y_fit)
    popt_T2_map[label] = popt
    popt_avg, _ = curve_fit(model_T2_avg, (x_fit, T_fit), y_fit)
    popt_T2_avg_map[label] = popt_avg

# 4) RH²‐model
for label, df_q, df_T, df_RH in csv_sources:
    m = pd.merge(df_cal, df_q, on="Time_ISO", how="inner").merge(df_RH, on="Time_ISO", how="inner")
    x_vals, y_vals, RH_vals = m[var].values, m[var_q].values, m["RH"].values
    mask = (~np.isnan(x_vals) & ~np.isnan(y_vals) & ~np.isnan(RH_vals) & 
            np.isfinite(x_vals) & np.isfinite(y_vals) & np.isfinite(RH_vals))
    x_fit, y_fit, RH_fit = x_vals[mask], y_vals[mask], RH_vals[mask]
    if len(x_fit) < 2:
        continue
    popt, _ = curve_fit(model_RH2, (x_fit, RH_fit), y_fit)
    popt_RH2_map[label] = popt
    popt_avg, _ = curve_fit(model_RH2_avg, (x_fit, RH_fit), y_fit)
    popt_RH2_avg_map[label] = popt_avg

# 5) RH + T‐model
for label, df_q, df_T, df_RH in csv_sources:
    m = (pd.merge(df_cal, df_q, on="Time_ISO", how="inner")
         .merge(df_T, on="Time_ISO", how="inner")
         .merge(df_RH, on="Time_ISO", how="inner"))
    x_vals, y_vals, T_vals, RH_vals = m[var].values, m[var_q].values, m[var_t].values, m["RH"].values
    mask = (~np.isnan(x_vals) & ~np.isnan(y_vals) & ~np.isnan(T_vals) & ~np.isnan(RH_vals) &
            np.isfinite(x_vals) & np.isfinite(y_vals) & np.isfinite(T_vals) & np.isfinite(RH_vals))
    x_fit, y_fit = x_vals[mask], y_vals[mask]
    T_fit, RH_fit = T_vals[mask], RH_vals[mask]
    if len(x_fit) < 2:
        continue
    popt, _ = curve_fit(model_RH_plus_T, (x_fit, T_fit, RH_fit), y_fit)
    popt_RH_plus_T_map[label] = popt
    popt_avg, _ = curve_fit(model_RH_plus_T_avg, (x_fit, T_fit, RH_fit), y_fit)
    popt_RH_plus_T_avg_map[label] = popt_avg

# 6) RH × T‐model
for label, df_q, df_T, df_RH in csv_sources:
    m = (pd.merge(df_cal, df_q, on="Time_ISO", how="inner")
         .merge(df_T, on="Time_ISO", how="inner")
         .merge(df_RH, on="Time_ISO", how="inner"))
    x_vals, y_vals, T_vals, RH_vals = m[var].values, m[var_q].values, m[var_t].values, m["RH"].values
    mask = (~np.isnan(x_vals) & ~np.isnan(y_vals) & ~np.isnan(T_vals) & ~np.isnan(RH_vals) &
            np.isfinite(x_vals) & np.isfinite(y_vals) & np.isfinite(T_vals) & np.isfinite(RH_vals))
    x_fit, y_fit = x_vals[mask], y_vals[mask]
    T_fit, RH_fit = T_vals[mask], RH_vals[mask]
    if len(x_fit) < 2:
        continue
    popt, _ = curve_fit(model_RH_x_T, (x_fit, T_fit, RH_fit), y_fit)
    popt_RH_x_T_map[label] = popt
    popt_avg, _ = curve_fit(model_RH_x_T_avg, (x_fit, T_fit, RH_fit), y_fit)
    popt_RH_x_T_avg_map[label] = popt_avg

# 7) Simple linear fits
for label, df_q, df_T, df_RH in csv_sources:
    merged = pd.merge(df_averaged[["Time_ISO", var]],
                      df_q[["Time_ISO", var_q]],
                      on="Time_ISO", how="inner")
    x = merged[var].values
    y = merged[var_q].values
    mask = (~np.isnan(x) & ~np.isnan(y) & np.isfinite(x) & np.isfinite(y))
    x_fit, y_fit = x[mask], y[mask]
    if len(x_fit) < 2:
        continue
    popt, _ = curve_fit(Linear_model, x_fit, y_fit)
    popt_Linear_map[label] = tuple(popt)
    x_store[label] = x_fit
    y_store[label] = y_fit


# === CORRECTION FUNCTIONS ===

def corr_Linear(raw_o3: np.ndarray, a: float, b: float) -> np.ndarray:
    return (raw_o3 - b) / a

def corr_T(raw_o3: np.ndarray, T: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return (raw_o3 - b - c * T) / a

def corr_T2(raw_o3: np.ndarray, T: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    return (raw_o3 - b - c * T - d * (T ** 2)) / a

def corr_RH(raw_o3: np.ndarray, RH: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return (raw_o3 - b - c * RH) / a

def corr_RH2(raw_o3: np.ndarray, RH: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    return (raw_o3 - b - c * RH - d * (RH ** 2)) / a

def corr_RH_plus_T(raw_o3: np.ndarray, T: np.ndarray, RH: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    return (raw_o3 - b - c * T - d * RH) / a

def corr_RH_x_T(raw_o3: np.ndarray, T: np.ndarray, RH: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return (raw_o3 - b - c * (T * RH)) / a


def corr_T_avg(raw_o3: np.ndarray, T: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    t0 = T - T.mean()
    return (raw_o3 - b - c * t0) / a

def corr_T2_avg(raw_o3: np.ndarray, T: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    t0 = T - T.mean()
    return (raw_o3 - b - c * t0 - d * (t0 ** 2)) / a

def corr_RH_avg(raw_o3: np.ndarray, RH: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    rh0 = RH - RH.mean()
    return (raw_o3 - b - c * rh0) / a

def corr_RH2_avg(raw_o3: np.ndarray, RH: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    rh0 = RH - RH.mean()
    return (raw_o3 - b - c * rh0 - d * (rh0 ** 2)) / a

def corr_RH_plus_T_avg(raw_o3: np.ndarray, T: np.ndarray, RH: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    t0 = T - T.mean()
    rh0 = RH - RH.mean()
    return (raw_o3 - b - c * t0 - d * rh0) / a

def corr_RH_x_T_avg(raw_o3: np.ndarray, T: np.ndarray, RH: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    t0 = T - T.mean()
    rh0 = RH - RH.mean()
    return (raw_o3 - b - c * (t0 * rh0)) / a


# === METRICS FUNCTION ===

def compute_metrics_training():

    for label, df_q, df_t, df_rh in csv_sources:
        print(f"{label}:")

        # 1) Merge calibrated + raw O₃
        merged_base = (
            df_cal.rename(columns={var: "calibrated_O3"})
                  .merge(df_q.rename(columns={var_q: var_q}), on="Time_ISO", how="inner")
        ).dropna(subset=["calibrated_O3", var_q])
        if len(merged_base) < 2:
            print("  Raw vs Calibrated: not enough points to compute metrics\n")
            continue

        x_cal = merged_base["calibrated_O3"].values
        y_raw = merged_base[var_q].values
        N_base = len(x_cal)

        # 1a) Raw vs Calibrated (no correction)
        r_raw, _   = pearsonr(x_cal, y_raw)
        r2_raw     = r_raw**2
        bias_raw   = np.mean(y_raw - x_cal)
        res_raw    = y_raw - x_cal
        chi2_raw   = np.sum(res_raw**2) / (N_base - 0)
        rmse_raw   = np.sqrt(np.mean(res_raw**2))
        print(
            f"  Raw:          R² = {r2_raw:.6f}, bias = {bias_raw:.4f} ppb,"
            f" χ²_red = {chi2_raw:.6f}, RMSE = {rmse_raw:.4f} ppb"
        )

        def _print_metrics(name, x_corr, x_true, p):
            valid_mask = x_corr > 0
            x_c = x_corr[valid_mask]
            x_t = x_true[valid_mask]
            N   = len(x_c)
            if N < 2:
                print(f"  {name:12s}: not enough positive corrected points for metrics")
                return
            r, _      = pearsonr(x_t, x_c)
            r2        = r**2
            bias      = np.mean(x_c - x_t)
            res       = x_c - x_t
            chi2      = np.sum(res**2) / (N - p)
            rmse      = np.sqrt(np.mean(res**2))
            print(
                f"  {name:12s}: R² = {r2:.4f}, bias = {bias:.4f} ppb, "
                f"χ²_red = {chi2:.4f}, RMSE = {rmse:.4f} ppb"
            )

        # 2) LINEAR (p=2) – ordinary
        if label in popt_Linear_map:
            a, b      = popt_Linear_map[label]
            x_corr_L  = corr_Linear(y_raw, a, b)
            _print_metrics("Linear", x_corr_L, x_cal, p=2)
        else:
            print("  Linear      : parameters missing")

        
        # 3) T-only correction (p=3)
        if label in popt_T_map:
            merged_T   = (
                merged_base
                .merge(df_t.rename(columns={var_t: var_t}), on="Time_ISO", how="inner")
            ).dropna(subset=[var_t])
            if len(merged_T) < 2:
                print("  T           : not enough points to compute metrics")
                print("  T_avg       : not enough points to compute metrics")
            else:
                x_cal_T   = merged_T["calibrated_O3"].values
                y_T_raw   = merged_T[var_q].values
                T_vals    = merged_T[var_t].values

                # — ordinary T
                a, b, c       = popt_T_map[label]
                x_corr_T      = corr_T(y_T_raw, T_vals, a, b, c)
                _print_metrics("T", x_corr_T, x_cal_T, p=3)

                # — T_avg
                a2, b2, c2       = popt_T_avg_map[label]
                x_corr_Ta       = corr_T_avg(y_T_raw, T_vals, a2, b2, c2)
                _print_metrics("T_avg", x_corr_Ta, x_cal_T, p=3)
        else:
            print("  T           : parameters missing")
            print("  T_avg       : parameters missing")

        # 4) RH-only (p=3)
        if label in popt_RH_map:
            merged_RH   = (
                merged_base
                .merge(df_rh.rename(columns={"RH": "RH"}), on="Time_ISO", how="inner")
            ).dropna(subset=["RH"])
            if len(merged_RH) < 2:
                print("  RH          : not enough points to compute metrics")
                print("  RH_avg      : not enough points to compute metrics")
            else:
                x_cal_RH   = merged_RH["calibrated_O3"].values
                y_RH_raw   = merged_RH[var_q].values
                RH_vals    = merged_RH["RH"].values

                # — ordinary RH
                a, b, c       = popt_RH_map[label]
                x_corr_RH     = corr_RH(y_RH_raw, RH_vals, a, b, c)
                _print_metrics("RH", x_corr_RH, x_cal_RH, p=3)

                # — RH_avg
                a2, b2, c2       = popt_RH_avg_map[label]
                x_corr_RHa       = corr_RH_avg(y_RH_raw, RH_vals, a2, b2, c2)
                _print_metrics("RH_avg", x_corr_RHa, x_cal_RH, p=3)
        else:
            print("  RH          : parameters missing")
            print("  RH_avg      : parameters missing")

        # 5) T² (p=4)
        if label in popt_T2_map:
            merged_T2  = (
                merged_base
                .merge(df_t.rename(columns={var_t: var_t}), on="Time_ISO", how="inner")
            ).dropna(subset=[var_t])
            if len(merged_T2) < 2:
                print("  T²          : not enough points to compute metrics")
                print("  T2_avg      : not enough points to compute metrics")
            else:
                x_cal_T2   = merged_T2["calibrated_O3"].values
                y_T2_raw   = merged_T2[var_q].values
                T_vals2    = merged_T2[var_t].values

                # — ordinary T²
                a, b, c, d     = popt_T2_map[label]
                x_corr_T2      = corr_T2(y_T2_raw, T_vals2, a, b, c, d)
                _print_metrics("T²", x_corr_T2, x_cal_T2, p=4)

                # — T2_avg
                a2, b2, c2, d2     = popt_T2_avg_map[label]
                x_corr_T2a         = corr_T2_avg(y_T2_raw, T_vals2, a2, b2, c2, d2)
                _print_metrics("T2_avg", x_corr_T2a, x_cal_T2, p=4)
        else:
            print("  T²          : parameters missing")
            print("  T2_avg      : parameters missing")

        # 6) RH² (p=4)
        if label in popt_RH2_map:
            merged_RH2 = (
                merged_base
                .merge(df_rh.rename(columns={"RH": "RH"}), on="Time_ISO", how="inner")
            ).dropna(subset=["RH"])
            if len(merged_RH2) < 2:
                print("  RH²         : not enough points to compute metrics")
                print("  RH2_avg     : not enough points to compute metrics")
            else:
                x_cal_RH2  = merged_RH2["calibrated_O3"].values
                y_RH2_raw  = merged_RH2[var_q].values
                RH_vals2   = merged_RH2["RH"].values

                # — ordinary RH²
                a, b, c, d     = popt_RH2_map[label]
                x_corr_RH2     = corr_RH2(y_RH2_raw, RH_vals2, a, b, c, d)
                _print_metrics("RH²", x_corr_RH2, x_cal_RH2, p=4)

                # — RH2_avg
                a2, b2, c2, d2 = popt_RH2_avg_map[label]
                x_corr_RH2a    = corr_RH2_avg(y_RH2_raw, RH_vals2, a2, b2, c2, d2)
                _print_metrics("RH2_avg", x_corr_RH2a, x_cal_RH2, p=4)
        else:
            print("  RH²         : parameters missing")
            print("  RH2_avg     : parameters missing")

        # 7) RH + T (p=4)
        if label in popt_RH_plus_T_map:
            merged_add = (
                merged_base
                .merge(df_t.rename(columns={var_t: var_t}), on="Time_ISO", how="inner")
                .merge(df_rh.rename(columns={"RH": "RH"}), on="Time_ISO", how="inner")
            ).dropna(subset=[var_t, "RH"])
            if len(merged_add) < 2:
                print("  RH + T      : not enough points to compute metrics")
                print("  RH+T_avg    : not enough points to compute metrics")
            else:
                x_cal_add  = merged_add["calibrated_O3"].values
                y_add_raw  = merged_add[var_q].values
                T_vals3    = merged_add[var_t].values
                RH_vals3   = merged_add["RH"].values

                # — ordinary RH + T
                a, b, c, d       = popt_RH_plus_T_map[label]
                x_corr_add       = corr_RH_plus_T(y_add_raw, T_vals3, RH_vals3, a, b, c, d)
                _print_metrics("RH + T", x_corr_add, x_cal_add, p=4)

                # — RH+T_avg
                a2, b2, c2, d2   = popt_RH_plus_T_avg_map[label]
                x_corr_add_avg   = corr_RH_plus_T_avg(y_add_raw, T_vals3, RH_vals3, a2, b2, c2, d2)
                _print_metrics("RH+T_avg", x_corr_add_avg, x_cal_add, p=4)
        else:
            print("  RH + T      : parameters missing")
            print("  RH+T_avg    : parameters missing")

        # 8) RH x T (p=3)
        if label in popt_RH_x_T_map:
            merged_mul = (
                merged_base
                .merge(df_t.rename(columns={var_t: var_t}), on="Time_ISO", how="inner")
                .merge(df_rh.rename(columns={"RH": "RH"}), on="Time_ISO", how="inner")
            ).dropna(subset=[var_t, "RH"])
            if len(merged_mul) < 2:
                print("  RH x T      : not enough points to compute metrics")
                print("  RHxT_avg    : not enough points to compute metrics")
            else:
                x_cal_mul  = merged_mul["calibrated_O3"].values
                y_mul_raw  = merged_mul[var_q].values
                T_vals4    = merged_mul[var_t].values
                RH_vals4   = merged_mul["RH"].values

                # — ordinary RH x T
                a, b, c         = popt_RH_x_T_map[label]
                x_corr_mul      = corr_RH_x_T(y_mul_raw, T_vals4, RH_vals4, a, b, c)
                _print_metrics("RH x T", x_corr_mul, x_cal_mul, p=3)

                # — RHxT_avg
                a2, b2, c2      = popt_RH_x_T_avg_map[label]
                x_corr_mul_avg  = corr_RH_x_T_avg(y_mul_raw, T_vals4, RH_vals4, a2, b2, c2)
                _print_metrics("RHxT_avg", x_corr_mul_avg, x_cal_mul, p=3)
        else:
            print("  RH x T      : parameters missing")
            print("  RHxT_avg    : parameters missing")

        print()  # blank line between sensors

def compute_metrics_testing(start, end, interval, tz=None, source_list=None):
    """
    For each sensor in source_list (or global csv_sources), compute and print
    R², bias, χ²_red, and RMSE for each correction method over the testing interval.
    """
    # 1) Prepare test-interval calibrated data
    start_ts = pd.to_datetime(start).tz_localize("UTC")
    end_ts   = pd.to_datetime(end).tz_localize("UTC") + pd.Timedelta(days=1)

    base_folders    = ["2024", "2025"]
    hdf_files_local = find_files_in_date_range(base_folders, start_ts, end_ts)
    df_raw_local    = read_variable_from_files(hdf_files_local, var, start_ts, end_ts)
    df_clean_local  = remove_outliers_rolling_zscore(df_raw_local, var)
    df_averaged_local = time_average(df_clean_local, var, interval)
    df_cal_local    = df_averaged_local.copy()

    if df_cal_local.empty:
        print("No calibrated data in the testing interval.")
        return

    # Rename calibrated column
    df_cal_local = df_cal_local.rename(columns={var: "calibrated_O3"})
    if tz:
        df_cal_local["Time_ISO"] = df_cal_local["Time_ISO"].dt.tz_convert(tz)

    # 2) Loop over sensors
    for label, _, _, _ in (source_list or csv_sources):
        print(f"{label}:")
        # Load test-interval raw data for this sensor
        path = next(fn for fn in csv_files if label in fn)
        df_sensor = pd.read_csv(path)
        df_sensor["Time_ISO"] = pd.to_datetime(df_sensor["timestamp"], utc=True)
        df_sensor = df_sensor[
            (df_sensor["Time_ISO"] >= start_ts) & (df_sensor["Time_ISO"] < end_ts)
        ]
        if df_sensor.empty:
            print("  No sensor data in testing interval.\n")
            continue

        # Resample raw O₃, temp, RH
        df_o3  = time_average(df_sensor, var_q, interval)
        df_tn  = time_average(df_sensor, var_t, interval)
        df_rhn = time_average(df_sensor, var_rh, interval).rename(columns={var_rh: "RH"})
        if tz:
            df_o3["Time_ISO"] = df_o3["Time_ISO"].dt.tz_convert(tz)
            df_tn["Time_ISO"] = df_tn["Time_ISO"].dt.tz_convert(tz)
            df_rhn["Time_ISO"] = df_rhn["Time_ISO"].dt.tz_convert(tz)

        # Merge calibrated + raw sensor + covariates
        merged = (
            df_cal_local[["Time_ISO", "calibrated_O3"]]
            .merge(df_o3.rename(columns={var_q: var_q}), on="Time_ISO", how="inner")
            .merge(df_tn.rename(columns={var_t: var_t}), on="Time_ISO", how="inner")
            .merge(df_rhn, on="Time_ISO", how="inner")
        ).dropna(subset=["calibrated_O3", var_q, var_t, "RH"])
        if len(merged) < 2:
            print("  Not enough merged points to compute metrics.\n")
            continue

        x_true = merged["calibrated_O3"].values
        y_raw  = merged[var_q].values
        t_vals = merged[var_t].values
        rh_vals= merged["RH"].values
        N = len(x_true)

        def _print(name, x_corr):
            valid = x_corr > 0
            x_c = x_corr[valid]
            x_t = x_true[valid]
            n   = len(x_c)
            if n < 2:
                print(f"  {name:12s}: not enough positive points")
                return
            r = pearsonr(x_t, x_c)[0]
            r2 = r**2
            bias = np.mean(x_c - x_t)
            res = x_c - x_t
            p = {
                "linear": 2,
                "T": 3,     "T_avg": 3,
                "RH": 3,    "RH_avg": 3,
                "T2": 4,    "T2_avg": 4,
                "RH2": 4,   "RH2_avg": 4,
                "RH+T": 4,  "RH+T_avg": 4,
                "RHxT": 3,  "RHxT_avg": 3,
            }[name]
            chi2 = np.sum(res**2) / (n - p)
            rmse = np.sqrt(np.mean(res**2))
            print(f"  {name:12s}: R² = {r2:.4f}, bias = {bias:.4f} ppb, χ²_red = {chi2:.4f}, RMSE = {rmse:.4f} ppb")

        # 1) Raw vs Calibrated (no correction)
        r0 = pearsonr(x_true, y_raw)[0]
        r2_0 = r0**2
        bias0 = np.mean(y_raw - x_true)
        res0 = y_raw - x_true
        chi2_0 = np.sum(res0**2) / N
        rmse0 = np.sqrt(np.mean(res0**2))
        print(f"  Raw vs Calibrated: R² = {r2_0:.4f}, bias = {bias0:.4f} ppb, χ²_red = {chi2_0:.4f}, RMSE = {rmse0:.4f} ppb")

        # 2) Linear
        if label in popt_Linear_map:
            a, b = popt_Linear_map[label]
            x_corr = corr_Linear(y_raw, a, b)
            _print("linear", x_corr)
        else:
            print("  linear      : parameters missing")

        # 2b) T
        if label in popt_T_map:
            a, b, c = popt_T_map[label]
            x_corr = corr_T(y_raw, t_vals, a, b, c)
            _print("T", x_corr)
        else:
            print("  T           : parameters missing")

        # 2c) T_avg
        if label in popt_T_avg_map:
            a, b, c = popt_T_avg_map[label]
            x_corr = corr_T_avg(y_raw, t_vals, a, b, c)
            _print("T_avg", x_corr)
        else:
            print("  T_avg       : parameters missing")

        # 3) RH
        if label in popt_RH_map:
            a, b, c = popt_RH_map[label]
            x_corr = corr_RH(y_raw, rh_vals, a, b, c)
            _print("RH", x_corr)
        else:
            print("  RH          : parameters missing")

        # 3b) RH_avg
        if label in popt_RH_avg_map:
            a, b, c = popt_RH_avg_map[label]
            x_corr = corr_RH_avg(y_raw, rh_vals, a, b, c)
            _print("RH_avg", x_corr)
        else:
            print("  RH_avg      : parameters missing")

        # 4) T2
        if label in popt_T2_map:
            a, b, c, d = popt_T2_map[label]
            x_corr = corr_T2(y_raw, t_vals, a, b, c, d)
            _print("T2", x_corr)
        else:
            print("  T2          : parameters missing")

        # 4b) T2_avg
        if label in popt_T2_avg_map:
            a, b, c, d = popt_T2_avg_map[label]
            x_corr = corr_T2_avg(y_raw, t_vals, a, b, c, d)
            _print("T2_avg", x_corr)
        else:
            print("  T2_avg      : parameters missing")

        # 5) RH2
        if label in popt_RH2_map:
            a, b, c, d = popt_RH2_map[label]
            x_corr = corr_RH2(y_raw, rh_vals, a, b, c, d)
            _print("RH2", x_corr)
        else:
            print("  RH2         : parameters missing")

        # 5b) RH2_avg
        if label in popt_RH2_avg_map:
            a, b, c, d = popt_RH2_avg_map[label]
            x_corr = corr_RH2_avg(y_raw, rh_vals, a, b, c, d)
            _print("RH2_avg", x_corr)
        else:
            print("  RH2_avg     : parameters missing")

        # 6) RH+T
        if label in popt_RH_plus_T_map:
            a, b, c, d = popt_RH_plus_T_map[label]
            x_corr = corr_RH_plus_T(y_raw, t_vals, rh_vals, a, b, c, d)
            _print("RH+T", x_corr)
        else:
            print("  RH+T        : parameters missing")

        # 6b) RH+T_avg
        if label in popt_RH_plus_T_avg_map:
            a, b, c, d = popt_RH_plus_T_avg_map[label]
            x_corr = corr_RH_plus_T_avg(y_raw, t_vals, rh_vals, a, b, c, d)
            _print("RH+T_avg", x_corr)
        else:
            print("  RH+T_avg    : parameters missing")

        # 7) RHxT
        if label in popt_RH_x_T_map:
            a, b, c = popt_RH_x_T_map[label]
            x_corr = corr_RH_x_T(y_raw, t_vals, rh_vals, a, b, c)
            _print("RHxT", x_corr)
        else:
            print("  RHxT        : parameters missing")

        # 7b) RHxT_avg
        if label in popt_RH_x_T_avg_map:
            a, b, c = popt_RH_x_T_avg_map[label]
            x_corr = corr_RH_x_T_avg(y_raw, t_vals, rh_vals, a, b, c)
            _print("RHxT_avg", x_corr)
        else:
            print("  RHxT_avg    : parameters missing")

        print()  # blank line between sensors


# === Timeseries before and after correction ===
def plot_csv_corrected_timeseries_best(
    start, end, interval,
    tz=None,
    source_list=None,
    save=True,
    out_path="corrected_timeseries_best.png"
):
    """
    For each sensor in source_list (or global csv_sources), pick the best correction model
    based on reduced χ² on training data. Ties go to the non-avg version. Then plot raw vs.
    best-corrected time series on [start, end).
    """
    # 1) Identify best model for each sensor using reduced χ² on training data
    best_model_map = {}

    for label, df_q, df_T, df_RH in (source_list or csv_sources):
        # Merge global training-interval calibrated O₃ (df_cal) with this sensor’s raw, T, RH
        m = (
            df_cal.rename(columns={var: "calibrated_O3"})
                  .merge(df_q.rename(columns={var_q: var_q}), on="Time_ISO", how="inner")
                  .merge(df_T.rename(columns={var_t: var_t}), on="Time_ISO", how="inner")
                  .merge(df_RH, on="Time_ISO", how="inner")
        )
        x_vals = m["calibrated_O3"].values
        y_vals = m[var_q].values
        t_vals = m[var_t].values
        rh_vals = m["RH"].values

        mask = (
            np.isfinite(x_vals) & np.isfinite(y_vals) &
            np.isfinite(t_vals) & np.isfinite(rh_vals)
        )
        x_fit, y_fit = x_vals[mask], y_vals[mask]
        t_fit, rh_fit = t_vals[mask], rh_vals[mask]
        N = len(x_fit)
        if N < 5:
            continue

        # Compute reduced χ² for each model (only if popt exists)
        chi2_entries = []

        # 1) Linear (p=2)
        if label in popt_Linear_map:
            a, b = popt_Linear_map[label]
            x_corr = (y_fit - b) / a
            res = x_fit - x_corr
            chi2 = np.sum(res**2) / (N - 2)
            chi2_entries.append(("linear", chi2, False))

        # 2) Temp (non-avg, p=3)
        if label in popt_T_map:
            a, b, c = popt_T_map[label]
            x_corr = (y_fit - b - c * t_fit) / a
            res = x_fit - x_corr
            chi2 = np.sum(res**2) / (N - 3)
            chi2_entries.append(("T", chi2, False))

        # 2b) Temp_avg (p=3)
        if label in popt_T_avg_map:
            a, b, c = popt_T_avg_map[label]
            t0 = t_fit - t_fit.mean()
            x_corr = (y_fit - b - c * t0) / a
            res = x_fit - x_corr
            chi2 = np.sum(res**2) / (N - 3)
            chi2_entries.append(("T_avg", chi2, True))

        # 3) RH (non-avg, p=3)
        if label in popt_RH_map:
            a, b, c = popt_RH_map[label]
            x_corr = (y_fit - b - c * rh_fit) / a
            res = x_fit - x_corr
            chi2 = np.sum(res**2) / (N - 3)
            chi2_entries.append(("RH", chi2, False))

        # 3b) RH_avg (p=3)
        if label in popt_RH_avg_map:
            a, b, c = popt_RH_avg_map[label]
            rh0 = rh_fit - rh_fit.mean()
            x_corr = (y_fit - b - c * rh0) / a
            res = x_fit - x_corr
            chi2 = np.sum(res**2) / (N - 3)
            chi2_entries.append(("RH_avg", chi2, True))

        # 4) Temp² (non-avg, p=4)
        if label in popt_T2_map:
            a, b, c, d = popt_T2_map[label]
            x_corr = (y_fit - b - c * t_fit - d * (t_fit**2)) / a
            res = x_fit - x_corr
            chi2 = np.sum(res**2) / (N - 4)
            chi2_entries.append(("T2", chi2, False))

        # 4b) T2_avg (p=4)
        if label in popt_T2_avg_map:
            a, b, c, d = popt_T2_avg_map[label]
            t0 = t_fit - t_fit.mean()
            x_corr = (y_fit - b - c * t0 - d * (t0**2)) / a
            res = x_fit - x_corr
            chi2 = np.sum(res**2) / (N - 4)
            chi2_entries.append(("T2_avg", chi2, True))

        # 5) RH² (non-avg, p=4)
        if label in popt_RH2_map:
            a, b, c, d = popt_RH2_map[label]
            x_corr = (y_fit - b - c * rh_fit - d * (rh_fit**2)) / a
            res = x_fit - x_corr
            chi2 = np.sum(res**2) / (N - 4)
            chi2_entries.append(("RH2", chi2, False))

        # 5b) RH2_avg (p=4)
        if label in popt_RH2_avg_map:
            a, b, c, d = popt_RH2_avg_map[label]
            rh0 = rh_fit - rh_fit.mean()
            x_corr = (y_fit - b - c * rh0 - d * (rh0**2)) / a
            res = x_fit - x_corr
            chi2 = np.sum(res**2) / (N - 4)
            chi2_entries.append(("RH2_avg", chi2, True))

        # 6) RH+T (non-avg, p=4)
        if label in popt_RH_plus_T_map:
            a, b, c, d = popt_RH_plus_T_map[label]
            x_corr = (y_fit - b - c * t_fit - d * rh_fit) / a
            res = x_fit - x_corr
            chi2 = np.sum(res**2) / (N - 4)
            chi2_entries.append(("RH+T", chi2, False))

        # 6b) RH+T_avg (p=4)
        if label in popt_RH_plus_T_avg_map:
            a, b, c, d = popt_RH_plus_T_avg_map[label]
            t0 = t_fit - t_fit.mean()
            rh0 = rh_fit - rh_fit.mean()
            x_corr = (y_fit - b - c * t0 - d * rh0) / a
            res = x_fit - x_corr
            chi2 = np.sum(res**2) / (N - 4)
            chi2_entries.append(("RH+T_avg", chi2, True))

        # 7) RH×T (non-avg, p=3)
        if label in popt_RH_x_T_map:
            a, b, c = popt_RH_x_T_map[label]
            x_corr = (y_fit - b - c * (t_fit * rh_fit)) / a
            res = x_fit - x_corr
            chi2 = np.sum(res**2) / (N - 3)
            chi2_entries.append(("RHxT", chi2, False))

        # 7b) RHxT_avg (p=3)
        if label in popt_RH_x_T_avg_map:
            a, b, c = popt_RH_x_T_avg_map[label]
            t0 = t_fit - t_fit.mean()
            rh0 = rh_fit - rh_fit.mean()
            x_corr = (y_fit - b - c * (t0 * rh0)) / a
            res = x_fit - x_corr
            chi2 = np.sum(res**2) / (N - 3)
            chi2_entries.append(("RHxT_avg", chi2, True))

        if not chi2_entries:
            continue

        # Choose minimum (chi2, is_avg) so non-avg wins ties
        best_label = min(chi2_entries, key=lambda tup: (tup[1], tup[2]))[0]
        best_model_map[label] = best_label

    # 2) Plot on test interval [start, end), using each sensor’s best model
    start_ts = pd.to_datetime(start).tz_localize("UTC")
    end_ts   = pd.to_datetime(end).tz_localize("UTC") + pd.Timedelta(days=1)

    # Load test-interval calibrated data
    base_folders    = ["2024", "2025"]
    hdf_files_local = find_files_in_date_range(base_folders, start_ts, end_ts)
    df_raw_local    = read_variable_from_files(hdf_files_local, var, start_ts, end_ts)
    df_clean_local  = remove_outliers_rolling_zscore(df_raw_local, var)
    df_averaged_local = time_average(df_clean_local, var, interval)
    df_cal_local    = df_averaged_local.copy()

    if df_cal_local.empty:
        df_cal_resampled = None
    else:
        df_cal_local = df_cal_local.rename(columns={var: "calibrated_O3"})
        if tz:
            df_cal_local["Time_ISO"] = df_cal_local["Time_ISO"].dt.tz_convert(tz)
        df_cal_resampled = df_cal_local.copy()

    source_list = source_list or csv_sources
    n           = len(source_list)
    colors      = plt.cm.tab10(np.linspace(0, 1, n))
    fig, (ax_raw, ax_corr) = plt.subplots(
        2, 1, sharex=True, figsize=(14, 8),
        gridspec_kw={"height_ratios": [1, 1]}
    )
    fig.suptitle(
        f"Best-Model Correction Based on Reduced χ²\n"
        f"Testing: {start_ts.date()} → {(end_ts - pd.Timedelta(days=1)).date()}, "
        f"interval = {interval} min",
        fontsize=16
    )

    # Plot calibrated O₃ reference
    if df_cal_resampled is not None:
        ax_raw.plot(
            df_cal_resampled["Time_ISO"],
            df_cal_resampled["calibrated_O3"],
            label="Calibrated O₃",
            color="black",
            linewidth=1.5
        )
        ax_corr.plot(
            df_cal_resampled["Time_ISO"],
            df_cal_resampled["calibrated_O3"],
            label="Calibrated O₃",
            color="black",
            linewidth=1.5
        )

    # Loop over each sensor CSV to plot raw & best-corrected
    for (label, df_q, df_T, df_RH), color in zip(source_list, colors):
        best_model = best_model_map.get(label)
        if best_model is None:
            continue

        path = next(fn for fn in csv_files if label in fn)
        df_sensor = pd.read_csv(path)
        df_sensor["Time_ISO"] = pd.to_datetime(df_sensor["timestamp"], utc=True)
        df_sensor = df_sensor[
            (df_sensor["Time_ISO"] >= start_ts) & (df_sensor["Time_ISO"] < end_ts)
        ]

        # Resample raw series & covariates
        df_o3  = time_average(df_sensor, var_q, interval)
        df_tn  = time_average(df_sensor, var_t, interval)
        df_rhn = time_average(df_sensor, var_rh, interval)
        df_rhn = df_rhn.rename(columns={var_rh: "RH"})
        if tz:
            df_o3["Time_ISO"] = df_o3["Time_ISO"].dt.tz_convert(tz)
            df_tn["Time_ISO"] = df_tn["Time_ISO"].dt.tz_convert(tz)
            df_rhn["Time_ISO"] = df_rhn["Time_ISO"].dt.tz_convert(tz)

        # Plot raw O₃
        ax_raw.scatter(
            df_o3["Time_ISO"],
            df_o3[var_q],
            label=label + " (raw)",
            color=color,
            s=0.7,
            alpha=0.6
        )

        # Merge covariates as needed
        merged = df_o3.rename(columns={var_q: var_q})
        if best_model in ["T", "T_avg", "T2", "T2_avg", "RH+T", "RH+T_avg", "RHxT", "RHxT_avg"]:
            merged = merged.merge(df_tn.rename(columns={var_t: var_t}), on="Time_ISO", how="inner")
        if best_model in ["RH", "RH_avg", "RH2", "RH2_avg", "RH+T", "RH+T_avg", "RHxT", "RHxT_avg"]:
            merged = merged.merge(df_rhn.rename(columns={"RH": "RH"}), on="Time_ISO", how="inner")
        if merged.empty:
            continue

        # Apply chosen correction
        if best_model == "linear":
            a, b = popt_Linear_map[label]
            merged["o3_corr_best"] = (merged[var_q] - b) / a

        elif best_model == "T":
            a, b, c = popt_T_map[label]
            merged["o3_corr_best"] = (merged[var_q] - b - c * merged[var_t]) / a

        elif best_model == "T_avg":
            a, b, c = popt_T_avg_map[label]
            t0 = merged[var_t] - merged[var_t].mean()
            merged["o3_corr_best"] = (merged[var_q] - b - c * t0) / a

        elif best_model == "RH":
            a, b, c = popt_RH_map[label]
            merged["o3_corr_best"] = (merged[var_q] - b - c * merged["RH"]) / a

        elif best_model == "RH_avg":
            a, b, c = popt_RH_avg_map[label]
            rh0 = merged["RH"] - merged["RH"].mean()
            merged["o3_corr_best"] = (merged[var_q] - b - c * rh0) / a

        elif best_model == "T2":
            a, b, c, d = popt_T2_map[label]
            merged["o3_corr_best"] = (merged[var_q] - b - c * merged[var_t] - d * (merged[var_t]**2)) / a

        elif best_model == "T2_avg":
            a, b, c, d = popt_T2_avg_map[label]
            t0 = merged[var_t] - merged[var_t].mean()
            merged["o3_corr_best"] = (merged[var_q] - b - c * t0 - d * (t0**2)) / a

        elif best_model == "RH2":
            a, b, c, d = popt_RH2_map[label]
            merged["o3_corr_best"] = (merged[var_q] - b - c * merged["RH"] - d * (merged["RH"]**2)) / a

        elif best_model == "RH2_avg":
            a, b, c, d = popt_RH2_avg_map[label]
            rh0 = merged["RH"] - merged["RH"].mean()
            merged["o3_corr_best"] = (merged[var_q] - b - c * rh0 - d * (rh0**2)) / a

        elif best_model == "RH+T":
            a, b, c, d = popt_RH_plus_T_map[label]
            merged["o3_corr_best"] = (merged[var_q] - b - c * merged[var_t] - d * merged["RH"]) / a

        elif best_model == "RH+T_avg":
            a, b, c, d = popt_RH_plus_T_avg_map[label]
            t0 = merged[var_t] - merged[var_t].mean()
            rh0 = merged["RH"] - merged["RH"].mean()
            merged["o3_corr_best"] = (merged[var_q] - b - c * t0 - d * rh0) / a

        elif best_model == "RHxT":
            a, b, c = popt_RH_x_T_map[label]
            merged["o3_corr_best"] = (merged[var_q] - b - c * (merged[var_t] * merged["RH"])) / a

        elif best_model == "RHxT_avg":
            a, b, c = popt_RH_x_T_avg_map[label]
            t0 = merged[var_t] - merged[var_t].mean()
            rh0 = merged["RH"] - merged["RH"].mean()
            merged["o3_corr_best"] = (merged[var_q] - b - c * (t0 * rh0)) / a

        else:
            continue

        merged = merged[merged["o3_corr_best"] > 0]
        ax_corr.scatter(
            merged["Time_ISO"],
            merged["o3_corr_best"],
            label=f"{label} ({best_model})",
            color=color,
            s=0.7,
            alpha=0.6
        )

    # Final formatting
    fmt = mdates.DateFormatter("%H:%M\n%d-%b", tz=tz)
    ax_corr.xaxis.set_major_formatter(fmt)
    fig.autofmt_xdate()

    ax_raw.set_ylabel("O₃ (ppb) – Before Correction")
    ax_corr.set_ylabel("O₃ (ppb) – After Best Correction")
    ax_corr.set_xlabel("Time")

    ax_raw.legend(ncol=2, fontsize="small", loc="upper left")
    ax_corr.legend(ncol=2, fontsize="small", loc="upper left")

    if tz:
        ax_raw.set_xlim(start_ts.tz_convert(tz), (end_ts - pd.Timedelta(days=1)).tz_convert(tz))
        ax_corr.set_xlim(start_ts.tz_convert(tz), (end_ts - pd.Timedelta(days=1)).tz_convert(tz))
    else:
        ax_raw.set_xlim(start_ts, end_ts - pd.Timedelta(days=1))
        ax_corr.set_xlim(start_ts, end_ts - pd.Timedelta(days=1))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
#===========
#linear correction all:

def plot_linear_corrected_timeseries(
    start: str,
    end: str,
    interval: int,
    tz: str = None,
    source_list=None,
    save: bool = False,
    out_path: str = "linear_corrected_timeseries.png"
):
    """
    Plot test-period time series before and after Linear correction for each sensor.
    """
    # 1) Prepare test-interval calibrated reference
    start_ts = pd.to_datetime(start).tz_localize("UTC")
    end_ts   = pd.to_datetime(end).tz_localize("UTC") + pd.Timedelta(days=1)

    # Load calibrated data for test interval
    base_folders    = ["2024", "2025"]
    hdf_files_local = find_files_in_date_range(base_folders, start_ts, end_ts)
    df_raw_local    = read_variable_from_files(hdf_files_local, var, start_ts, end_ts)
    df_clean_local  = remove_outliers_rolling_zscore(df_raw_local, var)
    df_averaged_local = time_average(df_clean_local, var, interval)
    df_cal_local    = df_averaged_local.copy()

    if df_cal_local.empty:
        print("No calibrated data in the testing interval.")
        return

    df_cal_local = df_cal_local.rename(columns={var: "calibrated_O3"})
    if tz:
        df_cal_local["Time_ISO"] = df_cal_local["Time_ISO"].dt.tz_convert(tz)

    # 2) Setup plotting canvas
    source_list = source_list or csv_sources
    n_sources   = len(source_list)
    colors      = plt.cm.tab10(np.linspace(0, 1, n_sources))

    fig, (ax_raw, ax_corr) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 8),
        gridspec_kw={"height_ratios": [1, 1]}
    )
    fig.suptitle(
        f"Linear Correction: Test Interval {start_ts.date()}→{(end_ts - pd.Timedelta(days=1)).date()}, interval={interval}min",
        fontsize=14
    )

    # Plot calibrated reference
    ax_raw.plot(
        df_cal_local["Time_ISO"],
        df_cal_local["calibrated_O3"],
        label="Calibrated O₃",
        color="black",
        linewidth=1.5
    )
    ax_corr.plot(
        df_cal_local["Time_ISO"],
        df_cal_local["calibrated_O3"],
        label="Calibrated O₃",
        color="black",
        linewidth=1.5
    )

    # 3) Loop through each sensor and plot raw & linear-corrected
    for (label, df_q, df_T, df_RH), color in zip(source_list, colors):
        # Skip if no Linear-fit parameters
        if label not in popt_Linear_map:
            continue
        a, b = popt_Linear_map[label]

        # Load sensor CSV and restrict to test interval
        path = next(fn for fn in csv_files if label in fn)
        df_sensor = pd.read_csv(path)
        df_sensor["Time_ISO"] = pd.to_datetime(df_sensor["timestamp"], utc=True)
        df_sensor = df_sensor[
            (df_sensor["Time_ISO"] >= start_ts) & (df_sensor["Time_ISO"] < end_ts)
        ]
        if df_sensor.empty:
            continue

        # Resample raw O₃
        df_o3 = time_average(df_sensor, var_q, interval)
        if tz:
            df_o3["Time_ISO"] = df_o3["Time_ISO"].dt.tz_convert(tz)

        # Plot raw series
        ax_raw.scatter(
            df_o3["Time_ISO"],
            df_o3[var_q],
            label=f"{label} (raw)",
            color=color,
            s=1,
            alpha=0.6
        )

        # Compute linear correction: (raw - b) / a
        df_o3["o3_corr_linear"] = (df_o3[var_q] - b) / a
        df_o3 = df_o3[df_o3["o3_corr_linear"] > 0]

        # Plot corrected series
        ax_corr.scatter(
            df_o3["Time_ISO"],
            df_o3["o3_corr_linear"],
            label=f"{label} ",
            color=color,
            s=1,
            alpha=0.6
        )

    # 4) Format axes
    fmt = mdates.DateFormatter("%H:%M\n%d-%b", tz=tz)
    ax_corr.xaxis.set_major_formatter(fmt)
    fig.autofmt_xdate()

    ax_raw.set_ylabel("O₃ (ppb) - Before correction")
    ax_corr.set_ylabel("O₃ (ppb) — Linear correction")
    ax_corr.set_xlabel("Time")

    ax_raw.legend(ncol=2, fontsize="small", loc="upper left")
    ax_corr.legend(ncol=2, fontsize="small", loc="upper left")

    if tz:
        ax_raw.set_xlim(start_ts.tz_convert(tz), (end_ts - pd.Timedelta(days=1)).tz_convert(tz))
        ax_corr.set_xlim(start_ts.tz_convert(tz), (end_ts - pd.Timedelta(days=1)).tz_convert(tz))
    else:
        ax_raw.set_xlim(start_ts, end_ts - pd.Timedelta(days=1))
        ax_corr.set_xlim(start_ts, end_ts - pd.Timedelta(days=1))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
#==== linear a and b:
def print_linear_fit_params():
    """
    For each sensor, fit a linear model y = a*x + b on the training data,
    and print a, b, and their uncertainties (±σ).
    """
    for label, df_q, df_T, df_RH in csv_sources:
        # Merge calibrated series (df_averaged) with raw sensor O₃ on training interval
        merged = df_averaged.merge(
            df_q.rename(columns={var_q: var_q}),
            on="Time_ISO", how="inner"
        ).dropna(subset=[var, var_q])
        if len(merged) < 2:
            print(f"{label}: not enough data for linear fit")
            continue

        x = merged[var].values
        y = merged[var_q].values
        mask = (~np.isnan(x) & ~np.isnan(y) & np.isfinite(x) & np.isfinite(y))
        x_fit, y_fit = x[mask], y[mask]
        if len(x_fit) < 2:
            print(f"{label}: not enough valid points for linear fit")
            continue

        # Perform curve_fit to get covariance matrix
        popt, pcov = curve_fit(Linear_model, x_fit, y_fit)
        a, b = popt
        sigma_a, sigma_b = np.sqrt(np.diag(pcov))

        print(f"{label}: a = {a:.4f} ± {sigma_a:.4f}, b = {b:.4f} ± {sigma_b:.4f}")

def print_all_model_params():
    """
    For each sensor and each model type, fit on the training data and print
    the best-fit parameters ± their uncertainties (±σ).
    """
    for label, df_q, df_T, df_RH in csv_sources:
        print(f"\n{label}:")
        
        # 1) LINEAR model: y = a*x + b
        merged_lin = (
            df_averaged.rename(columns={var: var})
                       .merge(df_q.rename(columns={var_q: var_q}), on="Time_ISO", how="inner")
        ).dropna(subset=[var, var_q])
        if len(merged_lin) >= 2:
            x = merged_lin[var].values
            y = merged_lin[var_q].values
            try:
                popt, pcov = curve_fit(Linear_model, x, y)
                a, b = popt
                sigma_a, sigma_b = np.sqrt(np.diag(pcov))
                print(f"  Linear:             a = {a:.4f} ± {sigma_a:.4f}, b = {b:.4f} ± {sigma_b:.4f}")
            except Exception:
                print("  Linear:             fit failed")
        else:
            print("  Linear:             not enough data")

        # 2) T-only models: model_T and model_T_avg
        merged_T = (
            df_averaged.rename(columns={var: var})
                       .merge(df_q.rename(columns={var_q: var_q}), on="Time_ISO", how="inner")
                       .merge(df_T.rename(columns={var_t: var_t}), on="Time_ISO", how="inner")
        ).dropna(subset=[var, var_q, var_t])
        if len(merged_T) >= 3:
            x = merged_T[var].values
            y = merged_T[var_q].values
            T_vals = merged_T[var_t].values

            # model_T: y = a*x + b + c*T
            try:
                popt_T, pcov_T = curve_fit(model_T, (x, T_vals), y)
                a, b, c = popt_T
                sigma_a, sigma_b, sigma_c = np.sqrt(np.diag(pcov_T))
                print(f"  T:                  a = {a:.4f} ± {sigma_a:.4f}, b = {b:.4f} ± {sigma_b:.4f}, c = {c:.4f} ± {sigma_c:.4f}")
            except Exception:
                print("  T:                  fit failed")

            # model_T_avg: y = a*x + b + c*(T - mean(T))
            try:
                popt_Ta, pcov_Ta = curve_fit(model_T_avg, (x, T_vals), y)
                a2, b2, c2 = popt_Ta
                sigma_a2, sigma_b2, sigma_c2 = np.sqrt(np.diag(pcov_Ta))
                print(f"  T_avg:              a = {a2:.4f} ± {sigma_a2:.4f}, b = {b2:.4f} ± {sigma_b2:.4f}, c = {c2:.4f} ± {sigma_c2:.4f}")
            except Exception:
                print("  T_avg:              fit failed")
        else:
            print("  T:                  not enough data")
            print("  T_avg:              not enough data")

        # 3) T² models: model_T2 and model_T2_avg
        if len(merged_T) >= 4:
            x = merged_T[var].values
            y = merged_T[var_q].values
            T_vals = merged_T[var_t].values

            # model_T2: y = a*x + b + c*T + d*T²
            try:
                popt_T2, pcov_T2 = curve_fit(model_T2, (x, T_vals), y)
                a, b, c, d = popt_T2
                sigmas = np.sqrt(np.diag(pcov_T2))
                print(f"  T2:                 a = {a:.4f} ± {sigmas[0]:.4f}, b = {b:.4f} ± {sigmas[1]:.4f}, c = {c:.4f} ± {sigmas[2]:.4f}, d = {d:.4f} ± {sigmas[3]:.4f}")
            except Exception:
                print("  T2:                 fit failed")

            # model_T2_avg: y = a*x + b + c*(T - mean(T)) + d*(T - mean(T))²
            try:
                popt_T2a, pcov_T2a = curve_fit(model_T2_avg, (x, T_vals), y)
                a2, b2, c2, d2 = popt_T2a
                sig2 = np.sqrt(np.diag(pcov_T2a))
                print(f"  T2_avg:             a = {a2:.4f} ± {sig2[0]:.4f}, b = {b2:.4f} ± {sig2[1]:.4f}, c = {c2:.4f} ± {sig2[2]:.4f}, d = {d2:.4f} ± {sig2[3]:.4f}")
            except Exception:
                print("  T2_avg:             fit failed")
        else:
            print("  T2:                 not enough data")
            print("  T2_avg:             not enough data")

        # 4) RH-only models: model_RH and model_RH_avg
        merged_RH = (
            df_averaged.rename(columns={var: var})
                       .merge(df_q.rename(columns={var_q: var_q}), on="Time_ISO", how="inner")
                       .merge(df_RH.rename(columns={"RH": "RH"}), on="Time_ISO", how="inner")
        ).dropna(subset=[var, var_q, "RH"])
        if len(merged_RH) >= 3:
            x = merged_RH[var].values
            y = merged_RH[var_q].values
            RH_vals = merged_RH["RH"].values

            # model_RH: y = a*x + b + c*RH
            try:
                popt_RH, pcov_RH = curve_fit(model_RH, (x, RH_vals), y)
                a, b, c = popt_RH
                sigma_a, sigma_b, sigma_c = np.sqrt(np.diag(pcov_RH))
                print(f"  RH:                 a = {a:.4f} ± {sigma_a:.4f}, b = {b:.4f} ± {sigma_b:.4f}, c = {c:.4f} ± {sigma_c:.4f}")
            except Exception:
                print("  RH:                 fit failed")

            # model_RH_avg: y = a*x + b + c*(RH - mean(RH))
            try:
                popt_RHa, pcov_RHa = curve_fit(model_RH_avg, (x, RH_vals), y)
                a2, b2, c2 = popt_RHa
                sigma_a2, sigma_b2, sigma_c2 = np.sqrt(np.diag(pcov_RHa))
                print(f"  RH_avg:             a = {a2:.4f} ± {sigma_a2:.4f}, b = {b2:.4f} ± {sigma_b2:.4f}, c = {c2:.4f} ± {sigma_c2:.4f}")
            except Exception:
                print("  RH_avg:             fit failed")
        else:
            print("  RH:                 not enough data")
            print("  RH_avg:             not enough data")

        # 5) RH² models: model_RH2 and model_RH2_avg
        if len(merged_RH) >= 4:
            x = merged_RH[var].values
            y = merged_RH[var_q].values
            RH_vals = merged_RH["RH"].values

            # model_RH2: y = a*x + b + c*RH + d*RH²
            try:
                popt_RH2, pcov_RH2 = curve_fit(model_RH2, (x, RH_vals), y)
                a, b, c, d = popt_RH2
                sigmas = np.sqrt(np.diag(pcov_RH2))
                print(f"  RH2:                a = {a:.4f} ± {sigmas[0]:.4f}, b = {b:.4f} ± {sigmas[1]:.4f}, c = {c:.4f} ± {sigmas[2]:.4f}, d = {d:.4f} ± {sigmas[3]:.4f}")
            except Exception:
                print("  RH2:                fit failed")

            # model_RH2_avg: y = a*x + b + c*(RH - mean(RH)) + d*(RH - mean(RH))²
            try:
                popt_RH2a, pcov_RH2a = curve_fit(model_RH2_avg, (x, RH_vals), y)
                a2, b2, c2, d2 = popt_RH2a
                sig2 = np.sqrt(np.diag(pcov_RH2a))
                print(f"  RH2_avg:            a = {a2:.4f} ± {sig2[0]:.4f}, b = {b2:.4f} ± {sig2[1]:.4f}, c = {c2:.4f} ± {sig2[2]:.4f}, d = {d2:.4f} ± {sig2[3]:.4f}")
            except Exception:
                print("  RH2_avg:            fit failed")
        else:
            print("  RH2:                not enough data")
            print("  RH2_avg:            not enough data")

        # 6) RH + T models: model_RH_plus_T and model_RH_plus_T_avg
        merged_RT = (
            df_averaged.rename(columns={var: var})
                       .merge(df_q.rename(columns={var_q: var_q}), on="Time_ISO", how="inner")
                       .merge(df_T.rename(columns={var_t: var_t}), on="Time_ISO", how="inner")
                       .merge(df_RH.rename(columns={"RH": "RH"}), on="Time_ISO", how="inner")
        ).dropna(subset=[var, var_q, var_t, "RH"])
        if len(merged_RT) >= 4:
            x = merged_RT[var].values
            y = merged_RT[var_q].values
            T_vals = merged_RT[var_t].values
            RH_vals = merged_RT["RH"].values

            # model_RH_plus_T: y = a*x + b + c*T + d*RH
            try:
                popt_RTp, pcov_RTp = curve_fit(model_RH_plus_T, (x, T_vals, RH_vals), y)
                a, b, c, d = popt_RTp
                sigmas = np.sqrt(np.diag(pcov_RTp))
                print(f"  RH + T:             a = {a:.4f} ± {sigmas[0]:.4f}, b = {b:.4f} ± {sigmas[1]:.4f}, c = {c:.4f} ± {sigmas[2]:.4f}, d = {d:.4f} ± {sigmas[3]:.4f}")
            except Exception:
                print("  RH + T:             fit failed")

            # model_RH_plus_T_avg: y = a*x + b + c*(T - mean(T)) + d*(RH - mean(RH))
            try:
                popt_RTpa, pcov_RTpa = curve_fit(model_RH_plus_T_avg, (x, T_vals, RH_vals), y)
                a2, b2, c2, d2 = popt_RTpa
                sig2 = np.sqrt(np.diag(pcov_RTpa))
                print(f"  RH+T_avg:           a = {a2:.4f} ± {sig2[0]:.4f}, b = {b2:.4f} ± {sig2[1]:.4f}, c = {c2:.4f} ± {sig2[2]:.4f}, d = {d2:.4f} ± {sig2[3]:.4f}")
            except Exception:
                print("  RH+T_avg:           fit failed")
        else:
            print("  RH + T:             not enough data")
            print("  RH + T_avg:           not enough data")

        # 7) RH × T models: model_RH_x_T and model_RH_x_T_avg
        if len(merged_RT) >= 3:
            x = merged_RT[var].values
            y = merged_RT[var_q].values
            T_vals = merged_RT[var_t].values
            RH_vals = merged_RT["RH"].values

            # model_RH_x_T: y = a*x + b + c*(T * RH)
            try:
                popt_RTx, pcov_RTx = curve_fit(model_RH_x_T, (x, T_vals, RH_vals), y)
                a, b, c = popt_RTx
                sigma_a, sigma_b, sigma_c = np.sqrt(np.diag(pcov_RTx))
                print(f"  RH × T:             a = {a:.4f} ± {sigma_a:.4f}, b = {b:.4f} ± {sigma_b:.4f}, c = {c:.4f} ± {sigma_c:.4f}")
            except Exception:
                print("  RH × T:             fit failed")

            # model_RH_x_T_avg: y = a*x + b + c*((T - mean(T)) * (RH - mean(RH)))
            try:
                popt_RTxa, pcov_RTxa = curve_fit(model_RH_x_T_avg, (x, T_vals, RH_vals), y)
                a2, b2, c2 = popt_RTxa
                sigma_a2, sigma_b2, sigma_c2 = np.sqrt(np.diag(pcov_RTxa))
                print(f"  RH × T_avg:           a = {a2:.4f} ± {sigma_a2:.4f}, b = {b2:.4f} ± {sigma_b2:.4f}, c = {c2:.4f} ± {sigma_c2:.4f}")
            except Exception:
                print("  RHxT_avg:           fit failed")
        else:
            print("  RH × g:             not enough data")
            print("  RH × T_avg:           not enough data")


######################################################################
######################################################################

######################################################################
######################################################################


# compute_reduced_chi_squared_training()
# compute_metrics_training()
# compute_metrics_testing("2025-01-15", "2025-02-01", 2)
# plot_csv_corrected_timeseries_best("2025-01-15", "2025-02-01", 2)
# plot_linear_corrected_timeseries("2025-01-15", "2025-02-01", 2)
# print_linear_fit_params()

print_all_model_params()

