import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
from scipy.optimize import OptimizeWarning
from scipy.stats import pearsonr
from functools import reduce
import math
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline
from typing import Dict, List, Tuple
from math import sqrt
warnings.simplefilter("ignore", OptimizeWarning)
import matplotlib
# matplotlib.use("Agg")
matplotlib.use("TkAgg")  # or "Qt5Agg" depending on your system



# === SETTINGS ===


# Reference instrument(T200U) variables:
ref_no2  = "NO2 concentration for range 1"
ref_no   = "NO concentration for range 1"
ref_nox  = "NOX concentration for range 1"
ref_temp = "Box temperature in degree C"
ref_rh   = "Sample pressure in In.Hg"
vars_to_load = [ref_no2, ref_no, ref_nox, ref_temp, ref_rh]

# Final variables:
no2           = "no2"

# Same for Raw and Final variables:
var_t         = "temp"
var_rh        = "rh"

# Raw variables:
co_we         = "co_we"
co_ae         = "co_ae"
co_d            = "co_diff"

o3_we         = "o3_we"
o3_ae         = "o3_ae"
o3_d          = "ox_diff"


no2_we         = "no2_we"
no2_ae         = "no2_ae"
no2_d         = "no2_diff"


no_we         = "no_we"
no_ae         = "no_ae"
no_d         = "no_diff"

# Time related variables:
# start_date    = "2024-08-01"
start_date    = "2024-08-03"
end_date      = "2024-12-31"
# end_date      = "2025-07-20"
time_interval = 60

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

def read_variables_from_files(files, start_date, end_date):
    frames = []
    for fpath in files:
        try:
            df = pd.read_hdf(fpath)
            if "Time_ISO" in df.columns:
                cols = ["Time_ISO"] + [v for v in vars_to_load if v in df.columns]
                df = df[cols].dropna()
                for v in vars_to_load:
                    if v in df.columns:
                        df[v] = pd.to_numeric(df[v], errors="coerce")
                df["Time_ISO"] = pd.to_datetime(df["Time_ISO"])
                if df["Time_ISO"].dt.tz is None:
                    df["Time_ISO"] = df["Time_ISO"].dt.tz_localize("UTC")
                else:
                    df["Time_ISO"] = df["Time_ISO"].dt.tz_convert("UTC")
                df = df[(df["Time_ISO"] >= start_date) & (df["Time_ISO"] < end_date)]
                frames.append(df)
        except:
            pass
    if frames:
        return pd.concat(frames).sort_values("Time_ISO").reset_index(drop=True)
    return pd.DataFrame(columns=["Time_ISO"] + vars_to_load)




def remove_outliers_rolling_zscore(df, var, window_size=30, z_thresh=1.0):
    m   = df[var].rolling(window=window_size, center=True, min_periods=1).median()
    mad = df[var].rolling(window=window_size, center=True, min_periods=1) \
                 .apply(lambda w: np.median(np.abs(w - np.median(w))), raw=True)
    s   = mad * 1.4826
    z   = (df[var] - m) / s
    return df[z.abs() <= z_thresh].reset_index(drop=True)


def time_average(df, var, interval):
    df = df.copy()
    df["Time_ISO"] = pd.to_datetime(df["Time_ISO"])
    df = df.set_index("Time_ISO")
    return df[[var]].resample(f"{interval}min").mean().reset_index()





# === LOAD RAW DATA ===
csv_files   = glob.glob("R-*.csv")
var_list = [var_t, var_rh, co_ae, co_we, co_d, o3_we, o3_ae, o3_d, no2_we, no2_ae, no2_d, no_we, no_ae, no_d]
csv_sources = []

for f in csv_files:
    label = os.path.basename(f).replace(".csv", "")
    df = pd.read_csv(f, usecols=["timestamp"] + var_list)
    df["Time_ISO"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[(df["Time_ISO"] >= start_date) & (df["Time_ISO"] < end_date)]

    avg_dfs = [
        time_average(df[["Time_ISO", v]], v, time_interval)
        for v in var_list
    ]
    df_avg = reduce(lambda L, R: pd.merge(L, R, on="Time_ISO"), avg_dfs)
    csv_sources.append((label, df_avg))


# === LOAD AND PROCESS REFERENCE DATA ===
base_folders = ["NOx/2024", "NOx/2025"]
hdf_files    = find_files_in_date_range(base_folders, start_date, end_date)
df_raw_ref   = read_variables_from_files(hdf_files, start_date, end_date)

df_clean_ref = df_raw_ref.copy()
for var in vars_to_load:
    if var in df_clean_ref.columns:
        df_clean_ref = remove_outliers_rolling_zscore(df_clean_ref, var)

# Average each variable, then merge all at once
avg_dfs = [
    time_average(df_clean_ref[["Time_ISO", v]], v, time_interval)
    for v in vars_to_load if v in df_clean_ref.columns
]
df_averaged_ref = reduce(lambda L, R: pd.merge(L, R, on="Time_ISO"), avg_dfs)
df_cal_ref = df_averaged_ref.copy()




def remove_negative_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df = df[df[col] >= 0]
    return df.reset_index(drop=True)


df_cal_ref = remove_negative_values(df_cal_ref)


# === LOAD FINAL DATA ===
csv_files    = glob.glob("F-*.csv")
final_vars   = [no2]
csv_final    = []

for f in csv_files:
    label = os.path.basename(f).replace(".csv", "")
    df = pd.read_csv(f, usecols=["timestamp"] + final_vars)
    df["Time_ISO"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[(df["Time_ISO"] >= start_date) & (df["Time_ISO"] < end_date)]

    avg_dfs = [
        time_average(df[["Time_ISO", v]], v, time_interval)
        for v in final_vars
    ]
    df_avg = reduce(lambda L, R: pd.merge(L, R, on="Time_ISO"), avg_dfs)
    csv_final.append((label, df_avg))




# === PLOT NAPS DATA USING GLOBAL START/END ===

def load_naps_data(path: str, pollutant: str,
                   start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=11, header=None)
    base_cols = ["Station_ID", "Pollutant", "Date"] + [f"H{str(i).zfill(2)}" for i in range(1,25)]
    df.columns = base_cols + (["extra"] if df.shape[1] > len(base_cols) else [])
    val_col = f"{pollutant.lower()}_naps"

    df_long = df.melt(
        id_vars=["Date"],
        value_vars=[f"H{str(i).zfill(2)}" for i in range(1,25)],
        var_name="hour",
        value_name=val_col
    )
    df_long["hour"] = df_long["hour"].str.extract("(\d+)").astype(int) - 1
    df_long[val_col] = pd.to_numeric(df_long[val_col], errors="coerce")
    df_long = df_long[(df_long[val_col] < 9999) & df_long[val_col].notna()]

    df_long["timestamp"] = (
        pd.to_datetime(df_long["Date"], errors="coerce")
        + pd.to_timedelta(df_long["hour"], unit="h")
    )
    df_long["timestamp"] = (
        df_long["timestamp"]
        .dt.tz_localize("Canada/Eastern", ambiguous="NaT", nonexistent="NaT")
        .dt.tz_convert("UTC")
    )
    df_long = df_long.dropna(subset=["timestamp"])
    mask = (df_long["timestamp"] >= start) & (df_long["timestamp"] <= end)
    return df_long.loc[mask].sort_values("timestamp").reset_index(drop=True)

# load all three pollutants into a dict
naps_root  = "NAPS"
pollutants = ["NO", "NO2", "NOx"]
naps_data  = {}

for pol in pollutants:
    files = glob.glob(os.path.join(naps_root, pol, "*.csv"))
    dfs   = [load_naps_data(f, pol, start_date, end_date) for f in files]
    naps_data[pol] = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()





# import matplotlib.pyplot as plt
# from zoneinfo import ZoneInfo

# # Plot NO time series from HDF and NAPS in Toronto local time, without any additional filtering

# local_tz = ZoneInfo("America/Toronto")

# # 1) HDF data (already loaded into df_cal_ref)
# df_hdf_no = df_cal_ref[[ "Time_ISO", ref_no ]].dropna().copy()
# df_hdf_no["local_time"] = df_hdf_no["Time_ISO"].dt.tz_convert(local_tz)

# # 2) NAPS data (already loaded into naps_data["NO"])
# df_naps_no = naps_data["NO"][["timestamp", "no_naps"]].dropna().copy()
# df_naps_no["local_time"] = df_naps_no["timestamp"].dt.tz_convert(local_tz)

# # 3) Plot them together
# plt.figure(figsize=(14, 5))
# plt.plot(
#     df_hdf_no["local_time"],
#     df_hdf_no[ref_no],
#     linewidth=1.5,
#     label="HDF NO (range 1)"
# )
# plt.plot(
#     df_naps_no["local_time"],
#     df_naps_no["no_naps"],
#     linewidth=1.0,
#     label="NAPS NO"
# )

# plt.xlabel("Time (America/Toronto)")
# plt.ylabel("NO concentration (ppb)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# local_tz = ZoneInfo("America/Toronto")

# # 1) HDF NO₂ data
# df_hdf_no2 = df_cal_ref[["Time_ISO", ref_no2]].dropna().copy()
# df_hdf_no2["local_time"] = df_hdf_no2["Time_ISO"].dt.tz_convert(local_tz)

# # 2) NAPS NO₂ data
# df_naps_no2 = naps_data["NO2"][["timestamp", "no2_naps"]].dropna().copy()
# df_naps_no2["local_time"] = df_naps_no2["timestamp"].dt.tz_convert(local_tz)

# # 3) Plot both
# plt.figure(figsize=(14, 5))
# plt.plot(
#     df_hdf_no2["local_time"],
#     df_hdf_no2[ref_no2],
#     linewidth=1.5,
#     label="HDF NO₂ (range 1)"
# )
# plt.plot(
#     df_naps_no2["local_time"],
#     df_naps_no2["no2_naps"],
#     linewidth=1.0,
#     label="NAPS NO₂"
# )

# plt.xlabel("Time (America/Toronto)")
# plt.ylabel("NO₂ concentration (ppb)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()



# local_tz = ZoneInfo("America/Toronto")

# # 1) HDF NOx data
# df_hdf_nox = df_cal_ref[["Time_ISO", ref_nox]].dropna().copy()
# df_hdf_nox["local_time"] = df_hdf_nox["Time_ISO"].dt.tz_convert(local_tz)

# # 2) NAPS NOx data
# df_naps_nox = naps_data["NOx"][["timestamp", "nox_naps"]].dropna().copy()
# df_naps_nox["local_time"] = df_naps_nox["timestamp"].dt.tz_convert(local_tz)

# # 3) Plot both
# plt.figure(figsize=(14, 5))
# plt.plot(
#     df_hdf_nox["local_time"],
#     df_hdf_nox[ref_nox],
#     linewidth=1.5,
#     label="HDF NOx (range 1)"
# )
# plt.plot(
#     df_naps_nox["local_time"],
#     df_naps_nox["nox_naps"],
#     linewidth=1.0,
#     label="NAPS NOx"
# )

# plt.xlabel("Time (America/Toronto)")
# plt.ylabel("NOx concentration (ppb)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1) Prepare the two series
df_t200u_no = (
    df_cal_ref[["Time_ISO", ref_no]]
    .dropna()
    .rename(columns={"Time_ISO": "timestamp", ref_no: "t200u_no"})
    .copy()
)
df_naps_no = (
    naps_data["NO"][["timestamp", "no_naps"]]
    .dropna()
    .copy()
)

# 2) Inner‐join on timestamp to drop unmatched
df_merge = pd.merge(df_t200u_no, df_naps_no, on="timestamp")

# 3) Extract x (T200U) and y (NAPS)
x = df_merge["t200u_no"].values.reshape(-1, 1)
y = df_merge["no_naps"].values

# 4) Fit a linear model
model = LinearRegression().fit(x, y)
slope = model.coef_[0]
intercept = model.intercept_

# 5) Determine plotting range
mn = min(df_merge["t200u_no"].min(), df_merge["no_naps"].min())
mx = max(df_merge["t200u_no"].max(), df_merge["no_naps"].max())

# 6) Plot scatter, 1:1 line, and fit line
plt.figure(figsize=(7, 7))
plt.scatter(df_merge["t200u_no"], df_merge["no_naps"], s=20, alpha=0.6, label="Data")
plt.plot([mn, mx], [mn, mx], linestyle="--", label="1:1 Line")
y_fit = model.predict(np.array([mn, mx]).reshape(-1, 1))
plt.plot([mn, mx], y_fit, label=f"Fit: y = {slope:.2f}x + {intercept:.2f}")

plt.xlabel("T200U NO (ppb)")
plt.ylabel("NAPS NO (ppb)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1) Prepare the two series
df_t200u_no2 = (
    df_cal_ref[["Time_ISO", ref_no2]]
    .dropna()
    .rename(columns={"Time_ISO": "timestamp", ref_no2: "t200u_no2"})
    .copy()
)
df_naps_no2 = (
    naps_data["NO2"][["timestamp", "no2_naps"]]
    .dropna()
    .copy()
)

# 2) Inner-join on timestamp to drop unmatched
df_merge = pd.merge(df_t200u_no2, df_naps_no2, on="timestamp")

# 3) Extract x (T200U) and y (NAPS)
x = df_merge["t200u_no2"].values.reshape(-1, 1)
y = df_merge["no2_naps"].values

# 4) Fit a linear model
model = LinearRegression().fit(x, y)
slope = model.coef_[0]
intercept = model.intercept_

# 5) Determine plotting range
mn = min(df_merge["t200u_no2"].min(), df_merge["no2_naps"].min())
mx = max(df_merge["t200u_no2"].max(), df_merge["no2_naps"].max())

# 6) Plot scatter, 1:1 line, and fit line
plt.figure(figsize=(7, 7))
plt.scatter(df_merge["t200u_no2"], df_merge["no2_naps"], s=20, alpha=0.6, label="Data")
plt.plot([mn, mx], [mn, mx], linestyle="--", label="1:1 Line")
y_fit = model.predict(np.array([mn, mx]).reshape(-1, 1))
plt.plot([mn, mx], y_fit, label=f"Fit: y = {slope:.2f}x + {intercept:.2f}")

plt.xlabel("T200U NO₂ (ppb)")
plt.ylabel("NAPS NO₂ (ppb)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1) Prepare the two series
df_t200u_nox = (
    df_cal_ref[["Time_ISO", ref_nox]]
    .dropna()
    .rename(columns={"Time_ISO": "timestamp", ref_nox: "t200u_nox"})
    .copy()
)
df_naps_nox = (
    naps_data["NOx"][["timestamp", "nox_naps"]]
    .dropna()
    .copy()
)

# 2) Inner-join on timestamp to drop unmatched
df_merge = pd.merge(df_t200u_nox, df_naps_nox, on="timestamp")

# 3) Extract x (T200U) and y (NAPS)
x = df_merge["t200u_nox"].values.reshape(-1, 1)
y = df_merge["nox_naps"].values

# 4) Fit a linear model
model = LinearRegression().fit(x, y)
slope = model.coef_[0]
intercept = model.intercept_

# 5) Determine plotting range
mn = min(df_merge["t200u_nox"].min(), df_merge["nox_naps"].min())
mx = max(df_merge["t200u_nox"].max(), df_merge["nox_naps"].max())

# 6) Plot scatter, 1:1 line, and fit line
plt.figure(figsize=(7, 7))
plt.scatter(df_merge["t200u_nox"], df_merge["nox_naps"], s=20, alpha=0.6, label="Data")
plt.plot([mn, mx], [mn, mx], linestyle="--", label="1:1 Line")
y_fit = model.predict(np.array([mn, mx]).reshape(-1, 1))
plt.plot([mn, mx], y_fit, label=f"Fit: y = {slope:.2f}x + {intercept:.2f}")

plt.xlabel("T200U NOₓ (ppb)")
plt.ylabel("NAPS NOₓ (ppb)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
