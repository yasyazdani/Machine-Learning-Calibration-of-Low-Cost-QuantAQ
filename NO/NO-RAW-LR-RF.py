# I changed the name of csv files from QuantAQ website to:
# R-???. csv where ??? is the sensor ID and R is Raw data
# F-???.cav where ??? is the sensor ID and F is Final data data
# I didnt change the name of hdf files
# Assuing  all files and this code are in the same directory



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
from typing import List, Tuple, Dict, Iterable
# add at the top with your other imports:
from typing import Optional


NO_UPPER_THRESHOLD = 175 
# === SETTINGS ===


# Reference instrument(T200U) variables:
ref_no2  = "NO2 concentration for range 1"
ref_no   = "NO concentration for range 1"
ref_nox  = "NOX concentration for range 1"
ref_temp = "Box temperature in degree C"
ref_rh   = "Sample pressure in In.Hg"
vars_to_load = [ ref_no, ref_nox, ref_temp, ref_rh]

# Final variables:
no           = "no"

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
start_date    = "2024-11-13"
end_date      = "2025-08-21"

time_interval = 2.  # Take the time average of both QuantAQ and reference instrument data to synchronize them


# Defining global start and end date
start_date = pd.to_datetime(start_date).tz_localize("UTC")
end_date   = pd.to_datetime(end_date).tz_localize("UTC") + pd.Timedelta(days=1)



# Taking time average for both QuantAQ and reference instrument data 
# I will use this function after screening out the unacceptable values in each dataset not before
def time_average(df, var, interval):
    df = df.copy()
    df["Time_ISO"] = pd.to_datetime(df["Time_ISO"])
    df = df.set_index("Time_ISO")
    return df[[var]].resample(f"{interval}min").mean().reset_index()



# Converting time to UTC
def _local_to_utc_str(ts_str: str) -> str:
    t = pd.to_datetime(ts_str)
    t = t.tz_localize("America/Toronto")  # interpret your inputs as local time
    t = t.tz_convert("UTC")
    return t.strftime("%Y-%m-%d %H:%M")   # keep as string for later tz_localize("UTC")



# ==================  Refrence instrument :  ================== #

# 1-Reading teh data from start and end global date
# 2- Reading the Variables I defined on the top
# 3- removing out lier before taking the time average
# 4- removing spikes based on the threshold I defined for each gas

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

def remove_outliers_rolling_zscore(df, var, window_size=30, z_thresh=2.0):
# Remove local outliers from a time series using a rolling median absolute deviation (MAD) filter.
# For each point, compute the median and MAD within a sliding window (window_size = 30 samples). 
# Calculate a robust z-score: (x - median) / (1.4826 * MAD).
# Drop any rows where |z| > z_thresh (default = 2.0), keeping only values close to the local baseline.
    m   = df[var].rolling(window=window_size, center=True, min_periods=1).median()
    mad = df[var].rolling(window=window_size, center=True, min_periods=1) \
                 .apply(lambda w: np.median(np.abs(w - np.median(w))), raw=True)
    s   = mad * 1.4826
    z   = (df[var] - m) / s
    return df[z.abs() <= z_thresh].reset_index(drop=True)

def remove_spikes(df, var, max_jump):
    df = df.copy()
    # compute absolute jump from the previous point
    jump = df[var].diff().abs().fillna(0)
    # keep rows where jump is ≤ max_jump
    return df[jump <= max_jump].reset_index(drop=True)

# thresholds for “big jumps” you want to drop per 30 seconds for refernce instrument
thresholds = { "no_diff": 100,   # ppb
    "temp":      5,   # °C
    "rh":       10,   # %
}


# === LOAD AND PROCESS REFERENCE DATA (HDF)  ===
base_folders = ["NOx/2024", "NOx/2025"]
hdf_files    = find_files_in_date_range(base_folders, start_date, end_date)
df_raw_ref   = read_variables_from_files(hdf_files, start_date, end_date).copy()

# 1) Force UTC exactly like CSV does
df_raw_ref["Time_ISO"] = pd.to_datetime(df_raw_ref["Time_ISO"], utc=True)


def remove_negative_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df = df[df[col] >= 0]
    return df.reset_index(drop=True)


# Remove bad dates (half-open [start, end), fix reversed)
# DROP ANY BAD PERIODS  for the reference  instrument
# some of these dates are based on the log notes, and some of them by visual inspection of the time series of refernce data
# these time intervals will not be dropped from the quantaq data

bad_periods = [
    ("2025-02-16 14:30", "2025-02-16 17:00"),
    ("2025-02-19 16:00", "2025-02-19 17:00"),
    ("2025-03-14 00:00", "2025-03-06 21:00"),
    ("2025-03-06 15:00", "2025-03-14 12:00"),
    ("2025-03-20 14:00", "2025-03-20 21:00"),
    ("2025-07-04 10:00", "2025-07-04 10:15"),("2025-07-07 13:20", "2025-07-07 13:40"),
    ("2025-07-10 13:10", "2025-07-10 13:55"),
    ("2025-07-01 13:20", "2025-07-01 13:40"),
    ("2025-07-04 10:00", "2025-07-04 10:15"),
    ("2025-07-09 05:40", "2025-07-09 06:05"),
    ("2024-12-30 09:00", "2024-12-30 10:05"),
    ("2024-12-16 14:00", "2024-12-16 20:00"),
    ("2025-04-24 14:00", "2025-04-24 18:00"),
    ("2025-03-20 16:00", "2025-03-20 19:00"),
    ("2025-05-05 16:00", "2025-05-05 23:00"),
    ("2025-05-27 05:00", "2025-05-27 06:00"),
    ("2025-03-06 16:00", "2025-03-14 17:00"),
    ("2025-08-07 16:00", "2025-08-08 21:00"),
    ("2025-08-11 13:00", "2025-08-11 23:00"),
    ("2025-03-20 00:00", "2025-03-21 00:00"),
    ("2025-05-05 00:00", "2025-05-06 00:00")
]
for s_str, e_str in bad_periods:
    s = pd.to_datetime(s_str, utc=True)
    e = pd.to_datetime(e_str, utc=True)
    if s > e:  # fix reversed ranges
        s, e = e, s
    df_raw_ref = df_raw_ref[~((df_raw_ref["Time_ISO"] >= s) & (df_raw_ref["Time_ISO"] < e))]

# Drop NaNs (only in vars we use) and remove negatives in the refrence instrument
keep_vars = [v for v in vars_to_load if v in df_raw_ref.columns]
df_ref = df_raw_ref.dropna(subset=["Time_ISO"] + keep_vars).copy()

for v in keep_vars:
    df_ref[v] = pd.to_numeric(df_ref[v], errors="coerce")
df_ref = df_ref.dropna(subset=keep_vars)

# keep only non-negative (and optional upper cap for NO)
for v in keep_vars:
    df_ref = df_ref[df_ref[v] >= 0]
if 'NO_UPPER_THRESHOLD' in globals() and NO_UPPER_THRESHOLD is not None:
    df_ref = df_ref[df_ref[ref_no] <= NO_UPPER_THRESHOLD]

# rolling outlier removal (still BEFORE averaging)
df_roll = df_ref.copy()
for v in keep_vars:
    df_roll = remove_outliers_rolling_zscore(df_roll, v)

# remove spikes on NO BEFORE averaging
df_no_spikes = remove_spikes(df_roll, ref_no, max_jump=10)

# Time-average each variable, then merge
avg_dfs = [
    time_average(df_no_spikes[["Time_ISO", v]], v, time_interval)
    for v in keep_vars
]
df_averaged_ref = reduce(lambda L, R: pd.merge(L, R, on="Time_ISO"), avg_dfs)

# Final HDF reference for training/plots
df_cal_ref = df_averaged_ref.copy()




# ================== QuantAQ instrument : ================== #



# define manual bad windows per QuantAQ sensor: (By visual inspection)
# diffrent sensors can have unacceptable data at diffrent time. 
# I will not necessarily drop a particular time for all sensors
# you can add more time interval for each sensor , or add new sensors
bad_periods_per_sensor = {
    "R-798": [
        ("2024-11-07 03:40", "2024-11-07 04:20"), ("2024-11-13 10:00", "2024-11-13 18:00"),("2024-11-27 12:00", "2024-11-27 21:00"),("2025-01-30 17:00", "2025-01-30 17:30"),("2024-11-07 03:55", "2024-11-07 04:04"),("2024-11-27 18:00", "2024-11-27 18:15"),("2025-01-15 17:20", "2025-01-15 17:40"),("2025-04-27 05:10", "2025-04-27 07:20"),("2025-05-23 22:00", "2025-05-23 23:20"),("2025-05-18 01:00", "2025-05-18 16:00"),("2025-05-20 14:40", "2025-05-20 15:00"),("2025-03-28 00:10", "2025-03-28 00:20"),("2025-05-28 06:00", "2025-05-28 06:15"),("2025-01-07 05:00", "2025-01-07 07:40"),("2025-07-04 10:00", "2025-07-04 10:15"),("2025-07-07 21:00", "2025-07-07 21:15"),("2025-03-11 13:05", "2025-03-11 13:20"),
    ],
    "R-797": [
        ("2024-11-07 03:40", "2024-11-07 04:20"), ("2024-11-13 10:00", "2024-11-13 18:00"),("2024-11-27 12:00", "2024-11-27 21:00"),("2025-01-30 17:00", "2025-01-30 17:30"),("2024-11-07 03:55", "2024-11-07 04:04"),("2024-11-27 18:00", "2024-11-27 18:15"),("2025-01-15 17:20", "2025-01-15 17:40"),("2025-04-27 05:10", "2025-04-27 07:20"),("2025-05-23 22:00", "2025-05-23 23:20"),("2025-05-18 01:00", "2025-05-18 16:00"),("2025-05-20 14:40", "2025-05-20 15:00"),("2025-03-28 00:10", "2025-03-28 00:20"),("2025-05-28 06:00", "2025-05-28 06:15"),("2025-01-07 05:00", "2025-01-07 07:40"),("2025-07-04 10:00", "2025-07-04 10:15"),("2025-07-07 21:00", "2025-07-07 21:15"),("2025-03-11 13:05", "2025-03-11 13:20"),
    ],
    "R-796": [
        ("2024-11-07 03:40", "2024-11-07 04:20"), ("2024-11-13 10:00", "2024-11-13 18:00"),("2024-11-27 12:00", "2024-11-27 21:00"),("2025-01-30 17:00", "2025-01-30 17:30"),("2024-11-07 03:55", "2024-11-07 04:04"),("2024-11-27 18:00", "2024-11-27 18:15"),("2025-01-15 17:20", "2025-01-15 17:40"),("2025-04-27 05:10", "2025-04-27 07:20"),("2025-05-23 22:00", "2025-05-23 23:20"),("2025-05-18 01:00", "2025-05-18 16:00"),("2025-05-20 14:40", "2025-05-20 15:00"),("2025-03-28 00:10", "2025-03-28 00:20"),("2025-05-28 06:00", "2025-05-28 06:15"),("2025-01-07 05:00", "2025-01-07 07:40"),("2025-07-04 10:00", "2025-07-04 10:15"),("2025-07-07 21:00", "2025-07-07 21:15"),("2025-06-26 14:40", "2025-06-26 15:15"),("2025-03-11 13:05", "2025-03-11 13:20"),
    ],

}


# === LOAD RAW DATA ===
csv_files = glob.glob("R-*.csv")
var_list = [var_t, var_rh, no_we, no_ae, no_d, no2_d , o3_d ]
csv_sources = []
for f in csv_files:
    label = os.path.basename(f).replace(".csv", "")

    # load raw columns
    df = pd.read_csv(f, usecols=["timestamp"] + var_list)

    # timestamp + date clipping
    df["Time_ISO"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[(df["Time_ISO"] >= start_date) & (df["Time_ISO"] < end_date)]

    # remove manual bad windows (raw level)
    for s_str, e_str in bad_periods_per_sensor.get(label, []):
        s = pd.to_datetime(s_str, utc=True)
        e = pd.to_datetime(e_str, utc=True)
        df = df[~((df["Time_ISO"] >= s) & (df["Time_ISO"] < e))]

    # remove NaNs based ONLY on var_list
    df = df.dropna(subset=var_list).reset_index(drop=True)

    # average to the target interval (filtered data only)
    avg_vars = var_list
    avg_dfs = [time_average(df[["Time_ISO", v]], v, time_interval) for v in avg_vars]
    df_avg = reduce(lambda L, R: pd.merge(L, R, on="Time_ISO"), avg_dfs)

    # remove manual bad windows AGAIN after averaging
    for s_str, e_str in bad_periods_per_sensor.get(label, []):
        s = pd.to_datetime(s_str, utc=True)
        e = pd.to_datetime(e_str, utc=True)
        df_avg = df_avg[~((df_avg["Time_ISO"] >= s) & (df_avg["Time_ISO"] < e))]

    # ---- drop negatives BEFORE defining power features ----
    for c in ("no_diff", "no2_diff"):
        if c in df_avg.columns:
            df_avg[c] = pd.to_numeric(df_avg[c], errors="coerce")
    if "no_diff" in df_avg.columns:
        df_avg = df_avg[df_avg["no_diff"] > 0]
    if "no2_diff" in df_avg.columns:
        df_avg = df_avg[df_avg["no2_diff"] >= 0]
    df_avg = df_avg.reset_index(drop=True)
    # ------------------------------------------------------

    # add derived features AFTER averaging and final filtering
    df_avg["temp_K2"] = (df_avg["temp"] + 273.15) ** 2
    df_avg["day_index"] = (df_avg["Time_ISO"] - start_date).dt.days

    if "no_diff" in df_avg.columns:
        df_avg["no_diff_sq"] = df_avg["no_diff"] ** 2
        df_avg["no_diff_cb"] = df_avg["no_diff"] ** 3
        df_avg["log_NO"]     = np.log(df_avg["no_diff"] + 1)
        df_avg["no_diff_15"] = df_avg["no_diff"] ** 1.5
        df_avg["no_diff_11"] = df_avg["no_diff"] ** 1.1

    if "no2_diff" in df_avg.columns:
        df_avg["no2_diff_15"] = df_avg["no2_diff"] ** 1.5

    # append cleaned DataFrame for modeling
    csv_sources.append((label, df_avg))


# === LOAD Final DATA ===
csv_files_finals = glob.glob("F-*.csv")
var_list_final = [no]
csv_sources_finals = []
for f in csv_files_finals:
    label = os.path.basename(f).replace(".csv", "")

    # load final columns
    df_f = pd.read_csv(f, usecols=["timestamp"] + var_list_final)

    # timestamp + date clipping
    df_f["Time_ISO"] = pd.to_datetime(df_f["timestamp"], utc=True)
    df_f = df_f[(df_f["Time_ISO"] >= start_date) & (df_f["Time_ISO"] < end_date)]

    # remove manual bad windows (reuse your dict)
    for s_str, e_str in bad_periods_per_sensor.get(label, []):
        s = pd.to_datetime(s_str, utc=True)
        e = pd.to_datetime(e_str, utc=True)
        df_f = df_f[~((df_f["Time_ISO"] >= s) & (df_f["Time_ISO"] < e))]

    # remove NaNs based ONLY on final var(s)
    df_f = df_f.dropna(subset=var_list_final).reset_index(drop=True)

    # average to the target interval (filtered data only)
    avg_vars_f = var_list_final
    avg_dfs_f = [time_average(df_f[["Time_ISO", v]], v, time_interval) for v in avg_vars_f]
    df_avg_f = reduce(lambda L, R: pd.merge(L, R, on="Time_ISO"), avg_dfs_f)

    # remove manual bad windows AGAIN after averaging
    for s_str, e_str in bad_periods_per_sensor.get(label, []):
        s = pd.to_datetime(s_str, utc=True)
        e = pd.to_datetime(e_str, utc=True)
        df_avg_f = df_avg_f[~((df_avg_f["Time_ISO"] >= s) & (df_avg_f["Time_ISO"] < e))]

    # collect for plotting
    csv_sources_finals.append((label, df_avg_f))



# Removing the bad periods in quantaq files
for sensor, periods in bad_periods_per_sensor.items():
    bad_periods_per_sensor[sensor] = [
        (_local_to_utc_str(s), _local_to_utc_str(e))
        for s, e in periods
    ]




# ================== time series of Final Data by quantaq + reference data :==================
def plot_f_and_hdf_together(
    csv_sources_finals: List[Tuple[str, pd.DataFrame]],
    var_name: str,
    df_ref: pd.DataFrame,
    ref_var: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> None:
    """
    Overlay all prepared F CSV series with the HDF reference on one plot,
    EXACTLY matching training behavior:
      - UTC enforced
      - [start_date, end_date) clipping
      - INNER JOIN each F series to the reference timestamps
    """
    # Reference in UTC + clamp
    ref_plot = (
        df_ref[["Time_ISO", ref_var]]
        .dropna(subset=[ref_var])
        .assign(Time_ISO=lambda d: pd.to_datetime(d["Time_ISO"], utc=True))
    )
    ref_plot = ref_plot[(ref_plot["Time_ISO"] >= start_date) & (ref_plot["Time_ISO"] < end_date)]
    ref_plot = ref_plot.sort_values("Time_ISO").reset_index(drop=True)

    plt.figure(figsize=(14, 6))
    plt.plot(ref_plot["Time_ISO"], ref_plot[ref_var], "k-", lw=2, label="T200U (ref)")

    # Plot only F points that align with reference timestamps (like training merges)
    ref_times = ref_plot[["Time_ISO"]]  # used for inner join
    for label, df_f in csv_sources_finals:
        if df_f is None or df_f.empty or var_name not in df_f.columns:
            continue
        dfp = df_f[["Time_ISO", var_name]].copy()
        dfp["Time_ISO"] = pd.to_datetime(dfp["Time_ISO"], utc=True)
        dfp = dfp[(dfp["Time_ISO"] >= start_date) & (dfp["Time_ISO"] < end_date)]
        if dfp.empty:
            continue

        dfp = pd.merge(ref_times, dfp, on="Time_ISO", how="inner")
        if dfp.empty:
            continue

        dfp = dfp.sort_values("Time_ISO").reset_index(drop=True)
        plt.plot(dfp["Time_ISO"], dfp[var_name], alpha=0.9, label=label)

    plt.xlabel("Time (UTC)")
    plt.ylabel("NO (ppb)")
    title_start = start_date.strftime("%Y-%m-%d")
    title_end   = (end_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    plt.title(f"QQ calibrated values + Reference — {title_start} to {title_end}")
    plt.legend(fontsize="small", ncol=2)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"F_HDF_timeseries_{title_start}_{title_end}.png", dpi=200)
    plt.show()

# You can call it by :
# plot_f_and_hdf_together(csv_sources_finals, no, df_cal_ref, ref_no, start_date, end_date)




# ================== machin learning(ml):==================
# You can see the arguments I tried
# you can uncoment the argument you are interested in to see the results
# if you want to add new arguments you should define them as a new columns in the QuantAQ section


# arguments for ml:
main_arguments = ["no_diff", "temp", "rh","no2_diff","ox_diff"]
# main_arguments = ["no_diff",  "temp", "rh"]
# main_arguments = ["no_diff",  "temp", "rh","no_diff_11",]
# main_arguments = ["no_diff", "temp", "rh", "ox_diff", "no2_diff"]
# main_arguments = ["no_diff", "temp", "rh", "no2_diff", "no_diff_cb"]
# main_arguments = ["no_diff", "temp", "rh", "no2_diff","log_NO"]
# main_arguments = ["no_diff", "temp", "rh","no2_diff", "ox_diff"]
# main_arguments = ["no_diff", "temp", "rh","no2_diff", "no_diff_15"]
# main_arguments = ["no_diff", "temp", "rh","no2_diff","ox_diff", "no_diff_15","no2_diff_15"]



#  Splits the data set into train/test by time and percentage
def split_train_test(
    df: pd.DataFrame,
    time_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits df into train/test by time, using the first 35% and last 35%
    of the records for training, and the middle 30% for testing.
    """
    # ensure chronological order
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    n = len(df_sorted)
    # how many rows in each tail?
    n_tail = int(n * 0.35)
    # carve out first 35% and last 35% for training
    train = pd.concat([
        df_sorted.iloc[:n_tail],
        df_sorted.iloc[-n_tail:]
    ]).reset_index(drop=True)
    # the middle 30% for testing
    test = df_sorted.iloc[n_tail: n - n_tail].reset_index(drop=True)
    return train, test



# ===========================
# Linear Regression trainer
# ===========================

# train model for linear regression
# plot the time series of predicted values for all sensors + refrence data in one plot
# prints the satistical metrics 
# it uses the main arguments i defined above so you dont need to change anything each time you want to test a new model


def train_models(
    csv_sources: List[Tuple[str, pd.DataFrame]],
    df_ref: pd.DataFrame,
    ref_var: str,
    main_args: List[str],
    tolerance_minutes: int = 3
) -> Dict[str, Dict]:
  
    results: Dict[str, Dict] = {}

    # use the provided reference as-is (keep only what we need)
    ref_plot = df_ref[["Time_ISO", ref_var]].dropna(subset=[ref_var])
    ref_plot = ref_plot.sort_values("Time_ISO")

    for label, df_src in csv_sources:
        # keep only required columns; assume already clean/averaged
        use_cols = ["Time_ISO"] + [c for c in main_args if c in df_src.columns]
        if len(use_cols) <= 1:
            continue

        df = df_src[use_cols].dropna()
        if df.empty:
            continue

        # fast inner-join on timestamps (no set ops)
        merged = pd.merge(df, ref_plot, on="Time_ISO", how="inner")
        if len(merged) < 2:
            continue

        # split / train / eval (unchanged structure)
        train_df, test_df = split_train_test(merged, "Time_ISO")
        if train_df.empty or test_df.empty:
            continue

        Xtr, ytr = train_df[main_args].values, train_df[ref_var].values
        Xte, yte = test_df[main_args].values, test_df[ref_var].values

        model = LinearRegression().fit(Xtr, ytr)
        ypred = model.predict(Xte)

        rmse = sqrt(mean_squared_error(yte, ypred))
        r2   = r2_score(yte, ypred)
        p2   = pearsonr(yte, ypred)[0]**2 if len(yte) > 1 else np.nan
        mean_ref = float(np.mean(yte))
        nrmse = rmse / mean_ref if mean_ref != 0 else np.nan

        coeffs = dict(zip(main_args, model.coef_))

        # prints EXACTLY as before
        print(f"{label}:")
        print(f"  Total: {train_df['Time_ISO'].min()} → {train_df['Time_ISO'].max()}")
        print(f"  Test : {test_df['Time_ISO'].min()} → {test_df['Time_ISO'].max()}")
        print(f"  Bias (intercept): {model.intercept_:.4f}")
        print("  Coeffs:")
        for v, c in coeffs.items():
            print(f"    {v}: {c:.4f}")
        print(
            f"  Metrics: RMSE={rmse:.2f}, R²={r2:.2f}, Pearson²={p2:.2f}, "
            f"NRMSE={nrmse:.2f}\n"
        )

        # cache sorted + predictions for plotting without recomputation
        merged_sorted = merged.sort_values("Time_ISO")
        yhat_full = model.predict(merged_sorted[main_args].values)

        results[label] = {
            "model":    model,
            "merged":   merged_sorted,
            "train_df": train_df,
            "test_df":  test_df,
            "yhat_full": yhat_full
        }

    # plot EXACTLY as before (using cached predictions)
    plt.figure(figsize=(14, 6))
    plt.plot(ref_plot["Time_ISO"], ref_plot[ref_var], "k-", lw=2, label="T200U (ref)")
    for label, res in results.items():
        yhat = np.clip(res["yhat_full"], 0, None)
        plt.plot(res["merged"]["Time_ISO"], yhat, alpha=0.7, label=f"{label} (pred)")
    plt.xlabel("Time (UTC)")
    plt.ylabel(f"{ref_var} (ppb)")
    plt.legend(fontsize="small", ncol=2)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"all_timeseries_{start_date}_{end_date}")
    plt.show()

    return results


# ===========================
#  Random-Forest trainer
# ===========================

# similar to train_model function but it uses random forest instead of linear regression


def train_models_rf(
    csv_sources: List[Tuple[str, pd.DataFrame]],
    df_ref: pd.DataFrame,
    ref_var: str,
    main_args: List[str],
    tolerance_minutes: int = 3
) -> Dict[str, Dict]:
    """
    Same flow as train_models(), but using RandomForestRegressor.
    - Uses the already filtered/averaged CSVs (csv_sources)
    - Uses the already filtered & corrected HDF reference df_ref (must include ref_var)
    - Uses your existing split_train_test() for the time-based split
    - Prints metrics in the same style and makes the same style plot
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.exceptions import NotFittedError

    results: Dict[str, Dict] = {}

    # reference series (already filtered & corrected upstream)
    ref_plot = (
        df_ref[["Time_ISO", ref_var]]
        .dropna(subset=[ref_var])
        .sort_values("Time_ISO")
    )

    for label, df_src in csv_sources:
        # Only keep features that actually exist for this sensor
        feat_cols = [c for c in main_args if c in df_src.columns]
        use_cols  = ["Time_ISO"] + feat_cols
        if len(use_cols) <= 1:  # no usable features
            continue

        df = df_src[use_cols].dropna()
        if df.empty:
            continue

        # Inner-join on timestamps to align with reference
        merged = pd.merge(df, ref_plot, on="Time_ISO", how="inner")
        if len(merged) < 2:
            continue

        # Same split as your linear version
        train_df, test_df = split_train_test(merged, "Time_ISO")
        if train_df.empty or test_df.empty:
            continue

        Xtr, ytr = train_df[feat_cols].values, train_df[ref_var].values
        Xte, yte = test_df[feat_cols].values,  test_df[ref_var].values

        # Random Forest (robust, literature-style defaults)
        model = RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            max_features="sqrt",   # (avoid 'auto' which is invalid in recent sklearn)
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )

        # Fit before predict
        model.fit(Xtr, ytr)
        ypred = model.predict(Xte)

        # Metrics (same style)
        rmse = sqrt(mean_squared_error(yte, ypred))
        r2   = r2_score(yte, ypred)
        try:
            from scipy.stats import pearsonr
            p = pearsonr(yte, ypred)[0]
            p2 = p**2
        except Exception:
            p2 = np.nan
        mean_ref = float(np.mean(yte))
        nrmse = rmse / mean_ref if mean_ref != 0 else np.nan

        # Print block (mirrors your style)
        print(f"{label}:")
        print(f"  Total: {train_df['Time_ISO'].min()} → {train_df['Time_ISO'].max()}")
        print(f"  Test : {test_df['Time_ISO'].min()} → {test_df['Time_ISO'].max()}")
        print(f"  Bias (intercept): n/a (tree-based)")
        print("  Feature importances:")
        for v, imp in zip(feat_cols, model.feature_importances_):
            print(f"    {v}: {imp:.4f}")
        print(
            f"  Metrics: RMSE={rmse:.2f}, R²={r2:.2f}, Pearson²={p2:.2f}, "
            f"NRMSE={nrmse:.2f}\n"
        )

        # Cache predictions for the full merged series for plotting
        merged_sorted = merged.sort_values("Time_ISO")
        try:
            yhat_full = model.predict(merged_sorted[feat_cols].values)
        except NotFittedError:
            # Shouldn't happen since we fit above, but guard anyway
            yhat_full = np.full(len(merged_sorted), np.nan)

        results[label] = {
            "model":     model,
            "merged":    merged_sorted,
            "train_df":  train_df,
            "test_df":   test_df,
            "yhat_full": yhat_full
        }

    # Plot like your linear version
    plt.figure(figsize=(14, 6))
    plt.plot(ref_plot["Time_ISO"], ref_plot[ref_var], "k-", lw=2, label="T200U (ref)")
    for label, res in results.items():
        yhat = np.clip(res["yhat_full"], 0, None)
        plt.plot(res["merged"]["Time_ISO"], yhat, alpha=0.7, label=f"{label} (RF pred)")
    plt.xlabel("Time (UTC)")
    plt.ylabel(f"{ref_var} (ppb)")
    plt.legend(fontsize="small", ncol=2)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"all_timeseries_RF_{start_date}_{end_date}")
    plt.show()

    return results






# —————————————————————————————
# NO correction (T200U calibration)
# —————————————————————————————
def apply_no_correction(
    df: pd.DataFrame,
    ref_no: str,
    ref_nox: str
) -> pd.DataFrame:
    df = df.copy()
    df["Time_ISO"] = pd.to_datetime(df["Time_ISO"], utc=True)

    cal_periods = [
        ("2024-10-16", "2025-03-20", {"no": (1.5433, -0.2344), "nox": (2.1357, -1.2044)}),
        ("2025-03-20", "2025-05-05", {"no": (1.6417, -0.0635), "nox": (2.1081, -0.5515)}),
        ("2025-05-05", None,         {"no": (1.9335, -0.1934), "nox": (2.4268, -0.1941)})
    ]


    for start, end, coefs in cal_periods:
        t0 = pd.to_datetime(start).tz_localize("UTC")
        t1 = pd.to_datetime(end).tz_localize("UTC") if end else df["Time_ISO"].max() + pd.Timedelta("1D")
        mask = (df["Time_ISO"] >= t0) & (df["Time_ISO"] < t1)
        m_no,  b_no  = coefs["no"]
        m_nox, b_nox = coefs["nox"]
        df.loc[mask, "NO_corr"]  = m_no  * df.loc[mask, ref_no]  + b_no
        df.loc[mask, "NOX_corr"] = m_nox * df.loc[mask, ref_nox] + b_nox
        # df.loc[mask, "NO2_corr"] = df.loc[mask, "NOX_corr"] - df.loc[mask, "NO_corr"]

    return df


def filter_negative_no(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Removes any rows where corrected NO is negative.
    """
    return df[df["NO_corr"] >= 0].reset_index(drop=True)

df_cal_ref = apply_no_correction(df_cal_ref, ref_no, ref_nox)
df_cal_ref = filter_negative_no(df_cal_ref)


# —————————————————————————————
# TRAIN & EVALUATE ON CORRECTED NO
# —————————————————————————————
# —————————————————————————————
# 1) OVERRIDE YOUR REFERENCE COLUMN NAME
# —————————————————————————————
# point to your corrected NO column

ref_no = "NO_corr"
ref_nox = "NOx_corr"
# —————————————————————————————
# 2) TRAIN (linear regression ) USING THE CORRECTED COLUMN
# —————————————————————————————
results_corrected = train_models(
    csv_sources,
    df_cal_ref,         # this DataFrame must already have NO2_corr
    ref_no,            # "NO_corr"
    main_arguments,
    tolerance_minutes=2
)

# —————————————————————————————
# 2) TRAIN (Random forest ) USING THE CORRECTED COLUMN
# —————————————————————————————
results_rf = train_models_rf(
    csv_sources,
    df_cal_ref,     # already filtered/averaged and NO-corrected upstream
    ref_no,         # e.g., "NO_corr"
    main_arguments,
    tolerance_minutes=2
)





# ===================== Plots functions ===================

def plot_residuals_subplot(
    results: Dict[str, Dict],
    ref_var: str = ref_no
):
    """
    Creates one subplot per sensor in results, showing residual time series.
    """
    import matplotlib.pyplot as plt

    labels = list(results.keys())
    n = len(labels)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)

    if n == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        res = results[label]
        m = res["merged"]
        y_pred = res["model"].predict(m[main_arguments].values)
        residual = m[ref_var] - y_pred

        ax.plot(m["Time_ISO"], residual, label=f"{label} residual")
        ax.axhline(0, linestyle="--", color="gray", linewidth=1)
        ax.set_ylabel("Residual (ppb)")
        ax.set_title(label)
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel("Time (UTC)")
    plt.tight_layout()
    plt.savefig("residuals_subplot.png", dpi=300)
    plt.show()

def plot_residuals_histogram(
    results: Dict[str, Dict],
    ref_var: str = ref_no
):
    """
    Plots overlaid histograms of residuals for each sensor in results.
    Legend shows mean ± standard deviation.
    Saves figure as 'residuals_histogram.png'.
    """
    plt.figure(figsize=(8, 5))
    for label, res in results.items():
        m = res["merged"]
        y_pred = res["model"].predict(m[main_arguments].values)
        residuals = m[ref_var] - y_pred
        mu, std = residuals.mean(), residuals.std()
        plt.hist(
            residuals,
            bins=50,
            density=True,
            alpha=0.5,
            label=f"{label}: μ={mu:.2f}±{std:.2f}"
        )

    plt.xlabel("Residual (ppb)")
    plt.ylabel("Density")
    plt.legend(fontsize="small")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("residuals_histogram.png", dpi=300)
    plt.show()

def plot_residuals_vs_temp_subplot(
    results: Dict[str, Dict],
    temp_var: str = var_t,      # "temp"
    ref_var: str = ref_no      # "NO concentration for range 1"
):
    """
    For each sensor in results, plot residual (ref – predicted) 
    against temperature in its own subplot.
    """
    import matplotlib.pyplot as plt

    labels = list(results.keys())
    n = len(labels)
    fig, axes = plt.subplots(n, 1, figsize=(8, 4 * n), sharex=False)

    if n == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        res = results[label]
        m = res["merged"]
        # predicted and residual
        y_pred = res["model"].predict(m[main_arguments].values)
        residual = m[ref_var] - y_pred
        # temperature
        temp = m[temp_var]

        ax.scatter(temp, residual, s=10, alpha=0.6)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(label)
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Residual (ppb)")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("residuals_vs_temperature.png", dpi=300)
    plt.show()

def plot_residuals_colored_by_temp(
    results: Dict[str, Dict],
    ref_var: str = ref_no,
    temp_var: str = var_t
):
    """
    Grid of subplots—one per sensor—showing residual vs time,
    with each point colored by temperature. Uses constrained_layout
    so the colorbar sits cleanly in its own margin.
    """
    import matplotlib.pyplot as plt
    import math

    labels = list(results.keys())
    n = len(labels)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 3 * nrows),
        sharex=True,
        constrained_layout=True
    )
    axes = axes.flatten()

    scatter = None
    for ax, label in zip(axes, labels):
        res = results[label]
        m = res["merged"]
        y_pred = res["model"].predict(m[main_arguments].values)
        residual = m[ref_var] - y_pred
        temp = m[temp_var]

        scatter = ax.scatter(
            m["Time_ISO"], residual,
            c=temp, cmap="viridis",
            s=10, edgecolors="none"
        )
        ax.axhline(0, linestyle="--", color="gray", linewidth=1)
        ax.set_title(label)
        ax.set_ylabel("Residual (ppb)")
        ax.grid(True)
        ax.tick_params(axis="x", rotation=45)  # rotate x-axis ticks

    # Hide any unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    # Label bottom row
    for ax in axes[-ncols:]:
        ax.set_xlabel("Time (UTC)")

    # Single colorbar on the right
    fig.colorbar(
        scatter,
        ax=axes[:n],
        orientation="vertical",
        label="Temperature (°C)",
        shrink=0.8
    )

    fig.autofmt_xdate()  # auto-format x-axis date labels
    plt.savefig("residuals_colored_by_temp_grid.png", dpi=300)
    plt.show()

global_slopes_intercepts = {}
def plot_sensors_vs_t200u(results, out_file="sensors_vs_t200u.png"):
    """
    Creates a single figure with one subplot per sensor:
      - X axis: sensor NO (predicted)
      - Y axis: T200U NO (reference)
      - adds 1:1 line and linear fit with equation in legend
    Saves the figure to `out_file` and displays it.
    """
    sensor_var = main_arguments[0]
    ref_var    = ref_no

    labels = list(results.keys())
    n = len(labels)
    fig, axes = plt.subplots(n, 1, figsize=(8, 4 * n), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    # compute global limits
    all_true = []
    all_pred = []
    for res in results.values():
        m = res["merged"]
        all_true.extend(m[ref_var].tolist())
        all_pred.extend(res["model"].predict(m[main_arguments].values).tolist())
    vmin = min(all_true + all_pred)
    vmax = max(all_true + all_pred)

    for ax, label in zip(axes, labels):
        res = results[label]
        m = res["merged"]

        x_pred = res["model"].predict(m[main_arguments].values)  # predicted
        y_true = m[ref_var].values                               # T200U reference

        ax.scatter(x_pred, y_true, s=20, alpha=0.7)
        ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1, label="1:1")

        slope, intercept = np.polyfit(x_pred, y_true, 1)
        global_slopes_intercepts[label] = (slope, intercept)
        y_fit = slope * np.array([vmin, vmax]) + intercept
        ax.plot([vmin, vmax], y_fit, color="red", linewidth=1.5,
                label=f"y = {slope:.2f}x + {intercept:.2f}")

        ax.set_title(label)
        ax.set_ylabel("T200U NO (ppb)")
        ax.grid(True)
        ax.legend(fontsize="small")

    axes[-1].set_xlabel("Sensor NO (ppb)")
    plt.setp(axes, xlim=(vmin, vmax), ylim=(vmin, vmax))
    fig.tight_layout()

    # --- Save exactly to the name you pass ---
    out_path = (out_file or "sensors_vs_t200u.png").strip()
    root, ext = os.path.splitext(out_path)
    if ext == "":
        out_path = out_path + ".png"
    fig.savefig(out_path, dpi=300)

    plt.show()

def plot_residuals_vs_reference(
    results: Dict[str, Dict],
    ref_var: str,
    main_args: List[str]
) -> None:
    """
    For each sensor in `results`, scatter-plot residual = (true – pred)
    against the true reference value ref_var.
    """
    plt.figure(figsize=(8, 6))
    for label, res in results.items():
        m = res["merged"]
        y_pred = res["model"].predict(m[main_args].values)
        residual = m[ref_var] - y_pred
        plt.scatter(
            m[ref_var],
            residual,
            s=10,
            alpha=0.6,
            label=label
        )

    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel(f"{ref_var} (ppb)")
    plt.ylabel("Residual (ppb)")
    plt.title("Residual vs. Reference")
    plt.legend(fontsize="small", ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()





# ==================== plot results =====================

# —————————————————————————————
# Plot corrected NO for linear regression
# —————————————————————————————

# plot_sensors_vs_t200u(results_corrected, out_file="sensors_vs_t200u_corr.png")
# plot_residuals_subplot(results_corrected,   ref_var="NO_corr")
# plot_residuals_histogram(results_corrected, ref_var="NO_corr")
# plot_residuals_vs_temp_subplot(results_corrected, ref_var="NO_corr")
# plot_residuals_colored_by_temp(results_corrected,  ref_var="NO_corr")
# plot_residuals_vs_reference(results_corrected,"NO_corr", main_arguments)


# —————————————————————————————
# Plot corrected NO for Random forest
# —————————————————————————————

# plot_sensors_vs_t200u(results_rf , out_file="sensors_vs_t200u_corr.png")
# plot_residuals_subplot(results_rf,   ref_var="NO_corr")
# plot_residuals_histogram(results_rf, ref_var="NO_corr")
# plot_residuals_vs_temp_subplot(results_rf, ref_var="NO_corr")
# plot_residuals_colored_by_temp(results_rf,  ref_var="NO_corr")
# plot_residuals_vs_reference(results_rf,  "NO_corr", main_args=main_arguments)


