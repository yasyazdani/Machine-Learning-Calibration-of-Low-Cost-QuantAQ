# I changed the name of csv files from QuantAQ website to:
# R-???.csv where ??? is the sensor ID and R is Raw data
# I didn't change the name of hdf files
# Assuming all files and this code are in the same directory



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

# === SETTINGS ===
var           = "CO"
var_t         = "temp"
var_rh        = "rh"
co_we         = "co_we"
co_ae         = "co_ae"
co            = "co_diff"
o3_we         = "o3_we"
o3_ae         = "o3_ae"
no2            = "no2_diff"
start_date    = "2024-10-30"
end_date      = "2025-03-02"




time_interval = 2 # Take the time average of both QuantAQ and reference instrument data to synchronize them



# Defining global start and end date
start_date = pd.to_datetime(start_date).tz_localize("UTC")
end_date   = pd.to_datetime(end_date).tz_localize("UTC") + pd.Timedelta(days=1)




# Taking time average for both QuantAQ and reference instrument data 
# I will use this function after screening out the unacceptable values in each dataset, not before
def time_average(df, var, interval):
    df = df.copy()
    df["Time_ISO"] = pd.to_datetime(df["Time_ISO"])
    df = df.set_index("Time_ISO")
    return df[[var]].resample(f"{interval}min").mean().reset_index()




# ==================  Refrence instrument :  ================== #

# 1-Reading the data from the start and end global date
# 2- Reading the Variables I defined at the top
# 3- removing outliers before taking the time average
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

def read_variable_from_files(files, var, start_date, end_date):
    frames = []
    for fpath in files:
        try:
            df = pd.read_hdf(fpath)
            if var in df.columns and "Time_ISO" in df.columns:
                # select and drop missing
                df = df[["Time_ISO", var]].dropna()
                # ensure numeric
                df[var] = pd.to_numeric(df[var], errors="coerce")
                # parse and localize Time_ISO
                df["Time_ISO"] = pd.to_datetime(df["Time_ISO"])
                if df["Time_ISO"].dt.tz is None:
                    df["Time_ISO"] = df["Time_ISO"].dt.tz_localize("UTC")
                else:
                    df["Time_ISO"] = df["Time_ISO"].dt.tz_convert("UTC")
                # **time crop on df** (was mistakenly on 'f')
                df = df[(df["Time_ISO"] >= start_date) & (df["Time_ISO"] < end_date)]
                frames.append(df)
        except:
            pass
    if frames:
        return pd.concat(frames).sort_values("Time_ISO").reset_index(drop=True)
    return pd.DataFrame(columns=["Time_ISO", var])

def remove_outliers_rolling_zscore(df, var, window_size=30, z_thresh=2.0):
# For each point, compute the median and MAD within a sliding window (window_size = 30 samples). 
# Calculate a robust z-score: (x - median) / (1.4826 * MAD).
# Drop any rows where |z| > z_thresh (default = 2.0), keeping only values close to the local baseline.

    m   = df[var].rolling(window=window_size, center=True, min_periods=1).median()
    mad = df[var].rolling(window=window_size, center=True, min_periods=1) \
                  .apply(lambda w: np.median(np.abs(w - np.median(w))), raw=True)
    s   = mad * 1.4826
    z   = (df[var] - m) / s
    return df[z.abs() <= z_thresh].reset_index(drop=True)


    df = df.copy()
    df["Time_ISO"] = pd.to_datetime(df["Time_ISO"])
    df = df.set_index("Time_ISO")
    return df[[var]].resample(f"{interval}min").mean().reset_index()

# === Bad intervals that should be removed. drop_interval_map for quantaq, hdf_interval for Picarro) ===

drop_interval_map = {
        "Raw-794": [("2024-11-12 00:00", "2024-11-13 20:00"), ("2024-11-27 17:55", "2024-11-27 18:50"),("2024-12-08 01:14", "2024-12-08 01:20"),("2025-01-07 05:00", "2025-01-07 07:30"),("2025-01-15 17:15", "2025-01-15 17:38")],
        "Raw-798": [("2024-11-13 14:00", "2024-11-13 16:30"), ("2024-11-27 17:55", "2024-11-27 18:50"),("2024-12-08 01:14", "2024-12-08 01:20"),("2025-01-07 05:00", "2025-01-07 07:30"),("2025-01-15 17:15", "2025-01-15 17:38"),("2025-01-30 17:00", "2025-01-30 17:20"), ("2025-05-13 15:00", "2025-05-13 20:30")],
        "Raw-796": [("2024-11-13 14:20", "2024-11-13 16:30"), ("2024-11-27 17:55", "2024-11-27 18:50"),("2024-12-08 01:14", "2024-12-08 01:20"),("2025-01-07 05:00", "2025-01-07 07:30"),("2025-01-15 17:15", "2025-01-15 17:38")],
        "Raw-793": [("2024-12-11 01:00", "2024-12-11 01:20"),("2024-11-27 17:55", "2024-11-27 18:50"),("2024-12-11 00:00", "2024-12-14 00:00"),("2024-12-08 01:14", "2024-12-08 01:20"),("2025-01-07 05:00", "2025-01-07 07:30"),("2025-01-15 17:15", "2025-01-15 17:38"),("2025-04-07 17:15", "2025-04-19 17:00"),("2025-05-05 15:00", "2025-05-05 17:00")],
        "Raw-795": [("2024-11-27 17:55", "2024-11-27 18:50"),("2024-12-08 01:14", "2024-12-08 01:20"),("2025-01-07 05:00", "2025-01-07 08:00"),("2025-01-15 17:15", "2025-01-15 17:38")],
        "Raw-797": [("2024-11-27 17:55", "2024-11-27 18:50"),("2024-12-08 01:14", "2024-12-08 01:20"),("2025-01-07 05:00", "2025-01-07 07:30"),("2025-01-15 17:15", "2025-01-15 17:38"),("2025-01-27 14:30", "2025-01-27 15:25"),("2025-01-30 17:00", "2025-01-30 17:20")],
        "Raw-792": [("2024-11-27 17:55", "2024-11-27 18:50"),("2024-12-08 01:14", "2024-12-08 01:20"),("2025-01-07 05:00", "2025-01-07 07:30"),("2025-01-15 17:15", "2025-01-15 17:38")],
    }

hdf_intervals = [
        ("2024-11-27 18:00", "2024-11-27 18:15"),
        ("2024-11-08 01:00", "2024-11-08 01:25"),
        ("2025-04-22 17:00", "2025-04-22 18:00"),
        ("2025-05-15 17:00", "2025-05-15 22:00"),
    ]



# === LOAD AND PROCESS CO DATA ===
base_folders = ["2024", "2025"]
hdf_files     = find_files_in_date_range(base_folders, start_date, end_date)
df_raw        = read_variable_from_files(hdf_files, var, start_date, end_date)
df_clean      = remove_outliers_rolling_zscore(df_raw, var)
df_averaged   = time_average(df_clean, var, time_interval)
df_cal        = df_averaged.copy()




# === LOAD RAW DATA ===
csv_files   = glob.glob("Raw-*.csv")
var_list = [var_t, var_rh, co_ae, co_we, co, o3_we, o3_ae, no2]
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




# ============ screaning and processing the data for both Picarro and quantaq: ============
# when I wrote this code I didint know Picarro was not calibrated so I had to screen out the data more than ususal
# also, the I first took the time avarge and synchronoize them and then I applied some of the screening out methods
# but it is probabaly better if you screen them out first and then take the time average
# since I am taking the time average first, I expected to have smoother data so the z threshold is much lower that of NO and NO2 scripts

# these come from your SETTINGS block
time_interval = 2
min_ppb       = 0
max_ppb       = 1000
window_minutes= 30
z_thresh      = 0.8
upper_thresh  = 20
lower_thresh  = 20

def screen_and_prepare():
    # — prepare HDF intervals as UTC timestamps once —
    global hdf_intervals, drop_interval_map
    hdf_utc = [(pd.to_datetime(s).tz_localize("UTC"),
                pd.to_datetime(e).tz_localize("UTC"))
               for s,e in hdf_intervals]
    # and per‐CSV intervals
    for label, rngs in drop_interval_map.items():
        drop_interval_map[label] = [
            (pd.to_datetime(s).tz_localize("UTC"),
             pd.to_datetime(e).tz_localize("UTC"))
            for s,e in rngs
        ]

    # 1) SCREEN HDF (df_raw → dfh_filtered)
    df_clean   = remove_outliers_rolling_zscore(df_raw, var)
    df_avg     = time_average(df_clean, var, time_interval)
    dfh        = df_avg[(df_avg["Time_ISO"] >= start_date)
                       & (df_avg["Time_ISO"] < end_date)].copy()
    dfh["co_ppb"] = dfh["CO"] * 1000

    # rolling‐MAD outliers
    window_samples = max(1, int(window_minutes/time_interval))
    m_h = dfh["co_ppb"].rolling(window_samples,center=True,min_periods=1).median()
    s_h = ( dfh["co_ppb"]
          .rolling(window_samples,center=True,min_periods=1)
          .apply(lambda w: np.median(np.abs(w-np.median(w))), raw=True)
          * 1.4826 )
    dfh = dfh.loc[(dfh["co_ppb"]-m_h).abs() <= z_thresh*s_h]

    # hard bounds
    dfh = dfh[(dfh["co_ppb"]>=min_ppb)&(dfh["co_ppb"]<=max_ppb)]

    # manual HDF intervals
    for s_ts, e_ts in hdf_utc:
        dfh = dfh[~((dfh["Time_ISO"]>=s_ts)&(dfh["Time_ISO"]<e_ts))]

    # gaps & spikes
    dfh = dfh.dropna(subset=["co_ppb"])
    dt       = dfh["Time_ISO"].diff().dt.total_seconds().fillna(0)
    gap_mask = dt <= time_interval*60
    vals     = dfh["co_ppb"].values
    prev     = np.roll(vals,1); prev[0]=vals[0]
    spike    = ~(((vals-prev)>upper_thresh) | ((vals-prev)<-lower_thresh))
    dfh_filtered = dfh.loc[gap_mask & spike].reset_index(drop=True)


    # 2) SCREEN EACH CSV SERIES
    csv_filtered = {}
    for label, df in csv_sources:
        df2 = df[(df["Time_ISO"]>=start_date)&(df["Time_ISO"]<end_date)].copy()
        df2 = df2.dropna(subset=["co_diff","temp","rh","co_we","co_ae","o3_we","o3_ae","no2_diff"])
        df2["o3"]      = df2["o3_we"] - df2["o3_ae"] 
        df2["no2_diff"]= df2["no2_diff"]  
        df2["o3-no2"] = df2["o3"] - df2["no2_diff"]
        # df2["temp2"] = df2["temp"]**2                      # not helpful + temp is in C , it should have been converted to kelvin first
        # df2["temp3"] = df2["temp"]**3                      # not helpful 
        # df2["rh*temp"] = df2["rh"] * df2["temp"]           # not helpful
        # df2["rh*temp2"] = df2["rh"] * df2["temp2"]         # not helpful
        # df2["hour"] = df2["Time_ISO"].dt.hour              # did not improve the performance  
        # df2["sin_h"] = np.sin(2*np.pi*df2["hour"]/24)      # did not improve the performance  
        # df2["cos_h"] = np.cos(2*np.pi*df2["hour"]/24)      # did not improve the performance  
        # df2["we_lag1"]  = df2["co_we"].shift(1)            # I was considering time lag
        # df2["co_lag2"] = df2["co_diff"].shift(2)           # # I was considering time lag
        # df2["co_ewma"] = df2["co_diff"].ewm(span=window_samples).mean() #Exponential moving average    # not helpful
        # df2 = pd.get_dummies(df2, columns=["dow"], drop_first=True) #Categorical time features   # not helpful
        # df2["we_ae_ratio"] = df2["co_we"].rolling(window_samples).mean() / df2["co_ae"].rolling(window_samples).mean() # not helpful
        # df2["co_diff_diff"] = df2["co_diff"].diff()        # not helpful

        # Rolling‐baseline (median AE & WE over past 1 h)
        window_samples = int(60 / time_interval)  # e.g. 60 min ÷ 2 min intervals = 30 samples
        # df2["we_base"]  = df2["co_we"].rolling(window_samples, center=False, min_periods=1).median()
        df2["ae_base"]  = df2["co_ae"].rolling(window_samples, center=False, min_periods=1).median()    # not helpful
        df2["co_diff"] = df2["co_we"] - df2["ae_base"]  # not helpful

        # hard CO bounds
        df2 = df2[(df2["co_diff"]>=min_ppb)&(df2["co_diff"]<=max_ppb)]
        # manual CSV intervals
        for s_ts, e_ts in drop_interval_map.get(label,[]):
            df2 = df2[~((df2["Time_ISO"]>=s_ts)&(df2["Time_ISO"]<e_ts))]

        # gaps & spikes on co_diff
        dt       = df2["Time_ISO"].diff().dt.total_seconds().fillna(0)
        gap_mask = dt<=time_interval*60
        vals     = df2["co_diff"].values
        prev     = np.roll(vals,1); prev[0]=vals[0]
        spike    = ~(((vals-prev)>upper_thresh)|((vals-prev)<-lower_thresh))
        df2_filtered = df2.loc[gap_mask & spike].reset_index(drop=True)

        # if an HDF timestamp was removed, drop that timestamp here too
        df2_filtered = df2_filtered[df2_filtered["Time_ISO"].isin(dfh_filtered["Time_ISO"])]

        csv_filtered[label] = df2_filtered

    return dfh_filtered, csv_filtered



# 1) screen & prepare
dfh_cleaned, csv_cleaned = screen_and_prepare()

sensor_datasets = {}
for label, df_csv in csv_cleaned.items():
    # 1) merge Picarro target
    df = pd.merge(
        df_csv,
        dfh_cleaned[["Time_ISO","co_ppb"]].rename(columns={"co_ppb":"target"}),
        on="Time_ISO",
        how="inner"
    )
    if df.empty:
        # no overlapping times → skip
        continue

    # 2) now that target exists, add your drift/time feature
    df["time_days"]   = (df["Time_ISO"] - start_date).dt.total_seconds() / 86400
    # df["time_days2"]  = df["time_days"] ** 2
    # df["time_days3"]  = df["time_days"] ** 3


    sensor_datasets[label] = df



#=============================================
# machin learning
#=============================================

from sklearn.preprocessing import StandardScaler

#============== init_fract + final_frac of interval is training, the rest is test ==============
def split_all_sensors(
    sensor_datasets: Dict[str, pd.DataFrame],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    feature_cols: List[str],
    init_frac: float = 0.35,
    final_frac: float = 0.35,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:

    splits = {}
    total_span = end_date - start_date
    t1 = start_date + total_span * init_frac
    t2 = start_date + total_span * (1 - final_frac)

    for label, df in sensor_datasets.items():
        # initial slice
        df_init = df[(df["Time_ISO"] >= start_date) & (df["Time_ISO"] < t1)].copy()
    
        # final slice
        df_final = df[(df["Time_ISO"] >= t2) & (df["Time_ISO"] < end_date)].copy()
        # middle slice = test
        df_test = df[(df["Time_ISO"] >= t1) & (df["Time_ISO"] < t2)].copy()

        # combine initial + final for training
        train_df = pd.concat([df_init, df_final], ignore_index=True)

        # drop any NaNs in features/target
        train_df.dropna(subset=feature_cols + ["target"], inplace=True)
        df_test .dropna(subset=feature_cols + ["target"], inplace=True)

        # extract arrays
        X_train = train_df[feature_cols].values
        y_train = train_df["target"].values
        X_test  = df_test [feature_cols].values
        y_test  = df_test ["target"].values

        splits[label] = (train_df, df_test, X_train, X_test, y_train, y_test)

    return splits


# arguments for linear regression models I tried 
feature_cols = ["temp","rh","co_diff","time_days" ]  # only this improved the results
# feature_cols = ["temp","rh","co_diff","o3-no2","co_lag1", "co_lag2", "we_lag1", "we_base", "ae_base","co_ewma","dow","we_ae_ratio"]  
# feature_cols = ["temp","rh","co_diff","o3-no2","co_lag1","ae_base","co_ewma"]  
# feature_cols = ["temp","rh","co_diff" ,"co_lag1","co_ewma","ae_base",]  
# feature_cols = ["temp","rh","co_diff","time_days", "rh*temp"]  
# feature_cols = ["temp","rh","co_diff","time_days"] #, "temp2", "temp3" ]  
# feature_cols = ["temp", "rh", "co_diff", "time_days","temp2", "temp3","rh*temp"]
# feature_cols = ["temp", "rh", "co_diff", "time_days","temp2", "temp3"]
# feature_cols = ["temp","rh", "co_diff"]  

splits = split_all_sensors(sensor_datasets, start_date, end_date, feature_cols)



scalers = {}
for label, (train_df, test_df, X_train, X_test, y_train, y_test) in splits.items():
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    scalers[label] = scaler
    
    # if you want to keep the scaled arrays alongside:
    splits[label] = (
        train_df, test_df,
        X_train_scaled, X_test_scaled,
        y_train, y_test
    )



# plot the predicted data + print the metrics
def run_lr_all_single(
    sensor_datasets,
    start_date,
    end_date,
    feature_cols,
    out_dir="figures"
):
    print(f"Total: from {start_date} to {end_date}")
    # prepare output directory
    os.makedirs(out_dir, exist_ok=True)

    # split datasets
    splits = split_all_sensors(sensor_datasets, start_date, end_date, feature_cols)

    # single plot sized to accommodate all series
    fig, ax = plt.subplots(figsize=(10, 4))

    # plot actual once (same for all sensors)
    first_label, (_, test_df, _, _, _, _) = next(iter(splits.items()))
    df_actual = test_df.copy()
    ax.plot(df_actual['Time_ISO'], df_actual['target'],color='black', label='Picarro')

    for label, (_, test_df, X_train, X_test, y_train, y_test) in splits.items():
        # pipeline: scale + linear regression
        pipe = make_pipeline(StandardScaler(), LinearRegression())
        pipe.fit(X_train, y_train)

        # predictions & metrics
        y_pred = pipe.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        r_raw, _ = pearsonr(y_test, y_pred)
        print(f"{label} → RMSE={rmse:.4f}, R²={r2:.4f}, Pearson²={(r_raw**2):.4f}")
        coefs = pipe.named_steps['linearregression'].coef_
        for feat, coef in zip(feature_cols, coefs):
            print(f"  {feat}: {coef:+.4f}")
        print(f"  Intercept: {pipe.named_steps['linearregression'].intercept_:+.4f}\n")

        # plot predicted only
        df_pred = test_df.copy()
        df_pred['predicted'] = y_pred
        ax.plot(df_pred['Time_ISO'], df_pred['predicted'], linestyle='--', label=f"{label} Predicted")

    ax.set_title("Actual vs Predicted for All Sensors", fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylabel('CO (ppb)')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=6)

    # save figure
    fname = f"lr_correction_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
    fpath = os.path.join(out_dir, fname)
    fig.savefig(fpath, dpi=300, bbox_inches='tight')
    print(f"Saved results to: {fpath}")
    plt.show()
    

# Computing the metrix in 2-weeks interval to find out the possible time of drift

def run_lr_by_intervals(
    sensor_datasets: Dict[str,pd.DataFrame],
    feature_cols: List[str],
    boundaries: List[Tuple[pd.Timestamp,pd.Timestamp]]
):
    """
    For each sensor, fit on the whole period then for each (start,end) boundary:
      • compute RMSE, R², Pearson², bias, normalized RMSE
      • print them.
    """
    # 1) train one model per sensor on its full train/test split
    models = {}
    for label, df in sensor_datasets.items():
        X = df[feature_cols].values
        y = df['target'].values
        pipe = make_pipeline(StandardScaler(), LinearRegression())
        pipe.fit(X, y)
        models[label] = pipe

    # 2) for each interval, compute metrics on the slice
    for i, (t0, t1) in enumerate(boundaries, 1):
        print(f"\n=== Interval {i}: {t0.date()} → {t1.date()} ===")
        for label, df in sensor_datasets.items():
            mask = (df['Time_ISO'] >= t0) & (df['Time_ISO'] < t1)
            df_i = df.loc[mask]
            if df_i.empty:
                continue

            X_i = df_i[feature_cols].values
            y_i = df_i['target'].values
            y_pred = models[label].predict(X_i)

            rmse = np.sqrt(mean_squared_error(y_i, y_pred))
            r2   = r2_score(y_i, y_pred)
            p    = pearsonr(y_i, y_pred)[0]**2
            bias = np.mean(y_pred - y_i)

            # normalized RMSE = RMSE / mean(actual)
            norm_rmse = rmse / np.mean(y_i)

            print(
                f"{label:8s} | RMSE={rmse:6.2f} ppb  | R²={r2:5.3f}  | "
                f"Pearson²={p:5.3f}  | bias={bias:6.2f} ppb  | "
                f"nRMSE={norm_rmse:5.3f}"
            )

# Defineing the two-week boundaries:
boundaries = [
    (start_date,   pd.Timestamp("2024-11-15", tz="UTC")),
    (pd.Timestamp("2024-11-15", tz="UTC"), pd.Timestamp("2024-12-01", tz="UTC")),
    (pd.Timestamp("2024-12-01", tz="UTC"), pd.Timestamp("2024-12-15", tz="UTC")),
    (pd.Timestamp("2024-12-15", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")),
    (pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-01-15", tz="UTC")),
    (pd.Timestamp("2025-01-15", tz="UTC"), pd.Timestamp("2025-02-01", tz="UTC")),
    (pd.Timestamp("2025-02-01", tz="UTC"), pd.Timestamp("2025-02-15", tz="UTC")),
    (pd.Timestamp("2025-02-15", tz="UTC"), pd.Timestamp("2025-03-01", tz="UTC")),
    (pd.Timestamp("2025-03-01", tz="UTC"), pd.Timestamp("2025-03-15", tz="UTC")),
    (pd.Timestamp("2025-03-15", tz="UTC"), pd.Timestamp("2025-04-01", tz="UTC")),
    (pd.Timestamp("2025-04-01", tz="UTC"), pd.Timestamp("2025-04-15", tz="UTC")),
    (pd.Timestamp("2025-04-15", tz="UTC"), pd.Timestamp("2025-05-01", tz="UTC")),
]


def run_lr_detailed(
    sensor_datasets: dict,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    feature_cols: list,
    boundaries: list[tuple[pd.Timestamp,pd.Timestamp]],
    out_dir="figures",
):
    # 1) one train/test split per sensor
    splits = split_all_sensors(sensor_datasets, start_date, end_date, feature_cols)
    
    for label, (train_df, test_df, X_train, X_test, y_train, y_test) in splits.items():
        print(f"\n=== Sensor: {label} ===")
        # train model
        pipe = make_pipeline(StandardScaler(), LinearRegression())
        pipe.fit(X_train, y_train)
        
        def _print(name: str, X: np.ndarray, y: np.ndarray, norm: bool=False):
            y_pred = pipe.predict(X)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2   = r2_score(y, y_pred)
            p2   = pearsonr(y, y_pred)[0]**2
            bias = np.mean(y_pred - y)
            line = f"{name:16s} | RMSE={rmse:6.2f} ppb | R²={r2:5.3f} | Pearson²={p2:5.3f} | bias={bias:6.2f} ppb"
            if norm:
                nrmse = rmse / np.mean(y)
                line += f" | nRMSE={nrmse:5.3f}"
            print(line)
        
        # overall metrics
        _print("Train period", X_train, y_train)
        _print("Test period",  X_test,  y_test)
        X_full = np.vstack([X_train, X_test])
        y_full = np.concatenate([y_train, y_test])
        _print(" Full period", X_full, y_full)
        
        # two-week slices
        for t0, t1 in boundaries:
            df_i = sensor_datasets[label]
            mask = (df_i["Time_ISO"] >= t0) & (df_i["Time_ISO"] < t1)
            Xi, yi = df_i.loc[mask, feature_cols].values, df_i.loc[mask, "target"].values
            if yi.size:
                name = f"{t0.strftime('%Y-%m-%d')}→{t1.strftime('%Y-%m-%d')}"
                _print(name, Xi, yi, norm=True)
    
    # 2) plot as in your original run_lr_all_single
    run_lr_all_single(sensor_datasets, start_date, end_date, feature_cols, out_dir)

# call it:
run_lr_detailed(sensor_datasets, start_date, end_date, feature_cols, boundaries)


def run_final_evaluation(
    dfh_cleaned,             # DataFrame with ["Time_ISO","co_ppb"]
    start_date,              # pd.Timestamp (UTC-aware)
    end_date,                # pd.Timestamp (UTC-aware)
    time_interval,           # int minutes for resampling (e.g. 2)
    boundaries,              # list of (t0,t1) two-week pd.Timestamp tuples
    train_fraction=0.7       # fraction of rows for “train” split
):
 

    def _print(name, y_true, y_pred, norm=False):
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        p2   = pearsonr(y_true, y_pred)[0]**2
        bias = np.mean(y_pred - y_true)
        line = f"{name:15s}| RMSE={rmse:6.2f} ppb | R²={r2:5.3f} | Pearson²={p2:5.3f} | bias={bias:6.2f} ppb"
        if norm:
            line += f" | nRMSE={rmse/np.mean(y_true):5.3f}"
        print(line)

    for path in sorted(glob.glob("Final-*.csv")):
        label = os.path.basename(path).replace(".csv","")
        # 1) load + crop
        df = pd.read_csv(path, usecols=["timestamp","co"])
        df["Time_ISO"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df[(df["Time_ISO"] >= start_date) & (df["Time_ISO"] < end_date)]
        if df.empty:
            continue
        # 2) resample to 2 min
        df = (
            df.set_index("Time_ISO")[["co"]]
              .resample(f"{time_interval}min")
              .mean()
              .reset_index()
        )
        # 3) merge & drop any NaNs
        df = (
            pd.merge(
                df,
                dfh_cleaned[["Time_ISO","co_ppb"]]
                           .rename(columns={"co_ppb":"target"}),
                on="Time_ISO",
                how="inner"
            )
            .sort_values("Time_ISO")
            .dropna(subset=["co","target"])
            .reset_index(drop=True)
        )
        if df.empty:
            continue

        # 4) split Train/Test/Full
        n       = len(df)
        n_train = int(n * train_fraction)
        df_train = df.iloc[:n_train]
        df_test  = df.iloc[n_train:]
        y_train  = df_train["target"].values
        y_test   = df_test["target"].values
        y_full   = df["target"].values
        y_pred_train = df_train["co"].values
        y_pred_test  = df_test ["co"].values
        y_pred_full  = df    ["co"].values

        print(f"\n=== Sensor: {label} ===")
        _print("Initial 70%   ", y_train,    y_pred_train)
        _print("Final 30%    ", y_test,     y_pred_test)
        _print("Full period    ", y_full,     y_pred_full)

        # 5) two-week slices
        for t0, t1 in boundaries:
            sub = df[(df["Time_ISO"] >= t0) & (df["Time_ISO"] < t1)]
            if len(sub) < 2:
                continue
            _print(f"{t0.date()}→{t1.date()}",
                   sub["target"].values,
                   sub["co"].values,
                   norm=True)


# run_final_evaluation(dfh_cleaned, start_date, end_date, time_interval, boundaries, train_fraction=0.7)


def plot_residuals_over_time(
    sensor_datasets,
    start_date,
    end_date,
    out_dir="figures"
):
    """
    Plots residuals (actual – predicted) over time for each Raw-*.csv sensor,
    using the global feature_cols list.
    Residual unit: ppb
    Time axis: UTC-aware timestamps
    """
    os.makedirs(out_dir, exist_ok=True)
    splits = split_all_sensors(sensor_datasets, start_date, end_date, feature_cols)
    n = len(splits)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3), sharex=False)
    axes = axes.flatten()

    for ax, (label, (train_df, test_df, X_train, X_test, y_train, y_test)) in zip(axes, splits.items()):
        pipe = make_pipeline(StandardScaler(), LinearRegression())
        pipe.fit(X_train, y_train)

        y_pred_train = pipe.predict(X_train)
        y_pred_test  = pipe.predict(X_test)
        res_train = y_train - y_pred_train
        res_test  = y_test  - y_pred_test

        ax.plot(train_df['Time_ISO'], res_train, marker='.', linestyle='-', color='C0')
        ax.plot(test_df ['Time_ISO'], res_test,  marker='.', linestyle='--', color='C0')

        ax.axhline(0, linestyle='--', color='C0')
        ax.set_title(label, fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=6)
        ax.set_ylabel('Residual (ppb)')

    for i in range(n, len(axes)):
        fig.delaxes(axes[i])

    fig.tight_layout()
    fname = f"residuals_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
    fpath = os.path.join(out_dir, fname)
    fig.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.show()

# call it:
# plot_residuals_over_time(sensor_datasets, start_date, end_date)


def plot_residuals_with_temp_heatmap(
    sensor_datasets,
    start_date,
    end_date,
    out_dir="figures"
):
    """
    Plots residuals over time with temperature heatmap for each sensor,
    using global feature_cols and var_t.
    """

    os.makedirs(out_dir, exist_ok=True)
    splits = split_all_sensors(sensor_datasets, start_date, end_date, feature_cols)
    n = len(splits)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*3), sharex=False)
    axes = axes.flatten()

    for ax, (label, (train_df, test_df, X_train, X_test, y_train, y_test)) in zip(axes, splits.items()):
        # train model
        pipe = make_pipeline(StandardScaler(), LinearRegression())
        pipe.fit(X_train, y_train)

        # assemble full dataset for plotting
        full_df = pd.concat([train_df, test_df]).reset_index(drop=True)
        X_full  = full_df[feature_cols].values
        y_full  = full_df["target"].values
        temps   = full_df[var_t].values

        # compute predictions and residuals
        preds   = pipe.predict(X_full)
        resids  = y_full - preds

        # temperature vs. residual metrics
        r_temp       = np.corrcoef(temps, resids)[0,1]
        slope_temp, _ = np.polyfit(temps, resids, 1)
        print(f"{label} → Temp→residual: r={r_temp:.2f}, slope={slope_temp:.2f} ppb/°C")

        # scatter over time, colored by temperature
        sc = ax.scatter(full_df["Time_ISO"], resids, c=temps, cmap="viridis", s=10)
        ax.axhline(0, linestyle="--", color="gray", linewidth=0.5)

        ax.set_title(label, fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=6)
        ax.set_ylabel("Residual (ppb)")

    # remove any empty subplots
    for i in range(n, len(axes)):
        fig.delaxes(axes[i])

    fig.tight_layout()
    cbar = fig.colorbar(sc, ax=axes.tolist(), label="Temperature (°C)")
    fname = f"residuals_temp_heatmap_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")
    plt.show()

def plot_residuals_vs_temp(
    sensor_datasets,
    start_date,
    end_date,
    out_dir="figures"
):
    """
    For each sensor:
      • fits LinearRegression on train→test
      • computes:
         – r_temp, slope_temp between temp and residual
         – r_time,  slope_time  between time_days and residual
      • prints those
      • scatter‐plots residual vs temp
    """

    os.makedirs(out_dir, exist_ok=True)

    splits = split_all_sensors(sensor_datasets, start_date, end_date, feature_cols)
    n = len(splits)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4))
    axes = axes.flatten()

    for ax, (label, (train_df, test_df, X_train, X_test, y_train, y_test)) \
            in zip(axes, splits.items()):
        # 1) train model
        pipe = make_pipeline(StandardScaler(), LinearRegression())
        pipe.fit(X_train, y_train)

        # 2) assemble full_df
        full_df = pd.concat([train_df, test_df]).reset_index(drop=True)
        times   = full_df["time_days"].values
        temps   = full_df[var_t].values
        y_full  = full_df["target"].values
        preds   = pipe.predict(full_df[feature_cols].values)
        resids  = y_full - preds

        # 3) metrics
        r_temp, _    = pearsonr(temps, resids)
        slope_temp, _= np.polyfit(temps, resids, 1)
        r_time, _    = pearsonr(times, resids)
        slope_time,_ = np.polyfit(times, resids, 1)
        print(f"{label}: Temp→resid r={r_temp:.2f}, slope={slope_temp:.2f} ppb/°C; "
              f"Time→resid r={r_time:.2f}, slope={slope_time:.2f} ppb/day")

        # 4) plot
        ax.scatter(temps, resids, s=10, color='C0')
        ax.axhline(0, linestyle='--', color='C0')
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Residual (ppb)")

    # drop empty panels
    for i in range(n, len(axes)):
        fig.delaxes(axes[i])

    fig.tight_layout()
    out = os.path.join(out_dir,
        f"residuals_vs_temp_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
    )
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()

# call them 
plot_residuals_vs_temp(sensor_datasets, start_date, end_date)
plot_residuals_with_temp_heatmap(sensor_datasets, start_date, end_date)












