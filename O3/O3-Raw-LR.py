# Linear regression on raw O3 
# quantaq raw csv files name: "Raw-*.csv"  where * is the sensor ID




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
from functools import reduce
import math

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

warnings.simplefilter("ignore", OptimizeWarning)

# === SETTINGS ===
var           = "O3 CONC"
o3_we         = "o3_we"
o3_ae         = "o3_ae"
no2_we         = "no2_we"
no2_ae         = "no2_ae"
var_t         = "temp"
var_no2         = "no2"
var_rh        = "rh"
co            = "co_diff"

start_date    = "2024-10-30"
end_date      = "2025-02-01"
time_interval = 2



start_date = pd.to_datetime(start_date).tz_localize("UTC")
end_date   = pd.to_datetime(end_date).tz_localize("UTC") + pd.Timedelta(days=1)


def find_files_in_date_range(base_dirs, start_date, end_date):
    selected = []
    for base in base_dirs:
        for f in glob.glob(os.path.join(base, "**", "*.hdf"), recursive=True):
            try:
                d = pd.to_datetime(os.path.basename(f)[:10], errors="coerce")
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


def remove_outliers_rolling_zscore(df, var, window_size=30, z_thresh=1.0):
    m = df[var].rolling(window=window_size, center=True, min_periods=1).mean()
    s = df[var].rolling(window=window_size, center=True, min_periods=1).std()
    z = (df[var] - m) / s
    return df[z.abs() <= z_thresh].reset_index(drop=True)


def time_average(df, var, interval):
    df = df.copy()
    df["Time_ISO"] = pd.to_datetime(df["Time_ISO"])
    df = df.set_index("Time_ISO")
    return df[[var]].resample(f"{interval}min").mean().reset_index()


def force_utc(df, col="Time_ISO"):
    df[col] = pd.to_datetime(df[col])
    if df[col].dt.tz is None:
        df[col] = df[col].dt.tz_localize("UTC")
    else:
        df[col] = df[col].dt.tz_convert("UTC")
    return df


# === CALIBRATED TARGET FROM HDF ===
base_folders = ["2024", "2025"]
hdf_files    = find_files_in_date_range(base_folders, start_date, end_date)

df_raw      = read_variable_from_files(hdf_files, var, start_date, end_date)
df_clean    = remove_outliers_rolling_zscore(df_raw, var)
df_averaged = time_average(df_clean, var, time_interval)
df_cal      = df_averaged.rename(columns={var: "target"})
df_cal      = force_utc(df_cal, "Time_ISO")


# === BUILD RAW SENSOR DATASETS ===
sensor_datasets = {}
csv_files = glob.glob("Raw-*.csv")
var_list = [var_t, var_rh, o3_ae, o3_we, no2_ae, no2_we, co]

for f in csv_files:
    label = os.path.splitext(os.path.basename(f))[0]
    df = pd.read_csv(f, usecols=["timestamp"] + var_list)
    df["Time_ISO"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[(df["Time_ISO"] >= start_date) & (df["Time_ISO"] < end_date)]

    avg_dfs = [time_average(df[["Time_ISO", v]], v, time_interval) for v in var_list]
    df_avg  = reduce(lambda L, R: pd.merge(L, R, on="Time_ISO"), avg_dfs)

    df_avg["o3"]  = df_avg[o3_we]  - df_avg[o3_ae]
    df_avg["no2"] = df_avg[no2_we] - df_avg[no2_ae]
    
    df_full = pd.merge(df_avg, df_cal[["Time_ISO", "target"]], on="Time_ISO", how="inner")
    sensor_datasets[label] = df_full



def split_all_sensors(sensor_datasets, start_date, end_date, feature_cols, train_ratio=0.7):
    splits = {}
    total_days = (end_date - start_date).days
    train_days = int(total_days * train_ratio)
    split_date = start_date + pd.Timedelta(days=train_days)
  
    for label, df in sensor_datasets.items():
        train_df = df[(df["Time_ISO"] >= start_date) & (df["Time_ISO"] < split_date)]
        test_df  = df[(df["Time_ISO"] >= split_date) & (df["Time_ISO"] < end_date)]
        train_df = train_df.dropna(subset=feature_cols + ["target"])
        test_df  = test_df.dropna(subset=feature_cols + ["target"])
        X_train, y_train = train_df[feature_cols].values, train_df["target"].values
        X_test,  y_test  = test_df[feature_cols].values,  test_df["target"].values
        splits[label] = (train_df, test_df, X_train, y_train, X_test, y_test)
    return splits



def run_rf_all(out_dir="figures"):

    os.makedirs(out_dir, exist_ok=True)

    feature_cols = ["temp", "rh", "o3", "no2"]
    splits = split_all_sensors(sensor_datasets, start_date, end_date, feature_cols)
    n = len(splits)
    ncols, nrows = math.ceil(n/2), 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*7, nrows*4), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax, (label, (_, test_df, X_train, y_train, X_test, y_test)) in zip(axes, splits.items()):
        # Train RF
        pipe = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42))
        pipe.fit(X_train, y_train)

        # Predict & metrics
        y_pred = pipe.predict(X_test)
        rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
        r2     = r2_score(y_test, y_pred)
        r_raw, _ = pearsonr(y_test, y_pred)
        r2p    = r_raw**2
        print(f"{label} → RF: RMSE={rmse:.4f}, R²={r2:.4f}, pearson²={r2p:.4f}")

        # Feature importances
        rf = pipe.named_steps["randomforestregressor"]
        importances = pd.Series(rf.feature_importances_, index=feature_cols)
        importances = importances.sort_values(ascending=False)
        for feat, imp in importances.items():
            print(f"{feat:<6} {imp:.6f}")
        print()  # blank line

        # Surrogate linear fit to RF predictions
        y_rf_train = pipe.predict(X_train)
        sur = Ridge().fit(X_train, y_rf_train)
        terms = [f"{coef:+.4f}·{f}" for coef, f in zip(sur.coef_, feature_cols)]
        eqn = f"y = {sur.intercept_:.4f} " + " ".join(terms)
        print(eqn, "\n")

        # Plot actual vs predicted
        df_plot = test_df.copy()
        df_plot["predicted"] = y_pred
        ax.plot(df_plot["Time_ISO"], df_plot["target"], label="Actual")
        ax.plot(df_plot["Time_ISO"], df_plot["predicted"], label="Predicted")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=6)

    # Remove unused subplots
    for i in range(n, len(axes)):
        fig.delaxes(axes[i])

    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    fig.tight_layout()

    fname = f"rf_timeseries_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")
    print(f"Saved RF timeseries plot to {os.path.join(out_dir, fname)}")
    plt.show()


def plot_residuals_all(out_dir="figures"):
   
    os.makedirs(out_dir, exist_ok=True)

    feature_cols = ["temp", "rh", "o3", "no2"]
    splits = split_all_sensors(sensor_datasets, start_date, end_date, feature_cols)
    n = len(splits)
    ncols, nrows = math.ceil(n/2), 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4), sharex=True)
    axes = axes.flatten()

    for ax, (label, (_, test_df, X_train, y_train, X_test, y_test)) in zip(axes, splits.items()):
        pipe = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42))
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        residuals = y_test - y_pred

        df_res = test_df.copy()
        df_res['residual'] = residuals
        ax.plot(df_res['Time_ISO'], df_res['residual'], marker='.', linestyle='none')
        ax.axhline(0, color='gray', linewidth=0.8)
        ax.set_title(label)
        ax.set_ylabel('Residual (obs – pred)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    for i in range(n, len(axes)):
        fig.delaxes(axes[i])

    fig.tight_layout()
    fname = f"rf_residuals_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches='tight')
    print(f"Saved RF residuals plot to {os.path.join(out_dir, fname)}")
    plt.show()


def run_lr_all(out_dir="figures"):
  
    os.makedirs(out_dir, exist_ok=True)

    feature_cols = ["temp", "rh", "o3", "no2"]
    splits = split_all_sensors(sensor_datasets, start_date, end_date, feature_cols)
    n = len(splits)
    ncols, nrows = math.ceil(n/2), 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*6, nrows*4), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax, (label, (_, test_df, X_train, y_train, X_test, y_test)) in zip(axes, splits.items()):
        # train linear regression
        pipe = make_pipeline(StandardScaler(), LinearRegression())
        pipe.fit(X_train, y_train)

        # predict & metrics
        y_pred = pipe.predict(X_test)
        rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
        r2     = r2_score(y_test, y_pred)
        r_raw, _ = pearsonr(y_test, y_pred)
        r2p    = r_raw**2
        print(f"{label} → LR: RMSE={rmse:.4f}, R²={r2:.4f}, pearson²={r2p:.4f}")

        # coefficients
        lr = pipe.named_steps["linearregression"]
        coefs = pd.Series(lr.coef_, index=feature_cols).sort_values(key=abs, ascending=False)
        for feat, coef in coefs.items():
            print(f"{feat:<6} {coef:+.4f}")
        print(f"Intercept {lr.intercept_:+.4f}\n")

        # plot actual vs predicted
        df_plot = test_df.copy()
        df_plot['predicted'] = y_pred
        ax.plot(df_plot['Time_ISO'], df_plot['target'], label='Actual')
        ax.plot(df_plot['Time_ISO'], df_plot['predicted'], label='Predicted')
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=6)

    # remove unused axes
    for i in range(n, len(axes)):
        fig.delaxes(axes[i])

    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    fig.tight_layout()

    fname = f"lr_timeseries_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches='tight')
    print(f"Saved LR timeseries plot to {os.path.join(out_dir, fname)}")
    plt.show()


def run_lr_all_filtered(min_val=0, max_val=50, out_dir="figures"):
    """
    Train Linear Regression on all sensors, then for each sensor:
    - Train on full training set
    - Predict on full test set
    - Filter to predictions in [min_val, max_val]
    - Print RMSE, R², Pearson², coefficients, and intercept on filtered data
    - Plot Actual vs Predicted for filtered predictions in a 2-row grid
    - Save the combined figure
    """
    os.makedirs(out_dir, exist_ok=True)

    feature_cols = ["temp", "rh", "o3", "no2"]
    splits = split_all_sensors(sensor_datasets, start_date, end_date, feature_cols)
    n = len(splits)
    ncols, nrows = math.ceil(n / 2), 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*6, nrows*4), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax, (label, (_, test_df, X_train, y_train, X_test, y_test)) in zip(axes, splits.items()):
        # train linear regression
        pipe = make_pipeline(StandardScaler(), LinearRegression())
        pipe.fit(X_train, y_train)

        # predict on full test set
        y_pred = pipe.predict(X_test)

        # align in DataFrame
        df_test = test_df.reset_index(drop=True).copy()
        df_test['predicted'] = y_pred

        # filter by predicted values
        mask = (df_test['predicted'] >= min_val) & (df_test['predicted'] <= max_val)
        if not mask.any():
            print(f"{label} → no predictions in [{min_val}, {max_val}], skipping.")
            fig.delaxes(ax)
            continue

        df_plot = df_test.loc[mask]
        y_true_f = df_plot['target'].values
        y_pred_f = df_plot['predicted'].values

        # compute and print metrics
        rmse = np.sqrt(mean_squared_error(y_true_f, y_pred_f))
        r2   = r2_score(y_true_f, y_pred_f)
        r_raw, _ = pearsonr(y_true_f, y_pred_f)
        r2p = r_raw**2
        print(f"{label} → LR pred-filtered [{min_val}, {max_val}]: RMSE={rmse:.4f}, R²={r2:.4f}, pearson²={r2p:.4f}")

        # print coefficients & intercept
        lr = pipe.named_steps['linearregression']
        coefs = pd.Series(lr.coef_, index=feature_cols).sort_values(key=abs, ascending=False)
        print("Coefficients:")
        for feat, coef in coefs.items():
            print(f"{feat:<6} {coef:+.4f}")
        print(f"Intercept {lr.intercept_:+.4f}\n")

        # plot filtered data
        ax.plot(df_plot['Time_ISO'], df_plot['target'], label='Actual')
        ax.plot(df_plot['Time_ISO'], df_plot['predicted'], label='Predicted', color="grey")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=6)

    # remove unused axes
    for i in range(n, len(axes)):
        if axes[i] in fig.axes:
            fig.delaxes(axes[i])

    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    fig.tight_layout()

    fname = f"lr_timeseries_pred-filtered_{min_val}_{max_val}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches='tight')
    print(f"Saved pred-filtered LR plot to {os.path.join(out_dir, fname)}")
    plt.show()



def run_rf_all_filtered(min_val=0, max_val=50, out_dir="figures"):
    """
    Train Random Forest on all sensors, then for each sensor:
    - Predict on the full test set
    - Filter to predictions in [min_val, max_val]
    - Print RMSE, R², Pearson², feature importances, and surrogate equation
    - Plot Actual vs Predicted for filtered predictions in a 2-row grid
    - Save the combined figure
    """
    os.makedirs(out_dir, exist_ok=True)

    feature_cols = ["temp", "rh", "o3", "no2"]
    splits = split_all_sensors(sensor_datasets, start_date, end_date, feature_cols)
    n = len(splits)
    ncols, nrows = math.ceil(n / 2), 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*6, nrows*4), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax, (label, (_, test_df, X_train, y_train, X_test, y_test)) in zip(axes, splits.items()):
        # train Random Forest
        pipe = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42))
        pipe.fit(X_train, y_train)

        # predict on full test set
        y_pred = pipe.predict(X_test)

        # align in DataFrame
        df_test = test_df.reset_index(drop=True).copy()
        df_test['predicted'] = y_pred

        # filter by predicted values
        mask = (df_test['predicted'] >= min_val) & (df_test['predicted'] <= max_val)
        if not mask.any():
            print(f"{label} → no predictions in [{min_val}, {max_val}], skipping.")
            fig.delaxes(ax)
            continue

        df_plot = df_test.loc[mask]
        y_true_f = df_plot['target'].values
        y_pred_f = df_plot['predicted'].values

        # compute and print metrics
        rmse = np.sqrt(mean_squared_error(y_true_f, y_pred_f))
        r2   = r2_score(y_true_f, y_pred_f)
        r_raw, _ = pearsonr(y_true_f, y_pred_f)
        r2p = r_raw**2
        print(f"{label} → RF pred-filtered [{min_val}, {max_val}]: RMSE={rmse:.4f}, R²={r2:.4f}, pearson²={r2p:.4f}")

        # print feature importances
        rf = pipe.named_steps['randomforestregressor']
        importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
        print("Feature importances:")
        for feat, imp in importances.items():
            print(f"{feat:<6} {imp:.6f}")

        # surrogate linear equation
        y_rf_train = pipe.predict(X_train)
        sur = Ridge().fit(X_train, y_rf_train)
        terms = [f"{coef:+.4f}·{f}" for coef, f in zip(sur.coef_, feature_cols)]
        eqn = f"y = {sur.intercept_:+.4f} " + " ".join(terms)
        print(eqn, "\n")

        # plot filtered data
        ax.plot(df_plot['Time_ISO'], df_plot['target'], label='Actual')
        ax.plot(df_plot['Time_ISO'], df_plot['predicted'], label='Predicted')
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=6)

    # remove unused axes
    for i in range(n, len(axes)):
        if axes[i] in fig.axes:
            fig.delaxes(axes[i])

    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    fig.tight_layout()

    fname = f"rf_timeseries_pred-filtered_{min_val}_{max_val}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches='tight')
    print(f"Saved pred-filtered RF plot to {os.path.join(out_dir, fname)}")
    plt.show()


# run_rf_all_filtered(min_val=0, max_val=50, out_dir="figures")


def plot_rf_residuals_filtered(min_val=0, max_val=50, out_dir="figures"):
    """
    Generate residuals vs time for Random Forest predictions filtered by prediction bounds.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    feature_cols = ["temp", "rh", "o3", "no2"]
    splits = split_all_sensors(sensor_datasets, start_date, end_date, feature_cols)
    n = len(splits)
    ncols, nrows = math.ceil(n/2), 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*6, nrows*4), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax, (label, (_, test_df, X_train, y_train, X_test, y_test)) in zip(axes, splits.items()):
        pipe = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42))
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        df_test = test_df.reset_index(drop=True).copy()
        df_test['predicted'] = y_pred
        mask = (df_test['predicted'] >= min_val) & (df_test['predicted'] <= max_val)
        if not mask.any():
            fig.delaxes(ax)
            continue

        df_res = df_test.loc[mask].copy()
        df_res['residual'] = df_res['target'] - df_res['predicted']

        ax.scatter(df_res['Time_ISO'], df_res['residual'], s=5)
        ax.axhline(0, linestyle='--', linewidth=1)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel('Residual')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=6)

    for i in range(n, len(axes)):
        fig.delaxes(axes[i])

    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    fig.tight_layout()
    fname = f"rf_residuals_pred-filtered_{min_val}_{max_val}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches='tight')
    plt.show()


def plot_lr_residuals_filtered(min_val=0, max_val=50, out_dir="figures"):
    """
    Generate residuals vs time for Linear Regression predictions filtered by prediction bounds.
    """
    os.makedirs(out_dir, exist_ok=True)

    feature_cols = ["temp", "rh", "o3", "no2"]
    splits = split_all_sensors(sensor_datasets, start_date, end_date, feature_cols)
    n = len(splits)
    ncols, nrows = math.ceil(n/2), 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*6, nrows*4), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax, (label, (_, test_df, X_train, y_train, X_test, y_test)) in zip(axes, splits.items()):
        pipe = make_pipeline(StandardScaler(), LinearRegression())
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        df_test = test_df.reset_index(drop=True).copy()
        df_test['predicted'] = y_pred
        mask = (df_test['predicted'] >= min_val) & (df_test['predicted'] <= max_val)
        if not mask.any():
            fig.delaxes(ax)
            continue

        df_res = df_test.loc[mask].copy()
        df_res['residual'] = df_res['target'] - df_res['predicted']

        ax.scatter(df_res['Time_ISO'], df_res['residual'], s=5)
        ax.axhline(0, linestyle='--', linewidth=1)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel('Residual')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=6)

    for i in range(n, len(axes)):
        fig.delaxes(axes[i])

    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    fig.tight_layout()
    fname = f"lr_residuals_pred-filtered_{min_val}_{max_val}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches='tight')
    plt.show()


def plot_lr_residuals_histogram(
    start: str,
    end: str,
    interval: int,
    tz: str = None,
    source_list=None,
    figsize=(12, 8),
    bins: int = 50,
    color: str = "tab:green",
    out_path: str = "lr_residuals_hist.png"
):
    """
    For each sensor, train a LinearRegression on training interval,
    predict on the test interval, compute residuals (obs - pred),
    and plot a 2×3 grid of residual histograms, saving the figure.
    """
    # 1) Prepare data splits
    feature_cols = ["temp", "rh", "o3", "no2"]
    splits = split_all_sensors(sensor_datasets, start_date, end_date, feature_cols)
    source_list = source_list or splits.keys()
    
    # 2) Setup subplots
    n = len(source_list)
    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=False)
    axes = axes.flatten()
    fig.suptitle(
        f"LR Prediction Residuals ({pd.to_datetime(start).date()} → {pd.to_datetime(end).date()})",
        fontsize=16
    )
    
    # 3) Loop over sensors and plot
    for ax, label in zip(axes, source_list):
        train_df, test_df, X_train, y_train, X_test, y_test = splits[label]
        # train linear model
        pipe = make_pipeline(StandardScaler(), LinearRegression())
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        # compute residuals
        residuals = y_test - y_pred
        
        # plot histogram
        ax.hist(residuals, bins=bins, color=color, alpha=0.8)
        ax.set_title(label)
        ax.set_xlabel("Residual (obs − pred) [ppb]")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", labelbottom=True)
    
    # 4) hide unused axes
    for ax in axes[n:]:
        ax.axis("off")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # always save
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()




def plot_lr_residuals_histogram(
    start: str,
    end: str,
    interval: int,
    tz: str = None,
    source_list=None,
    figsize=(12, 8),
    bins: int = 50,
    color: str = "tab:green",
    out_path: str = "lr_residuals_hist.png"
):
    """
    For each sensor, train a LinearRegression on training interval,
    predict on the test interval, compute residuals (obs - pred),
    and plot a 2×3 grid of residual histograms (with mean and std), saving the figure.
    """
    feature_cols = ["temp", "rh", "o3", "no2"]
    splits = split_all_sensors(sensor_datasets, start_date, end_date, feature_cols)
    source_list = source_list or list(splits.keys())

    # Setup subplots
    n = len(source_list)
    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=False)
    axes = axes.flatten()
    fig.suptitle(
        f"LR Prediction Residuals ({pd.to_datetime(start).date()} → {pd.to_datetime(end).date()})",
        fontsize=16
    )

    for ax, label in zip(axes, source_list):
        train_df, test_df, X_train, y_train, X_test, y_test = splits[label]
        pipe = make_pipeline(StandardScaler(), LinearRegression())
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        residuals = y_test - y_pred

        # Plot histogram
        ax.hist(residuals, bins=bins, color=color, alpha=0.8)
        ax.set_title(label)
        ax.set_xlabel("Residual (obs − pred) [ppb]")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", labelbottom=True)

        # Compute and display mean and standard deviation
        mu = residuals.mean()
        sigma = residuals.std()
        ax.axvline(mu, color='black', linestyle='--', linewidth=1)
        ax.axvline(mu + sigma, color='red', linestyle=':', linewidth=1)
        ax.axvline(mu - sigma, color='red', linestyle=':', linewidth=1)
        ax.text(
            0.95, 0.95,
            f"μ = {mu:.2f}\nσ = {sigma:.2f}",
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=9,
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
        )

    # Hide unused axes
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()



def plot_rf_residuals_histogram(
    start: str,
    end: str,
    interval: int,
    tz: str = None,
    source_list=None,
    figsize=(12, 8),
    bins: int = 50,
    color: str = "tab:blue",
    out_path: str = "rf_residuals_hist.png"
):
    """
    For each sensor, train a RandomForestRegressor on training interval,
    predict on the test interval, compute residuals (obs - pred),
    and plot a 2×3 grid of residual histograms (with mean and std), saving the figure.
    """
    feature_cols = ["temp", "rh", "o3", "no2"]
    splits = split_all_sensors(sensor_datasets, start_date, end_date, feature_cols)
    source_list = source_list or list(splits.keys())

    # Setup subplots
    n = len(source_list)
    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=False)
    axes = axes.flatten()
    fig.suptitle(
        f"RF Prediction Residuals ({pd.to_datetime(start).date()} → {pd.to_datetime(end).date()})",
        fontsize=16
    )

    for ax, label in zip(axes, source_list):
        train_df, test_df, X_train, y_train, X_test, y_test = splits[label]
        pipe = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42))
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        residuals = y_test - y_pred

        # Plot histogram
        ax.hist(residuals, bins=bins, color=color, alpha=0.8)
        ax.set_title(label)
        ax.set_xlabel("Residual (obs − pred) [ppb]")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", labelbottom=True)

        # Compute and display mean and standard deviation
        mu = residuals.mean()
        sigma = residuals.std()
        ax.axvline(mu, color='black', linestyle='--', linewidth=1)
        ax.axvline(mu + sigma, color='red', linestyle=':', linewidth=1)
        ax.axvline(mu - sigma, color='red', linestyle=':', linewidth=1)
        ax.text(
            0.95, 0.95,
            f"μ = {mu:.2f}\nσ = {sigma:.2f}",
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=9,
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
        )

    # Hide unused axes
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()





def plot_lr_residuals_filtered_histogram(
    min_val=0,
    max_val=50,
    source_list=None,
    figsize=(12, 8),
    bins=50,
    color="tab:green",
    out_path="lr_residuals_filtered_hist.png"
):
    """
    For each sensor:
    - Train LinearRegression on training interval
    - Predict on test interval
    - Filter predictions to [min_val, max_val]
    - Compute residuals (obs - pred)
    - Plot a 2×3 grid of residual histograms (with mean and std), saving the figure
    """
    feature_cols = ["temp", "rh", "o3", "no2"]
    splits = split_all_sensors(sensor_datasets, start_date, end_date, feature_cols)
    sources = source_list or list(splits.keys())
    n = len(sources)
    ncols, nrows = 3, math.ceil(n / 3)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=False)
    axes = axes.flatten()
    fig.suptitle(
        f"LR Filtered Residuals [{min_val}, {max_val}] ({start_date.date()} → {end_date.date()})",
        fontsize=16
    )

    for ax, label in zip(axes, sources):
        train_df, test_df, X_train, y_train, X_test, y_test = splits[label]
        pipe = make_pipeline(StandardScaler(), LinearRegression())
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        df = test_df.reset_index(drop=True).copy()
        df["predicted"] = y_pred
        mask = (df["predicted"] >= min_val) & (df["predicted"] <= max_val)
        if not mask.any():
            ax.axis("off")
            continue

        residuals = df.loc[mask, "target"].values - df.loc[mask, "predicted"].values

        ax.hist(residuals, bins=bins, color=color, alpha=0.8)
        ax.set_title(label)
        ax.set_xlabel("Residual (obs − pred)")
        ax.set_ylabel("Count")

        mu, sigma = residuals.mean(), residuals.std()
        ax.axvline(mu, color="black", linestyle="--", linewidth=1)
        ax.axvline(mu + sigma, color="red", linestyle=":", linewidth=1)
        ax.axvline(mu - sigma, color="red", linestyle=":", linewidth=1)
        ax.text(
            0.95, 0.95,
            f"μ={mu:.2f}\nσ={sigma:.2f}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
        )

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


# plot_lr_residuals_filtered(min_val=0, max_val=50)
# plot_rf_residuals_histogram("2025-01-15","2025-02-01", 2)
# plot_lr_residuals_filtered_histogram()

#===== new

def run_lr_all(out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)

    feature_cols = ["temp", "rh", "o3", "no2"]
    splits = split_all_sensors(sensor_datasets, start_date, end_date, feature_cols)
    n = len(splits)
    ncols, nrows = math.ceil(n/2), 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*3), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax, (label, (_, test_df, X_train, y_train, X_test, y_test)) in zip(axes, splits.items()):
        # train linear regression
        pipe = make_pipeline(StandardScaler(), LinearRegression())
        pipe.fit(X_train, y_train)

        # predict & metrics
        y_pred = pipe.predict(X_test)
        rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
        r2     = r2_score(y_test, y_pred)
        r_raw, _ = pearsonr(y_test, y_pred)
        r2p    = r_raw**2

        # plot actual vs predicted
        df_plot = test_df.copy()
        df_plot['predicted'] = y_pred
        ax.plot(df_plot['Time_ISO'], df_plot['target'], label='Actual')
        ax.plot(df_plot['Time_ISO'], df_plot['predicted'], label='Predicted')
        ax.set_title(label, fontsize=10)
        ax.set_ylabel('O3 (ppb)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=6)

        # annotate metrics
        ax.text(
            0.05, 0.95,
            f"$r_{{pearson}}^2$ = {r2p:.2f}\n"
            f"$R^2$ = {r2:.2f}\n"
            
            f"RMSE = {rmse:.2f}",
            transform=ax.transAxes,
            va='top', ha='left', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='none', edgecolor='black')

        )

        ax.legend(fontsize=8)

    # remove unused axes
    for i in range(n, len(axes)):
        fig.delaxes(axes[i])

    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    fig.tight_layout()

    fname = f"lr_timeseries_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches='tight')
    plt.show()


def run_lr_all_filtered(min_val=0, max_val=50, out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)

    feature_cols = ["temp", "rh", "o3", "no2"]
    splits = split_all_sensors(sensor_datasets, start_date, end_date, feature_cols)
    n = len(splits)
    ncols, nrows = math.ceil(n/2), 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*3), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax, (label, (_, test_df, X_train, y_train, X_test, y_test)) in zip(axes, splits.items()):
        # train linear regression
        pipe = make_pipeline(StandardScaler(), LinearRegression())
        pipe.fit(X_train, y_train)

        # predict on full test set
        y_pred = pipe.predict(X_test)
        df_test = test_df.reset_index(drop=True).copy()
        df_test['predicted'] = y_pred

        # filter by predicted values
        mask = (df_test['predicted'] >= min_val) & (df_test['predicted'] <= max_val)
        if not mask.any():
            fig.delaxes(ax)
            continue

        df_plot = df_test.loc[mask]
        y_true_f = df_plot['target'].values
        y_pred_f = df_plot['predicted'].values

        # compute metrics
        rmse = np.sqrt(mean_squared_error(y_true_f, y_pred_f))
        r2   = r2_score(y_true_f, y_pred_f)
        r_raw, _ = pearsonr(y_true_f, y_pred_f)
        r2p = r_raw**2

        # plot filtered data
        ax.plot(df_plot['Time_ISO'], df_plot['target'], label='Actual')
        ax.plot(df_plot['Time_ISO'], df_plot['predicted'], label='Predicted')
        ax.set_title(label, fontsize=10)
        ax.set_ylabel('O3 (ppb)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=6)

        # annotate metrics
        
        ax.text(
            0.05, 0.95,
            f"$r_{{pearson}}^2$ = {r2p:.2f}\n"
            f"$R^2$ = {r2:.2f}\n"
        
            f"RMSE = {rmse:.2f}",
            transform=ax.transAxes,
            va='top', ha='left', fontsize=14,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='none', edgecolor='black')

        )

        ax.legend(fontsize=8)

    # remove unused axes
    for i in range(n, len(axes)):
        if axes[i] in fig.axes:
            fig.delaxes(axes[i])

    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    fig.tight_layout()

    fname = f"lr_timeseries_pred-filtered_{min_val}_{max_val}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches='tight')
    plt.show()



# call them 
run_rf_all()
run_lr_all_filtered()






def run_rf_all_in_one_filtered(min_val=0, max_val=50, out_dir="figures"):

    os.makedirs(out_dir, exist_ok=True)

    feature_cols = ["temp", "rh", "o3", "no2"]
    splits = split_all_sensors(sensor_datasets, start_date, end_date, feature_cols)
    n = len(splits)
    colors = plt.cm.tab10(np.linspace(0, 1, n))

    fig, ax = plt.subplots(figsize=(12, 4))
    plotted_actual = False

    for (label, (_, test_df, X_train, y_train, X_test, y_test)), col in zip(splits.items(), colors):
        # train Random Forest
        pipe = make_pipeline(StandardScaler(),
                             RandomForestRegressor(n_estimators=100, random_state=42))
        pipe.fit(X_train, y_train)

        # predict on full test set
        y_pred = pipe.predict(X_test)

        # align in DataFrame
        df_test = test_df.reset_index(drop=True).copy()
        df_test['predicted'] = y_pred

        # filter by predicted values
        mask = (df_test['predicted'] >= min_val) & (df_test['predicted'] <= max_val)
        if not mask.any():
            print(f"{label} → no predictions in [{min_val}, {max_val}], skipping.")
            continue

        df_plot = df_test.loc[mask]
        y_true_f = df_plot['target'].values
        y_pred_f = df_plot['predicted'].values

        # compute and print metrics
        rmse = np.sqrt(mean_squared_error(y_true_f, y_pred_f))
        r2   = r2_score(y_true_f, y_pred_f)
        r_raw, _ = pearsonr(y_true_f, y_pred_f)
        r2p = r_raw**2
        print(f"{label} → RF pred-filtered [{min_val}, {max_val}]: "
              f"RMSE={rmse:.4f}, R²={r2:.4f}, pearson²={r2p:.4f}")

        # print feature importances
        rf = pipe.named_steps['randomforestregressor']
        importances = pd.Series(rf.feature_importances_, index=feature_cols)\
                        .sort_values(ascending=False)
        print("Feature importances:")
        for feat, imp in importances.items():
            print(f"  {feat:<6} {imp:.6f}")

        # surrogate linear equation
        y_rf_train = pipe.predict(X_train)
        sur = Ridge().fit(X_train, y_rf_train)
        terms = [f"{coef:+.4f}·{f}" for coef, f in zip(sur.coef_, feature_cols)]
        eqn = f"y = {sur.intercept_:+.4f} " + " ".join(terms)
        print(eqn, "\n")

        # plot actual once, with high zorder
        if not plotted_actual:
            ax.plot(df_plot['Time_ISO'], df_plot['target'],
                    color='k', label='49i', zorder=10)
            plotted_actual = True

        # plot this sensor's predictions below actual
        ax.plot(df_plot['Time_ISO'], df_plot['predicted'],
                label=f"{label} Predicted", color=col, zorder=1)

    # finalize plot
    ax.set_title(f"Filtered RF [{min_val}, {max_val}]")
    ax.set_xlabel("Time")
    ax.set_ylabel("O3 (ppb)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
    ax.legend(fontsize=8, ncol=2, loc='upper left')
    fig.tight_layout()

    # save
    fname = (f"rf_all-in-one_pred-filtered_{min_val}_{max_val}_"
             f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png")
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined filtered RF plot to {out_path}")
    plt.show()




def run_lr_all_in_one_filtered(min_val=0, max_val=50, out_dir="figures"):
    """
    Train Linear Regression on all sensors, filter predictions in [min_val, max_val],
    and plot all sensors' filtered predicted vs actual in one plot, printing metrics,
    coefficients, intercept (bias) for each sensor.
    """
    os.makedirs(out_dir, exist_ok=True)

    feature_cols = ["temp", "rh", "o3", "no2"]
    splits = split_all_sensors(sensor_datasets, start_date, end_date, feature_cols)
    n = len(splits)
    colors = plt.cm.tab10(np.linspace(0, 1, n))

    fig, ax = plt.subplots(figsize=(12, 4))
    plotted_actual = False

    for (label, (_, test_df, X_train, y_train, X_test, y_test)), col in zip(splits.items(), colors):
        pipe = make_pipeline(StandardScaler(), LinearRegression())
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        df_test = test_df.reset_index(drop=True).copy()
        df_test['predicted'] = y_pred

        mask = (df_test['predicted'] >= min_val) & (df_test['predicted'] <= max_val)
        if not mask.any():
            print(f"{label} → no predictions in [{min_val}, {max_val}], skipping.")
            continue

        df_plot = df_test.loc[mask]
        y_true_f = df_plot['target'].values
        y_pred_f = df_plot['predicted'].values

        # compute metrics
        rmse = np.sqrt(mean_squared_error(y_true_f, y_pred_f))
        r2   = r2_score(y_true_f, y_pred_f)
        r_raw, _ = pearsonr(y_true_f, y_pred_f)
        r2p = r_raw**2
        print(f"{label} → LR pred-filtered [{min_val}, {max_val}]: "
              f"RMSE={rmse:.4f}, R²={r2:.4f}, pearson²={r2p:.4f}")

        # coefficients and bias (intercept)
        lr = pipe.named_steps['linearregression']
        coefs = pd.Series(lr.coef_, index=feature_cols).sort_values(key=abs, ascending=False)
        print("Coefficients:")
        for feat, coef in coefs.items():
            print(f"  {feat:<6} {coef:+.4f}")
        bias = lr.intercept_
        print(f"Bias (intercept): {bias:+.4f}\n")

        # plot actual once
        if not plotted_actual:
            ax.plot(df_plot['Time_ISO'], df_plot['target'],
                    color='k', label='Actual')
            plotted_actual = True

        # plot this sensor's predictions
        z = 10 if label.lower() == "bcal" else 1
        ax.plot(df_plot['Time_ISO'], df_plot['predicted'],
                label=f"{label} Predicted", color=col, zorder=z)

    ax.set_title(f"Filtered LR [{min_val}, {max_val}]")
    ax.set_xlabel("Time")
    ax.set_ylabel("O3 (ppb)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
    ax.legend(fontsize=8, ncol=2, loc='upper left')
    fig.tight_layout()

    fname = (f"lr_all-in-one_pred-filtered_{min_val}_{max_val}_"
             f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png")
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined filtered LR plot to {out_path}")
    plt.show()




# call it
run_lr_all_in_one_filtered(min_val=0, max_val=100, out_dir="figures")
