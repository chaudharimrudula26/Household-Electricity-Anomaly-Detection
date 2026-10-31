import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeRegressor

BASE_DIR = Path(__file__).parent
DEMO_FILE = BASE_DIR / "data" / "sample_daily_consumption.csv"

st.set_page_config(page_title="Household Electricity Anomaly Detection", page_icon="⚡", layout="wide")
st.title("⚡ Household Electricity Anomaly Detection Using AI")
st.caption("Daily electricity-consumption monitoring with unsupervised anomaly-detection methods.")

def make_demo_data():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=180, freq="D")
    weekly = 7 + 1.2 * np.sin(np.arange(len(dates)) * 2 * np.pi / 7)
    trend = np.linspace(0, 0.8, len(dates))
    values = weekly + trend + rng.normal(0, 0.65, len(dates))
    # Illustrative spikes for a dashboard demo only
    for idx, amount in [(23, 5.5), (61, 4.2), (99, 6.0), (142, 5.0), (165, 4.7)]:
        values[idx] += amount
    return pd.DataFrame({"date": dates, "consumption_kwh": np.maximum(values, 0.2)})

def parse_uploaded_file(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded, sep=None, engine="python")
    else:
        # UCI household_power_consumption.txt is semicolon-delimited and uses '?' for missing values.
        df = pd.read_csv(uploaded, sep=";", na_values=["?"], low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}

    if "date" in lower and "time" in lower and "global_active_power" in lower:
        date_col, time_col, power_col = lower["date"], lower["time"], lower["global_active_power"]
        df[power_col] = pd.to_numeric(df[power_col], errors="coerce")
        df["timestamp"] = pd.to_datetime(
            df[date_col].astype(str) + " " + df[time_col].astype(str),
            dayfirst=True, errors="coerce"
        )
        df = df.dropna(subset=["timestamp", power_col])
        if df.empty:
            raise ValueError("No valid Date/Time/Global_active_power records found.")
        # Dataset power is in kW; readings are typically one minute apart.
        df["date"] = df["timestamp"].dt.floor("D")
        daily = df.groupby("date")[power_col].sum(min_count=1).reset_index()
        daily["consumption_kwh"] = daily[power_col] / 60.0
        return daily[["date", "consumption_kwh"]].dropna()

    date_col = lower.get("date")
    consumption_col = lower.get("consumption_kwh") or lower.get("consumption") or lower.get("daily_kwh")
    if date_col and consumption_col:
        out = df[[date_col, consumption_col]].copy()
        out.columns = ["date", "consumption_kwh"]
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["consumption_kwh"] = pd.to_numeric(out["consumption_kwh"], errors="coerce")
        return out.dropna().sort_values("date")
    raise ValueError(
        "Could not recognise the file. Upload the original household_power_consumption.txt "
        "with Date, Time and Global_active_power columns, or a CSV with date and consumption_kwh."
    )

def add_features(df):
    out = df.copy().sort_values("date").drop_duplicates("date")
    out["day_of_week"] = out["date"].dt.dayofweek
    out["previous_day"] = out["consumption_kwh"].shift(1)
    out["rolling_mean_7"] = out["consumption_kwh"].rolling(7, min_periods=1).mean()
    out["rolling_std_7"] = out["consumption_kwh"].rolling(7, min_periods=2).std().fillna(0)
    out["change_from_rolling_mean"] = out["consumption_kwh"] - out["rolling_mean_7"]
    # Shifted features are used by forecasting models so they only use information
    # available before the day being predicted.
    out["previous_rolling_mean_7"] = out["consumption_kwh"].shift(1).rolling(7, min_periods=1).mean()
    out["previous_rolling_std_7"] = out["consumption_kwh"].shift(1).rolling(7, min_periods=2).std().fillna(0)
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["consumption_kwh"])

def detect(df, contamination):
    out = df.copy()
    features = ["consumption_kwh", "day_of_week", "previous_day", "rolling_mean_7", "rolling_std_7"]
    model_data = out[features].copy()
    model_data = model_data.fillna(model_data.median(numeric_only=True)).fillna(0)
    n = len(out)
    if n < 10:
        out["IF_anomaly"] = False
        out["LOF_anomaly"] = False
        out["SVM_anomaly"] = False
        out["ZScore_anomaly"] = False
        return out
    out["IF_anomaly"] = IsolationForest(contamination=contamination, random_state=42, n_estimators=150).fit_predict(model_data) == -1
    out["LOF_anomaly"] = LocalOutlierFactor(n_neighbors=min(20, n - 1), contamination=contamination).fit_predict(model_data) == -1
    out["SVM_anomaly"] = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale").fit_predict(model_data) == -1
    mean = out["consumption_kwh"].mean()
    std = out["consumption_kwh"].std(ddof=0)
    out["ZScore_anomaly"] = ((out["consumption_kwh"] - mean).abs() > 3 * std) if std else False
    return out

def add_ml_results(df, n_clusters):
    """Add consumption forecasts and K-Means usage-pattern labels.

    Regression is evaluated on the latest 20% of days to preserve time order.
    The models are then fitted on all available days for dashboard predictions.
    """
    out = df.copy()
    forecast_features = ["previous_day", "previous_rolling_mean_7", "previous_rolling_std_7", "day_of_week"]
    cluster_features = ["consumption_kwh", "rolling_mean_7", "rolling_std_7", "day_of_week"]
    usable = out.dropna(subset=forecast_features).copy()
    models = {
        "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=max(1, min(7, len(usable) - 1)), weights="distance")),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, min_samples_leaf=3, random_state=42),
    }
    scores = []
    for name, model in models.items():
        out[f"{name}_prediction"] = np.nan
        if len(usable) < 12:
            scores.append({"Model": name, "MAE (kWh)": np.nan, "RMSE (kWh)": np.nan, "R²": np.nan})
            continue
        split = max(8, int(len(usable) * 0.8))
        split = min(split, len(usable) - 2)
        train, test = usable.iloc[:split], usable.iloc[split:]
        model.fit(train[forecast_features], train["consumption_kwh"])
        predicted = model.predict(test[forecast_features])
        scores.append({
            "Model": name,
            "MAE (kWh)": round(float(mean_absolute_error(test["consumption_kwh"], predicted)), 3),
            "RMSE (kWh)": round(float(mean_squared_error(test["consumption_kwh"], predicted) ** 0.5), 3),
            "R²": round(float(r2_score(test["consumption_kwh"], predicted)), 3),
        })
        model.fit(usable[forecast_features], usable["consumption_kwh"])
        out.loc[usable.index, f"{name}_prediction"] = model.predict(usable[forecast_features])

    cluster_data = out[cluster_features].fillna(out[cluster_features].median(numeric_only=True)).fillna(0)
    actual_clusters = min(n_clusters, len(cluster_data))
    if actual_clusters >= 2:
        out["usage_cluster"] = KMeans(n_clusters=actual_clusters, random_state=42, n_init=20).fit_predict(
            StandardScaler().fit_transform(cluster_data)
        )
        means = out.groupby("usage_cluster")["consumption_kwh"].mean().sort_values()
        names = {cluster: f"Usage pattern {position + 1}" for position, cluster in enumerate(means.index)}
        out["usage_cluster_label"] = out["usage_cluster"].map(names)
    else:
        out["usage_cluster"] = 0
        out["usage_cluster_label"] = "Usage pattern 1"
    return out, pd.DataFrame(scores)

st.sidebar.header("Project controls")
uploaded = st.sidebar.file_uploader("Upload household_power_consumption.txt or a daily CSV", type=["txt", "csv"])
contamination = st.sidebar.slider("Expected anomaly fraction (ML models)", min_value=0.01, max_value=0.15, value=0.05, step=0.01)
n_clusters = st.sidebar.slider("K-Means usage patterns", min_value=2, max_value=6, value=3)
use_demo = uploaded is None
if use_demo:
    st.sidebar.info("Showing included illustrative demo data. Upload your Kaggle/UCI dataset for real analysis.")
    # Keep the dashboard usable if the optional packaged CSV is absent.
    raw = pd.read_csv(DEMO_FILE, parse_dates=["date"]) if DEMO_FILE.exists() else make_demo_data()
else:
    try:
        raw = parse_uploaded_file(uploaded)
    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")
        st.stop()

if len(raw) == 0:
    st.error("The file contains no usable rows.")
    st.stop()

raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
raw["consumption_kwh"] = pd.to_numeric(raw["consumption_kwh"], errors="coerce")
raw = raw.dropna(subset=["date", "consumption_kwh"]).sort_values("date")
daily = add_features(raw)
results = detect(daily, contamination)
results, forecast_scores = add_ml_results(results, n_clusters)
source_label = "Illustrative demo data" if use_demo else "Uploaded dataset"

st.caption(f"Data source: **{source_label}** · {len(results):,} daily records")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Days analysed", f"{len(results):,}")
c2.metric("Isolation Forest anomalies", f"{int(results['IF_anomaly'].sum()):,}")
c3.metric("Anomaly rate", f"{results['IF_anomaly'].mean() * 100:.2f}%")
c4.metric("Average daily consumption", f"{results['consumption_kwh'].mean():.2f} kWh")

min_date = results["date"].min().date()
max_date = results["date"].max().date()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
filtered = results
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    filtered = results[(results["date"].dt.date >= date_range[0]) & (results["date"].dt.date <= date_range[1])]

st.subheader("Daily consumption and detected anomalies")
plot_df = filtered.copy()
plot_df["Status"] = np.where(plot_df["IF_anomaly"], "Detected anomaly", "Normal")
fig = px.line(plot_df, x="date", y="consumption_kwh", title="Daily electricity consumption", markers=True)
anoms = plot_df[plot_df["IF_anomaly"]]
if not anoms.empty:
    fig.add_scatter(x=anoms["date"], y=anoms["consumption_kwh"], mode="markers",
                    name="Isolation Forest anomaly", marker=dict(size=10, symbol="x"))
fig.update_layout(xaxis_title="Date", yaxis_title="Consumption (kWh)", legend_title="")
st.plotly_chart(fig, use_container_width=True)

st.subheader("KNN, Linear Regression, and Decision Tree forecasts")
st.caption("Each model predicts a day's consumption from only earlier consumption and day-of-week features. Metrics use the newest 20% of records as a time-ordered test set.")
st.dataframe(forecast_scores, use_container_width=True, hide_index=True)
forecast_cols = ["date", "consumption_kwh", "KNN_prediction", "Linear Regression_prediction", "Decision Tree_prediction"]
forecast_plot = filtered.dropna(subset=["KNN_prediction"]).copy()
if not forecast_plot.empty:
    forecast_long = forecast_plot[forecast_cols].melt(id_vars=["date"], var_name="Series", value_name="Consumption (kWh)")
    forecast_long["Series"] = forecast_long["Series"].replace({
        "consumption_kwh": "Actual consumption", "KNN_prediction": "KNN", 
        "Linear Regression_prediction": "Linear Regression", "Decision Tree_prediction": "Decision Tree",
    })
    st.plotly_chart(px.line(forecast_long, x="date", y="Consumption (kWh)", color="Series",
                            title="Actual consumption and model forecasts"), use_container_width=True)
else:
    st.info("At least 12 daily records are needed to train and evaluate the forecasting models.")

st.subheader("K-Means consumption patterns")
cluster_summary = (filtered.groupby("usage_cluster_label", as_index=False)
                   .agg(Days=("date", "count"), Average_consumption_kwh=("consumption_kwh", "mean"))
                   .sort_values("Average_consumption_kwh"))
cluster_summary["Average_consumption_kwh"] = cluster_summary["Average_consumption_kwh"].round(2)
st.dataframe(cluster_summary, use_container_width=True, hide_index=True)
cluster_fig = px.scatter(filtered, x="date", y="consumption_kwh", color="usage_cluster_label",
                         title="Daily consumption grouped by K-Means", labels={"usage_cluster_label": "Usage pattern"})
st.plotly_chart(cluster_fig, use_container_width=True)

st.subheader("Model comparison")
model_cols = ["IF_anomaly", "LOF_anomaly", "SVM_anomaly", "ZScore_anomaly"]
labels = {"IF_anomaly": "Isolation Forest", "LOF_anomaly": "Local Outlier Factor",
          "SVM_anomaly": "One-Class SVM", "ZScore_anomaly": "Z-score"}
comparison = pd.DataFrame([
    {"Model": labels[col], "Anomaly days": int(filtered[col].sum()),
     "Normal days": int(len(filtered) - filtered[col].sum()),
     "Anomaly percentage": round(float(filtered[col].mean() * 100), 2)}
    for col in model_cols
])
st.dataframe(comparison, use_container_width=True, hide_index=True)

st.subheader("Detected anomaly records")
anomaly_rows = filtered[filtered["IF_anomaly"]].copy()
show_cols = ["date", "consumption_kwh", "rolling_mean_7", "rolling_std_7", "IF_anomaly"]
if anomaly_rows.empty:
    st.info("No Isolation Forest anomalies found in this date range. Try widening the date range or adjusting the expected anomaly fraction.")
else:
    st.dataframe(anomaly_rows[show_cols].sort_values("date", ascending=False), use_container_width=True, hide_index=True)

with st.expander("View all daily records"):
    st.dataframe(filtered[["date", "consumption_kwh", "day_of_week", "previous_day", "rolling_mean_7", "rolling_std_7", "usage_cluster_label", "KNN_prediction", "Linear Regression_prediction", "Decision Tree_prediction"]], use_container_width=True, hide_index=True)

csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button("Download analysed daily data (CSV)", data=csv, file_name="electricity_anomaly_results.csv", mime="text/csv")
st.caption("Important: these models identify unusual patterns; without verified ground-truth labels, anomaly detection percentage is not model accuracy.")
