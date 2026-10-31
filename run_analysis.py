"""Optional command-line analysis for the UCI household power text file."""
import sys
from pathlib import Path
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

if len(sys.argv) < 2:
    print("Usage: python run_analysis.py path\\to\\household_power_consumption.txt")
    raise SystemExit(1)

path = Path(sys.argv[1])
df = pd.read_csv(path, sep=";", na_values=["?"], low_memory=False)
df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")
df["timestamp"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str),
                                 dayfirst=True, errors="coerce")
df = df.dropna(subset=["timestamp", "Global_active_power"])
df["date"] = df["timestamp"].dt.floor("D")
daily = df.groupby("date")["Global_active_power"].sum().div(60).rename("consumption_kwh").reset_index()
daily["previous_day"] = daily["consumption_kwh"].shift(1)
daily["rolling_mean_7"] = daily["consumption_kwh"].rolling(7, min_periods=1).mean()
daily["rolling_std_7"] = daily["consumption_kwh"].rolling(7, min_periods=2).std().fillna(0)
daily["day_of_week"] = daily["date"].dt.dayofweek
daily["previous_rolling_mean_7"] = daily["consumption_kwh"].shift(1).rolling(7, min_periods=1).mean()
daily["previous_rolling_std_7"] = daily["consumption_kwh"].shift(1).rolling(7, min_periods=2).std().fillna(0)
features = ["consumption_kwh", "previous_day", "rolling_mean_7", "rolling_std_7", "day_of_week"]
model_data = daily[features].fillna(daily[features].median()).fillna(0)
daily["IF_anomaly"] = IsolationForest(contamination=0.05, random_state=42).fit_predict(model_data) == -1

# Forecasting methods use only values that would have been known before the
# consumption day. KNN is configured as a regression model because this data
# has no labelled consumption classes.
forecast_features = ["previous_day", "previous_rolling_mean_7", "previous_rolling_std_7", "day_of_week"]
usable = daily.dropna(subset=forecast_features)
if len(usable) >= 12:
    models = {
        "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=min(7, len(usable) - 1), weights="distance")),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, min_samples_leaf=3, random_state=42),
    }
    for name, model in models.items():
        model.fit(usable[forecast_features], usable["consumption_kwh"])
        daily.loc[usable.index, f"{name}_prediction"] = model.predict(usable[forecast_features])

cluster_features = ["consumption_kwh", "rolling_mean_7", "rolling_std_7", "day_of_week"]
cluster_data = daily[cluster_features].fillna(daily[cluster_features].median()).fillna(0)
cluster_count = min(3, len(cluster_data))
if cluster_count >= 2:
    daily["usage_cluster"] = KMeans(n_clusters=cluster_count, random_state=42, n_init=20).fit_predict(
        StandardScaler().fit_transform(cluster_data)
    )
out = Path("outputs")
out.mkdir(exist_ok=True)
daily.to_csv(out / "daily_anomaly_results.csv", index=False)
print(f"Saved {len(daily)} daily rows to {out / 'daily_anomaly_results.csv'}")
print(f"Isolation Forest flagged {daily['IF_anomaly'].sum()} days ({daily['IF_anomaly'].mean()*100:.2f}%).")
