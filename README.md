# Household Electricity Anomaly Detection Using AI

## What this project does
This project analyses daily electricity consumption, forecasts upcoming usage, groups recurring usage patterns, and flags unusual days.

Anomaly detection uses four methods:
- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM
- Z-score baseline

It also implements the following machine-learning methods:
- KNN regression, Linear Regression, and Decision Tree regression to predict daily consumption from historical values and day of week
- K-Means clustering to group daily electricity-use patterns

The Streamlit dashboard displays summary metrics, a daily consumption chart, model comparison, anomaly records, date filtering, and CSV export.

## Important data note
The included `data/sample_daily_consumption.csv` is **illustrative demo data** so the dashboard can open immediately for a presentation. It is not the original Kaggle/UCI household dataset and must not be presented as real measured results.

To analyse your downloaded `household_power_consumption.txt`, upload it using the dashboard sidebar. The file should contain the original columns `Date`, `Time`, and `Global_active_power`. The app aggregates minute-level active power (kW) into approximate daily kWh by summing readings and dividing by 60.

## Run on your personal laptop (Windows)
1. Install Python 3.10 or newer from https://www.python.org/downloads/ if it is not already installed.
2. Extract `household_anomaly_project.zip`.
3. Open the extracted folder.
4. Click the folder address bar in File Explorer, type `cmd`, and press Enter.
5. Run:
   ```bash
   py -m pip install -r requirements.txt
   py -m streamlit run app.py
   ```
   If `py` is not recognised, try `python` instead.
6. A browser page should open at `http://localhost:8501`. If it does not, copy that address into your browser.

## Run in Google Colab
The most dependable way to show the graphs without a tunnel is to run the analysis in the notebook and display the plots inline. Streamlit normally runs on the Colab virtual machine, so `localhost:8501` on your own laptop is not the same machine. Public tunnels may be blocked or unreliable. For a laptop presentation, running Streamlit locally is recommended.

## Input formats
- Original household file: `.txt` or `.csv` with `Date`, `Time`, and `Global_active_power`.
- Daily CSV: must contain `date` and `consumption_kwh` columns.

## Features
- Data upload
- Daily consumption aggregation for the original minute-level dataset
- Rolling mean and rolling standard deviation features
- KNN, Linear Regression, and Decision Tree forecast comparison with time-ordered holdout metrics
- K-Means usage-pattern clustering (2 to 6 groups, selected in the sidebar)
- Four anomaly detection methods
- Date filter
- Summary metrics and anomaly chart
- Model comparison table
- Downloadable CSV results

## Limitations
- The models are unsupervised; without ground-truth labels, anomaly percentage is not classification accuracy.
- This dataset represents whole-household consumption, not a separately metered AC or another single appliance.
- Incomplete days and missing readings can distort daily totals. Check data completeness before reporting conclusions.
- An anomaly is an unusual pattern, not proof of a faulty appliance.

## Suggested presentation explanation
"We use historical electricity consumption readings, aggregate them by day, create time-based and rolling statistical features, and apply four anomaly-detection methods. The dashboard highlights days that differ from learned usage patterns and lets a user inspect or download the results. The system is a prototype and requires verified labels for a formal accuracy evaluation."
