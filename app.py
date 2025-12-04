import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---- Streamlit Page Setup ----
st.set_page_config(page_title="Meru Potato Yield & Storage Intelligence", layout="wide")
st.title("🥔 Meru Potato Yield Forecasting (Option A - Datasets Only)")
st.markdown("""
Upload your **hvstat_meru_potato.csv** and **adm_meru_potato.csv** files to explore
EDA, yield prediction, and cold storage recommendations.
""")

# ---- File Upload Section ----
hv_file = st.file_uploader("Upload hvstat_meru_potato.csv", type=["csv"])
adm_file = st.file_uploader("Upload adm_meru_potato.csv", type=["csv"])

if hv_file is None:
    st.stop()

# Load datasets
hv = pd.read_csv(hv_file)
hv.columns = hv.columns.str.lower()

if adm_file:
    adm = pd.read_csv(adm_file)
    adm.columns = adm.columns.str.lower()

# ---- Helper Functions ----

def extract_year(df):
    if "harvest_year" in df.columns:
        return df["harvest_year"]
    if "period_date" in df.columns:
        return pd.to_datetime(df["period_date"], errors="coerce").dt.year
    if "start_date" in df.columns:
        return pd.to_datetime(df["start_date"], errors="coerce").dt.year
    return None

hv["year"] = extract_year(hv)

# ---- EDA Section ----
st.header("📊 Exploratory Data Analysis (HVSTAT)")

if "yield" in hv.columns:
    hv_year = hv.groupby("year")["yield"].mean()
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(hv_year.index, hv_year.values, marker="o")
    ax.set_title("Mean Potato Yield Over Time (Meru)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Yield (t/ha)")
    ax.grid(True)
    st.pyplot(fig)

if "area" in hv.columns and "production" in hv.columns:
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(hv["area"], hv["production"])
    ax.set_title("Area vs Production")
    ax.set_xlabel("Area (ha)")
    ax.set_ylabel("Production (tonnes)")
    ax.grid(True)
    st.pyplot(fig)

if "admin_2" in hv.columns:
    sub = hv.groupby("admin_2")["yield"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(12,4))
    sub.plot(kind="bar", ax=ax)
    ax.set_title("Yield by Sub-county (Meru)")
    ax.set_ylabel("Yield (t/ha)")
    plt.xticks(rotation=45)
    st.pyplot(fig)

st.markdown("### Summary Statistics")
st.write(hv[["year","area","production","yield"]].describe())

# ---- MODEL PREPARATION ----
st.header("🤖 Yield Modeling (Option A)")

grouped = hv.groupby("year").agg({
    "area": "sum",
    "production": "sum"
}).reset_index()

grouped["yield"] = grouped["production"] / grouped["area"]
grouped = grouped.sort_values("year").reset_index(drop=True)

st.write("### Aggregated Dataset")
st.dataframe(grouped)

# Feature engineering
df = grouped.copy()
for lag in [1,2,3]:
    df[f"yield_lag_{lag}"] = df["yield"].shift(lag)

df["yield_roll3"] = df["yield"].rolling(3).mean().shift(1)
df = df.dropna().reset_index(drop=True)

FEATURES = ["yield_lag_1","yield_lag_2","yield_lag_3","yield_roll3"]

# Train-test split (last year → test)
train = df.iloc[:-1]
test = df.iloc[-1:]

X_train = train[FEATURES]
y_train = train["yield"]
X_test = test[FEATURES]
y_test = test["yield"]

# Train models
lr = LinearRegression().fit(X_train, y_train)
rf = RandomForestRegressor(n_estimators=300, random_state=42).fit(X_train, y_train)

lr_pred = lr.predict(X_test)[0]
rf_pred = rf.predict(X_test)[0]

def metrics(y, yhat):
    mae = mean_absolute_error([y], [yhat])
    mse = mean_squared_error([y], [yhat])
    rmse = np.sqrt(mse)  # Calculate RMSE manually
    mape = np.mean(np.abs((y - yhat) / y)) * 100
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape
    }

st.subheader("📈 Model Performance on Test Year")
st.write("### Linear Regression")
st.write(metrics(y_test, lr_pred))
st.write("### Random Forest")
st.write(metrics(y_test, rf_pred))

# ---- FORECAST NEXT YEAR ----
st.header("📅 Forecast Next Year Yield")

last = grouped.iloc[-1]

vals = {
    "yield_lag_1": grouped["yield"].iloc[-1],
    "yield_lag_2": grouped["yield"].iloc[-2],
    "yield_lag_3": grouped["yield"].iloc[-3],
    "yield_roll3": grouped["yield"].iloc[-3:].mean()
}

X_fore = np.array(list(vals.values())).reshape(1,-1)

rf_forecast = rf.predict(X_fore)[0]

st.metric("Forecasted Yield (t/ha)", f"{rf_forecast:.2f}")

predicted_tonnage = rf_forecast * last["area"]
st.metric("Forecasted Production (tonnes)", f"{predicted_tonnage:,.0f}")

# ---- STORAGE ENGINE ----
st.header("❄️ Cold Storage Requirement Engine")

def pack_storage(total_tonnes, chamber_sizes=[1000,500,250], max_fill=0.9):
    required_capacity = int(np.ceil(total_tonnes / max_fill))
    allocation = []
    remaining = required_capacity

    for size in chamber_sizes:
        count = remaining // size
        if count > 0:
            allocation.append({"size": size, "count": count})
            remaining -= count * size

    if remaining > 0:
        allocation.append({"size": chamber_sizes[-1], "count": 1})

    capacity = sum(a["size"] * a["count"] for a in allocation)
    utilization = total_tonnes / capacity

    return {
        "predicted_tonnes": total_tonnes,
        "required_capacity": required_capacity,
        "allocation": allocation,
        "total_allocated": capacity,
        "utilization": utilization
    }

plan = pack_storage(predicted_tonnage)

st.subheader("📦 Recommended Storage Allocation")
st.json(plan)

# ---- DOWNLOAD RESULTS ----
st.header("⬇ Download Results")

results = {
    "forecast_yield_t_per_ha": rf_forecast,
    "predicted_tonnage": predicted_tonnage,
    "storage_plan": plan
}

st.download_button(
    "Download Forecast + Storage Plan (JSON)",
    data=str(results),
    file_name="meru_yield_storage_plan.json",
    mime="application/json"
)
