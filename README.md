# 🥔 Meru Potato Yield & Cold Storage Intelligence (Option A)

This repository contains a Streamlit application for:
- Historical potato yield analysis (Meru County)
- Yield forecasting (Linear Regression + RandomForest)
- Cold storage recommendation engine

## 🚀 Running Locally

Install dependencies:
pip install -r requirements.txt


Run Streamlit app:
streamlit run app.py


## 📂 Upload Required Datasets
Upload the following files when the app runs:

- `hvstat_meru_potato.csv`
- `adm_meru_potato.csv`

## 📊 Features
- Exploratory Data Analysis (HVSTAT)
- Yield modeling (lags + rolling mean)
- Next-year forecast
- Cold storage chamber sizing (1000, 500, 250 MT)

## 📦 Output
The app generates:
- Yield forecast
- Recommended cold storage allocation
- Downloadable JSON report

---

## 🔮 Roadmap
Option B will extend the system using:
- NDVI (Sentinel-2)
- Weather forecasts
- More advanced modeling
