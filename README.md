# AI-Driven Short-Term Electricity Demand Forecasting for Smart Buildings
### A Machine Learning Approach to Sustainable Urban Energy Management

Forecasting hourly building electricity demand from meter, weather, and building-metadata,
comparing a statistical baseline, a gradient-boosting model, and a deep learning model —
and translating the forecasts into a **derived carbon-and-cost impact layer**.

> Final-year **Urban Computing** research project · ASHRAE Great Energy Predictor III dataset.

---

## 🎯 Overview

Buildings consume a large share of urban electricity and produce significant carbon emissions.
Their hourly demand is volatile (time of day, weekday, weather, building type), so smart-building
automation needs **accurate short-term forecasts** to cut avoidable waste. This project builds a
reproducible pipeline that:

1. Cleans and prepares the ASHRAE electricity data (20.2M → 11.6M readings).
2. Engineers leakage-safe calendar, lag, rolling, and building features.
3. Compares three models under a strict time-based protocol.
4. Interprets the best model with feature importance + SHAP.
5. **Derives** a CO₂ / cost layer and estimates the savings from forecast-driven automation.

---

## 📊 Key Results

**Full test set (1,413 buildings, Oct–Dec 2016):**

| Model | RMSE | MAE | MAPE % | RMSLE |
|---|---|---|---|---|
| Seasonal-naïve | **100.5** | 18.45 | 37.2 | 0.397 |
| **LightGBM** | 161.2 | **16.09** | **16.7** | **0.238** |

- LightGBM cuts **RMSLE by 40%** and **MAPE from 37% → 17%**; **R² = 0.97** (log), 0.81 (raw).
- Higher RMSE is a known trade-off of the log objective on skewed data (RMSLE is the official ASHRAE metric).

**Three-model comparison (identical 40-building subset, 77,534 test points):**

| Model | RMSE | MAE | MAPE % | RMSLE |
|---|---|---|---|---|
| Seasonal-naïve | 90.95 | 23.80 | 14.19 | 0.293 |
| **LightGBM** | **70.10** | **19.71** | **11.06** | 0.218 |
| LSTM | 153.90 | 35.13 | 11.69 | **0.199** |

> LightGBM is the strongest overall model and far cheaper to train than the LSTM.

**Carbon impact:** the portfolio emits ≈ **715,000 t CO₂/year** (band 357k–893k). Forecast-driven
automation could avoid ≈ **71,000 t CO₂ and \$27M/year** (5–15% band). Education accounts for **56%**
of total emissions.

---

## 📁 Repository Structure

| File | Purpose | Output |
|---|---|---|
| `finalize_dataset.ipynb` | Load, filter to electricity, merge, clean (unit fix, dead/outlier/weather) | `electricity_clean.parquet` |
| `EDA.ipynb` | Exploratory analysis (hourly, weekday, seasonal, building-type, temperature) | figures |
| `Feature_Engineering.ipynb` | Calendar, lag (24h/168h), rolling, metadata features | `model_ready.parquet` |
| `Modelling.ipynb` | Baseline vs LightGBM, frozen-meter fix, metrics, importance, SHAP | `model_ready_clean.parquet`, `test_predictions.parquet` |
| `Emissions.ipynb` | Derived CO₂/cost, sensitivity band, savings scenario | figures |
| `LSTM.ipynb` | Deep-learning model + fair 3-way comparison | — |
| `ashrae-energy-prediction/` | Raw dataset (see note below) | — |

---

## 🗂️ Dataset

**ASHRAE Great Energy Predictor III** — 1,449 buildings, 16 sites, hourly readings for 2016,
with per-site weather and building metadata. This project forecasts **electricity only** (meter type 0).

Download from Kaggle: <https://www.kaggle.com/c/ashrae-energy-prediction/data> and place the CSVs in
`ashrae-energy-prediction/`.

> ⚠️ **Note on large files:** `train.csv` (~1.2 GB) and `test.csv` exceed GitHub's 100 MB per-file
> limit. Do **not** commit them directly — either use **Git LFS** or keep them out of the repo (see
> `.gitignore` below) and download from Kaggle.

---

## 🔧 Pipeline Highlights

- **Unit correction:** Site 0 electricity is in kBTU → converted to kWh (×0.2931). Critical — otherwise
  emissions are 3.4× too high.
- **Cleaning:** removed dead-meter zero runs (459k rows) and frozen-meter runs (5.4%, 257 buildings),
  capped outliers, interpolated weather.
- **Features:** hour, weekday, month, weekend; lag-24h / lag-168h; 24h rolling mean (leakage-safe);
  temperature, dew point; log area, building type, building ID (global model).
- **Target:** `log(1 + kWh)` (raw skew ≈ 9.5).
- **Evaluation:** strict time-based split (train Jan–Sep, test Oct–Dec) — no leakage.

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the ASHRAE data into ashrae-energy-prediction/

# 3. Run the notebooks in order:
#    finalize_dataset → EDA → Feature_Engineering → Modelling → Emissions → LSTM
```

**Requirements** (`requirements.txt`):
```
pandas
numpy
pyarrow
scikit-learn
lightgbm
shap
matplotlib
seaborn
torch
```

---

## ⚠️ Limitations & Future Work
- One year of data; anonymised grids (hence the emission-factor band).
- Regime shifts and frozen-meter artefacts bound achievable accuracy.
- The LSTM is a compact, CPU-trained model — a larger GPU model may narrow the gap.
- Savings are scenario estimates from published ranges, not measured causal outcomes.
- **Future:** thermal meters, per-region emission factors, probabilistic forecasting, closed-loop control.

---

## 🛠️ Tech Stack
Python · pandas · NumPy · LightGBM · PyTorch · scikit-learn · SHAP · matplotlib · seaborn

## 👤 Author
**[Your Name]** — [Your University], Urban Computing

## 📄 License
Released under the MIT License (optional — add a `LICENSE` file if you want one).
